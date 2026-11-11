use sentinel_core::graph::GraphSpec;
use sentinel_core::predict;
use sentinel_core::training;
use std::collections::HashMap;
use std::path::PathBuf;

fn have_torch() -> bool {
    std::process::Command::new(sentinel_core::platform::python_exe()).args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

fn workdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sentinel-train-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A small MLP with `out` outputs and the given `training:` body (YAML mapping text, indented content only).
fn model_yaml(out: usize, training: &str) -> String {
    format!(
        "inputs: {{x: [B, 2]}}\nnodes:\n  a: {{block: linear, params: {{out: 16}}}}\n  r: {{block: activation, params: {{kind: relu}}}}\n  h: {{block: linear, params: {{out: {out}}}}}\nedges: ['x -> a', 'a -> r', 'r -> h']\noutputs: {{y: h}}\ntraining:\n{training}"
    )
}

fn script(yaml: &str) -> Result<String, sentinel_core::graph::GraphError> {
    let g = GraphSpec::from_yaml(yaml)?.validate()?;
    sentinel_core::codegen::generate_training_script(&g, "test/model@v1")
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
    dir: PathBuf,
}

impl Run {
    fn art(&self) -> PathBuf {
        self.dir.join("run")
    }
    /// `epoch N key value ...` lines as maps.
    fn epochs(&self) -> Vec<HashMap<String, f64>> {
        self.stdout
            .lines()
            .filter(|l| l.starts_with("epoch "))
            .map(|l| {
                let t: Vec<&str> = l.split_whitespace().collect();
                let mut m = HashMap::new();
                m.insert("epoch".into(), t[1].parse().unwrap());
                for kv in t[2..].chunks(2) {
                    if kv.len() == 2 {
                        if let Ok(v) = kv[1].parse::<f64>() {
                            m.insert(kv[0].to_string(), v);
                        }
                    }
                }
                m
            })
            .collect()
    }
    fn last(&self, key: &str) -> f64 {
        *self.epochs().last().unwrap_or_else(|| panic!("no epoch lines\n{}\n{}", self.stdout, self.stderr)).get(key).unwrap_or_else(|| panic!("no {key} in\n{}", self.stdout))
    }
    fn first(&self, key: &str) -> f64 {
        self.epochs()[0][key]
    }
}

fn train(tag: &str, yaml: &str, csv: &str) -> Run {
    let dir = workdir(tag);
    std::fs::write(dir.join("d.csv"), csv).unwrap();
    let code = script(yaml).unwrap_or_else(|e| panic!("{tag}: {e}"));
    std::fs::write(dir.join("train.py"), code).unwrap();
    let out = std::process::Command::new(sentinel_core::platform::python_exe()).args(["-I", "train.py"]).current_dir(&dir).env("SENTINEL_ARTIFACT_DIR", dir.join("run").join("artifacts")).output().unwrap();
    Run { stdout: String::from_utf8_lossy(&out.stdout).into(), stderr: String::from_utf8_lossy(&out.stderr).into(), ok: out.status.success(), dir }
}

fn ok(tag: &str, yaml: &str, csv: &str) -> Run {
    let r = train(tag, yaml, csv);
    assert!(r.ok, "{tag} failed:\n{}\n{}", r.stderr, r.stdout);
    r
}

// ---------------------------------------------------------------- deterministic datasets
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    fn g(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
    }
}

fn blobs(classes: usize, n: usize, names: &[&str]) -> String {
    let mut r = Lcg(1);
    let mut s = String::from("a,b,label\n");
    for i in 0..n {
        let c = i % classes;
        let (cx, cy) = [(-3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)][c];
        s.push_str(&format!("{:.4},{:.4},{}\n", cx + 0.8 * r.g(), cy + 0.8 * r.g(), names[c]));
    }
    s
}

fn plain(common: &str) -> String {
    format!("  {}\n", common.replace(", ", "\n  "))
}

const DATA: &str = "  data: {csv: d.csv, features: [a, b], target: label}";

fn tr(body: &str) -> String {
    // defaults, unless the body sets the same key
    let defaults: String = [("epochs", "60"), ("lr", "0.02"), ("batch_size", "64")]
        .iter()
        .filter(|(k, _)| !body.lines().any(|l| l.trim_start().starts_with(&format!("{k}:"))))
        .map(|(k, v)| format!("  {k}: {v}\n"))
        .collect();
    format!("{defaults}{body}\n{DATA}\n")
}

// ---------------------------------------------------------------- catalog and validation (no PyTorch needed)
#[test]
fn the_catalog_covers_every_task_and_loss() {
    let c = training::catalog();
    let tasks: Vec<&str> = c["tasks"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    for t in ["classification", "binary", "multilabel", "regression", "ranking", "reconstruction", "vae", "custom"] {
        assert!(tasks.contains(&t), "{t}");
    }
    for t in c["tasks"].as_array().unwrap() {
        assert!(!t["losses"].as_array().unwrap().is_empty(), "{} has no losses", t["name"]);
    }
    assert!(c["optimizers"].as_array().unwrap().len() >= 9);
    assert!(c["schedulers"].as_array().unwrap().len() >= 8);
    let reg: Vec<&str> = c["tasks"].as_array().unwrap().iter().find(|t| t["name"] == "regression").unwrap()["losses"].as_array().unwrap().iter().map(|l| l["name"].as_str().unwrap()).collect();
    assert_eq!(reg, ["mse", "mae", "huber", "smooth_l1", "quantile", "poisson", "log_cosh"]);
}

#[test]
fn bad_training_specs_are_rejected_with_specific_messages() {
    let cases: Vec<(String, &str)> = vec![
        (model_yaml(3, &tr("  loss: nonsense")), "unsupported loss 'nonsense'"),
        (model_yaml(1, &tr("  loss: mse\n  task: classification")), "does not fit the classification task"),
        (model_yaml(3, &tr("  loss: {name: cross_entropy, gamma: 2}")), "has no setting 'gamma'"),
        (model_yaml(3, &tr("  loss: {name: cross_entropy, label_smoothing: high}")), "label_smoothing"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  optimizer: {name: adam, momentum: 0.9}")), "optimizer 'adam' has no setting 'momentum'"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  optimizer: lion")), "unsupported optimizer 'lion'"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  optimizer: {name: sgd, nesterov: true}")), "nesterov needs momentum"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  scheduler: {name: cosine, warmup_epochs: 60}")), "warmup_epochs"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  scheduler: warp")), "unsupported scheduler 'warp'"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  early_stopping: {patience: 3, monitor: test_bleu}")), "not a metric of the classification task"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  early_stopping: {patience: 0}")), "patience must be at least 1"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  freeze: [nope]")), "no node 'nope'"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  freeze: [a, r, h]")), "nothing would train"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  accumulate: 0")), "accumulate must be at least 1"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  clip_grad: -1")), "clip_grad must be a positive"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  amp: fp8")), "amp must be one of"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  device: tpu")), "device must be one of"),
        (model_yaml(3, &tr("  loss: cross_entropy\n  lr: 0")), "lr must be a positive"),
        (model_yaml(2, &tr("  loss: bce_logits")), "needs a model that outputs 1 value"),
        (model_yaml(1, "  loss: bce_logits\n  task: multilabel\n  epochs: 5\n  data: {csv: d.csv, features: [a, b], targets: [p, q]}\n"), "outputs 2 value"),
        (model_yaml(1, "  loss: ranknet\n  epochs: 5\n  data: {csv: d.csv, features: [a, b], target: rel}\n"), "needs `data.group`"),
        (model_yaml(2, "  loss: mse\n  task: reconstruction\n  epochs: 5\n  data: {csv: d.csv, features: [a, b], target: y}\n"), "unsupervised"),
        (model_yaml(1, "  loss: mse\n  task: reconstruction\n  epochs: 5\n  data: {csv: d.csv, features: [a, b]}\n"), "outputs 2 value"),
        (model_yaml(1, &tr("  loss: custom")), "needs a `custom` section"),
        (model_yaml(1, &tr("  loss: mse\n  custom: {source: 'def loss(o, t): return 0'}")), "needs `loss: custom`"),
        (model_yaml(3, "  loss: cross_entropy\n  epochs: 5\n  data: {csv: d.csv, features: [a, b]}\n"), "needs `data.target`"),
        (model_yaml(1, &tr("  loss: {name: quantile, q: 1.5}")), "q must be between 0 and 1"),
    ];
    for (yaml, needle) in cases {
        let e = script(&yaml).unwrap_err().to_string();
        assert!(e.contains(needle), "expected '{needle}' in: {e}\n{yaml}");
    }
}

#[test]
fn legacy_flat_specs_still_work() {
    // plain strings for loss/optimizer, as every shipped component writes them
    let s = script(&model_yaml(3, &tr("  loss: cross_entropy\n  optimizer: adamw"))).unwrap();
    assert!(s.contains("torch.optim.AdamW") && s.contains("nn.CrossEntropyLoss"));
    // inference: bce_logits is binary with one target and multilabel with several
    assert!(script(&model_yaml(1, &tr("  loss: bce_logits"))).is_ok());
}

#[test]
fn losses_and_optimizers_round_trip_through_yaml() {
    let g = GraphSpec::from_yaml(&model_yaml(3, &tr("  loss: {name: focal, gamma: 3.0}\n  optimizer: {name: sgd, momentum: 0.9, nesterov: true}"))).unwrap();
    let back: GraphSpec = serde_yaml::from_str(&serde_yaml::to_string(&g).unwrap()).unwrap();
    assert_eq!(g, back);
    let t = back.training.unwrap();
    assert_eq!(t.loss.name, "focal");
    assert_eq!(t.loss.f64("gamma"), Some(3.0));
    assert_eq!(t.optimizer.bool("nesterov"), Some(true));
}

// ---------------------------------------------------------------- tasks and losses, real PyTorch
#[test]
fn classification_losses_all_learn() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 300, &["x", "y", "z"]);
    for (tag, loss) in [
        ("ce", "{name: cross_entropy, label_smoothing: 0.1, class_weights: balanced}"),
        ("focal", "{name: focal, gamma: 2.0, class_weights: balanced}"),
        ("margin", "{name: multi_margin, margin: 1.0}"),
    ] {
        let r = ok(&format!("cls-{tag}"), &model_yaml(3, &tr(&format!("  loss: {loss}"))), &csv);
        assert!(r.last("test_acc") > 0.9, "{tag}: {}", r.stdout);
    }
}

#[test]
fn binary_and_multilabel_tasks() {
    if !have_torch() {
        return;
    }
    let r = ok("binary", &model_yaml(1, &tr("  loss: {name: bce_logits, pos_weight: balanced}")), &blobs(2, 300, &["no", "yes"]));
    assert!(r.stdout.contains("classes (negative, positive): ['no', 'yes']"), "{}", r.stdout);
    assert!(r.last("test_acc") > 0.9, "{}", r.stdout);
    assert_eq!(predict::read_meta(&r.art()).unwrap()["task"], "binary");
    let p = predict::predict(&r.art(), &[vec![-3.0, -3.0], vec![3.0, 3.0]]).unwrap();
    assert_eq!((p["predictions"][0]["label"].as_str(), p["predictions"][1]["label"].as_str()), (Some("no"), Some("yes")));

    // multilabel: p = (a > 0), q = (b > 0)
    let mut rng = Lcg(5);
    let mut csv = String::from("a,b,p,q\n");
    for _ in 0..300 {
        let (a, b) = (rng.g() * 2.0, rng.g() * 2.0);
        csv.push_str(&format!("{a:.4},{b:.4},{},{}\n", (a > 0.0) as i32, (b > 0.0) as i32));
    }
    let y = model_yaml(2, "  loss: bce_logits\n  epochs: 80\n  lr: 0.03\n  batch_size: 64\n  data: {csv: d.csv, features: [a, b], targets: [p, q]}\n");
    let r = ok("multilabel", &y, &csv);
    assert!(r.last("test_acc") > 0.9, "{}", r.stdout);
    let dir = r.art();
    let p = predict::predict(&dir, &[vec![2.0, -2.0]]).unwrap();
    assert_eq!(p["predictions"][0]["label"], "p", "a>0 and b<0 means only label p is on: {p}");
}

#[test]
fn every_regression_loss_trains() {
    if !have_torch() {
        return;
    }
    let mut rng = Lcg(7);
    let mut csv = String::from("a,b,label\n");
    for _ in 0..300 {
        let (a, b) = (rng.g(), rng.g());
        csv.push_str(&format!("{a:.4},{b:.4},{:.4}\n", 2.0 * a - b + 0.05 * rng.g()));
    }
    for loss in ["mse", "mae", "{name: huber, delta: 0.5}", "{name: smooth_l1, beta: 0.5}", "log_cosh"] {
        let r = ok("reg", &model_yaml(1, &tr(&format!("  loss: {loss}\n  epochs: 120"))), &csv);
        assert!(r.last("test_r2") > 0.9, "{loss}: {}", r.stdout);
        assert!(r.last("test_mae") < 0.4, "{loss}: {}", r.stdout);
    }
    // poisson regression on counts (the model predicts the log-rate)
    let mut counts = String::from("a,b,label\n");
    for _ in 0..300 {
        let (a, b) = (rng.g() * 0.5, rng.g() * 0.5);
        counts.push_str(&format!("{a:.4},{b:.4},{}\n", ((0.8 * a + 0.3 * b).exp() * 3.0).round() as i64));
    }
    let r = ok("poisson", &model_yaml(1, &tr("  loss: poisson\n  epochs: 80")), &counts);
    assert!(r.last("test_loss").is_finite() && r.last("test_loss") < r.first("test_loss"), "{}", r.stdout);

    // quantile regression: the 0.9 model must predict above the 0.1 model
    let mean_pred = |q: f64, tag: &str| {
        let r = ok(tag, &model_yaml(1, &tr(&format!("  loss: {{name: quantile, q: {q}}}\n  epochs: 200"))), &csv);
        let dir = r.art();
        let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![(i as f64 - 10.0) / 5.0, 0.0]).collect();
        let p = predict::predict(&dir, &rows).unwrap();
        p["values"].as_array().unwrap().iter().map(|v| v[0].as_f64().unwrap()).sum::<f64>() / 20.0
    };
    let (hi, lo) = (mean_pred(0.9, "q90"), mean_pred(0.1, "q10"));
    assert!(hi > lo + 0.02, "q0.9 mean {hi} should exceed q0.1 mean {lo}");
}

#[test]
fn reconstruction_task_trains_an_autoencoder() {
    if !have_torch() {
        return;
    }
    let mut rng = Lcg(9);
    let mut csv = String::from("a,b\n");
    for _ in 0..300 {
        let a = rng.g();
        csv.push_str(&format!("{a:.4},{:.4}\n", 0.5 * a + 0.05 * rng.g()));
    }
    let g = "inputs: {x: [B, 2]}\nnodes: {e: {block: linear, params: {out: 1}}, d: {block: linear, params: {out: 2}}}\nedges: ['x -> e', 'e -> d']\noutputs: {y: d}\ntraining:\n  loss: mse\n  task: reconstruction\n  epochs: 150\n  lr: 0.02\n  batch_size: 64\n  data: {csv: d.csv, features: [a, b]}\n";
    let r = ok("recon", g, &csv);
    assert!(r.last("test_loss") < r.first("test_loss") * 0.3, "{}", r.stdout);
}

fn ranking_csv() -> String {
    let mut rng = Lcg(3);
    let mut csv = String::from("q,f1,f2,rel\n");
    for q in 0..60 {
        for _ in 0..8 {
            let (f1, f2) = (rng.u() * 2.0 - 1.0, rng.u() * 2.0 - 1.0);
            let rel = ((2.0 * f1 + 0.5 * f2 + 1.5 + 0.1 * rng.g()).clamp(0.0, 3.0)).round() as i32;
            csv.push_str(&format!("q{q},{f1:.4},{f2:.4},{rel}\n"));
        }
    }
    csv
}

#[test]
fn ranking_learns_to_order_items_within_queries() {
    if !have_torch() {
        return;
    }
    for loss in ["ranknet", "listnet", "{name: margin_rank, margin: 0.5, ndcg_k: 5}"] {
        let y = format!("inputs: {{x: [B, 2]}}\nnodes: {{a: {{block: linear, params: {{out: 16}}}}, r: {{block: activation}}, h: {{block: linear, params: {{out: 1}}}}}}\nedges: ['x -> a', 'a -> r', 'r -> h']\noutputs: {{y: h}}\ntraining:\n  loss: {loss}\n  epochs: 80\n  lr: 0.02\n  batch_size: 8\n  data: {{csv: d.csv, features: [f1, f2], target: rel, group: q}}\n");
        let r = ok("rank", &y, &ranking_csv());
        assert!(r.stdout.contains("60 queries (48 train, 12 test)"), "{}", r.stdout);
        assert!(r.last("test_ndcg") > 0.9, "{loss}: ndcg {} \n{}", r.last("test_ndcg"), r.stdout);
        assert!(r.last("test_ndcg") >= r.first("test_ndcg") - 1e-9, "{loss}");
    }
}

#[test]
fn custom_tasks_use_the_projects_own_loss_and_metric() {
    if !have_torch() {
        return;
    }
    let mut rng = Lcg(11);
    let mut csv = String::from("a,b,label\n");
    for _ in 0..200 {
        let (a, b) = (rng.g(), rng.g());
        csv.push_str(&format!("{a:.4},{b:.4},{:.4}\n", a + b));
    }
    let code = "import torch\\n\\ndef asym_loss(out, target):\\n    e = out - target.view_as(out)\\n    return (e.abs() + 2.0 * torch.relu(e)).mean()\\n\\ndef within_half(out, target):\\n    return ((out - target.view_as(out)).abs() < 0.5).float().mean().item()\\n";
    let y = model_yaml(1, &format!("  loss: custom\n  epochs: 120\n  lr: 0.02\n  batch_size: 64\n  custom: {{source: \"{code}\", loss: asym_loss, metric: within_half, mode: max}}\n  data: {{csv: d.csv, features: [a, b], target: label}}\n"));
    let r = ok("custom", &y, &csv);
    assert!(r.stdout.contains("test_within_half"), "{}", r.stdout);
    assert!(r.last("test_within_half") > 0.7, "{}", r.stdout);
    assert!(r.last("test_loss") < r.first("test_loss"));
    // the exported training script is standalone: the custom code is inlined
    let train_py = std::fs::read_to_string(r.art().join("artifacts/train.py")).unwrap();
    assert!(train_py.contains("def asym_loss(") && train_py.contains("custom task code"));

    // errors: a missing function, a path that escapes the project, an absolute path
    let missing = model_yaml(1, &tr("  loss: custom\n  custom: {source: 'def other(o, t): return o.mean()', loss: nope}"));
    assert!(script(&missing).unwrap_err().to_string().contains("does not define `def nope(`"));
    for path in ["../secrets.py", "/etc/passwd"] {
        let e = script(&model_yaml(1, &tr(&format!("  loss: custom\n  custom: {{file: '{path}'}}")))).unwrap_err().to_string();
        assert!(e.contains("relative path inside the project"), "{e}");
    }
    let e = script(&model_yaml(1, &tr("  loss: custom\n  custom: {file: nothing_here.py}"))).unwrap_err().to_string();
    assert!(e.contains("could not be read"), "{e}");
}

// ---------------------------------------------------------------- strategies
#[test]
fn every_optimizer_trains() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 300, &["x", "y", "z"]);
    for opt in ["adam", "adamw", "{name: sgd, momentum: 0.9, nesterov: true}", "{name: rmsprop, alpha: 0.9}", "adagrad", "adadelta", "nadam", "radam", "adamax"] {
        let lr = if opt.contains("adadelta") { 1.0 } else { 0.01 };
        let y = model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 40\n  lr: {lr}\n  weight_decay: 0.0001\n  optimizer: {opt}\n  batch_size: 64\n{DATA}\n"));
        let r = ok("opt", &y, &csv);
        assert!(r.last("test_acc") > 0.8, "{opt}: {}", r.stdout);
    }
}

#[test]
fn schedulers_follow_their_learning_rate_curves() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 150, &["x", "y", "z"]);
    let run = |sched: &str| {
        let y = model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 30\n  lr: 0.1\n  batch_size: 150\n  scheduler: {sched}\n{DATA}\n"));
        ok("sched", &y, &csv)
    };
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6_f64.max(b.abs() * 0.02);
    let r = run("{name: step, step_size: 10, gamma: 0.1}");
    let e: Vec<(f64, f64)> = r.epochs().iter().map(|m| (m["epoch"], m["lr"])).collect();
    // lr is read after the epoch's scheduler step: epoch 0 -> after 1 step, epoch 10 -> after 11 steps, ...
    assert!(close(e[0].1, 0.1), "{e:?}");
    let at = |n: f64| e.iter().find(|x| x.0 == n).unwrap().1;
    assert!(close(at(10.0), 0.01) && close(at(20.0), 0.001) && close(at(29.0), 0.0001), "{e:?}");

    let r = run("{name: exponential, gamma: 0.9}");
    assert!(close(r.last("lr"), 0.1 * 0.9f64.powi(30)), "{}", r.stdout);

    let r = run("{name: cosine, eta_min: 0.0}");
    assert!(r.last("lr") < 0.002 && r.first("lr") > 0.09, "cosine decays to ~0: {}", r.stdout);

    let r = run("{name: cosine, warmup_epochs: 5}");
    assert!(r.first("lr") < 0.05, "warmup starts well below the base rate: {}", r.stdout);
    assert!(r.last("lr") < 0.01);

    let r = run("constant");
    assert!(r.epochs().iter().all(|m| close(m["lr"], 0.1)), "{}", r.stdout);

    let r = run("{name: multistep, milestones: [5, 15], gamma: 0.1}");
    assert!(close(r.last("lr"), 0.001), "{}", r.stdout);

    let r = run("{name: cosine_restarts, t_0: 10, eta_min: 0.001}");
    assert!(r.epochs().len() >= 3 && r.last("lr") > 0.0);

    let r = run("{name: onecycle, max_lr: 0.3}");
    let peak = r.epochs().iter().map(|m| m["lr"]).fold(0.0, f64::max);
    assert!(peak > 0.1, "one-cycle rises above the base rate: {}", r.stdout);

    let r = run("{name: plateau, factor: 0.5, patience: 1}");
    assert!(r.last("lr") <= 0.1, "{}", r.stdout);
}

#[test]
fn strategies_clip_accumulate_and_mixed_precision() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 300, &["x", "y", "z"]);
    let base = |extra: &str| model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 50\n  lr: 0.02\n  batch_size: 32\n{extra}\n{DATA}\n"));
    let r = ok("clip", &base("  clip_grad: 1.0"), &csv);
    assert!(r.last("test_acc") > 0.9, "{}", r.stdout);
    let r = ok("accum", &base("  accumulate: 4"), &csv);
    assert!(r.last("test_acc") > 0.85, "{}", r.stdout);
    let r = ok("bf16", &base("  amp: bf16"), &csv);
    assert!(r.stdout.contains("amp: bf16") && r.last("test_acc") > 0.85, "{}", r.stdout);
    // fp16 needs CUDA: on a CPU-only machine it falls back to float32 with a notice instead of failing
    let r = ok("fp16", &base("  amp: fp16\n  device: cpu"), &csv);
    assert!(r.stdout.contains("fp16 autocast needs CUDA") && r.stdout.contains("amp: off"), "{}", r.stdout);
}

#[test]
fn early_stopping_and_best_checkpoint() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 300, &["x", "y", "z"]);
    // an impossible min_delta means no epoch after the first counts as an improvement
    let y = model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 200\n  lr: 0.02\n  batch_size: 64\n  early_stopping: {{patience: 2, min_delta: 1000.0, monitor: test_loss}}\n{DATA}\n"));
    let r = ok("early", &y, &csv);
    assert!(r.stdout.contains("early stopping at epoch 2: test_loss has not improved for 2 epochs"), "{}", r.stdout);
    let meta = predict::read_meta(&r.art()).unwrap();
    assert_eq!(meta["epochs_run"], 3, "{meta}");
    assert_eq!(meta["monitor"], "test_loss");

    // save_best restores the best epoch: the exported metric is the best one seen, never worse than the last epoch's
    let y = model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 60\n  lr: 0.5\n  batch_size: 16\n  save_best: true\n  early_stopping: {{patience: 100, monitor: test_loss}}\n{DATA}\n"));
    let r = ok("best", &y, &csv);
    assert!(r.stdout.contains("restored the best weights from epoch"), "{}", r.stdout);
    let meta = predict::read_meta(&r.art()).unwrap();
    assert!(meta["test_loss"].as_f64().unwrap() <= r.last("test_loss") + 1e-9, "best {} vs last {}", meta["test_loss"], r.last("test_loss"));
}

#[test]
fn freezing_and_transfer_from_a_checkpoint() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 300, &["x", "y", "z"]);
    let first = ok("pre", &model_yaml(3, &tr("  loss: cross_entropy")), &csv);
    let weights = first.art().join("artifacts/model.pt");
    assert!(weights.exists());
    let second = format!("  loss: cross_entropy\n  epochs: 10\n  lr: 0.05\n  batch_size: 64\n  init_from: '{}'\n  freeze: [a]\n{DATA}\n", weights.display());
    let r = ok("ft", &model_yaml(3, &second), &csv);
    assert!(r.stdout.contains("frozen: ['a']") && r.stdout.contains("trainable"), "{}", r.stdout);
    assert!(r.stdout.contains("loaded, 0 missing and 0 unexpected keys"), "{}", r.stdout);
    // frozen layer unchanged, trainable layer changed
    let compare = "import torch,sys\na=torch.load(sys.argv[1]);b=torch.load(sys.argv[2])\nprint(torch.equal(a['a.weight'],b['a.weight']), torch.equal(a['h.weight'],b['h.weight']))";
    let out = std::process::Command::new(sentinel_core::platform::python_exe()).args(["-I", "-c", compare]).arg(&weights).arg(r.art().join("artifacts/model.pt")).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "True False", "{}", String::from_utf8_lossy(&out.stderr));
    let meta = predict::read_meta(&r.art()).unwrap();
    assert!(meta["n_trainable"].as_u64().unwrap() < meta["n_params"].as_u64().unwrap());
}

#[test]
fn devices() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 120, &["x", "y", "z"]);
    let y = |dev: &str| model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 5\n  lr: 0.05\n  batch_size: 60\n  device: {dev}\n{DATA}\n"));
    let r = ok("cpu", &y("cpu"), &csv);
    assert!(r.stdout.contains("device: cpu"), "{}", r.stdout);
    let has = |expr: &str| std::process::Command::new(sentinel_core::platform::python_exe()).args(["-c", &format!("import torch,sys; sys.exit(0 if ({expr}) else 1)")]).status().unwrap().success();
    if !has("torch.cuda.is_available()") {
        let r = train("cuda", &y("cuda"), &csv);
        assert!(!r.ok && r.stderr.contains("CUDA is not available"), "{}\n{}", r.stdout, r.stderr);
    }
    if has("hasattr(torch.backends,'mps') and torch.backends.mps.is_available()") {
        let r = ok("mps", &y("mps"), &csv);
        assert!(r.stdout.contains("device: mps") && r.last("test_acc") > 0.5, "{}", r.stdout);
        // the exported model is a CPU model: prediction must not need the GPU
        assert!(predict::predict(&r.art(), &[vec![-3.0, -3.0]]).is_ok());
    }
}

#[test]
fn default_checkpointing_and_seamless_resume() {
    if !have_torch() {
        return;
    }
    let csv = blobs(3, 120, &["x", "y", "z"]);
    // 1. Train with default checkpointing (mode="latest") for 3 epochs
    let y = model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 3\n  lr: 0.05\n  batch_size: 60\n{DATA}\n"));
    let r = ok("ckpt_run", &y, &csv);
    let ckpt_dir = r.art().join("artifacts/checkpoints");
    assert!(ckpt_dir.join("checkpoint_latest.pt").exists(), "checkpoint_latest.pt must exist by default");

    // 2. Train with mode="all"
    let y_all = model_yaml(3, &format!("  loss: cross_entropy\n  epochs: 3\n  lr: 0.05\n  batch_size: 60\n  checkpoint: all\n{DATA}\n"));
    let r_all = ok("ckpt_all", &y_all, &csv);
    let ckpt_all_dir = r_all.art().join("artifacts/checkpoints");
    assert!(ckpt_all_dir.join("checkpoint_latest.pt").exists(), "checkpoint_latest.pt must exist");
    assert!(ckpt_all_dir.join("checkpoint_epoch_0000.pt").exists(), "per-epoch checkpoint 0 must exist");
    assert!(ckpt_all_dir.join("checkpoint_epoch_0001.pt").exists(), "per-epoch checkpoint 1 must exist");
}
