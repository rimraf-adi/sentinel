use sentinel_core::codegen::generate_training_script;
use sentinel_core::components::ComponentRegistry;
use sentinel_core::predict;
use sentinel_core::tracking;
use std::collections::HashMap;
use std::path::PathBuf;

fn have_torch() -> bool {
    std::process::Command::new("python3").args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

fn workdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sentinel-export-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Train `component` on `csv` inside a fake run directory; returns that run directory (artifacts under it).
fn train(tag: &str, component: &str, csv: &str, overrides: &[(&str, serde_yaml::Value)]) -> PathBuf {
    let dir = workdir(tag);
    std::fs::write(dir.join("d.csv"), csv).unwrap();
    let mut ov: HashMap<String, serde_yaml::Value> = overrides.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    ov.insert("csv".into(), "d.csv".into());
    let g = ComponentRegistry::new().get(component).unwrap().build_graph(&ov).unwrap();
    std::fs::write(dir.join("train.py"), generate_training_script(&g, component).unwrap()).unwrap();
    let run = dir.join("run");
    let out = std::process::Command::new("python3")
        .args(["-I", "train.py"])
        .current_dir(&dir)
        .env("SENTINEL_ARTIFACT_DIR", run.join("artifacts"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}\n{}", String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout));
    run
}

fn three_class_csv() -> String {
    let mut csv = String::from("sepal,petal,species\n");
    for i in 0..40 {
        let f = i as f64 / 40.0;
        csv.push_str(&format!("{:.3},{:.3},low\n{:.3},{:.3},mid\n{:.3},{:.3},high\n", f, f * 0.2, 3.0 + f, 1.0 + f * 0.2, 6.0 + f, 4.0 + f * 0.2));
    }
    csv
}

#[test]
fn training_exports_weights_torchscript_and_metadata() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let run = train("ann", "tabular/ann-classifier@v1", &three_class_csv(), &[("feature_cols", "sepal,petal".into()), ("target", "species".into()), ("classes", 3.into()), ("epochs", 150.into())]);
    let files: Vec<String> = predict::list_artifacts(&run).into_iter().map(|f| f.name).collect();
    for needed in ["model.pt", "model_script.pt", "meta.json", "train.py"] {
        assert!(files.contains(&needed.to_string()), "{needed} missing from {files:?}");
    }
    let meta = predict::read_meta(&run).unwrap();
    assert_eq!(meta["task"], "classification");
    assert_eq!(meta["features"], serde_json::json!(["sepal", "petal"]));
    assert_eq!(meta["classes"], serde_json::json!(["high", "low", "mid"]));
    assert_eq!(meta["input_shape"], serde_json::json!([2]));
    assert!(meta["test_acc"].as_f64().unwrap() > 0.9, "{meta}");
    assert_eq!(meta["files"]["torchscript"], "model_script.pt");
    // ONNX is best-effort (it needs an exporter backend): when present it must be a real file.
    if meta["files"].get("onnx").is_some() {
        assert!(run.join("artifacts/model.onnx").metadata().unwrap().len() > 100);
    }
}

#[test]
fn predict_classifies_with_probabilities_and_any_batch_size() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let run = train("pred", "tabular/ann-classifier@v1", &three_class_csv(), &[("feature_cols", "sepal,petal".into()), ("target", "species".into()), ("classes", 3.into()), ("epochs", 200.into())]);
    // one row, then several: the traced model must not be pinned to the batch size it was traced with
    for rows in [vec![vec![0.2, 0.04]], vec![vec![0.2, 0.04], vec![3.5, 1.1], vec![6.5, 4.1], vec![0.5, 0.1], vec![3.1, 1.0], vec![6.9, 4.3], vec![0.3, 0.06]]] {
        let r = predict::predict(&run, &rows).unwrap();
        let preds = r["predictions"].as_array().unwrap();
        assert_eq!(preds.len(), rows.len());
        for p in preds {
            let total: f64 = p["probabilities"].as_object().unwrap().values().map(|v| v.as_f64().unwrap()).sum();
            assert!((total - 1.0).abs() < 1e-4, "probabilities sum to 1, got {total}");
        }
        assert_eq!(preds[0]["label"], "low");
        if rows.len() > 1 {
            assert_eq!(preds[1]["label"], "mid");
            assert_eq!(preds[2]["label"], "high");
        }
    }
    let e = predict::predict(&run, &[vec![1.0, 2.0, 3.0]]).unwrap_err();
    assert!(e.contains("expected rows of 2 values"), "{e}");
}

#[test]
fn lstm_exports_and_predicts_with_a_reshaped_sequence() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let run = train("lstm", "sequence/lstm-classifier@v1", &three_class_csv(), &[
        ("feature_cols", "sepal,petal".into()), ("target", "species".into()), ("classes", 3.into()), ("epochs", 120.into()), ("hidden", 16.into()),
    ]);
    let r = predict::predict(&run, &[vec![0.2, 0.04], vec![6.5, 4.1], vec![3.2, 1.0]]).unwrap();
    assert_eq!(r["predictions"].as_array().unwrap().len(), 3);
    assert_eq!(r["predictions"][1]["label"], "high");
}

#[test]
fn regression_and_vae_have_their_own_output_shapes() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let mut csv = String::from("a,b,y\n");
    for i in 0..200 {
        let (a, b) = (i as f64 / 100.0 - 1.0, ((i * 7) % 100) as f64 / 100.0);
        csv.push_str(&format!("{a:.4},{b:.4},{:.4}\n", 2.0 * a - b));
    }
    let run = train("reg", "tabular/ann-classifier@v1", &csv, &[("feature_cols", "a,b".into()), ("target", "y".into()), ("loss", "mse".into()), ("classes", 1.into()), ("epochs", 250.into())]);
    assert_eq!(predict::read_meta(&run).unwrap()["task"], "regression");
    let r = predict::predict(&run, &[vec![0.5, 0.25], vec![-0.5, 0.5]]).unwrap();
    let v: Vec<f64> = r["values"].as_array().unwrap().iter().map(|x| x[0].as_f64().unwrap()).collect();
    assert!((v[0] - 0.75).abs() < 0.25 && (v[1] + 1.5).abs() < 0.4, "2a - b should be learned: {v:?}");

    let run = train("vae", "generative/vae@v1", &csv, &[("feature_cols", "a,b".into()), ("epochs", 100.into()), ("beta", 0.01.into())]);
    let meta = predict::read_meta(&run).unwrap();
    assert_eq!(meta["task"], "vae");
    let r = predict::predict(&run, &[vec![0.5, 0.25]]).unwrap();
    assert_eq!(r["reconstruction"][0].as_array().unwrap().len(), 2);
    assert_eq!(r["latent_mean"][0].as_array().unwrap().len(), 2);
}

#[test]
fn run_lookup_accepts_prefixes_and_rejects_traversal() {
    let root = workdir("lookup").join("mlruns");
    for id in ["abc123def", "abc999aaa", "fff000111"] {
        let d = root.join("0").join(id);
        std::fs::create_dir_all(d.join("artifacts")).unwrap();
        std::fs::write(d.join("meta.yaml"), "run_name: r\nstatus: 3\nstart_time: 1\n").unwrap();
    }
    assert_eq!(tracking::resolve_run_id(&root, "fff").unwrap(), "fff000111");
    assert!(tracking::resolve_run_id(&root, "abc").unwrap_err().contains("matches 2"));
    assert!(tracking::resolve_run_id(&root, "zzz").unwrap_err().contains("no run"));
    assert!(tracking::resolve_run_id(&root, "../x").is_err());
    assert!(tracking::run_dir(&root, "abc123def").is_some());
    assert!(tracking::run_dir(&root, "..").is_none() && tracking::run_dir(&root, "a/b").is_none());
}
