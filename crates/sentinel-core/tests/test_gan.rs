use sentinel_core::components::{Built, Component, ComponentRegistry};
use sentinel_core::graph::{fmt_shape, GraphError};
use sentinel_core::predict;
use std::collections::HashMap;
use std::path::PathBuf;

fn have_torch() -> bool {
    std::process::Command::new("python3").args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

fn gan_component() -> Component {
    ComponentRegistry::new().get("generative/gan@v1").unwrap().clone()
}

fn build(c: &Component, ov: &[(&str, serde_yaml::Value)]) -> Result<Built, GraphError> {
    c.build(&ov.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<HashMap<_, _>>())
}

#[test]
fn a_gan_is_two_graphs_that_fit_together() {
    let Built::Gan(v) = build(&gan_component(), &[]).unwrap() else { panic!("expected a GAN") };
    assert_eq!(fmt_shape(&v.generator.shapes["out.out"]), "[B, 2]");
    assert_eq!(fmt_shape(&v.discriminator.shapes["out.out"]), "[B, 1]");
    // both networks are built from the compound mlp_block, so each has a compound inner graph
    assert!(v.generator.compounds.contains_key("g1") && v.discriminator.compounds.contains_key("d1"));
}

#[test]
fn the_generated_script_shares_classes_between_the_two_models() {
    let built = build(&gan_component(), &[]).unwrap();
    let code = built.training_script("generative/gan@v1").unwrap();
    assert!(code.contains("class Generator(nn.Module)") && code.contains("class Discriminator(nn.Module)"));
    // one class per distinct (parameters, input shape): g1, g2, d1 and d2 all differ, and no class is ever defined twice
    let names: Vec<&str> = code.lines().filter_map(|l| l.strip_prefix("class ")).map(|l| l.split('(').next().unwrap()).collect();
    let unique: std::collections::BTreeSet<&&str> = names.iter().collect();
    assert_eq!(names.len(), unique.len(), "duplicate class definitions: {names:?}");
    assert_eq!(names.iter().filter(|n| n.starts_with("MlpBlock")).count(), 4, "{names:?}");
    assert!(code.find("class MlpBlock").unwrap() < code.find("class Generator").unwrap(), "dependencies come first");
    assert!(code.contains("opt_g = torch.optim.Adam") && code.contains("opt_d = torch.optim.Adam"));
    assert!(code.contains("fake = G(torch.randn(n, LATENT)).detach()"), "discriminator step must not update the generator");
    assert!(built.model_code().contains("class Discriminator"));
}

#[test]
fn interface_mismatches_are_rejected_with_specific_errors() {
    let c = gan_component();
    // generator emits 2 features but the data has 3 columns
    let e = build(&c, &[("feature_cols", "a,b,c".into())]).unwrap_err().to_string();
    assert!(e.contains("generator must output [B, 3]"), "{e}");
    // generator and discriminator disagree about the data shape
    let mut bad = c.clone();
    let mut v = serde_yaml::to_string(bad.gan.as_ref().unwrap()).unwrap();
    v = v.replacen("x: [B, ${features}]", "x: [B, 5]", 1).replace("x:\n    - B\n    - ${features}", "x:\n    - B\n    - 5");
    bad.gan = Some(serde_yaml::from_str(&v).unwrap());
    let e = bad.build(&HashMap::new()).unwrap_err().to_string();
    assert!(e.contains("discriminator"), "{e}");
    // a latent size that disagrees with the generator's noise input
    let mut bad = c.clone();
    let s = serde_yaml::to_string(bad.gan.as_ref().unwrap()).unwrap().replace("latent: ${latent}", "latent: 3");
    bad.gan = Some(serde_yaml::from_str(&s).unwrap());
    assert!(bad.build(&HashMap::new()).unwrap_err().to_string().contains("generator's input must be [B, 3]"));
    // a discriminator that does not end in one logit
    let e = build(&c, &[]).map(|_| ()).err();
    assert!(e.is_none());
    let mut bad = c.clone();
    let s = serde_yaml::to_string(bad.gan.as_ref().unwrap()).unwrap().replacen("out: 1", "out: 4", 1);
    bad.gan = Some(serde_yaml::from_str(&s).unwrap());
    assert!(bad.build(&HashMap::new()).unwrap_err().to_string().contains("one logit per sample"));
    // unknown overrides still fail loudly
    assert!(build(&c, &[("nope", 1.into())]).unwrap_err().to_string().contains("unknown parameter"));
    // out-of-range d_steps is caught by validation
    assert!(build(&c, &[("d_steps", 0.into())]).unwrap_err().to_string().contains("at least 1"));
}

#[test]
fn a_component_cannot_be_both_a_graph_and_a_gan() {
    let y = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("builtin/components/generative/gan.yaml")).unwrap();
    let both = y.replace("gan:\n  generator:", "graph:\n  inputs: {x: [B, 2]}\n  nodes: {a: {block: linear, params: {out: 2}}}\n  edges: ['x -> a']\n  outputs: {y: a}\ngan:\n  generator:");
    assert!(Component::from_yaml(&both).unwrap_err().to_string().contains("not both"));
}

#[test]
fn adversarial_training_learns_a_distribution_and_the_generator_is_usable() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sentinel-gan-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // target: a Gaussian blob centred at (3, -2) with std (0.5, 1.5); a generator that outputs ~0 is far off at the start
    let mut csv = String::from("a,b\n");
    let mut state = 12345u64;
    let mut gauss = || {
        let mut u = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0) };
        (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
    };
    for _ in 0..600 {
        csv.push_str(&format!("{:.4},{:.4}\n", 3.0 + 0.5 * gauss(), -2.0 + 1.5 * gauss()));
    }
    std::fs::write(dir.join("d.csv"), csv).unwrap();

    let c = gan_component();
    let built = build(&c, &[("csv", "d.csv".into()), ("feature_cols", "a,b".into()), ("epochs", 500.into()), ("batch_size", 100.into())]).unwrap();
    std::fs::write(dir.join("train.py"), built.training_script("generative/gan@v1").unwrap()).unwrap();
    let run = dir.join("run");
    let out = std::process::Command::new("python3").args(["-I", "train.py"]).current_dir(&dir).env("SENTINEL_ARTIFACT_DIR", run.join("artifacts")).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}\n{stdout}", String::from_utf8_lossy(&out.stderr));

    let gaps: Vec<f64> = stdout.lines().filter(|l| l.contains("mean_gap")).map(|l| l.split("mean_gap ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap()).collect();
    assert!(gaps.len() > 5, "{stdout}");
    assert!(gaps[0] > 1.5, "the untrained generator should be far from (3, -2): {gaps:?}");
    assert!(*gaps.last().unwrap() < gaps[0] * 0.4, "the generated mean must close most of the gap: {gaps:?}");

    // the exported generator draws new samples near the data, deterministically for a seed
    let meta = predict::read_meta(&run).unwrap();
    assert_eq!(meta["task"], "gan");
    assert_eq!(meta["latent"], 8);
    let s1 = predict::generate(&run, 400, 7).unwrap();
    let s2 = predict::generate(&run, 400, 7).unwrap();
    assert_eq!(s1, s2, "same seed, same samples");
    let rows = s1["samples"].as_array().unwrap();
    assert_eq!(rows.len(), 400);
    let mean = |j: usize| rows.iter().map(|r| r[j].as_f64().unwrap()).sum::<f64>() / rows.len() as f64;
    assert!((mean(0) - 3.0).abs() < 1.2 && (mean(1) + 2.0).abs() < 1.5, "sample means ({:.2}, {:.2}) should be near (3, -2)", mean(0), mean(1));
    // the two entry points refuse each other's runs with a clear message
    assert!(predict::predict(&run, &[vec![1.0, 2.0]]).unwrap_err().contains("use generate"));
    assert!(predict::generate(&run, 0, 0).is_err());
}
