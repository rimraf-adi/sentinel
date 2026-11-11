use sentinel_core::codegen::{generate_model, generate_training_script};
use sentinel_core::components::ComponentRegistry;
use sentinel_core::compound::BlockLibrary;
use sentinel_core::graph::{fmt_shape, GraphError, ValidatedGraph};
use std::collections::HashMap;

fn build(id: &str, overrides: &[(&str, serde_yaml::Value)]) -> ValidatedGraph {
    let r = ComponentRegistry::new();
    let ov: HashMap<String, serde_yaml::Value> = overrides.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    r.get(id).unwrap_or_else(|| panic!("{id} not registered")).build_graph(&ov).unwrap_or_else(|e| panic!("{id}: {e}"))
}

fn have_torch() -> bool {
    std::process::Command::new(sentinel_core::platform::python_exe()).args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

fn forward(g: &ValidatedGraph, body: &str) {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let script = format!("import torch\nimport torch.nn as nn\n{}\nm = Model()\n{body}\nprint('ok')\n", generate_model(g));
    let out = std::process::Command::new(sentinel_core::platform::python_exe()).args(["-I", "-c", &script]).output().unwrap();
    assert!(out.status.success(), "{}\n--- script ---\n{script}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn compound_library_grew_a_real_second_tier() {
    let cat = BlockLibrary::builtin().catalog();
    let uses = |n: &str| cat.iter().find(|b| b.name == n).unwrap_or_else(|| panic!("{n}")).uses.clone();
    assert_eq!(uses("transformer_block"), ["add", "attention", "layernorm", "linear", "mlp_block"]);
    assert_eq!(uses("unet_up"), ["concat", "conv_block", "conv_transpose2d"]);
    assert_eq!(uses("vae_encoder"), ["linear", "mlp_block"]);
    assert!(cat.iter().filter(|b| b.tier == "compound").count() >= 9);
}

#[test]
fn vit_tiny_is_a_transformer_of_compound_blocks() {
    let g = build("vision/vit-tiny@v1", &[]);
    assert_eq!(fmt_shape(&g.shapes["patches.out"]), "[B, 16, 64]");
    assert_eq!(fmt_shape(&g.shapes["t2.out"]), "[B, 17, 64]");
    assert_eq!(fmt_shape(&g.shapes["head.out"]), "[B, 10]");
    // model -> transformer_block -> mlp_block -> linear: three levels of the hierarchy
    assert!(g.compounds["t1"].compounds.contains_key("ff"));
    assert!(g.compounds["t1"].compounds["ff"].kind("fc").tier() == "fundamental");
    forward(&g, "x = torch.randn(2, 3, 32, 32)\ny = m(x)\nassert y.shape == (2, 10)\ny.sum().backward()\n\
                 assert all(p.grad is not None for p in m.parameters())");
}

#[test]
fn unet_wires_multi_output_and_multi_input_compounds() {
    let g = build("vision/unet@v1", &[]);
    assert_eq!(fmt_shape(&g.shapes["d1.down"]), "[B, 16, 16, 16]");
    assert_eq!(fmt_shape(&g.shapes["d1.skip"]), "[B, 16, 32, 32]");
    assert_eq!(fmt_shape(&g.shapes["d2.down"]), "[B, 32, 8, 8]");
    assert_eq!(fmt_shape(&g.shapes["u2.out"]), "[B, 32, 16, 16]");
    assert_eq!(fmt_shape(&g.shapes["u1.out"]), "[B, 16, 32, 32]");
    assert_eq!(fmt_shape(&g.shapes["out.out"]), "[B, 2, 32, 32]");
    assert!(g.compounds["d1"].compounds.contains_key("c1"), "unet_down is built from conv_block");

    let code = generate_model(&g);
    assert!(code.contains("d1__down, d1__skip = self.d1(img)"), "multi-output compounds unpack in port order\n{code}");
    assert!(code.contains("u2 = UnetUp") && code.contains("class UnetUp"));
    forward(&g, "y = m(torch.randn(2, 3, 32, 32))\nassert y.shape == (2, 2, 32, 32)\ny.mean().backward()");
    // a different size works from the same definitions
    let big = build("vision/unet@v1", &[("size", serde_yaml::Value::from(64))]);
    assert_eq!(fmt_shape(&big.shapes["out.out"]), "[B, 2, 64, 64]");
}

#[test]
fn unet_rejects_sizes_that_do_not_survive_two_downsamplings() {
    let r = ComponentRegistry::new();
    let mut ov = HashMap::new();
    ov.insert("size".to_string(), serde_yaml::Value::from(30)); // 30 -> 15 -> 7: transposed conv gives 14, not 15
    let e = r.get("vision/unet@v1").unwrap().build_graph(&ov).unwrap_err().to_string();
    assert!(e.contains("compound block") && e.contains("differ"), "{e}");
}

#[test]
fn resnet_mini_runs() {
    let g = build("vision/resnet-mini@v1", &[]);
    assert_eq!(fmt_shape(&g.shapes["r2.out"]), "[B, 32, 16, 16]");
    assert_eq!(fmt_shape(&g.shapes["head.out"]), "[B, 10]");
    forward(&g, "assert m(torch.randn(4, 3, 32, 32)).shape == (4, 10)");
}

#[test]
fn vae_trains_on_a_csv_and_its_loss_falls() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sentinel-vae-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // two correlated features: a 1-D curve a VAE with a small latent space can learn
    let mut csv = String::from("a,b\n");
    for i in 0..300 {
        let t = (i as f64 / 300.0) * 6.0 - 3.0;
        csv.push_str(&format!("{:.4},{:.4}\n", t, t * t / 3.0));
    }
    std::fs::write(dir.join("d.csv"), csv).unwrap();
    let g = build(
        "generative/vae@v1",
        &[("csv", "d.csv".into()), ("feature_cols", "a,b".into()), ("epochs", 120.into()), ("beta", 0.01.into()), ("latent", 2.into())],
    );
    assert_eq!(fmt_shape(&g.shapes["z.out"]), "[B, 2]");
    let script = generate_training_script(&g, "generative/vae@v1").unwrap();
    assert!(script.contains("BETA = 0.01") && script.contains("kl = -0.5"), "{script}");
    std::fs::write(dir.join("train.py"), script).unwrap();
    let out = std::process::Command::new(sentinel_core::platform::python_exe()).args(["-I", "train.py"]).current_dir(&dir).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}\n{stdout}", String::from_utf8_lossy(&out.stderr));
    let vals: Vec<f64> = stdout.lines().filter(|l| l.contains("test_recon")).map(|l| l.rsplit(' ').next().unwrap().parse().unwrap()).collect();
    assert!(vals.len() >= 3, "{stdout}");
    assert!(vals.last().unwrap() < &(vals[0] * 0.5), "reconstruction error should at least halve: {vals:?}");
}

#[test]
fn vae_loss_has_its_own_validation() {
    let g = build("generative/vae@v1", &[]);
    // outputs must be recon / mu / logvar; the loss is unsupervised
    let mut spec = g.spec.clone();
    spec.training.as_mut().unwrap().data.target = "label".into();
    let e = generate_training_script(&spec.validate().unwrap(), "x").unwrap_err();
    assert!(matches!(e, GraphError::Training(ref m) if m.contains("unsupervised")), "{e}");

    let mut spec = g.spec.clone();
    spec.outputs.remove("mu");
    let e = generate_training_script(&spec.validate().unwrap(), "x").unwrap_err();
    assert!(e.to_string().contains("recon"), "{e}");

    // a supervised loss without a target is rejected too
    let mut spec = build("tabular/ann-classifier@v1", &[]).spec;
    spec.training.as_mut().unwrap().data.target = String::new();
    let e = generate_training_script(&spec.validate().unwrap(), "x").unwrap_err();
    assert!(e.to_string().contains("target"), "{e}");
}
