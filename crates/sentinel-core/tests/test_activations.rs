use sentinel_core::blocks::{Block, ACTIVATIONS};
use sentinel_core::codegen::generate_model;
use sentinel_core::compound::BlockLibrary;
use sentinel_core::graph::{fmt_shape, GraphSpec};

fn graph(yaml: &str) -> Result<sentinel_core::graph::ValidatedGraph, sentinel_core::graph::GraphError> {
    GraphSpec::from_yaml(yaml)?.validate_with(&BlockLibrary::builtin())
}

fn have_torch() -> bool {
    std::process::Command::new("python3").args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

fn act_graph(params: &str) -> String {
    format!("inputs: {{x: [B, 8]}}\nnodes: {{a: {{block: activation, params: {{{params}}}}}}}\nedges: ['x -> a']\noutputs: {{y: a}}")
}

#[test]
fn the_catalog_lists_every_activation() {
    assert!(ACTIVATIONS.len() >= 20);
    let info = Block::Activation.info();
    let kind = info.params.iter().find(|p| p.name == "kind").unwrap();
    assert_eq!(kind.options.len(), ACTIVATIONS.len(), "the UI gets the options from the API");
    assert!(info.params.iter().find(|p| p.name == "alpha").unwrap().default == Some(serde_json::Value::Null));
}

#[test]
fn every_activation_runs_in_pytorch_and_counts_parameters_correctly() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    // one process, every activation: shape, and the analytic parameter count against PyTorch's
    let mut cases = Vec::new();
    for k in ACTIVATIONS.iter().chain(["swish"].iter()) {
        let g = graph(&act_graph(&format!("kind: {k}"))).unwrap_or_else(|e| panic!("{k}: {e}"));
        let want_shape = if *k == "glu" { "[B, 4]" } else { "[B, 8]" };
        assert_eq!(fmt_shape(&g.shapes["a.out"]), want_shape, "{k}");
        cases.push(serde_json::json!({ "kind": k, "code": generate_model(&g), "params": g.stats().params, "out": if *k == "glu" { 4 } else { 8 } }));
    }
    let script = r#"
import json, sys, torch, torch.nn as nn
for c in json.load(sys.stdin):
    ns = {"torch": torch, "nn": nn}
    exec(c["code"], ns)
    m = ns["Model"]()
    y = m(torch.randn(3, 8))
    assert y.shape == (3, c["out"]), (c["kind"], y.shape)
    assert torch.isfinite(y).all(), c["kind"]
    real = sum(p.numel() for p in m.parameters())
    assert real == c["params"], (c["kind"], real, c["params"])
print("ok")
"#;
    let mut child = std::process::Command::new("python3").args(["-I", "-c", script]).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), serde_json::to_string(&cases).unwrap().as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // prelu is the one learnable activation
    assert_eq!(graph(&act_graph("kind: prelu")).unwrap().stats().params, 1);
    assert_eq!(graph(&act_graph("kind: relu")).unwrap().stats().params, 0);
}

#[test]
fn alpha_changes_the_parameterised_activations() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let check = |params: &str, body: &str| {
        let code = generate_model(&graph(&act_graph(params)).unwrap());
        let script = format!("import torch\nimport torch.nn as nn\n{code}\nm = Model()\n{body}\nprint('ok')");
        let out = std::process::Command::new("python3").args(["-I", "-c", &script]).output().unwrap();
        assert!(out.status.success(), "{params}: {}", String::from_utf8_lossy(&out.stderr));
    };
    let neg = "x = -torch.ones(1, 8)\ny = m(x)[0, 0].item()\n";
    check("kind: leaky_relu", &format!("{neg}assert abs(y + 0.01) < 1e-6, y"));
    check("kind: leaky_relu, alpha: 0.2", &format!("{neg}assert abs(y + 0.2) < 1e-6, y"));
    check("kind: elu", &format!("{neg}assert abs(y - (2.718281828 ** -1 - 1)) < 1e-5, y"));
    check("kind: elu, alpha: 2.0", &format!("{neg}assert abs(y - 2 * (2.718281828 ** -1 - 1)) < 1e-5, y"));
    check("kind: softplus, alpha: 2.0", "x = torch.zeros(1, 8)\nassert abs(m(x)[0, 0].item() - (0.6931471806 / 2)) < 1e-5");
    check("kind: swish", &format!("{neg}assert abs(y + 0.26894142) < 1e-5, y"));
    check("kind: glu", "x = torch.cat([torch.ones(1, 4), torch.zeros(1, 4)], dim=1)\nassert abs(m(x)[0, 0].item() - 0.5) < 1e-6");
    check("kind: softmax", "assert abs(m(torch.randn(2, 8)).sum(-1) - 1).max() < 1e-5");
}

#[test]
fn bad_activations_are_rejected_with_helpful_messages() {
    let e = graph(&act_graph("kind: relux")).unwrap_err().to_string();
    assert!(e.contains("unsupported activation 'relux'") && e.contains("mish") && e.contains("glu"), "{e}");
    for bad in ["kind: elu, alpha: 0", "kind: elu, alpha: -1.0", "kind: elu, alpha: fast"] {
        assert!(graph(&act_graph(bad)).unwrap_err().to_string().contains("positive number"), "{bad}");
    }
    let e = graph("inputs: {x: [B, 7]}\nnodes: {a: {block: activation, params: {kind: glu}}}\nedges: ['x -> a']\noutputs: {y: a}").unwrap_err().to_string();
    assert!(e.contains("must be even"), "{e}");
}

#[test]
fn compound_blocks_pass_any_activation_through() {
    let y = "inputs: {x: [B, 8]}\nnodes: {m: {block: mlp_block, params: {width: 16, activation: mish}}}\nedges: ['x -> m']\noutputs: {y: m}";
    let g = graph(y).unwrap();
    assert!(generate_model(&g).contains("nn.Mish()"));
    // and a typo inside the compound is reported with the compound's name
    let e = graph(&y.replace("mish", "mushy")).unwrap_err().to_string();
    assert!(e.contains("compound block 'mlp_block'") && e.contains("unsupported activation"), "{e}");
}
