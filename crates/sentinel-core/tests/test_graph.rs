use sentinel_core::blocks::Block;
use sentinel_core::codegen::{generate_model, generate_training_script};
use sentinel_core::components::{Component, ComponentRegistry};
use sentinel_core::graph::{fmt_shape, GraphError, GraphSpec};
use std::collections::HashMap;
use std::path::Path;

fn registry() -> ComponentRegistry {
    let mut r = ComponentRegistry::new();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../components");
    r.load_from_dir(&dir).unwrap();
    r
}

fn graph(yaml: &str) -> Result<sentinel_core::graph::ValidatedGraph, GraphError> {
    GraphSpec::from_yaml(yaml)?.validate()
}

#[test]
fn every_shipped_graph_component_validates() {
    let r = registry();
    let graphs: Vec<_> = r.list().into_iter().filter(|c| c.graph.is_some()).collect();
    assert!(graphs.len() >= 7, "expected >= 6 graph components, got {}", graphs.len());
    for c in graphs {
        c.build_graph(&HashMap::new()).unwrap_or_else(|e| panic!("{}: {e}", c.component_id));
    }
}

#[test]
fn ann_shapes_and_overrides() {
    let r = registry();
    let c = r.get("tabular/ann-classifier@v1").unwrap();
    let g = c.build_graph(&HashMap::new()).unwrap();
    assert_eq!(fmt_shape(&g.shapes["fc1.out"]), "[B, 32]");
    assert_eq!(fmt_shape(&g.shapes["head.out"]), "[B, 3]");
    let mut ov = HashMap::new();
    ov.insert("hidden".to_string(), serde_yaml::Value::from(64));
    let g = c.build_graph(&ov).unwrap();
    assert_eq!(fmt_shape(&g.shapes["fc2.out"]), "[B, 64]");
}

#[test]
fn lstm_has_two_output_ports() {
    let r = registry();
    let g = r.get("sequence/lstm-classifier@v1").unwrap().build_graph(&HashMap::new()).unwrap();
    assert_eq!(fmt_shape(&g.shapes["seq.out"]), "[B, 2, 1]");
    assert_eq!(fmt_shape(&g.shapes["rnn.out"]), "[B, 2, 32]");
    assert_eq!(fmt_shape(&g.shapes["rnn.last"]), "[B, 32]");
}

#[test]
fn cnn_shapes_propagate() {
    let r = registry();
    let g = r.get("vision/cnn-classifier@v1").unwrap().build_graph(&HashMap::new()).unwrap();
    assert_eq!(fmt_shape(&g.shapes["pool2.out"]), "[B, 32, 8, 8]");
    assert_eq!(fmt_shape(&g.shapes["flat.out"]), "[B, 2048]");
}

#[test]
fn gcn_and_transformer_validate() {
    let r = registry();
    let g = r.get("graph/gcn-classifier@v1").unwrap().build_graph(&HashMap::new()).unwrap();
    assert_eq!(fmt_shape(&g.shapes["g2.out"]), "[B, N, 4]");
    let t = r.get("sequence/transformer-classifier@v1").unwrap().build_graph(&HashMap::new()).unwrap();
    assert_eq!(fmt_shape(&t.shapes["pool.out"]), "[B, 64]");
}

#[test]
fn residual_skip_is_a_real_dag() {
    let r = registry();
    let g = r.get("tabular/residual-mlp@v1").unwrap().build_graph(&HashMap::new()).unwrap();
    let pos = |n: &str| g.order.iter().position(|x| x == n).unwrap();
    assert!(pos("embed") < pos("fc") && pos("fc") < pos("act") && pos("act") < pos("skip"));
    let code = generate_model(&g);
    assert!(code.contains("skip__out = act__out + embed__out"), "{code}");
}

#[test]
fn rejects_cycles() {
    let e = graph(
        "inputs: {x: [B, 4]}
nodes: {a: {block: linear, params: {out: 4}}, b: {block: linear, params: {out: 4}}, s: {block: add}}
edges: ['x -> a', 'a -> s.a', 'b -> s.b', 's -> b']
outputs: {y: s}",
    )
    .unwrap_err();
    assert!(matches!(e, GraphError::Cycle(_)), "{e}");
}

#[test]
fn rejects_shape_mismatch_on_add() {
    let e = graph(
        "inputs: {x: [B, 4]}
nodes: {a: {block: linear, params: {out: 8}}, s: {block: add}}
edges: ['x -> a', 'a -> s.a', 'x -> s.b']
outputs: {y: s}",
    )
    .unwrap_err();
    assert!(e.to_string().contains("cannot add [B, 8] and [B, 4]"), "{e}");
}

#[test]
fn rejects_missing_duplicate_and_bad_ports() {
    let base = |edges: &str| {
        graph(&format!(
            "inputs: {{x: [B, 4]}}\nnodes: {{a: {{block: linear, params: {{out: 4}}}}, s: {{block: add}}}}\nedges: {edges}\noutputs: {{y: s}}"
        ))
    };
    assert!(matches!(base("['x -> a', 'a -> s.a']").unwrap_err(), GraphError::MissingInput { .. }));
    assert!(matches!(base("['x -> a', 'x -> a', 'a -> s.a', 'a -> s.b']").unwrap_err(), GraphError::DuplicateInput { .. }));
    assert!(matches!(base("['x -> a.nope']").unwrap_err(), GraphError::EdgeError { .. }));
    assert!(matches!(base("['x a']").unwrap_err(), GraphError::BadEdge(_)));
}

#[test]
fn rejects_bad_params_and_unknown_blocks() {
    let e = graph("inputs: {x: [B, 4]}\nnodes: {a: {block: linear}}\nedges: ['x -> a']\noutputs: {y: a}").unwrap_err();
    assert!(matches!(e, GraphError::BadParam { .. }), "{e}");
    let e = graph("inputs: {x: [B, 4]}\nnodes: {a: {block: linear, params: {out: 2, nope: 1}}}\nedges: ['x -> a']\noutputs: {y: a}").unwrap_err();
    assert!(e.to_string().contains("unknown parameter"), "{e}");
    let e = graph("inputs: {x: [B, 4]}\nnodes: {a: {block: bogus}}\nedges: ['x -> a']\noutputs: {y: a}").unwrap_err();
    assert!(matches!(e, GraphError::UnknownBlock { .. }));
}

#[test]
fn multi_output_requires_explicit_port() {
    let e = graph(
        "inputs: {x: [B, 5, 3]}
nodes: {r: {block: lstm, params: {hidden: 8}}, h: {block: linear, params: {out: 2}}}
edges: ['x -> r', 'r -> h']
outputs: {y: h}",
    )
    .unwrap_err();
    assert!(e.to_string().contains("several outputs"), "{e}");
}

#[test]
fn attention_heads_must_divide_embedding() {
    let e = graph("inputs: {x: [B, 5, 30]}\nnodes: {a: {block: attention, params: {heads: 4}}}\nedges: ['x -> a']\noutputs: {y: a}").unwrap_err();
    assert!(e.to_string().contains("not divisible"), "{e}");
}

#[test]
fn weight_sharing_is_validated_and_emitted() {
    let ok = graph(
        "inputs: {x: [B, 4], y: [B, 4]}
nodes:
  a: {block: linear, params: {out: 4}}
  b: {block: linear, params: {out: 4}, share: a}
  s: {block: add}
edges: ['x -> a', 'y -> b', 'a -> s.a', 'b -> s.b']
outputs: {o: s}",
    )
    .unwrap();
    assert!(generate_model(&ok).contains("self.b = self.a"));
    let e = graph(
        "inputs: {x: [B, 4], y: [B, 6]}
nodes:
  a: {block: linear, params: {out: 4}}
  b: {block: linear, params: {out: 4}, share: a}
edges: ['x -> a', 'y -> b']
outputs: {o: a, p: b}",
    )
    .unwrap_err();
    assert!(matches!(e, GraphError::BadShare { .. }), "{e}");
}

#[test]
fn block_aliases_and_catalog() {
    assert_eq!(Block::parse("lstm@v1"), Some(Block::Lstm));
    assert!(Block::parse("nope").is_none());
    assert!(Block::all().len() >= 20);
    for b in Block::all() {
        let i = b.info();
        assert!(!i.inputs.is_empty() && !i.outputs.is_empty());
        assert_eq!(i.tier, "fundamental");
    }
}

#[test]
fn component_with_invalid_graph_fails_to_load() {
    let yaml = "component_id: x/y@v1\nname: Y\ndescription: d\ndomain: x\ngraph:\n  inputs: {x: [B, 4]}\n  nodes: {a: {block: bogus}}\n  edges: ['x -> a']\n  outputs: {y: a}\n";
    assert!(Component::from_yaml(yaml).is_err());
}

#[test]
fn training_script_requires_training_section() {
    let r = registry();
    let g = r.get("vision/cnn-classifier@v1").unwrap().build_graph(&HashMap::new()).unwrap();
    assert!(matches!(generate_training_script(&g, "x").unwrap_err(), GraphError::Training(_)));
}

fn have_torch() -> bool {
    std::process::Command::new("python3")
        .args(["-c", "import torch"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Generate every shipped model and instantiate it in real PyTorch with concrete shapes.
#[test]
fn generated_models_run_forward_in_pytorch() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let cases: &[(&str, &str)] = &[
        ("tabular/ann-classifier@v1", "m(torch.randn(5, 2)).shape == (5, 3)"),
        ("tabular/residual-mlp@v1", "m(torch.randn(5, 2)).shape == (5, 3)"),
        ("tabular/deep-residual-mlp@v1", "m(torch.randn(5, 2)).shape == (5, 3)"),
        ("sequence/lstm-classifier@v1", "m(torch.randn(5, 2)).shape == (5, 3)"),
        ("vision/cnn-classifier@v1", "m(torch.randn(2, 3, 32, 32)).shape == (2, 10)"),
        ("sequence/transformer-classifier@v1", "m(torch.randint(0, 1000, (2, 7))).shape == (2, 2)"),
        ("graph/gcn-classifier@v1", "m(x=torch.randn(2, 6, 16), adj=torch.rand(2, 6, 6)).shape == (2, 6, 4)"),
    ];
    let r = registry();
    for (id, check) in cases {
        let g = r.get(id).unwrap().build_graph(&HashMap::new()).unwrap();
        let script = format!(
            "import torch\nimport torch.nn as nn\n{}\nm = Model()\nassert {check}, 'bad output shape'\nprint('ok')\n",
            generate_model(&g)
        );
        let out = std::process::Command::new("python3").args(["-I", "-c", &script]).output().unwrap();
        assert!(
            out.status.success(),
            "{id} failed:\n{}\n--- script ---\n{script}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn training_script_logs_model_hyperparameters() {
    let r = registry();
    let c = r.get("tabular/ann-classifier@v1").unwrap();
    let mut ov = HashMap::new();
    ov.insert("hidden".to_string(), serde_yaml::Value::from(7));
    let g = c.build_graph(&ov).unwrap();
    let script = generate_training_script(&g, "tabular/ann-classifier@v1").unwrap();
    // fc1.out must be recorded with the run so runs can be compared by their hyperparameters
    assert!(script.contains(r#"\"fc1\":{"#) && script.contains(r#"\"out\":7"#), "{script}");
}

#[test]
fn features_accept_list_or_comma_string_and_component_columns_are_parameters() {
    let r = registry();
    let c = r.get("tabular/ann-classifier@v1").unwrap();
    let mut ov = HashMap::new();
    ov.insert("feature_cols".to_string(), serde_yaml::Value::from("a, b ,c"));
    ov.insert("target".to_string(), serde_yaml::Value::from("species"));
    ov.insert("features".to_string(), serde_yaml::Value::from(3));
    let t = c.build_graph(&ov).unwrap().spec.training.unwrap();
    assert_eq!(t.data.features, ["a", "b", "c"]);
    assert_eq!(t.data.target, "species");
}

/// Real PyTorch run: string class labels + a row with a missing value, on a 3-class CSV.
#[test]
fn training_script_handles_string_labels_and_missing_values_in_pytorch() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sentinel-strlabels-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut csv = String::from("sepal,petal,species\n");
    for i in 0..40 {
        let f = i as f64 / 40.0;
        csv.push_str(&format!("{:.3},{:.3},low\n{:.3},{:.3},mid\n{:.3},{:.3},high\n", f, f * 0.2, 3.0 + f, 1.0 + f * 0.2, 6.0 + f, 4.0 + f * 0.2));
    }
    csv.push_str("1.0,,low\n"); // missing petal: must be skipped, not crash
    std::fs::write(dir.join("d.csv"), csv).unwrap();

    let r = registry();
    let c = r.get("tabular/ann-classifier@v1").unwrap();
    let mut ov = HashMap::new();
    ov.insert("csv".to_string(), serde_yaml::Value::from("d.csv"));
    ov.insert("feature_cols".to_string(), serde_yaml::Value::from("sepal,petal"));
    ov.insert("target".to_string(), serde_yaml::Value::from("species"));
    ov.insert("classes".to_string(), serde_yaml::Value::from(3));
    ov.insert("epochs".to_string(), serde_yaml::Value::from(150));
    let script = generate_training_script(&c.build_graph(&ov).unwrap(), "t").unwrap();
    std::fs::write(dir.join("train.py"), &script).unwrap();
    let out = std::process::Command::new("python3").args(["-I", "train.py"]).current_dir(&dir).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}\n{}", String::from_utf8_lossy(&out.stderr), script);
    assert!(stdout.contains("skipped 1 rows with missing values"), "{stdout}");
    assert!(stdout.contains("classes: ['high', 'low', 'mid']"), "{stdout}");
    let acc: f64 = stdout.lines().filter(|l| l.contains("test_acc")).last().unwrap().rsplit(' ').next().unwrap().parse().unwrap();
    assert!(acc > 0.9, "accuracy {acc}\n{stdout}");

    // wrong output size must give a clear message, not a cryptic tensor error
    ov.insert("classes".to_string(), serde_yaml::Value::from(2));
    let bad = generate_training_script(&c.build_graph(&ov).unwrap(), "t").unwrap();
    std::fs::write(dir.join("bad.py"), bad).unwrap();
    let out = std::process::Command::new("python3").args(["-I", "bad.py"]).current_dir(&dir).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("data has 3 classes"), "{}", String::from_utf8_lossy(&out.stderr));
}
