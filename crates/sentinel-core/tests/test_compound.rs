use sentinel_core::codegen::generate_model;
use sentinel_core::compound::{BlockKind, BlockLibrary, CompoundDef};
use sentinel_core::graph::{fmt_shape, GraphError, GraphSpec};

fn graph(lib: &BlockLibrary, yaml: &str) -> Result<sentinel_core::graph::ValidatedGraph, GraphError> {
    GraphSpec::from_yaml(yaml)?.validate_with(lib)
}

fn lib_with(defs: &[&str]) -> BlockLibrary {
    let mut lib = BlockLibrary::builtin();
    for d in defs {
        lib.insert(CompoundDef::from_yaml(d).unwrap_or_else(|e| panic!("{e}\n{d}"))).unwrap();
    }
    lib
}

fn have_torch() -> bool {
    std::process::Command::new("python3").args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

fn run_python(body: &str) {
    let out = std::process::Command::new("python3").args(["-I", "-c", body]).output().unwrap();
    assert!(out.status.success(), "{}\n--- script ---\n{body}", String::from_utf8_lossy(&out.stderr));
}

const MODEL: &str = "
inputs: {x: [B, 16]}
nodes:
  r1: {block: residual_mlp_block, params: {width: 16, hidden: 32}}
  r2: {block: residual_mlp_block, params: {width: 16, hidden: 32}}
  head: {block: linear, params: {out: 3}}
edges: ['x -> r1', 'r1 -> r2', 'r2 -> head']
outputs: {y: head}";

#[test]
fn library_has_both_tiers_and_tags_them() {
    let lib = BlockLibrary::builtin();
    let cat = lib.catalog();
    assert!(cat.iter().any(|b| b.name == "linear" && b.tier == "fundamental"));
    let res = cat.iter().find(|b| b.name == "residual_mlp_block").unwrap();
    assert_eq!(res.tier, "compound");
    assert_eq!(res.uses, ["add", "linear", "mlp_block"], "a compound reports the tier below it");
    assert!(res.params.iter().any(|p| p.name == "width" && p.required));
    assert!(matches!(lib.resolve("mlp_block@v1"), Some(BlockKind::Compound(_))));
    assert!(matches!(lib.resolve("linear"), Some(BlockKind::Fundamental(_))));
    assert!(lib.resolve("nope").is_none());
}

#[test]
fn shapes_flow_through_nested_compounds() {
    let g = graph(&BlockLibrary::builtin(), MODEL).unwrap();
    assert_eq!(fmt_shape(&g.shapes["r1.out"]), "[B, 16]");
    assert_eq!(fmt_shape(&g.shapes["head.out"]), "[B, 3]");
    // depth 2: residual_mlp_block contains an mlp_block, which was validated and kept
    let inner = &g.compounds["r1"];
    assert_eq!(fmt_shape(&inner.shapes["mlp.out"]), "[B, 32]");
    assert!(inner.compounds.contains_key("mlp"), "nested compound expansions are stored");
    assert_eq!(fmt_shape(&inner.compounds["mlp"].shapes["fc.out"]), "[B, 32]");
}

#[test]
fn errors_name_every_compound_on_the_path() {
    let bad = MODEL.replace("width: 16, hidden: 32}}\n  r2", "width: 8, hidden: 32}}\n  r2");
    let e = graph(&BlockLibrary::builtin(), &bad).unwrap_err().to_string();
    assert!(e.contains("compound block 'residual_mlp_block' (node 'r1')"), "{e}");
    assert!(e.contains("cannot add"), "{e}");
}

#[test]
fn compound_parameters_are_checked() {
    let lib = BlockLibrary::builtin();
    let missing = "inputs: {x: [B, 4, 8, 8]}\nnodes: {r: {block: resnet_block}}\nedges: ['x -> r']\noutputs: {y: r}";
    assert!(matches!(graph(&lib, missing).unwrap_err(), GraphError::BadParam { .. }));
    let unknown = "inputs: {x: [B, 4, 8, 8]}\nnodes: {r: {block: resnet_block, params: {channels: 4, nope: 1}}}\nedges: ['x -> r']\noutputs: {y: r}";
    assert!(graph(&lib, unknown).unwrap_err().to_string().contains("unknown parameter"));
    let ok = "inputs: {x: [B, 4, 8, 8]}\nnodes: {r: {block: resnet_block, params: {channels: 4}}}\nedges: ['x -> r']\noutputs: {y: r}";
    assert_eq!(fmt_shape(&graph(&lib, ok).unwrap().shapes["r.out"]), "[B, 4, 8, 8]");
}

#[test]
fn compound_that_contains_itself_is_rejected() {
    let selfref = "compound: loopy\ndescription: x\ngraph:\n  inputs: {in: [B, 4]}\n  nodes: {a: {block: loopy}}\n  edges: ['in -> a']\n  outputs: {out: a}\n";
    let lib = lib_with(&[selfref]);
    let e = graph(&lib, "inputs: {x: [B, 4]}\nnodes: {n: {block: loopy}}\nedges: ['x -> n']\noutputs: {y: n}").unwrap_err().to_string();
    assert!(e.contains("cannot contain themselves") && e.contains("loopy -> loopy"), "{e}");

    let a = "compound: ping\ngraph:\n  inputs: {in: [B, 4]}\n  nodes: {n: {block: pong}}\n  edges: ['in -> n']\n  outputs: {out: n}\n";
    let b = "compound: pong\ngraph:\n  inputs: {in: [B, 4]}\n  nodes: {n: {block: ping}}\n  edges: ['in -> n']\n  outputs: {out: n}\n";
    let lib = lib_with(&[a, b]);
    let e = graph(&lib, "inputs: {x: [B, 4]}\nnodes: {n: {block: ping}}\nedges: ['x -> n']\noutputs: {y: n}").unwrap_err().to_string();
    assert!(e.contains("ping -> pong -> ping"), "{e}");
}

#[test]
fn definition_checks() {
    for (yaml, needle) in [
        ("compound: linear\ngraph: {inputs: {in: [B,1]}, nodes: {a: {block: linear, params: {out: 1}}}, edges: ['in -> a'], outputs: {out: a}}", "fundamental block"),
        ("compound: Bad-Name\ngraph: {inputs: {in: [B,1]}, nodes: {}, edges: [], outputs: {out: x}}", "lowercase"),
        ("compound: uses_undeclared\ngraph: {inputs: {in: [B,1]}, nodes: {a: {block: linear, params: {out: '${w}'}}}, edges: ['in -> a'], outputs: {out: a}}", "declares no parameter 'w'"),
        ("compound: no_ports\ngraph: {inputs: {}, nodes: {}, edges: [], outputs: {}}", "at least one input"),
    ] {
        let e = CompoundDef::from_yaml(yaml).unwrap_err();
        assert!(e.contains(needle), "{needle}: {e}");
    }
}

#[test]
fn project_directory_loads_good_blocks_and_reports_bad_ones() {
    let dir = std::env::temp_dir().join(format!("sentinel-blocks-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("core")).unwrap();
    std::fs::write(dir.join("core/tiny.yaml"), "compound: tiny\nfamily: core\ngraph:\n  inputs: {in: [B, 4]}\n  nodes: {a: {block: linear, params: {out: 2}}}\n  edges: ['in -> a']\n  outputs: {out: a}\n").unwrap();
    std::fs::write(dir.join("core/broken.yaml"), "compound: broken\ngraph: [not, a, mapping]").unwrap();
    std::fs::write(dir.join("core/shadow.yaml"), "compound: mlp_block\ngraph:\n  inputs: {in: [B, 4]}\n  nodes: {a: {block: linear, params: {out: 2}}}\n  edges: ['in -> a']\n  outputs: {out: a}\n").unwrap();
    let (lib, errors) = BlockLibrary::with_dir(&dir);
    assert!(lib.resolve("tiny").is_some(), "good project block loads");
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(errors.iter().any(|e| e.contains("broken.yaml")) && errors.iter().any(|e| e.contains("already exists")), "{errors:?}");
    let g = graph(&lib, "inputs: {x: [B, 4]}\nnodes: {t: {block: tiny}}\nedges: ['x -> t']\noutputs: {y: t}").unwrap();
    assert_eq!(fmt_shape(&g.shapes["t.out"]), "[B, 2]");
}

#[test]
fn nested_classes_are_emitted_once_per_configuration() {
    let g = graph(&BlockLibrary::builtin(), MODEL).unwrap();
    let code = generate_model(&g);
    // r1 and r2 are identical configurations, so they share one class definition
    assert_eq!(code.matches("class ResidualMlpBlock").count(), 1, "{code}");
    assert!(code.contains("class MlpBlock") && code.contains("class Model"));
    // dependencies come first: MlpBlock is defined before ResidualMlpBlock, which is before Model
    let pos = |s: &str| code.find(s).unwrap();
    assert!(pos("class MlpBlock") < pos("class ResidualMlpBlock") && pos("class ResidualMlpBlock") < pos("class Model"));
    assert!(code.contains("self.r1 = ResidualMlpBlock") && code.contains("self.r2 = ResidualMlpBlock"));

    let other = MODEL.replace("r2: {block: residual_mlp_block, params: {width: 16, hidden: 32}}", "r2: {block: residual_mlp_block, params: {width: 16, hidden: 8}}");
    let code2 = generate_model(&graph(&BlockLibrary::builtin(), &other).unwrap());
    assert_eq!(code2.matches("class ResidualMlpBlock").count(), 2, "different configurations get their own class");
}

#[test]
fn nested_compounds_run_in_real_pytorch() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let mlp = generate_model(&graph(&BlockLibrary::builtin(), MODEL).unwrap());
    run_python(&format!("import torch\nimport torch.nn as nn\n{mlp}\nm = Model()\nassert m(torch.randn(5, 16)).shape == (5, 3)\n\
                         assert sum(p.numel() for p in m.parameters()) > 0\nprint('ok')"));

    let conv = "inputs: {img: [B, 8, 16, 16]}\nnodes:\n  a: {block: resnet_block, params: {channels: 8}}\n  b: {block: resnet_block, params: {channels: 8}}\n  pool: {block: maxpool2d}\n  flat: {block: flatten}\n  head: {block: linear, params: {out: 4}}\nedges: ['img -> a', 'a -> b', 'b -> pool', 'pool -> flat', 'flat -> head']\noutputs: {y: head}";
    let code = generate_model(&graph(&BlockLibrary::builtin(), conv).unwrap());
    assert!(code.contains("class ConvBlock") && code.contains("class ResnetBlock"));
    run_python(&format!("import torch\nimport torch.nn as nn\n{code}\nm = Model()\nassert m(torch.randn(2, 8, 16, 16)).shape == (2, 4)\nprint('ok')"));
}

#[test]
fn sharing_weights_between_compound_instances_ties_their_parameters() {
    let shared = "inputs: {x: [B, 16], y: [B, 16]}\nnodes:\n  a: {block: mlp_block, params: {width: 16}}\n  b: {block: mlp_block, params: {width: 16}, share: a}\n  s: {block: add}\nedges: ['x -> a', 'y -> b', 'a -> s.a', 'b -> s.b']\noutputs: {o: s}";
    let g = graph(&BlockLibrary::builtin(), shared).unwrap();
    let code = generate_model(&g);
    assert!(code.contains("self.b = self.a"), "{code}");
    if have_torch() {
        run_python(&format!("import torch\nimport torch.nn as nn\n{code}\nm = Model()\nassert len(list(m.parameters())) == 2, 'shared module counted once'\n\
                             assert m(x=torch.randn(3, 16), y=torch.randn(3, 16)).shape == (3, 16)\nprint('ok')"));
    }
    // different configurations cannot share (same shapes, so the sharing check is what must reject it)
    let differ = shared.replace("b: {block: mlp_block, params: {width: 16}, share: a}", "b: {block: mlp_block, params: {width: 16, dropout: 0.5}, share: a}");
    assert!(matches!(graph(&BlockLibrary::builtin(), &differ).unwrap_err(), GraphError::BadShare { .. }));
}

#[test]
fn multi_output_compounds_and_python_keyword_ports_work() {
    // `in` is a Python keyword and the block has two outputs: both must still generate valid code.
    let split = "compound: fork\nfamily: core\ndescription: two linear heads\ngraph:\n  inputs: {in: [B, 8]}\n  nodes:\n    l: {block: linear, params: {out: 4}}\n    r: {block: linear, params: {out: 4}}\n  edges: ['in -> l', 'in -> r']\n  outputs: {left: l, right: r}\n";
    let lib = lib_with(&[split]);
    let m = "inputs: {x: [B, 8]}\nnodes:\n  f: {block: fork}\n  s: {block: add}\nedges: ['x -> f', 'f.left -> s.a', 'f.right -> s.b']\noutputs: {y: s}";
    let g = graph(&lib, m).unwrap();
    assert_eq!(fmt_shape(&g.shapes["f.left"]), "[B, 4]");
    let code = generate_model(&g);
    assert!(code.contains("def forward(self, in_)"), "{code}");
    assert!(code.contains("f__left, f__right = self.f(x)"), "{code}");
    if have_torch() {
        run_python(&format!("import torch\nimport torch.nn as nn\n{code}\nm = Model()\nassert m(torch.randn(3, 8)).shape == (3, 4)\nprint('ok')"));
    }
    // a multi-output compound must be wired by port, like a multi-output fundamental block
    let ambiguous = "inputs: {x: [B, 8]}\nnodes: {f: {block: fork}, h: {block: linear, params: {out: 2}}}\nedges: ['x -> f', 'f -> h']\noutputs: {y: h}";
    assert!(graph(&lib, ambiguous).unwrap_err().to_string().contains("several outputs"));
}
