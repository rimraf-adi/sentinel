//! The analytic parameter counts and MACs must agree with what PyTorch itself reports.

use sentinel_core::codegen::generate_model;
use sentinel_core::components::{Built, ComponentRegistry};
use sentinel_core::compound::BlockLibrary;
use sentinel_core::graph::{GraphSpec, ValidatedGraph};
use serde_json::{json, Value};
use std::collections::HashMap;

fn have_torch() -> bool {
    std::process::Command::new("python3").args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

const MEASURE: &str = r#"
import json, sys, torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
res = {}
for c in json.load(sys.stdin):
    ns = {"torch": torch, "nn": nn}
    exec(c["code"], ns)
    m = ns["Model"]()
    r = {"params": sum(p.numel() for p in m.parameters())}
    if c.get("input"):
        m.train(False)
        with FlopCounterMode(display=False) as fc:
            m(torch.zeros(*c["input"]))
        r["flops"] = fc.get_total_flops()
    res[c["name"]] = r
print(json.dumps(res))
"#;

fn measure(cases: &[(String, &ValidatedGraph, Option<Vec<i64>>)]) -> HashMap<String, Value> {
    let payload: Vec<Value> = cases.iter().map(|(n, g, input)| json!({ "name": n, "code": generate_model(g), "input": input })).collect();
    let mut child = std::process::Command::new("python3").args(["-I", "-c", MEASURE]).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), serde_json::to_string(&payload).unwrap().as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
}

fn graph(yaml: &str) -> ValidatedGraph {
    GraphSpec::from_yaml(yaml).unwrap().validate_with(&BlockLibrary::builtin()).unwrap_or_else(|e| panic!("{e}\n{yaml}"))
}

#[test]
fn parameter_counts_match_pytorch_for_every_shipped_component() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let reg = ComponentRegistry::new();
    let built: Vec<(String, ValidatedGraph)> = reg
        .list()
        .into_iter()
        .filter(|c| c.graph.is_some())
        .map(|c| (c.component_id.clone(), c.build_graph(&HashMap::new()).unwrap()))
        .collect();
    assert!(built.len() >= 11);
    let cases: Vec<_> = built.iter().map(|(n, g)| (n.clone(), g, None)).collect();
    let real = measure(&cases);
    for (name, g) in &built {
        let want = real[name]["params"].as_u64().unwrap();
        assert_eq!(g.stats().params, want, "{name}: analytic count differs from PyTorch");
    }
    // a GAN is two graphs; the total is their sum
    if let Built::Gan(v) = reg.get("generative/gan@v1").unwrap().build(&HashMap::new()).unwrap() {
        let real = measure(&[("g".into(), &v.generator, None), ("d".into(), &v.discriminator, None)]);
        assert_eq!(v.generator.stats().params, real["g"]["params"].as_u64().unwrap());
        assert_eq!(v.discriminator.stats().params, real["d"]["params"].as_u64().unwrap());
    }
}

#[test]
fn every_parameterised_block_counts_correctly() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let specs: &[(&str, &str)] = &[
        ("linear_nobias", "inputs: {x: [B, 7]}\nnodes: {a: {block: linear, params: {out: 5, bias: false}}, b: {block: linear, params: {out: 3}}}\nedges: ['x -> a', 'a -> b']\noutputs: {y: b}"),
        ("norms", "inputs: {x: [B, 6, 10]}\nnodes: {l: {block: layernorm}, b: {block: batchnorm}}\nedges: ['x -> l', 'l -> b']\noutputs: {y: b}"),
        ("lstm_deep_bi", "inputs: {x: [B, 5, 7]}\nnodes: {r: {block: lstm, params: {hidden: 6, layers: 3, bidirectional: true}}}\nedges: ['x -> r']\noutputs: {y: r.out}"),
        ("gru_deep_bi", "inputs: {x: [B, 5, 7]}\nnodes: {r: {block: gru, params: {hidden: 6, layers: 2, bidirectional: true}}}\nedges: ['x -> r']\noutputs: {y: r.out}"),
        ("lstm_uni", "inputs: {x: [B, 5, 7]}\nnodes: {r: {block: lstm, params: {hidden: 9}}}\nedges: ['x -> r']\noutputs: {y: r.last}"),
        ("convs", "inputs: {x: [B, 3, 12]}\nnodes: {c: {block: conv1d, params: {out_channels: 4, kernel: 3}}}\nedges: ['x -> c']\noutputs: {y: c}"),
        ("conv2d_family", "inputs: {x: [B, 3, 8, 8]}\nnodes: {c: {block: conv2d, params: {out_channels: 5, kernel: 3, padding: 1}}, t: {block: conv_transpose2d, params: {out_channels: 2}}}\nedges: ['x -> c', 'c -> t']\noutputs: {y: t}"),
        ("embedding_attention", "inputs: {t: [B, 9]}\nnodes: {e: {block: embedding, params: {num_embeddings: 50, dim: 16}}, a: {block: attention, params: {heads: 4}}, f: {block: transformer_encoder, params: {heads: 2, ff: 24}}}\nedges: ['t -> e', 'e -> a', 'a -> f']\noutputs: {y: f}"),
        ("patch_cls_pos", "inputs: {i: [B, 3, 16, 16]}\nnodes: {p: {block: patch_embed, params: {dim: 12, patch: 4}}, c: {block: cls_token}, l: {block: positional_encoding, params: {max_len: 40}}, s: {block: positional_encoding, params: {max_len: 40, kind: sinusoidal}}}\nedges: ['i -> p', 'p -> c', 'c -> l', 'l -> s']\noutputs: {y: s}"),
        ("graph_layers", "inputs: {x: [B, 6, 10], adj: [B, 6, 6]}\nnodes: {g: {block: gcn, params: {out: 7}}, a: {block: gat, params: {out: 5, heads: 3, concat: false}}}\nedges: ['x -> g.x', 'adj -> g.adj', 'g -> a.x', 'adj -> a.adj']\noutputs: {y: a}"),
        ("shared_weights", "inputs: {x: [B, 8], y: [B, 8]}\nnodes: {a: {block: mlp_block, params: {width: 8}}, b: {block: mlp_block, params: {width: 8}, share: a}, s: {block: add}}\nedges: ['x -> a', 'y -> b', 'a -> s.a', 'b -> s.b']\noutputs: {o: s}"),
    ];
    let graphs: Vec<(String, ValidatedGraph)> = specs.iter().map(|(n, y)| (n.to_string(), graph(y))).collect();
    let cases: Vec<_> = graphs.iter().map(|(n, g)| (n.clone(), g, None)).collect();
    let real = measure(&cases);
    for (name, g) in &graphs {
        assert_eq!(g.stats().params, real[name]["params"].as_u64().unwrap(), "{name}");
    }
    // weight sharing: the second mlp_block adds nothing
    let shared = graphs.iter().find(|(n, _)| n == "shared_weights").unwrap().1.stats();
    assert!(shared.by_node["b"].shared && shared.by_node["b"].params == 0);
    assert_eq!(shared.params, shared.by_node["a"].params);
}

#[test]
fn macs_match_pytorchs_flop_counter() {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let reg = ComponentRegistry::new();
    let ids = ["tabular/ann-classifier@v1", "tabular/residual-mlp@v1", "tabular/deep-residual-mlp@v1", "vision/cnn-classifier@v1", "vision/resnet-mini@v1", "vision/unet@v1"];
    let built: Vec<(String, ValidatedGraph, Vec<i64>)> = ids
        .iter()
        .map(|id| {
            let g = reg.get(id).unwrap().build_graph(&HashMap::new()).unwrap();
            let shape = g.spec.inputs.values().next().unwrap().iter().map(|d| match d { sentinel_core::graph::Dim::Const(n) => *n, _ => 1 }).collect();
            (id.to_string(), g, shape)
        })
        .collect();
    let cases: Vec<_> = built.iter().map(|(n, g, s)| (n.clone(), g, Some(s.clone()))).collect();
    let real = measure(&cases);
    for (name, g, _) in &built {
        let macs = g.stats().macs.unwrap_or_else(|| panic!("{name}: MACs unknown"));
        let flops = real[name]["flops"].as_u64().unwrap();
        assert_eq!(flops, 2 * macs, "{name}: PyTorch counts {flops} FLOPs, expected 2 x {macs} MACs");
    }
}

#[test]
fn stats_describe_the_hierarchy() {
    let g = ComponentRegistry::new().get("vision/unet@v1").unwrap().build_graph(&HashMap::new()).unwrap();
    let st = g.stats();
    assert!(st.depth >= 2, "model -> unet_down -> conv_block: depth {}", st.depth);
    assert!(st.compounds >= 10 && st.blocks > st.compounds);
    assert_eq!(st.bytes_fp32, st.params * 4);
    // the per-block-type breakdown accounts for every parameter
    assert_eq!(st.by_block.values().sum::<u64>(), st.params);
    assert!(st.by_block["conv2d"] > st.by_block["batchnorm"], "{:?}", st.by_block);
    // a compound node's parameters are the sum of its inner nodes
    let d1 = &st.by_node["d1"];
    let inner = d1.inner.as_ref().unwrap();
    assert_eq!(d1.params, inner.params);
    assert_eq!(inner.params, inner.by_node.values().map(|n| n.params).sum::<u64>());
    // symbolic sizes make MACs unknown rather than wrong
    let sym = graph("inputs: {x: [B, T, 8]}\nnodes: {a: {block: linear, params: {out: 4}}}\nedges: ['x -> a']\noutputs: {y: a}");
    assert_eq!(sym.stats().macs, None);
    assert_eq!(sym.stats().params, 8 * 4 + 4);
}
