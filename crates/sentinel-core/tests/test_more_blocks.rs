//! The second batch of fundamental blocks. Every block is built into a real graph, its generated PyTorch is run,
//! and the shapes, parameter counts, gradients and (where PyTorch's FLOP counter covers the layer) MACs the Rust side
//! inferred are compared with what PyTorch actually produced. Numeric behaviour is checked per block.

use sentinel_core::blocks::Block;
use sentinel_core::codegen::generate_model;
use sentinel_core::compound::BlockLibrary;
use sentinel_core::graph::{fmt_shape, Dim, GraphError, GraphSpec, ValidatedGraph};
use serde_json::{json, Value};

fn graph(yaml: &str) -> Result<ValidatedGraph, GraphError> {
    GraphSpec::from_yaml(yaml)?.validate_with(&BlockLibrary::builtin())
}

fn have_torch() -> bool {
    std::process::Command::new("python3").args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

const PRELUDE: &str = "import json\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.flop_counter import FlopCounterMode\n";

/// Instantiate the model, feed it random inputs, and record what the Rust side wants to compare. Placeholders:
/// __INPUTS__ (name/shape pairs), __SINGLE__ (name of a lone output), __FLOPS__.
const SETUP: &str = r#"
torch.manual_seed(0)
m = Model()
r = {"params": sum(p.numel() for p in m.parameters())}
xs = [(torch.rand(*s) > 0.5).float() if n.startswith("adj") else torch.randn(*s) for n, s in __INPUTS__]
def run():
    out = m(*xs)
    return out if isinstance(out, dict) else {__SINGLE__: out}
m.train(False)
y = run()
r["shapes"] = {k: list(v.shape) for k, v in y.items()}
if __FLOPS__:
    with FlopCounterMode(display=False) as fc:
        m(*xs)
    r["flops"] = fc.get_total_flops()
m.train(True)
floats = [v for v in run().values() if v.is_floating_point()]
if r["params"] > 0 and floats:
    sum(v.sum() for v in floats).backward()
    r["no_grad"] = [n for n, p in m.named_parameters() if p.grad is None]
m.train(False)
"#;

struct Case {
    name: &'static str,
    yaml: &'static str,
    /// Python run against the eval-mode model `m`, its inputs `xs` and its outputs `y` (a dict by output name).
    body: &'static str,
    /// Compare the MAC count with PyTorch's FLOP counter (only for layers it covers).
    flops: bool,
}

fn sub_b(shape: &[Dim]) -> Vec<i64> {
    shape
        .iter()
        .map(|d| match d {
            Dim::Const(n) => *n,
            Dim::Sym(s) if s == "B" => 2,
            Dim::Sym(s) => panic!("unexpected symbolic dimension {s}"),
        })
        .collect()
}

/// Build the script for one case and run it in its own interpreter.
fn run_case(c: &Case, g: &ValidatedGraph) -> Value {
    let inputs: Vec<Value> = g.spec.inputs.iter().map(|(k, s)| json!([k, sub_b(s)])).collect();
    let setup = SETUP
        .replace("__INPUTS__", &Value::Array(inputs).to_string())
        .replace("__SINGLE__", &json!(g.outputs.keys().next().unwrap()).to_string())
        .replace("__FLOPS__", if c.flops { "True" } else { "False" });
    let script = format!("{PRELUDE}{}\n{setup}\n{}\nprint('RESULT ' + json.dumps(r))\n", generate_model(g), c.body);
    let out = std::process::Command::new("python3").args(["-I", "-c", &script]).output().unwrap();
    assert!(out.status.success(), "[{}] PyTorch failed:\n{}\n--- script ---\n{script}", c.name, String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().rev().find_map(|l| l.strip_prefix("RESULT ")).unwrap_or_else(|| panic!("[{}] no result line in:\n{stdout}", c.name));
    serde_json::from_str(line).unwrap()
}

fn run_cases(cases: &[Case]) {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let graphs: Vec<ValidatedGraph> = cases.iter().map(|c| graph(c.yaml).unwrap_or_else(|e| panic!("{}: {e}", c.name))).collect();
    let results: Vec<Value> = std::thread::scope(|s| {
        let handles: Vec<_> = cases.iter().zip(&graphs).map(|(c, g)| s.spawn(move || run_case(c, g))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    });

    for ((c, g), r) in cases.iter().zip(&graphs).zip(&results) {
        let st = g.stats();
        assert_eq!(r["params"].as_u64().unwrap(), st.params, "[{}] analytic parameter count vs PyTorch", c.name);
        for (name, src) in &g.outputs {
            let want = sub_b(g.shape_of(src));
            let got: Vec<i64> = r["shapes"][name].as_array().unwrap_or_else(|| panic!("[{}] output {name} missing", c.name)).iter().map(|v| v.as_i64().unwrap()).collect();
            assert_eq!(want, got, "[{}] inferred shape of output '{name}' ({}) vs PyTorch", c.name, fmt_shape(g.shape_of(src)));
        }
        if let Some(missing) = r.get("no_grad") {
            assert!(missing.as_array().unwrap().is_empty(), "[{}] parameters without a gradient: {missing}", c.name);
        }
        if c.flops {
            let macs = st.macs.unwrap_or_else(|| panic!("[{}] macs unknown", c.name));
            assert_eq!(r["flops"].as_u64().unwrap(), 2 * 2 * macs, "[{}] FLOPs = 2 x MACs x batch (2)", c.name);
        }
    }
}

// ---- catalog ------------------------------------------------------------------------------------------------------

#[test]
fn catalog_lists_every_new_block_in_its_family() {
    let expect = [
        ("shape", "multiply subtract divide scale clamp reduce argmax permute transpose squeeze unsqueeze split slice stack repeat pad matmul"),
        ("core", "groupnorm instancenorm rmsnorm l2_normalize dropout2d alpha_dropout gaussian_noise bilinear"),
        ("vision", "maxpool1d avgpool1d adaptive_avgpool2d global_max_pool conv3d conv_transpose1d pixel_shuffle depthwise_conv2d"),
        ("sequence", "rnn"),
        ("attention", "rope cross_attention"),
        ("graph", "sage_conv gin_conv edge_dropout"),
        ("generative", "noise_like vector_quantizer gumbel_softmax"),
    ];
    let mut n = 0;
    for (family, names) in expect {
        for name in names.split_whitespace() {
            let b = Block::parse(name).unwrap_or_else(|| panic!("{name} missing"));
            let info = b.info();
            assert_eq!((info.family.as_str(), info.tier), (family, "fundamental"), "{name}");
            assert!(info.description.len() > 20, "{name} needs a real description");
            for p in &info.params {
                assert!(!p.help.is_empty(), "{name}.{} needs help text", p.name);
            }
            n += 1;
        }
    }
    assert_eq!(n, 42);
    assert!(Block::all().len() >= 72, "30 original + 42 new");
    let names: std::collections::BTreeSet<_> = Block::all().iter().map(|b| b.name()).collect();
    assert_eq!(names.len(), Block::all().len(), "block names are unique");
}

// ---- tensor operations --------------------------------------------------------------------------------------------

#[test]
fn elementwise_ops_and_broadcasting() {
    run_cases(&[
        Case {
            name: "squeeze_excitation",
            yaml: "
inputs: {x: [B, 8, 4, 4]}
nodes:
  gap: {block: global_avg_pool}
  fc1: {block: linear, params: {out: 2}}
  act: {block: activation, params: {kind: relu}}
  fc2: {block: linear, params: {out: 8}}
  gate: {block: activation, params: {kind: sigmoid}}
  u1: {block: unsqueeze, params: {dim: 2}}
  u2: {block: unsqueeze, params: {dim: 3}}
  mul: {block: multiply}
edges: ['x -> gap', 'gap -> fc1', 'fc1 -> act', 'act -> fc2', 'fc2 -> gate', 'gate -> u1', 'u1 -> u2', 'x -> mul.a', 'u2 -> mul.b']
outputs: {y: mul}",
            body: "q = y['y'] / xs[0]\nassert torch.allclose(q.amax((2, 3)), q.amin((2, 3)), atol=1e-4), 'one gate value per channel'\nassert (q > 0).all() and (q < 1).all()",
            flops: true,
        },
        Case {
            name: "binary_arithmetic",
            yaml: "
inputs: {a: [B, 3, 4], b: [B, 3, 4]}
nodes: {s: {block: subtract}, p: {block: multiply}, q: {block: divide}}
edges: ['a -> s.a', 'b -> s.b', 'a -> p.a', 'b -> p.b', 'a -> q.a', 'b -> q.b']
outputs: {s: s, p: p, q: q}",
            body: "a, b = xs\nassert torch.equal(y['s'], a - b) and torch.equal(y['p'], a * b) and torch.equal(y['q'], a / b)",
            flops: false,
        },
        Case {
            name: "scale_and_clamp",
            yaml: "
inputs: {x: [B, 6]}
nodes:
  sc: {block: scale, params: {factor: 2.0, shift: 1.0}}
  both: {block: clamp, params: {min: -1.0, max: 3.0}}
  lo: {block: clamp, params: {min: 0.0}}
  hi: {block: clamp, params: {max: 0.5}}
edges: ['x -> sc', 'sc -> both', 'x -> lo', 'x -> hi']
outputs: {scaled: sc, both: both, lo: lo, hi: hi}",
            body: "x = xs[0]\nassert torch.allclose(y['scaled'], x * 2 + 1)\nassert torch.equal(y['both'], (x * 2 + 1).clamp(-1, 3))\nassert torch.equal(y['lo'], x.clamp(min=0)) and torch.equal(y['hi'], x.clamp(max=0.5))",
            flops: false,
        },
    ]);
}

#[test]
fn reductions() {
    run_cases(&[Case {
        name: "reductions",
        yaml: "
inputs: {x: [B, 4, 5]}
nodes:
  s: {block: reduce, params: {op: sum, dim: 1}}
  mx: {block: reduce, params: {op: max, dim: 2, keepdim: true}}
  mn: {block: reduce, params: {op: min, dim: -1}}
  me: {block: reduce, params: {op: mean, dim: 1}}
  sd: {block: reduce, params: {op: std, dim: -1}}
  pr: {block: reduce, params: {op: prod, dim: 2}}
  am: {block: argmax, params: {dim: 1}}
edges: ['x -> s', 'x -> mx', 'x -> mn', 'x -> me', 'x -> sd', 'x -> pr', 'x -> am']
outputs: {s: s, mx: mx, mn: mn, me: me, sd: sd, pr: pr, am: am}",
        body: "x = xs[0]\nassert torch.allclose(y['s'], x.sum(1)) and torch.allclose(y['me'], x.mean(1))\nassert torch.equal(y['mx'], x.amax(2, keepdim=True)) and torch.equal(y['mn'], x.amin(-1))\nassert torch.allclose(y['sd'], x.std(-1)) and torch.allclose(y['pr'], x.prod(2))\nassert torch.equal(y['am'], x.argmax(1)) and y['am'].dtype == torch.int64",
        flops: false,
    }]);
}

#[test]
fn shape_manipulation() {
    run_cases(&[Case {
        name: "shape_moves",
        yaml: "
inputs: {x: [B, 2, 3, 4], z: [B, 1, 5]}
nodes:
  perm: {block: permute, params: {dims: [3, 1, 2]}}
  tr: {block: transpose, params: {dim0: 1, dim1: 3}}
  tr_neg: {block: transpose, params: {dim0: -1, dim1: -2}}
  sq: {block: squeeze, params: {dim: 1}}
  un: {block: unsqueeze, params: {dim: 3}}
  un_neg: {block: unsqueeze, params: {dim: -1}}
  rep: {block: repeat, params: {times: [1, 2, 1]}}
  sl: {block: slice, params: {dim: 3, start: 1, end: 3}}
  sl_open: {block: slice, params: {dim: 2, start: 1}}
edges: ['x -> perm', 'x -> tr', 'x -> tr_neg', 'z -> sq', 'z -> un', 'z -> un_neg', 'x -> rep', 'x -> sl', 'x -> sl_open']
outputs: {perm: perm, tr: tr, tr_neg: tr_neg, sq: sq, un: un, un_neg: un_neg, rep: rep, sl: sl, sl_open: sl_open}",
        body: "x, z = xs\nassert torch.equal(y['perm'], x.permute(0, 3, 1, 2)) and torch.equal(y['tr'], x.transpose(1, 3)) and torch.equal(y['tr_neg'], x.transpose(-1, -2))\nassert torch.equal(y['sq'], z.squeeze(1)) and torch.equal(y['un'], z.unsqueeze(3)) and torch.equal(y['un_neg'], z.unsqueeze(-1))\nassert torch.equal(y['rep'], x.repeat(1, 1, 2, 1)) and torch.equal(y['sl'], x[:, :, :, 1:3]) and torch.equal(y['sl_open'], x[:, :, 1:])",
        flops: false,
    }]);
}

#[test]
fn split_stack_matmul_pad() {
    run_cases(&[
        Case {
            name: "split_stack_matmul",
            yaml: "
inputs: {x: [B, 6, 8]}
nodes:
  sp: {block: split, params: {dim: 1}}
  st: {block: stack, params: {dim: 1}}
  qk: {block: matmul, params: {transpose_b: true}}
  av: {block: matmul}
edges: ['x -> sp', 'sp.a -> st.a', 'sp.b -> st.b', 'sp.a -> qk.a', 'sp.b -> qk.b', 'qk -> av.a', 'sp.b -> av.b']
outputs: {a: sp.a, b: sp.b, st: st, qk: qk, av: av}",
            body: "x = xs[0]\nassert torch.equal(torch.cat([y['a'], y['b']], 1), x)\nassert torch.equal(y['st'], torch.stack([y['a'], y['b']], 1))\nassert torch.allclose(y['qk'], y['a'] @ y['b'].transpose(1, 2)) and torch.allclose(y['av'], y['qk'] @ y['b'])",
            flops: true,
        },
        Case {
            name: "pad",
            yaml: "
inputs: {x: [B, 3, 4, 5]}
nodes: {p: {block: pad, params: {pad: [1, 2, 3, 0], value: 0.5}}, q: {block: pad, params: {pad: [2, 2]}}}
edges: ['x -> p', 'x -> q']
outputs: {p: p, q: q}",
            body: "x = xs[0]\nassert torch.equal(y['p'], F.pad(x, [1, 2, 3, 0], value=0.5)) and torch.equal(y['q'], F.pad(x, [2, 2]))",
            flops: false,
        },
    ]);
}

// ---- norms, dropouts, pools, basics -------------------------------------------------------------------------------

#[test]
fn normalization_layers() {
    run_cases(&[Case {
        name: "norms",
        yaml: "
inputs: {x: [B, 8, 6, 6], s: [B, 5, 16]}
nodes:
  gn: {block: groupnorm, params: {groups: 4}}
  inorm: {block: instancenorm, params: {affine: false}}
  inorm_a: {block: instancenorm}
  rms: {block: rmsnorm}
  l2: {block: l2_normalize}
edges: ['x -> gn', 'x -> inorm', 'x -> inorm_a', 's -> rms', 's -> l2']
outputs: {gn: gn, inorm: inorm, inorm_a: inorm_a, rms: rms, l2: l2}",
        body: "g = y['gn'].reshape(2, 4, -1)\nassert g.mean(-1).abs().max() < 1e-4 and (g.var(-1, unbiased=False) - 1).abs().max() < 1e-2\nassert y['inorm'].flatten(2).mean(-1).abs().max() < 1e-4\nassert ((y['rms'] ** 2).mean(-1) - 1).abs().max() < 1e-3\nassert (y['l2'].norm(dim=-1) - 1).abs().max() < 1e-5",
        flops: false,
    }]);
}

#[test]
fn dropouts_and_noise() {
    run_cases(&[Case {
        name: "dropouts",
        yaml: "
inputs: {x: [B, 8, 6, 6]}
nodes:
  d2: {block: dropout2d, params: {p: 0.5}}
  ad: {block: alpha_dropout, params: {p: 0.2}}
  gn: {block: gaussian_noise, params: {std: 0.5}}
edges: ['x -> d2', 'x -> ad', 'x -> gn']
outputs: {d2: d2, ad: ad, gn: gn}",
        body: "x = xs[0]\nassert all(torch.equal(y[k], x) for k in y), 'identity in eval mode'\nm.train(True)\ntorch.manual_seed(1)\nt = m(x)\nz = (t['d2'] == 0).flatten(2).float().mean(-1)\nassert ((z == 0) | (z == 1)).all() and z.max() == 1, 'whole channels are dropped'\nassert not torch.equal(t['ad'], x) and not torch.equal(t['gn'], x)\nassert 0.3 < (t['gn'] - x).std() < 0.7",
        flops: false,
    }]);
}

#[test]
fn pooling_layers() {
    run_cases(&[Case {
        name: "pools",
        yaml: "
inputs: {s: [B, 4, 10], x: [B, 4, 9, 9]}
nodes:
  mp: {block: maxpool1d}
  ap: {block: avgpool1d, params: {kernel: 3, stride: 2}}
  ada: {block: adaptive_avgpool2d, params: {size: 3}}
  gmp: {block: global_max_pool}
edges: ['s -> mp', 's -> ap', 'x -> ada', 'x -> gmp']
outputs: {mp: mp, ap: ap, ada: ada, gmp: gmp}",
        body: "s, x = xs\nassert torch.equal(y['mp'], F.max_pool1d(s, 2, 2)) and torch.allclose(y['ap'], F.avg_pool1d(s, 3, 2))\nassert torch.allclose(y['ada'], F.adaptive_avg_pool2d(x, 3)) and torch.equal(y['gmp'], x.amax((2, 3)))",
        flops: false,
    }]);
}

#[test]
fn bilinear_layer() {
    run_cases(&[Case {
        name: "bilinear",
        yaml: "
inputs: {a: [B, 5], b: [B, 3]}
nodes: {bl: {block: bilinear, params: {out: 7}}, nb: {block: bilinear, params: {out: 2, bias: false}}}
edges: ['a -> bl.a', 'b -> bl.b', 'a -> nb.a', 'b -> nb.b']
outputs: {bl: bl, nb: nb}",
        body: "a, b = xs\nassert torch.allclose(y['bl'], F.bilinear(a, b, m.bl.weight, m.bl.bias)) and m.nb.bias is None",
        flops: false,
    }]);
}

#[test]
fn plain_rnn() {
    run_cases(&[
        Case {
            name: "rnn_tanh",
            yaml: "
inputs: {x: [B, 6, 4]}
nodes: {r: {block: rnn, params: {hidden: 8}}}
edges: ['x -> r']
outputs: {seq: r.out, last: r.last}",
            body: "assert torch.equal(y['seq'][:, -1], y['last'])\nassert m.r.nonlinearity == 'tanh'",
            flops: false,
        },
        Case {
            name: "rnn_relu_bidirectional_stacked",
            yaml: "
inputs: {x: [B, 6, 4]}
nodes: {r: {block: rnn, params: {hidden: 8, layers: 2, bidirectional: true, nonlinearity: relu, dropout: 0.1}}, head: {block: linear, params: {out: 3}}}
edges: ['x -> r', 'r.last -> head']
outputs: {seq: r.out, logits: head}",
            body: "assert m.r.nonlinearity == 'relu' and y['seq'].shape == (2, 6, 16)",
            flops: false,
        },
    ]);
}

// ---- vision -------------------------------------------------------------------------------------------------------

#[test]
fn convolution_variants_match_pytorch_flops() {
    run_cases(&[
        Case {
            name: "conv3d",
            yaml: "
inputs: {x: [B, 2, 6, 6, 6]}
nodes: {c: {block: conv3d, params: {out_channels: 4, kernel: 3, padding: 1}}, d: {block: conv3d, params: {out_channels: 3, kernel: 2, stride: 2}}}
edges: ['x -> c', 'c -> d']
outputs: {y: d}",
            body: "assert y['y'].shape == (2, 3, 3, 3, 3)",
            flops: true,
        },
        Case {
            name: "conv_transpose1d",
            yaml: "
inputs: {x: [B, 3, 5]}
nodes: {u: {block: conv_transpose1d, params: {out_channels: 4, kernel: 4, stride: 2, padding: 1}}}
edges: ['x -> u']
outputs: {y: u}",
            body: "assert y['y'].shape == (2, 4, 10)",
            flops: true,
        },
        Case {
            name: "depthwise_conv2d",
            yaml: "
inputs: {x: [B, 3, 8, 8]}
nodes: {dw: {block: depthwise_conv2d, params: {multiplier: 2, padding: 1}}, pw: {block: conv2d, params: {out_channels: 5, kernel: 1}}}
edges: ['x -> dw', 'dw -> pw']
outputs: {y: pw}",
            body: "assert m.dw.groups == 3 and m.dw.out_channels == 6",
            flops: true,
        },
        Case {
            name: "pixel_shuffle",
            yaml: "
inputs: {x: [B, 8, 3, 3]}
nodes: {c: {block: conv2d, params: {out_channels: 8, kernel: 3, padding: 1}}, ps: {block: pixel_shuffle, params: {factor: 2}}}
edges: ['x -> c', 'c -> ps']
outputs: {y: ps}",
            body: "assert y['y'].shape == (2, 2, 6, 6)",
            flops: true,
        },
    ]);
}

// ---- attention ----------------------------------------------------------------------------------------------------

#[test]
fn rotary_embedding_rotates_by_position() {
    run_cases(&[Case {
        name: "rope",
        yaml: "
inputs: {x: [B, 7, 8]}
nodes: {r: {block: rope}}
edges: ['x -> r']
outputs: {y: r}",
        body: "x = xs[0]\nrot = y['y']\nassert torch.allclose(rot.norm(dim=-1), x.norm(dim=-1), atol=1e-5), 'a rotation keeps lengths'\nassert torch.allclose(rot[:, 0], x[:, 0], atol=1e-6), 'position 0 is not rotated'\nv = torch.randn(1, 1, 8).expand(1, 7, 8).contiguous()\nz = m(v)[0]\nd = z @ z.T\nfor i in range(4):\n    assert torch.allclose(d[i, i + 2], d[i + 1, i + 3], atol=1e-4), 'dot products depend only on the offset'\nassert len(list(m.parameters())) == 0",
        flops: false,
    }]);
}

#[test]
fn cross_attention_and_causal_masks() {
    run_cases(&[
        Case {
            name: "cross_attention",
            yaml: "
inputs: {q: [B, 5, 16], kv: [B, 9, 16]}
nodes: {ca: {block: cross_attention, params: {heads: 4}}}
edges: ['q -> ca.q', 'kv -> ca.kv']
outputs: {y: ca}",
            body: "kv, q = xs\nkv2 = kv.clone()\nkv2[:, 0] += 5\nassert not torch.allclose(m(kv2, q), y['y']), 'the output depends on the context'\nq2 = q.clone()\nq2[:, 3] += 5\nassert torch.allclose(m(kv, q2)[:, [0, 1, 2, 4]], y['y'][:, [0, 1, 2, 4]], atol=1e-5), 'each query attends independently'",
            flops: false,
        },
        Case {
            name: "causal_attention",
            yaml: "
inputs: {x: [B, 6, 16]}
nodes: {a: {block: attention, params: {heads: 4, causal: true}}}
edges: ['x -> a']
outputs: {y: a}",
            body: "x = xs[0].clone()\nx[:, -1] += 10\nz = m(x)\nassert torch.allclose(z[:, :-1], y['y'][:, :-1], atol=1e-5), 'earlier positions never see later ones'\nassert not torch.allclose(z[:, -1], y['y'][:, -1])",
            flops: false,
        },
        Case {
            name: "causal_transformer_encoder",
            yaml: "
inputs: {x: [B, 6, 16]}
nodes: {e: {block: transformer_encoder, params: {heads: 4, ff: 32, causal: true}}}
edges: ['x -> e']
outputs: {y: e}",
            body: "x = xs[0].clone()\nx[:, -1] += 10\nz = m(x)\nassert torch.allclose(z[:, :-1], y['y'][:, :-1], atol=1e-5)\nassert not torch.allclose(z[:, -1], y['y'][:, -1])",
            flops: false,
        },
        Case {
            name: "non_causal_attention_sees_the_future",
            yaml: "
inputs: {x: [B, 6, 16]}
nodes: {a: {block: attention, params: {heads: 4}}}
edges: ['x -> a']
outputs: {y: a}",
            body: "x = xs[0].clone()\nx[:, -1] += 10\nassert not torch.allclose(m(x)[:, :-1], y['y'][:, :-1], atol=1e-5)",
            flops: false,
        },
    ]);
}

// ---- graph --------------------------------------------------------------------------------------------------------

#[test]
fn graph_layers() {
    run_cases(&[Case {
        name: "graph_stack",
        yaml: "
inputs: {x: [B, 5, 6], adj: [B, 5, 5]}
nodes:
  ed: {block: edge_dropout, params: {p: 0.5}}
  sage: {block: sage_conv, params: {out: 8}}
  gin: {block: gin_conv, params: {out: 4}}
edges: ['x -> sage.x', 'adj -> sage.adj', 'sage -> gin.x', 'adj -> gin.adj', 'adj -> ed']
outputs: {nodes: gin, ed: ed}",
        body: "adj, x = xs\nassert torch.equal(y['ed'], adj), 'edge dropout is the identity at inference'\nm.train(True)\ntorch.manual_seed(2)\nt = m(adj, x)['ed']\nassert (t <= adj).all() and (t < adj).any() and set(t.unique().tolist()) <= {0.0, 1.0}",
        flops: true,
    }]);
}

// ---- generative ---------------------------------------------------------------------------------------------------

#[test]
fn quantizer_noise_and_gumbel() {
    run_cases(&[Case {
        name: "generative",
        yaml: "
inputs: {x: [B, 4, 8], l: [B, 5]}
nodes:
  vq: {block: vector_quantizer, params: {num_embeddings: 6}}
  nz: {block: noise_like, params: {std: 0.3}}
  gs: {block: gumbel_softmax, params: {tau: 0.5, hard: true}}
  soft: {block: gumbel_softmax}
edges: ['x -> vq', 'x -> nz', 'l -> gs', 'l -> soft']
outputs: {q: vq.out, idx: vq.indices, n: nz, g: gs, s: soft}",
        body: "l, x = xs\nm.train(False)\no = m(l, x)\ne = m.vq.embed\nassert o['idx'].dtype == torch.int64 and o['idx'].shape == (2, 4)\nassert torch.allclose(o['q'], e[o['idx']], atol=1e-5), 'every output vector is a codebook entry'\nd = ((x[:, :, None] - e) ** 2).sum(-1)\nassert torch.equal(d.argmin(-1), o['idx']), 'the nearest entry'\ne0 = e.clone()\nm.train(True)\nm(l, x)\nassert not torch.equal(e0, m.vq.embed), 'the codebook follows the data while training'\nm.train(False)\ng = o['g']\nassert (g.sum(-1) == 1).all() and ((g == 0) | (g == 1)).all(), 'hard samples are one-hot'\nassert torch.allclose(o['s'].sum(-1), torch.ones(2)), 'soft samples are distributions'\nassert 0.1 < o['n'].std() < 0.6 and o['n'].shape == x.shape",
        flops: false,
    }]);
}

#[test]
fn vector_quantizer_passes_gradients_straight_through() {
    if !have_torch() {
        return;
    }
    let g = graph(
        "inputs: {x: [B, 4, 8]}\nnodes: {enc: {block: linear, params: {out: 8}}, vq: {block: vector_quantizer, params: {num_embeddings: 6}}, dec: {block: linear, params: {out: 8}}}\nedges: ['x -> enc', 'enc -> vq', 'vq.out -> dec']\noutputs: {y: dec}",
    )
    .unwrap();
    let script = format!(
        "import torch\nimport torch.nn as nn\n{}\nm = Model()\nx = torch.randn(2, 4, 8)\nm(x).sum().backward()\nassert m.enc.weight.grad is not None and m.enc.weight.grad.abs().sum() > 0, 'the encoder trains through the quantizer'\nassert len(list(m.vq.parameters())) == 0\nprint('ok')",
        generate_model(&g)
    );
    let out = std::process::Command::new("python3").args(["-I", "-c", &script]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

// ---- errors -------------------------------------------------------------------------------------------------------

fn err(yaml: &str) -> String {
    graph(yaml).err().unwrap_or_else(|| panic!("expected an error for:\n{yaml}")).to_string()
}

#[test]
fn bad_shapes_and_parameters_are_refused_with_clear_messages() {
    let g = |inputs: &str, nodes: &str, edges: &str| format!("inputs: {{{inputs}}}\nnodes: {{{nodes}}}\nedges: [{edges}]\noutputs: {{y: n}}");
    let cases: Vec<(String, &str)> = vec![
        (g("a: [B, 3, 4], b: [B, 5, 4]", "n: {block: multiply}", "'a -> n.a', 'b -> n.b'"), "cannot broadcast"),
        (g("x: [B, 3, 4]", "n: {block: squeeze, params: {dim: 1}}", "'x -> n'"), "only size 1"),
        (g("x: [B, 3, 5]", "n: {block: split, params: {dim: -1}}", "'x -> n'"), "two equal halves"),
        (g("x: [B, 3, 4]", "n: {block: permute, params: {dims: [1, 1]}}", "'x -> n'"), "permutation"),
        (g("x: [B, 3, 4]", "n: {block: slice, params: {dim: 2, start: 1, end: 9}}", "'x -> n'"), "outside dimension"),
        (g("a: [B, 3, 4], b: [B, 5, 6]", "n: {block: matmul}", "'a -> n.a', 'b -> n.b'"), "inner dimensions differ"),
        (g("x: [B, 6, 4, 4]", "n: {block: groupnorm, params: {groups: 4}}", "'x -> n'"), "do not divide"),
        (g("x: [B, 6, 4, 4]", "n: {block: pixel_shuffle}", "'x -> n'"), "not divisible"),
        (g("x: [B, 5, 7]", "n: {block: rope}", "'x -> n'"), "must be even"),
        (g("x: [B, 5]", "n: {block: reduce, params: {dim: 0}}", "'x -> n'"), "non-batch"),
        (g("x: [B, 5]", "n: {block: clamp, params: {min: 2.0, max: 1.0}}", "'x -> n'"), "must not exceed max"),
        (g("x: [B, 5]", "n: {block: reduce, params: {op: median}}", "'x -> n'"), "sum, mean, max"),
        (g("q: [B, 4, 10], kv: [B, 4, 10]", "n: {block: cross_attention, params: {heads: 4}}", "'q -> n.q', 'kv -> n.kv'"), "not divisible"),
        (g("q: [B, 4, 8], kv: [B, 4, 6]", "n: {block: cross_attention}", "'q -> n.q', 'kv -> n.kv'"), "context feature size"),
        (g("x: [B, 3, 4]", "n: {block: pad, params: {pad: [1, 1, 1, 1, 1, 1, 1, 1]}}", "'x -> n'"), "pads"),
        (g("x: [B, 3, 4]", "n: {block: pad, params: {pad: [1, 2, 3]}}", "'x -> n'"), "even length"),
        (g("x: [B, 4]", "n: {block: gumbel_softmax, params: {tau: 0.0}}", "'x -> n'"), "> 0"),
        (g("x: [B, 4]", "n: {block: dropout2d}", "'x -> n'"), "rank 4"),
        (g("x: [B, 4, 4]", "n: {block: rnn, params: {hidden: 4, nonlinearity: gelu}}", "'x -> n'"), "tanh or relu"),
        (g("x: [B, 4]", "n: {block: vector_quantizer, params: {num_embeddings: 4, decay: 1.5}}", "'x -> n'"), "(0, 1)"),
    ];
    for (yaml, want) in &cases {
        let msg = err(yaml);
        assert!(msg.contains(want), "expected '{want}' in: {msg}\n{yaml}");
    }
}

#[test]
fn defaults_and_unknown_parameters() {
    let ok = graph("inputs: {x: [B, 8, 4, 4]}\nnodes: {n: {block: groupnorm}}\nedges: ['x -> n']\noutputs: {y: n}").unwrap();
    assert_eq!(ok.params["n"]["groups"].as_i64(), Some(8));
    let e = err("inputs: {x: [B, 4]}\nnodes: {n: {block: scale, params: {gain: 2}}}\nedges: ['x -> n']\noutputs: {y: n}");
    assert!(e.contains("unknown parameter") && e.contains("factor"), "{e}");
    let e = err("inputs: {x: [B, 4]}\nnodes: {n: {block: permute}}\nedges: ['x -> n']\noutputs: {y: n}");
    assert!(e.contains("required"), "{e}");
}

#[test]
fn symbolic_dimensions_survive_the_ops_that_can_keep_them() {
    // a symbolic sequence length flows through slice and reduce as an expression instead of being rejected
    let g = graph("inputs: {x: [B, T, 8]}\nnodes: {s: {block: slice, params: {dim: 2, start: 2, end: 6}}, r: {block: reduce, params: {op: mean, dim: 1}}}\nedges: ['x -> s', 'x -> r']\noutputs: {s: s, r: r}").unwrap();
    assert_eq!(fmt_shape(&g.shapes["s.out"]), "[B, T, 4]");
    assert_eq!(fmt_shape(&g.shapes["r.out"]), "[B, 8]");
}
