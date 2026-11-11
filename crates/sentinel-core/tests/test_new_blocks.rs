use sentinel_core::blocks::Block;
use sentinel_core::codegen::generate_model;
use sentinel_core::compound::BlockLibrary;
use sentinel_core::graph::{fmt_shape, GraphError, GraphSpec};

fn graph(yaml: &str) -> Result<sentinel_core::graph::ValidatedGraph, GraphError> {
    GraphSpec::from_yaml(yaml)?.validate_with(&BlockLibrary::builtin())
}

fn have_torch() -> bool {
    std::process::Command::new(sentinel_core::platform::python_exe()).args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false)
}

/// Generate the model from `yaml` and run `body` against it in real PyTorch (`m` is the instantiated Model).
fn torch(yaml: &str, body: &str) {
    if !have_torch() {
        eprintln!("skipping: torch unavailable");
        return;
    }
    let code = generate_model(&graph(yaml).unwrap());
    let script = format!("import torch\nimport torch.nn as nn\n{code}\ntorch.manual_seed(0)\nm = Model()\n{body}\nprint('ok')\n");
    let out = std::process::Command::new(sentinel_core::platform::python_exe()).args(["-I", "-c", &script]).output().unwrap();
    assert!(out.status.success(), "{}\n--- script ---\n{script}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn catalog_has_the_new_blocks() {
    for (name, family) in [
        ("avgpool2d", "vision"), ("global_avg_pool", "vision"), ("conv_transpose2d", "vision"), ("upsample", "vision"), ("patch_embed", "vision"),
        ("cls_token", "attention"), ("cls_select", "attention"), ("positional_encoding", "attention"), ("sampling", "generative"), ("gat", "graph"),
    ] {
        let b = Block::parse(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(b.info().family, family, "{name}");
        assert_eq!(b.info().tier, "fundamental");
    }
    assert!(Block::all().len() >= 30);
}

const VIT: &str = "
inputs: {img: [B, 3, 32, 32]}
nodes:
  patches: {block: patch_embed, params: {dim: 64, patch: 8}}
  cls: {block: cls_token}
  pos: {block: positional_encoding, params: {max_len: 64}}
  enc1: {block: transformer_encoder, params: {heads: 4}}
  enc2: {block: transformer_encoder, params: {heads: 4}}
  pick: {block: cls_select}
  head: {block: linear, params: {out: 10}}
edges: ['img -> patches', 'patches -> cls', 'cls -> pos', 'pos -> enc1', 'enc1 -> enc2', 'enc2 -> pick', 'pick -> head']
outputs: {logits: head}";

#[test]
fn vit_shapes_and_pytorch() {
    let g = graph(VIT).unwrap();
    assert_eq!(fmt_shape(&g.shapes["patches.out"]), "[B, 16, 64]");
    assert_eq!(fmt_shape(&g.shapes["cls.out"]), "[B, 17, 64]");
    assert_eq!(fmt_shape(&g.shapes["pick.out"]), "[B, 64]");
    let code = generate_model(&g);
    assert_eq!(code.matches("class ClsToken").count(), 1);
    assert_eq!(code.matches("class PositionalEncoding").count(), 1);
    torch(VIT, "x = torch.randn(2, 3, 32, 32)\ny = m(x)\nassert y.shape == (2, 10)\ny.sum().backward()\n\
                assert all(p.grad is not None for p in m.parameters()), 'every parameter, including the cls token and position table, gets a gradient'");
}

#[test]
fn unet_style_skip_with_transposed_convolution() {
    let unet = "
inputs: {img: [B, 3, 16, 16]}
nodes:
  d1: {block: conv_block, params: {channels: 8}}
  pool: {block: maxpool2d}
  d2: {block: conv_block, params: {channels: 16}}
  up: {block: conv_transpose2d, params: {out_channels: 8}}
  join: {block: concat, params: {axis: 1}}
  u1: {block: conv_block, params: {channels: 8}}
  out: {block: conv2d, params: {out_channels: 2, kernel: 1}}
edges: ['img -> d1', 'd1 -> pool', 'pool -> d2', 'd2 -> up', 'd1 -> join.a', 'up -> join.b', 'join -> u1', 'u1 -> out']
outputs: {mask: out}";
    let g = graph(unet).unwrap();
    assert_eq!(fmt_shape(&g.shapes["pool.out"]), "[B, 8, 8, 8]");
    assert_eq!(fmt_shape(&g.shapes["up.out"]), "[B, 8, 16, 16]", "transposed conv undoes the pooling");
    assert_eq!(fmt_shape(&g.shapes["join.out"]), "[B, 16, 16, 16]");
    assert_eq!(fmt_shape(&g.shapes["out.out"]), "[B, 2, 16, 16]");
    torch(unet, "assert m(torch.randn(2, 3, 16, 16)).shape == (2, 2, 16, 16)");
}

#[test]
fn pooling_and_upsampling_shapes() {
    let m = "
inputs: {x: [B, 4, 8, 8]}
nodes:
  avg: {block: avgpool2d}
  up: {block: upsample, params: {scale: 3, mode: bilinear}}
  gap: {block: global_avg_pool}
edges: ['x -> avg', 'avg -> up', 'up -> gap']
outputs: {o: gap, mid: up}";
    let g = graph(m).unwrap();
    assert_eq!(fmt_shape(&g.shapes["avg.out"]), "[B, 4, 4, 4]");
    assert_eq!(fmt_shape(&g.shapes["up.out"]), "[B, 4, 12, 12]");
    assert_eq!(fmt_shape(&g.shapes["gap.out"]), "[B, 4]");
    torch(m, "r = m(torch.randn(2, 4, 8, 8))\nassert r['o'].shape == (2, 4) and r['mid'].shape == (2, 4, 12, 12)");
}

#[test]
fn vae_reparameterization_trains_and_is_deterministic_at_inference() {
    let vae = "
inputs: {x: [B, 20]}
nodes:
  enc: {block: linear, params: {out: 16}}
  act: {block: activation}
  mu: {block: linear, params: {out: 4}}
  logvar: {block: linear, params: {out: 4}}
  z: {block: sampling}
  dec: {block: linear, params: {out: 16}}
  dact: {block: activation}
  recon: {block: linear, params: {out: 20}}
edges: ['x -> enc', 'enc -> act', 'act -> mu', 'act -> logvar', 'mu -> z.mu', 'logvar -> z.logvar', 'z -> dec', 'dec -> dact', 'dact -> recon']
outputs: {recon: recon, mu: mu, logvar: logvar}";
    let g = graph(vae).unwrap();
    assert_eq!(fmt_shape(&g.shapes["z.out"]), "[B, 4]");
    torch(vae, "x = torch.randn(5, 20)\nm.train()\na, b = m(x)['recon'], m(x)['recon']\nassert not torch.allclose(a, b), 'training samples noise'\n\
                m.eval()\nassert torch.allclose(m(x)['recon'], m(x)['recon']), 'inference uses the mean'\n\
                m.train()\nout = m(x)\n(out['recon'].pow(2).mean() + out['mu'].pow(2).mean() + out['logvar'].exp().mean()).backward()\n\
                assert m.mu.weight.grad is not None and m.logvar.weight.grad is not None");
}

#[test]
fn graph_attention_respects_the_adjacency_mask() {
    let gat = "
inputs: {x: [B, 6, 16], adj: [B, 6, 6]}
nodes: {g: {block: gat, params: {out: 8, heads: 4}}}
edges: ['x -> g.x', 'adj -> g.adj']
outputs: {y: g}";
    assert_eq!(fmt_shape(&graph(gat).unwrap().shapes["g.out"]), "[B, 6, 32]");
    assert_eq!(fmt_shape(&graph(&gat.replace("heads: 4}", "heads: 4, concat: false}")).unwrap().shapes["g.out"]), "[B, 6, 8]");
    torch(gat, "x = torch.randn(1, 6, 16)\nadj = torch.zeros(1, 6, 6)\nbase = m(x=x, adj=adj)\n\
                x2 = x.clone(); x2[0, 1] += 5.0\nmoved = m(x=x2, adj=adj)\n\
                assert torch.allclose(base[0, 0], moved[0, 0], atol=1e-6), 'no edge: node 0 must not see node 1'\n\
                assert not torch.allclose(base[0, 1], moved[0, 1]), 'a node still sees itself'\n\
                adj[0, 0, 1] = 1.0\nlinked = m(x=x2, adj=adj)\nassert not torch.allclose(base[0, 0], linked[0, 0]), 'with an edge node 0 sees node 1'\n\
                assert m(x=x, adj=torch.ones(1, 6, 6)).shape == (1, 6, 32)");
}

#[test]
fn sinusoidal_positions_are_fixed_not_learned() {
    let m = "inputs: {x: [B, 10, 7]}\nnodes: {p: {block: positional_encoding, params: {max_len: 32, kind: sinusoidal}}}\nedges: ['x -> p']\noutputs: {y: p}";
    torch(m, "assert len(list(m.parameters())) == 0, 'sinusoidal encoding has no parameters'\nx = torch.zeros(2, 10, 7)\ny = m(x)\n\
              assert y.shape == (2, 10, 7) and not torch.allclose(y[0, 0], y[0, 5]), 'positions differ'\nassert torch.allclose(y[0], y[1])");
}

#[test]
fn shape_rules_reject_bad_inputs() {
    let cases: &[(&str, &str)] = &[
        ("inputs: {x: [B, 3, 30, 30]}\nnodes: {p: {block: patch_embed, params: {dim: 8, patch: 8}}}\nedges: ['x -> p']\noutputs: {y: p}", "not divisible"),
        ("inputs: {a: [B, 4], b: [B, 5]}\nnodes: {s: {block: sampling}}\nedges: ['a -> s.mu', 'b -> s.logvar']\noutputs: {y: s}", "same shape"),
        ("inputs: {x: [B, 100, 8]}\nnodes: {p: {block: positional_encoding, params: {max_len: 64}}}\nedges: ['x -> p']\noutputs: {y: p}", "exceeds max_len"),
        ("inputs: {x: [B, 6, 8], adj: [B, 5, 5]}\nnodes: {g: {block: gat, params: {out: 4}}}\nedges: ['x -> g.x', 'adj -> g.adj']\noutputs: {y: g}", "must be [B, N, N]"),
        ("inputs: {x: [B, 3, 8, 8]}\nnodes: {u: {block: upsample, params: {mode: cubic}}}\nedges: ['x -> u']\noutputs: {y: u}", "nearest or bilinear"),
        ("inputs: {x: [B, 3, 8, 8]}\nnodes: {u: {block: conv_transpose2d, params: {out_channels: 0}}}\nedges: ['x -> u']\noutputs: {y: u}", "positive integer"),
        ("inputs: {x: [B, 8]}\nnodes: {c: {block: cls_token}}\nedges: ['x -> c']\noutputs: {y: c}", "expected rank 3"),
    ];
    for (yaml, needle) in cases {
        let e = graph(yaml).unwrap_err().to_string();
        assert!(e.contains(needle), "{needle}: {e}");
    }
}
