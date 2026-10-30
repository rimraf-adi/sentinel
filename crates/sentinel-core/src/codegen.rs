//! PyTorch code generation from a validated component graph.
//! Output depends only on `torch`; Sentinel tracking is used only if importable.

use crate::compound::BlockKind;
use crate::graph::{GraphError, Source, ValidatedGraph};
use std::collections::BTreeMap;

/// Python keywords cannot be argument names (`in` is a natural port name), so they get a trailing underscore.
fn py(name: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "in", "is", "as", "if", "or", "and", "not", "def", "for", "from", "pass", "class", "del", "try", "with", "else", "elif", "while", "global",
        "lambda", "return", "yield", "assert", "break", "continue", "except", "finally", "import", "nonlocal", "raise", "None", "True", "False", "await", "async",
    ];
    if KEYWORDS.contains(&name) { format!("{name}_") } else { name.to_string() }
}

fn var(s: &Source) -> String {
    match s {
        Source::Input { name } => py(name),
        Source::Port { node, port } => format!("{node}__{port}"),
    }
}

fn camel(s: &str) -> String {
    s.split('_').filter(|p| !p.is_empty()).map(|p| p[..1].to_uppercase() + &p[1..]).collect()
}

/// Compound blocks become nested classes, one per distinct (block, parameters, input shapes), emitted
/// dependencies-first so every class is defined before it is used.
#[derive(Default)]
struct ClassBook {
    by_key: BTreeMap<String, String>,
    sources: Vec<String>,
    /// Helper classes for blocks that are not stock torch.nn layers, emitted once each.
    helpers: BTreeMap<&'static str, &'static str>,
}

fn emit_class(class: &str, g: &ValidatedGraph, book: &mut ClassBook, tuple_out: bool) -> String {
    let mut init = String::new();
    let mut aliases = String::new();
    let mut fwd = String::new();

    for node in &g.order {
        let kind = g.kind(node);
        let params = &g.params[node];
        let mut ins = BTreeMap::new();
        for p in kind.inputs() {
            let src = &g.wires[&(node.clone(), p.clone())];
            ins.insert(p.clone(), g.shape_of(src).clone());
        }
        let spec = &g.spec.nodes[node];
        let iv: BTreeMap<String, String> = kind.inputs().iter().map(|p| (p.clone(), var(&g.wires[&(node.clone(), p.clone())]))).collect();
        let ov: BTreeMap<String, String> = kind.outputs().iter().map(|p| (p.clone(), format!("{node}__{p}"))).collect();

        match kind {
            BlockKind::Fundamental(block) => {
                if let Some((name, source)) = block.helper() {
                    book.helpers.insert(name, source);
                }
                if let Some(target) = &spec.share {
                    aliases.push_str(&format!("        self.{node} = self.{target}\n"));
                } else if let Some(c) = block.ctor(node, params, &ins) {
                    init.push_str(&format!("        self.{node} = {c}\n"));
                }
                fwd.push_str(&format!("        {}\n", block.forward(node, params, &iv, &ov)));
            }
            BlockKind::Compound(def) => {
                if let Some(target) = &spec.share {
                    aliases.push_str(&format!("        self.{node} = self.{target}\n"));
                } else {
                    let key = format!("{}|{params:?}|{ins:?}", def.compound);
                    let cname = match book.by_key.get(&key) {
                        Some(c) => c.clone(),
                        None => {
                            let c = format!("{}{}", camel(&def.compound), book.by_key.len());
                            book.by_key.insert(key, c.clone());
                            let src = emit_class(&c, &g.compounds[node], book, true);
                            book.sources.push(src);
                            c
                        }
                    };
                    init.push_str(&format!("        self.{node} = {cname}()\n"));
                }
                let args = kind.inputs().iter().map(|p| iv[p].clone()).collect::<Vec<_>>().join(", ");
                let outs = kind.outputs().iter().map(|p| ov[p].clone()).collect::<Vec<_>>().join(", ");
                fwd.push_str(&format!("        {outs} = self.{node}({args})\n"));
            }
        }
    }

    let args: Vec<String> = g.spec.inputs.keys().map(|k| py(k)).collect();
    let ret = if g.outputs.len() == 1 {
        var(g.outputs.values().next().unwrap())
    } else if tuple_out {
        g.outputs.values().map(var).collect::<Vec<_>>().join(", ")
    } else {
        let items: Vec<String> = g.outputs.iter().map(|(k, s)| format!("\"{k}\": {}", var(s))).collect();
        format!("{{{}}}", items.join(", "))
    };

    format!(
        "class {class}(nn.Module):\n    def __init__(self):\n        super().__init__()\n{init}{aliases}        pass\n\n    def forward(self, {}):\n{fwd}        return {ret}\n",
        args.join(", ")
    )
}

/// Emit `class Model(nn.Module)` for the graph, preceded by one class per compound block it uses.
pub fn generate_model(g: &ValidatedGraph) -> String {
    let mut book = ClassBook::default();
    let model = emit_class("Model", g, &mut book, false);
    let mut out = String::new();
    for src in book.helpers.values() {
        out.push_str(src);
        out.push_str("\n\n");
    }
    for src in &book.sources {
        out.push_str(src);
        out.push_str("\n\n");
    }
    out.push_str(&model);
    out
}

/// Emit a complete standalone training script. Requires a `training` section and one input/output.
pub fn generate_training_script(g: &ValidatedGraph, graph_id: &str) -> Result<String, GraphError> {
    let t = g
        .spec
        .training
        .as_ref()
        .ok_or_else(|| GraphError::Training("graph has no `training` section".into()))?;
    if g.spec.inputs.len() != 1 {
        return Err(GraphError::Training("training scripts need exactly one graph input".into()));
    }
    if g.outputs.len() != 1 {
        return Err(GraphError::Training("training scripts need exactly one graph output".into()));
    }
    if !["cross_entropy", "mse", "bce_logits"].contains(&t.loss.as_str()) {
        return Err(GraphError::Training(format!("unsupported loss '{}'", t.loss)));
    }
    let opt = match t.optimizer.as_str() {
        "adam" => "Adam",
        "adamw" => "AdamW",
        "sgd" => "SGD",
        o => return Err(GraphError::Training(format!("unsupported optimizer '{o}'"))),
    };
    if !(0.0..1.0).contains(&t.data.test_split) {
        return Err(GraphError::Training("test_split must be in [0, 1)".into()));
    }

    let classification = t.loss == "cross_entropy";
    let loss_ctor = match t.loss.as_str() {
        "cross_entropy" => "nn.CrossEntropyLoss()",
        "mse" => "nn.MSELoss()",
        _ => "nn.BCEWithLogitsLoss()",
    };
    let label_code = if classification {
        format!(
            r#"def _class_key(v):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


CLASSES = sorted({{r[{t:?}] for r in rows}}, key=_class_key)
_index = {{c: i for i, c in enumerate(CLASSES)}}
Y = torch.tensor([_index[r[{t:?}]] for r in rows], dtype=torch.long)
print("classes:", CLASSES, flush=True)"#,
            t = t.data.target
        )
    } else {
        format!("Y = torch.tensor([float(r[{:?}]) for r in rows], dtype=torch.float32)", t.data.target)
    };
    let class_check = if classification {
        "\nwith torch.no_grad():\n    _n_out = model(xtr[:1]).shape[-1]\nif _n_out != len(CLASSES):\n    raise SystemExit(f\"the model outputs {_n_out} values but the data has {len(CLASSES)} classes {CLASSES}; set the last linear block's `out` to {len(CLASSES)}\")\n"
    } else {
        ""
    };
    let loss_call = if classification { "loss_fn(out, yb)" } else { "loss_fn(out, yb.view_as(out))" };
    let metric = if classification {
        "(model(xte).argmax(1) == yte).float().mean().item()"
    } else {
        "loss_fn(model(xte), yte.view_as(model(xte))).item()"
    };
    let metric_name = if classification { "test_acc" } else { "test_loss" };
    let reshape = match &t.data.reshape {
        Some(r) => format!(
            "\nX = X.reshape(X.shape[0], {})",
            r.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
        ),
        None => String::new(),
    };
    let feats = t.data.features.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>().join(", ");
    let bs = t.batch_size.map(|b| b.to_string()).unwrap_or_else(|| "len(xtr)".into());
    let model_code = generate_model(g);
    // Record the model's resolved hyperparameters so runs can be compared by them.
    let model_params: serde_json::Map<String, serde_json::Value> = g
        .order
        .iter()
        .map(|n| (n.clone(), serde_json::to_value(&g.params[n]).unwrap_or_default()))
        .collect();
    let cfg = serde_json::to_string(&serde_json::json!({
        "graph": graph_id, "loss": t.loss, "optimizer": t.optimizer, "lr": t.lr,
        "epochs": t.epochs, "batch_size": t.batch_size, "model": model_params,
    }))
    .unwrap();

    Ok(format!(
        r#"# Generated by Sentinel from component graph: {graph_id}
# Standalone PyTorch. Sentinel tracking is used only if the `sentinel` SDK is importable.
import csv
import json
import torch
import torch.nn as nn

{model_code}

torch.manual_seed({seed})
with open({csv:?}, newline="") as f:
    rows = list(csv.DictReader(f, skipinitialspace=True))
_used = [{feats}, {target:?}]
_total = len(rows)
rows = [r for r in rows if all((r.get(c) or "").strip() != "" for c in _used)]
if len(rows) < _total:
    print(f"skipped {{_total - len(rows)}} rows with missing values", flush=True)
if not rows:
    raise SystemExit("no usable rows in " + {csv:?})
X = torch.tensor([[float(r[c]) for c in [{feats}]] for r in rows], dtype=torch.float32){reshape}
{label_code}

perm = torch.randperm(len(X))
n_test = int(len(X) * {split})
te, tr = perm[:n_test], perm[n_test:]
xtr, ytr, xte, yte = X[tr], Y[tr], X[te], Y[te]

model = Model(){class_check}
loss_fn = {loss_ctor}
optimizer = torch.optim.{opt}(model.parameters(), lr={lr})

try:
    import sentinel
    sentinel.init(run_name={graph_id:?}, config=json.loads({cfg:?}))
except Exception:
    sentinel = None

BATCH = {bs}
for epoch in range({epochs}):
    model.train(True)
    order = torch.randperm(len(xtr))
    for i in range(0, len(xtr), BATCH):
        idx = order[i:i + BATCH]
        xb, yb = xtr[idx], ytr[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = {loss_call}
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0 or epoch == {epochs} - 1:
        model.train(False)
        with torch.no_grad():
            metric = {metric}
        print(f"epoch {{epoch:3d}} loss {{loss.item():.4f}} {metric_name} {{metric:.4f}}", flush=True)
        if sentinel:
            sentinel.log({{"train_loss": loss.item(), "{metric_name}": metric}}, step=epoch)
if sentinel:
    sentinel.finish()
"#,
        seed = t.data.seed,
        csv = t.data.csv,
        target = t.data.target,
        split = t.data.test_split,
        lr = t.lr,
        epochs = t.epochs,
    ))
}
