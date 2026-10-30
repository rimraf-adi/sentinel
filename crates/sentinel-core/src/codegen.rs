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
    generate_models(&[("Model", g)])
}

/// Several named model classes sharing one set of helper and compound classes, so two graphs that use the same
/// compound block (a GAN's generator and discriminator) never define it twice.
pub fn generate_models(models: &[(&str, &ValidatedGraph)]) -> String {
    let mut book = ClassBook::default();
    let classes: Vec<String> = models.iter().map(|(name, g)| emit_class(name, g, &mut book, false)).collect();
    let mut out = String::new();
    for src in book.helpers.values() {
        out.push_str(src);
        out.push_str("\n\n");
    }
    for src in &book.sources {
        out.push_str(src);
        out.push_str("\n\n");
    }
    out.push_str(&classes.join("\n\n"));
    out
}

pub(crate) fn feats_literal(features: &[String]) -> String {
    features.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>().join(", ")
}

/// A complete standalone training script (see `training::training_script`).
pub fn generate_training_script(g: &ValidatedGraph, graph_id: &str) -> Result<String, GraphError> {
    crate::training::training_script(g, graph_id)
}
