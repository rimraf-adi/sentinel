//! The block hierarchy.
//!
//! * **Fundamental blocks** (`blocks::Block`): native, one PyTorch module each.
//! * **Compound blocks** (this module): reusable subgraphs written in YAML from fundamental blocks *or other
//!   compound blocks*, with their own ports and hyperparameters. They compile to nested `nn.Module` classes.
//! * **Models** (`components`): whole graphs that use blocks from either tier.

use crate::blocks::{Block, BlockInfo, ParamInfo};
use crate::graph::{substitute, GraphError, GraphSpec, Params, Shape, ValidatedGraph};
use serde::Deserialize;
use serde_yaml::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

const MAX_DEPTH: usize = 8;

#[derive(Debug, Clone, Deserialize)]
pub struct CompoundParam {
    /// Absent means the parameter is required.
    #[serde(default)]
    pub default: Option<Value>,
    #[serde(default)]
    pub help: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompoundDef {
    pub compound: String,
    #[serde(default = "default_family")]
    pub family: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub params: BTreeMap<String, CompoundParam>,
    /// A graph with `${param}` placeholders. Its `inputs` and `outputs` keys are this block's ports.
    pub graph: Value,
}

fn default_family() -> String {
    "core".into()
}

fn ident(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(x) if x.is_ascii_lowercase() || x == '_') && c.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '_')
}

fn collect_placeholders(v: &Value, out: &mut BTreeSet<String>) {
    match v {
        Value::String(s) => {
            let mut rest = s.as_str();
            while let Some(i) = rest.find("${") {
                if let Some(j) = rest[i..].find('}') {
                    out.insert(rest[i + 2..i + j].trim().to_string());
                    rest = &rest[i + j + 1..];
                } else {
                    break;
                }
            }
        }
        Value::Sequence(s) => s.iter().for_each(|x| collect_placeholders(x, out)),
        Value::Mapping(m) => m.values().for_each(|x| collect_placeholders(x, out)),
        _ => {}
    }
}

impl CompoundDef {
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        let def: CompoundDef = serde_yaml::from_str(yaml).map_err(|e| e.to_string())?;
        def.check_static()?;
        Ok(def)
    }

    /// Checks that need no instantiation: names, ports, placeholders, and that the graph parses.
    fn check_static(&self) -> Result<(), String> {
        if !ident(&self.compound) {
            return Err(format!("compound name '{}' must be lowercase letters, digits and underscores", self.compound));
        }
        if Block::parse(&self.compound).is_some() {
            return Err(format!("'{}' is the name of a fundamental block", self.compound));
        }
        if self.input_ports().is_empty() || self.output_ports().is_empty() {
            return Err("a compound block needs at least one input and one output".into());
        }
        let mut used = BTreeSet::new();
        collect_placeholders(&self.graph, &mut used);
        for u in &used {
            if !self.params.contains_key(u) {
                return Err(format!("graph uses ${{{u}}} but the block declares no parameter '{u}'"));
            }
        }
        for p in self.params.keys() {
            if !ident(p) {
                return Err(format!("parameter name '{p}' must be lowercase letters, digits and underscores"));
            }
        }
        GraphSpec::from_value(self.graph.clone()).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn keys(&self, section: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .graph
            .get(section)
            .and_then(|s| s.as_mapping())
            .map(|m| m.keys().filter_map(|k| k.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        v.sort();
        v
    }

    pub fn input_ports(&self) -> Vec<String> {
        self.keys("inputs")
    }

    pub fn output_ports(&self) -> Vec<String> {
        self.keys("outputs")
    }

    /// Distinct blocks this one is built from (the next tier down).
    pub fn uses(&self) -> Vec<String> {
        let mut u = BTreeSet::new();
        if let Some(nodes) = self.graph.get("nodes").and_then(|n| n.as_mapping()) {
            for n in nodes.values() {
                if let Some(b) = n.get("block").and_then(|b| b.as_str()) {
                    u.insert(b.split('@').next().unwrap_or(b).to_string());
                }
            }
        }
        u.into_iter().collect()
    }

    fn resolve_params(&self, node: &str, given: &Params) -> Result<Params, GraphError> {
        let bad = |param: &str, msg: String| GraphError::BadParam { node: node.into(), param: param.into(), msg };
        for k in given.keys() {
            if !self.params.contains_key(k) {
                return Err(bad(k, format!("unknown parameter for '{}' (known: {:?})", self.compound, self.params.keys().collect::<Vec<_>>())));
            }
        }
        let mut out = Params::new();
        for (name, decl) in &self.params {
            let v = match (given.get(name), &decl.default) {
                (Some(v), _) => v.clone(),
                (None, Some(d)) => d.clone(),
                (None, None) => return Err(bad(name, "required parameter is missing".into())),
            };
            out.insert(name.clone(), v);
        }
        Ok(out)
    }

    /// Build and validate this block's inner graph for one use: parameters substituted, input shapes taken from the
    /// wires that feed it, so shape inference sees the real shapes.
    pub fn instantiate(&self, node: &str, params: &Params, ins: &BTreeMap<String, Shape>, lib: &BlockLibrary, stack: &[String]) -> Result<ValidatedGraph, GraphError> {
        let wrap = |msg: String| GraphError::Compound { node: node.into(), compound: self.compound.clone(), msg };
        if stack.contains(&self.compound) || stack.len() >= MAX_DEPTH {
            let mut chain = stack.to_vec();
            chain.push(self.compound.clone());
            return Err(wrap(format!("compound blocks cannot contain themselves ({})", chain.join(" -> "))));
        }
        let vars: BTreeMap<String, Value> = params.clone().into_iter().collect();
        let mut spec = GraphSpec::from_value(substitute(&self.graph, &vars)).map_err(|e| wrap(e.to_string()))?;
        let ports: BTreeSet<&String> = spec.inputs.keys().collect();
        let given: BTreeSet<&String> = ins.keys().collect();
        if ports != given {
            return Err(wrap(format!("input ports {ports:?} do not match the wired inputs {given:?}")));
        }
        spec.inputs = ins.clone();
        spec.training = None;
        let mut next = stack.to_vec();
        next.push(self.compound.clone());
        spec.validate_stack(lib, &next).map_err(|e| wrap(e.to_string()))
    }
}

/// A resolvable block: either tier.
#[derive(Clone)]
pub enum BlockKind {
    Fundamental(Block),
    Compound(Arc<CompoundDef>),
}

impl fmt::Debug for BlockKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockKind({})", self.name())
    }
}

impl BlockKind {
    pub fn name(&self) -> String {
        match self {
            BlockKind::Fundamental(b) => b.name().into(),
            BlockKind::Compound(c) => c.compound.clone(),
        }
    }

    pub fn tier(&self) -> &'static str {
        match self {
            BlockKind::Fundamental(_) => "fundamental",
            BlockKind::Compound(_) => "compound",
        }
    }

    pub fn inputs(&self) -> Vec<String> {
        match self {
            BlockKind::Fundamental(b) => b.inputs().iter().map(|s| s.to_string()).collect(),
            BlockKind::Compound(c) => c.input_ports(),
        }
    }

    pub fn outputs(&self) -> Vec<String> {
        match self {
            BlockKind::Fundamental(b) => b.outputs().iter().map(|s| s.to_string()).collect(),
            BlockKind::Compound(c) => c.output_ports(),
        }
    }

    pub fn resolve_params(&self, node: &str, given: &Params) -> Result<Params, GraphError> {
        match self {
            BlockKind::Fundamental(b) => b.resolve_params(node, given),
            BlockKind::Compound(c) => c.resolve_params(node, given),
        }
    }

    /// Output shapes, plus the validated inner graph for compound blocks (kept for code generation).
    pub fn infer(&self, node: &str, params: &Params, ins: &BTreeMap<String, Shape>, lib: &BlockLibrary, stack: &[String]) -> Result<(BTreeMap<String, Shape>, Option<Box<ValidatedGraph>>), GraphError> {
        match self {
            BlockKind::Fundamental(b) => Ok((b.infer(node, params, ins)?, None)),
            BlockKind::Compound(c) => {
                let inner = c.instantiate(node, params, ins, lib, stack)?;
                let out = inner.outputs.iter().map(|(port, src)| (port.clone(), inner.shape_of(src).clone())).collect();
                Ok((out, Some(Box::new(inner))))
            }
        }
    }

    /// Two nodes with equal keys build identical modules, so they may share weights. `None`: nothing to share.
    pub fn instance_key(&self, node: &str, params: &Params, ins: &BTreeMap<String, Shape>) -> Option<String> {
        match self {
            BlockKind::Fundamental(b) => b.ctor(node, params, ins),
            BlockKind::Compound(c) => Some(format!("{}|{params:?}|{ins:?}", c.compound)),
        }
    }

    pub fn info(&self) -> BlockInfo {
        match self {
            BlockKind::Fundamental(b) => b.info(),
            BlockKind::Compound(c) => BlockInfo {
                name: c.compound.clone(),
                tier: "compound",
                family: c.family.clone(),
                description: c.description.clone(),
                inputs: c.input_ports(),
                outputs: c.output_ports(),
                params: c
                    .params
                    .iter()
                    .map(|(n, p)| ParamInfo { name: n.clone(), required: p.default.is_none(), default: p.default.as_ref().and_then(|v| serde_json::to_value(v).ok()), help: p.help.clone() })
                    .collect(),
                uses: c.uses(),
                definition: serde_json::to_value(&c.graph).ok(),
            },
        }
    }
}

/// Shipped compound blocks plus whatever the project defines under `./blocks`.
#[derive(Debug, Clone, Default)]
pub struct BlockLibrary {
    compounds: BTreeMap<String, Arc<CompoundDef>>,
}

const SHIPPED: &[&str] = &[
    include_str!("../../../blocks/core/mlp_block.yaml"),
    include_str!("../../../blocks/core/residual_mlp_block.yaml"),
    include_str!("../../../blocks/vision/conv_block.yaml"),
    include_str!("../../../blocks/vision/resnet_block.yaml"),
];

static GLOBAL: OnceLock<RwLock<Arc<BlockLibrary>>> = OnceLock::new();

impl BlockLibrary {
    pub fn builtin() -> Self {
        let mut lib = BlockLibrary::default();
        for y in SHIPPED {
            lib.insert(CompoundDef::from_yaml(y).expect("shipped compound block must be valid")).expect("unique shipped names");
        }
        lib
    }

    pub fn insert(&mut self, def: CompoundDef) -> Result<(), String> {
        if self.compounds.contains_key(&def.compound) {
            return Err(format!("a compound block named '{}' already exists", def.compound));
        }
        self.compounds.insert(def.compound.clone(), Arc::new(def));
        Ok(())
    }

    /// Shipped blocks plus every `*.yaml` under `dir` (recursively). Bad files are reported, not fatal.
    pub fn with_dir(dir: &Path) -> (Self, Vec<String>) {
        let mut lib = Self::builtin();
        let mut errors = Vec::new();
        fn walk(dir: &Path, lib: &mut BlockLibrary, errors: &mut Vec<String>) {
            let Ok(rd) = std::fs::read_dir(dir) else { return };
            let mut entries: Vec<_> = rd.flatten().collect();
            entries.sort_by_key(|e| e.path());
            for e in entries {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, lib, errors);
                } else if matches!(p.extension().and_then(|x| x.to_str()), Some("yaml" | "yml")) {
                    let res = std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| CompoundDef::from_yaml(&t)).and_then(|d| lib.insert(d));
                    if let Err(e) = res {
                        errors.push(format!("{}: {e}", p.display()));
                    }
                }
            }
        }
        walk(dir, &mut lib, &mut errors);
        (lib, errors)
    }

    pub fn global() -> Arc<BlockLibrary> {
        GLOBAL.get_or_init(|| RwLock::new(Arc::new(BlockLibrary::builtin()))).read().expect("block library lock").clone()
    }

    /// Make the shipped blocks plus the project's `dir` the library every `validate()` uses. Returns load errors.
    pub fn reload_project(dir: &Path) -> Vec<String> {
        let (lib, errors) = Self::with_dir(dir);
        *GLOBAL.get_or_init(|| RwLock::new(Arc::new(BlockLibrary::default()))).write().expect("block library lock") = Arc::new(lib);
        errors
    }

    pub fn resolve(&self, name: &str) -> Option<BlockKind> {
        let base = name.split('@').next().unwrap_or(name);
        Block::parse(base).map(BlockKind::Fundamental).or_else(|| self.compounds.get(base).cloned().map(BlockKind::Compound))
    }

    /// Drop a block (used when overwriting a project block so its replacement can be validated against the rest).
    pub fn remove(&mut self, name: &str) {
        self.compounds.remove(name);
    }

    pub fn compounds(&self) -> impl Iterator<Item = &Arc<CompoundDef>> {
        self.compounds.values()
    }

    /// Every block of both tiers, fundamentals first.
    pub fn catalog(&self) -> Vec<BlockInfo> {
        let mut compounds: Vec<&Arc<CompoundDef>> = self.compounds.values().collect();
        compounds.sort_by(|a, b| (&a.family, &a.compound).cmp(&(&b.family, &b.compound)));
        Block::all().iter().map(|b| b.info()).chain(compounds.into_iter().map(|c| BlockKind::Compound(c.clone()).info())).collect()
    }
}
