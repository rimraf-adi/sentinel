//! Training: the catalog of tasks, losses, optimizers, schedulers and strategies, their validation, and the
//! standalone PyTorch script that carries them out (see `training_script`).

use crate::graph::{Dim, GraphError, Named, TrainingSpec, ValidatedGraph};
use serde_json::{json, Value};

pub struct Param {
    pub name: &'static str,
    /// JSON default; `null` means "use the library default".
    pub default: &'static str,
    pub help: &'static str,
    pub options: &'static [&'static str],
}

const fn p(name: &'static str, default: &'static str, help: &'static str) -> Param {
    Param { name, default, help, options: &[] }
}
const fn choice(name: &'static str, default: &'static str, help: &'static str, options: &'static [&'static str]) -> Param {
    Param { name, default, help, options }
}

pub struct LossDef {
    pub name: &'static str,
    /// Tasks this loss can serve (the first is the default when the task is inferred).
    pub tasks: &'static [&'static str],
    pub help: &'static str,
    pub params: &'static [Param],
}

const NDCG: Param = p("ndcg_k", "10", "NDCG is computed over the top k items of each query");

pub const LOSSES: &[LossDef] = &[
    LossDef {
        name: "cross_entropy",
        tasks: &["classification"],
        help: "Softmax cross-entropy over class scores (the standard multi-class loss)",
        params: &[
            p("label_smoothing", "0.0", "Soften the one-hot targets by this amount (0 to 1)"),
            choice("class_weights", "\"none\"", "balanced weights classes by inverse frequency", &["none", "balanced"]),
        ],
    },
    LossDef {
        name: "focal",
        tasks: &["classification"],
        help: "Focal loss: down-weights easy examples so rare or hard classes matter more",
        params: &[p("gamma", "2.0", "Focusing strength (0 is plain cross-entropy)"), choice("class_weights", "\"none\"", "balanced weights classes by inverse frequency", &["none", "balanced"])],
    },
    LossDef { name: "multi_margin", tasks: &["classification"], help: "Multi-class hinge loss (SVM-style margin)", params: &[p("margin", "1.0", "Required score gap to the other classes")] },
    LossDef {
        name: "bce_logits",
        tasks: &["binary", "multilabel"],
        help: "Binary cross-entropy on logits: one yes/no target (binary) or several independent ones (multilabel)",
        params: &[p("pos_weight", "null", "Weight of positives: a number, or \"balanced\" (negatives / positives)")],
    },
    LossDef { name: "mse", tasks: &["regression", "reconstruction"], help: "Mean squared error", params: &[] },
    LossDef { name: "mae", tasks: &["regression", "reconstruction"], help: "Mean absolute error (robust to outliers)", params: &[] },
    LossDef { name: "huber", tasks: &["regression"], help: "Quadratic near zero, linear for large errors", params: &[p("delta", "1.0", "Where the loss turns linear")] },
    LossDef { name: "smooth_l1", tasks: &["regression"], help: "Smooth L1 (Huber scaled by 1/beta)", params: &[p("beta", "1.0", "Where the loss turns linear")] },
    LossDef { name: "quantile", tasks: &["regression"], help: "Pinball loss: predict a quantile instead of the mean", params: &[p("q", "0.5", "Quantile to predict, 0 to 1 (0.5 is the median)")] },
    LossDef { name: "poisson", tasks: &["regression"], help: "Poisson negative log-likelihood for counts (the model predicts the log-rate)", params: &[] },
    LossDef { name: "log_cosh", tasks: &["regression"], help: "log(cosh(error)): smooth, outlier-tolerant", params: &[] },
    LossDef { name: "ranknet", tasks: &["ranking"], help: "Pairwise logistic loss: more relevant items should score higher", params: &[NDCG] },
    LossDef { name: "margin_rank", tasks: &["ranking"], help: "Pairwise hinge loss on score differences", params: &[p("margin", "1.0", "Required score gap"), NDCG] },
    LossDef { name: "listnet", tasks: &["ranking"], help: "Listwise loss: match the softmax of the scores to that of the relevance labels", params: &[NDCG] },
    LossDef { name: "vae", tasks: &["vae"], help: "Reconstruction error plus beta times the KL divergence (the model outputs recon, mu, logvar)", params: &[] },
    LossDef { name: "custom", tasks: &["custom"], help: "Your own loss function (see the `custom` section)", params: &[] },
];

pub struct TaskDef {
    pub name: &'static str,
    pub help: &'static str,
    pub metrics: &'static [&'static str],
    pub primary: &'static str,
    pub needs: &'static str,
}

pub const TASKS: &[TaskDef] = &[
    TaskDef { name: "classification", help: "Pick one of several classes", metrics: &["test_loss", "test_acc"], primary: "test_acc", needs: "data.target; the model outputs one score per class" },
    TaskDef { name: "binary", help: "A single yes/no target", metrics: &["test_loss", "test_acc"], primary: "test_acc", needs: "data.target with exactly two values; the model outputs one logit" },
    TaskDef { name: "multilabel", help: "Several independent yes/no targets", metrics: &["test_loss", "test_acc"], primary: "test_acc", needs: "data.targets (0/1 columns); one logit per target" },
    TaskDef { name: "regression", help: "Predict one or more numbers", metrics: &["test_mae", "test_r2", "test_loss"], primary: "test_loss", needs: "data.target or data.targets; one output per target" },
    TaskDef { name: "ranking", help: "Order items within each query", metrics: &["test_loss", "test_ndcg"], primary: "test_ndcg", needs: "data.group (query id) and data.target (relevance); the model outputs one score per row" },
    TaskDef { name: "reconstruction", help: "Reproduce the input (autoencoders)", metrics: &["test_loss"], primary: "test_loss", needs: "no target; the model outputs as many values as there are features" },
    TaskDef { name: "vae", help: "Variational autoencoder", metrics: &["test_recon"], primary: "test_recon", needs: "no target; graph outputs recon, mu and logvar" },
    TaskDef { name: "custom", help: "Your own task: a Python loss (and optional metric) over the model output and target", metrics: &["test_loss"], primary: "test_loss", needs: "a `custom` section; data.target(s) optional" },
];

pub struct OptDef {
    pub name: &'static str,
    pub help: &'static str,
    pub params: &'static [Param],
}

const BETAS: [Param; 3] = [p("beta1", "0.9", "First-moment decay"), p("beta2", "0.999", "Second-moment decay"), p("eps", "1e-8", "Numerical stability term")];

pub const OPTIMIZERS: &[OptDef] = &[
    OptDef { name: "adam", help: "Adaptive moments: a robust default", params: &BETAS },
    OptDef { name: "adamw", help: "Adam with decoupled weight decay (set weight_decay)", params: &BETAS },
    OptDef { name: "sgd", help: "Stochastic gradient descent, optionally with momentum", params: &[p("momentum", "0.0", "Momentum factor (0.9 is common)"), choice("nesterov", "false", "Nesterov momentum (needs momentum > 0)", &["false", "true"]), p("dampening", "0.0", "Momentum dampening")] },
    OptDef { name: "rmsprop", help: "Scales steps by a running RMS of gradients", params: &[p("alpha", "0.99", "Smoothing constant"), p("momentum", "0.0", "Momentum factor"), p("eps", "1e-8", "Numerical stability term")] },
    OptDef { name: "adagrad", help: "Per-parameter rates that shrink with accumulated gradients", params: &[p("lr_decay", "0.0", "Learning-rate decay"), p("eps", "1e-10", "Numerical stability term")] },
    OptDef { name: "adadelta", help: "Adagrad variant with a moving window", params: &[p("rho", "0.9", "Decay of the running average"), p("eps", "1e-6", "Numerical stability term")] },
    OptDef { name: "nadam", help: "Adam with Nesterov momentum", params: &BETAS },
    OptDef { name: "radam", help: "Rectified Adam: more stable early training", params: &BETAS },
    OptDef { name: "adamax", help: "Adam variant based on the infinity norm", params: &BETAS },
];

pub const SCHEDULERS: &[OptDef] = &[
    OptDef { name: "constant", help: "Keep the learning rate fixed", params: &[p("warmup_epochs", "0", "Linear warmup before the rate is held")] },
    OptDef { name: "step", help: "Multiply the rate by gamma every step_size epochs", params: &[p("step_size", "0", "Epochs between drops (0 = a third of the run)"), p("gamma", "0.1", "Drop factor"), p("warmup_epochs", "0", "Linear warmup first")] },
    OptDef { name: "multistep", help: "Multiply the rate by gamma at chosen epochs", params: &[p("milestones", "null", "List of epochs (default: halfway and three quarters)"), p("gamma", "0.1", "Drop factor"), p("warmup_epochs", "0", "Linear warmup first")] },
    OptDef { name: "exponential", help: "Multiply the rate by gamma every epoch", params: &[p("gamma", "0.95", "Per-epoch factor"), p("warmup_epochs", "0", "Linear warmup first")] },
    OptDef { name: "cosine", help: "Cosine decay from lr to eta_min over the run", params: &[p("eta_min", "0.0", "Final learning rate"), p("warmup_epochs", "0", "Linear warmup first")] },
    OptDef { name: "cosine_restarts", help: "Cosine decay that periodically restarts", params: &[p("t_0", "10", "Epochs in the first cycle"), p("t_mult", "1", "Cycle length multiplier"), p("eta_min", "0.0", "Lowest learning rate"), p("warmup_epochs", "0", "Linear warmup first")] },
    OptDef { name: "onecycle", help: "Warm up to max_lr then anneal (stepped every batch; has its own warmup)", params: &[p("max_lr", "null", "Peak rate (default: 10 x lr)"), p("pct_start", "0.3", "Fraction of the run spent rising")] },
    OptDef { name: "plateau", help: "Cut the rate when the monitored metric stops improving", params: &[p("factor", "0.5", "Drop factor"), p("patience", "10", "Epochs without improvement before dropping"), p("min_lr", "0.0", "Lowest learning rate")] },
];

pub const AMP_MODES: &[&str] = &["off", "auto", "bf16", "fp16"];
pub const DEVICES: &[&str] = &["auto", "cpu", "cuda", "mps"];

fn param_json(pr: &Param) -> Value {
    json!({
        "name": pr.name, "default": serde_json::from_str::<Value>(pr.default).unwrap_or(Value::Null), "help": pr.help,
        "options": pr.options,
    })
}

/// Everything a UI needs to render the training form, and the single source of truth for validation.
pub fn catalog() -> Value {
    let named = |defs: &[OptDef]| -> Vec<Value> { defs.iter().map(|d| json!({ "name": d.name, "help": d.help, "params": d.params.iter().map(param_json).collect::<Vec<_>>() })).collect() };
    json!({
        "tasks": TASKS.iter().map(|t| json!({
            "name": t.name, "help": t.help, "metrics": t.metrics, "primary": t.primary, "needs": t.needs,
            "losses": LOSSES.iter().filter(|l| l.tasks.contains(&t.name)).map(|l| json!({ "name": l.name, "help": l.help, "params": l.params.iter().map(param_json).collect::<Vec<_>>() })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "optimizers": named(OPTIMIZERS),
        "schedulers": named(SCHEDULERS),
        "amp": AMP_MODES,
        "devices": DEVICES,
        "strategies": [
            { "name": "weight_decay", "help": "L2 / decoupled weight decay applied by the optimizer" },
            { "name": "clip_grad", "help": "Clip the gradient norm to this value (stabilises RNNs and transformers)" },
            { "name": "accumulate", "help": "Sum gradients over this many batches before each update (bigger effective batch)" },
            { "name": "amp", "help": "Mixed precision: bf16 (CPU/GPU) or fp16 (CUDA, with loss scaling)" },
            { "name": "early_stopping", "help": "Stop when the monitored metric has not improved for `patience` epochs" },
            { "name": "save_best", "help": "Restore the weights from the best epoch when training ends" },
            { "name": "freeze", "help": "Freeze top-level nodes (transfer learning)" },
            { "name": "init_from", "help": "Start from a saved state_dict, e.g. an earlier run's model.pt" },
            { "name": "checkpoint", "help": "Checkpoint strategy: 'latest' (default) keeps the most recent checkpoint for seamless resume; 'all' keeps all checkpoints; 'off' disables" },
            { "name": "resume", "help": "Seamlessly resume training from existing checkpoint if found (default: true)" },
        ],
    })
}

/// Settings are stored as YAML values; checks are easier on JSON.
fn jv(v: &serde_yaml::Value) -> Value {
    serde_json::to_value(v).unwrap_or(Value::Null)
}

fn bad(msg: impl Into<String>) -> GraphError {
    GraphError::Training(msg.into())
}

/// Reject unknown settings and wrongly typed values for a named choice (loss, optimizer, scheduler).
fn check_named(kind: &str, n: &Named, allowed: &[Param]) -> Result<(), GraphError> {
    for (k, v) in &n.params {
        let v = &jv(v);
        let Some(def) = allowed.iter().find(|d| d.name == k) else {
            let known: Vec<&str> = allowed.iter().map(|d| d.name).collect();
            return Err(bad(format!("{kind} '{}' has no setting '{k}' (known: {})", n.name, if known.is_empty() { "none".into() } else { known.join(", ") })));
        };
        let default: Value = serde_json::from_str(def.default).unwrap_or(Value::Null);
        let ok = match &default {
            Value::Number(_) => v.is_number(),
            Value::Bool(_) => v.is_boolean(),
            Value::String(_) => v.is_string(),
            _ => true, // null default: any
        };
        if !ok && !v.is_null() {
            return Err(bad(format!("{kind} '{}': setting '{k}' should be like {} (got {v:?})", n.name, def.default)));
        }
        if !def.options.is_empty() {
            let s = match v {
                Value::String(s) => s.clone(),
                Value::Bool(b) => b.to_string(),
                other => format!("{other:?}"),
            };
            if !def.options.contains(&s.as_str()) {
                return Err(bad(format!("{kind} '{}': setting '{k}' must be one of {} (got '{s}')", n.name, def.options.join(", "))));
            }
        }
    }
    Ok(())
}

/// Everything resolved and checked before any code is written.
#[derive(Debug, Clone)]
pub struct Plan {
    pub task: &'static str,
    /// Metrics `evaluate()` returns, primary last.
    pub metrics: Vec<String>,
    pub primary: String,
    pub monitor: String,
    pub monitor_max: bool,
    /// The columns the targets come from (`target` or `targets`).
    pub target_cols: Vec<String>,
    pub out_dim: Option<i64>,
}

pub fn infer_task(t: &TrainingSpec) -> Result<&'static str, GraphError> {
    if let Some(task) = &t.task {
        return TASKS.iter().find(|d| d.name == task).map(|d| d.name).ok_or_else(|| bad(format!("unknown task '{task}' (options: {})", TASKS.iter().map(|d| d.name).collect::<Vec<_>>().join(", "))));
    }
    let def = LOSSES.iter().find(|l| l.name == t.loss.name).ok_or_else(|| bad(format!("unsupported loss '{}' (options: {})", t.loss.name, LOSSES.iter().map(|l| l.name).collect::<Vec<_>>().join(", "))))?;
    Ok(match (def.name, t.data.targets.len()) {
        ("bce_logits", n) if n > 1 => "multilabel",
        _ => def.tasks[0],
    })
}

fn out_dim(g: &ValidatedGraph) -> Option<i64> {
    let src = g.outputs.values().next()?;
    match g.shape_of(src).last() {
        Some(Dim::Const(n)) => Some(*n),
        _ => None,
    }
}

/// Validate the whole training spec against the graph and resolve the task, metrics and column roles.
pub fn plan(g: &ValidatedGraph, t: &TrainingSpec) -> Result<Plan, GraphError> {
    let task = infer_task(t)?;
    let loss = LOSSES.iter().find(|l| l.name == t.loss.name).ok_or_else(|| bad(format!("unsupported loss '{}' (options: {})", t.loss.name, LOSSES.iter().map(|l| l.name).collect::<Vec<_>>().join(", "))))?;
    if !loss.tasks.contains(&task) {
        return Err(bad(format!("loss '{}' does not fit the {task} task (it serves: {}; losses for {task}: {})", loss.name, loss.tasks.join(", "),
            LOSSES.iter().filter(|l| l.tasks.contains(&task)).map(|l| l.name).collect::<Vec<_>>().join(", "))));
    }
    check_named("loss", &t.loss, loss.params)?;
    let opt = OPTIMIZERS.iter().find(|o| o.name == t.optimizer.name).ok_or_else(|| bad(format!("unsupported optimizer '{}' (options: {})", t.optimizer.name, OPTIMIZERS.iter().map(|o| o.name).collect::<Vec<_>>().join(", "))))?;
    check_named("optimizer", &t.optimizer, opt.params)?;

    if t.epochs == 0 {
        return Err(bad("epochs must be at least 1"));
    }
    if !(t.lr.is_finite() && t.lr > 0.0) {
        return Err(bad("lr must be a positive number"));
    }
    if t.weight_decay < 0.0 || !t.weight_decay.is_finite() {
        return Err(bad("weight_decay must be 0 or more"));
    }
    if t.batch_size == Some(0) {
        return Err(bad("batch_size must be at least 1"));
    }
    if t.accumulate == 0 {
        return Err(bad("accumulate must be at least 1"));
    }
    if let Some(c) = t.clip_grad {
        if !(c.is_finite() && c > 0.0) {
            return Err(bad("clip_grad must be a positive number"));
        }
    }
    if !AMP_MODES.contains(&t.amp.as_str()) {
        return Err(bad(format!("amp must be one of {}", AMP_MODES.join(", "))));
    }
    if !DEVICES.contains(&t.device.as_str()) {
        return Err(bad(format!("device must be one of {}", DEVICES.join(", "))));
    }
    if !(0.0..1.0).contains(&t.data.test_split) {
        return Err(bad("test_split must be in [0, 1)"));
    }
    if let Some(i) = &t.init_from {
        if i.trim().is_empty() {
            return Err(bad("init_from must be a path"));
        }
    }
    for node in &t.freeze {
        if !g.spec.nodes.contains_key(node) {
            return Err(bad(format!("freeze: the graph has no node '{node}' (nodes: {})", g.spec.nodes.keys().cloned().collect::<Vec<_>>().join(", "))));
        }
    }
    if t.freeze.len() == g.spec.nodes.len() && !t.freeze.is_empty() {
        return Err(bad("freeze covers every node, so nothing would train"));
    }

    // --- task-specific requirements
    let target_cols: Vec<String> = if !t.data.targets.is_empty() { t.data.targets.clone() } else if !t.data.target.is_empty() { vec![t.data.target.clone()] } else { vec![] };
    let od = out_dim(g);
    let need_target = || if target_cols.is_empty() { Err(bad(format!("the {task} task needs `data.target`"))) } else { Ok(()) };
    let expect_out = |want: i64, why: &str| -> Result<(), GraphError> {
        match od {
            Some(n) if n != want => Err(bad(format!("the {task} task needs a model that outputs {want} value(s) per row ({why}), but the graph outputs {n}"))),
            _ => Ok(()),
        }
    };
    match task {
        "classification" => {
            need_target()?;
            if target_cols.len() != 1 {
                return Err(bad("classification takes one target column (use the multilabel task for several)"));
            }
        }
        "binary" => {
            need_target()?;
            if target_cols.len() != 1 {
                return Err(bad("binary classification takes one target column (use the multilabel task for several)"));
            }
            expect_out(1, "one logit")?;
        }
        "multilabel" => {
            need_target()?;
            expect_out(target_cols.len() as i64, "one logit per target column")?;
        }
        "regression" => {
            need_target()?;
            expect_out(target_cols.len() as i64, "one output per target column")?;
        }
        "ranking" => {
            need_target()?;
            if target_cols.len() != 1 {
                return Err(bad("ranking takes one target column holding the relevance"));
            }
            if t.data.group.as_deref().unwrap_or("").is_empty() {
                return Err(bad("the ranking task needs `data.group`, the column that identifies each query"));
            }
            expect_out(1, "a relevance score")?;
        }
        "reconstruction" => {
            if !target_cols.is_empty() {
                return Err(bad("the reconstruction task is unsupervised: remove `data.target`"));
            }
            expect_out(t.data.features.len() as i64, "one per feature column")?;
        }
        "vae" => {
            let keys: Vec<&str> = g.outputs.keys().map(|k| k.as_str()).collect();
            if keys != ["logvar", "mu", "recon"] {
                return Err(bad(format!("the vae loss needs the graph outputs `recon`, `mu` and `logvar`, found {keys:?}")));
            }
            if !target_cols.is_empty() {
                return Err(bad("the vae loss is unsupervised: remove `data.target`"));
            }
        }
        "custom" => {
            let c = t.custom.as_ref().ok_or_else(|| bad("the custom task needs a `custom` section (file or source, and the loss function name)"))?;
            if c.file.is_none() && c.source.is_none() {
                return Err(bad("custom: provide `file` (a Python file in the project) or `source` (the code)"));
            }
            if !["min", "max"].contains(&c.mode.as_str()) {
                return Err(bad("custom.mode must be min or max"));
            }
            if !["float", "long"].contains(&c.target_dtype.as_str()) {
                return Err(bad("custom.target_dtype must be float or long"));
            }
        }
        _ => unreachable!(),
    }
    if task != "vae" && task != "custom" && g.outputs.len() != 1 {
        return Err(bad("training scripts need exactly one graph output"));
    }
    if task != "ranking" && t.data.group.is_some() {
        return Err(bad("`data.group` only applies to the ranking task"));
    }
    if t.custom.is_some() && task != "custom" {
        return Err(bad("a `custom` section needs `loss: custom`"));
    }

    // --- scheduler
    if let Some(s) = &t.scheduler {
        let def = SCHEDULERS.iter().find(|d| d.name == s.name).ok_or_else(|| bad(format!("unsupported scheduler '{}' (options: {})", s.name, SCHEDULERS.iter().map(|d| d.name).collect::<Vec<_>>().join(", "))))?;
        check_named("scheduler", s, def.params)?;
        if let Some(w) = s.i64("warmup_epochs") {
            if w < 0 || w as u32 >= t.epochs {
                return Err(bad(format!("scheduler warmup_epochs ({w}) must be between 0 and epochs - 1 ({})", t.epochs - 1)));
            }
        }
        if let Some(m) = s.params.get("milestones").map(jv) {
            let ok = m.as_array().is_some_and(|a| a.iter().all(|x| x.as_u64().is_some()));
            if !m.is_null() && !ok {
                return Err(bad("scheduler milestones must be a list of epoch numbers"));
            }
        }
        if s.f64("gamma").is_some_and(|g| g <= 0.0) {
            return Err(bad("scheduler gamma must be positive"));
        }
    }

    // --- metrics and what early stopping / best checkpoints watch
    let def = TASKS.iter().find(|d| d.name == task).unwrap();
    let mut metrics: Vec<String> = def.metrics.iter().map(|s| s.to_string()).collect();
    let mut primary = def.primary.to_string();
    let mut mode_max = matches!(primary.as_str(), "test_acc" | "test_ndcg" | "test_r2");
    if task == "custom" {
        let c = t.custom.as_ref().unwrap();
        if let Some(m) = &c.metric {
            let name = format!("test_{m}");
            metrics.push(name.clone());
            primary = name;
            mode_max = c.mode == "max";
        }
    }
    // the primary metric is printed last
    metrics.retain(|m| *m != primary);
    metrics.push(primary.clone());
    let monitor = match t.early_stopping.as_ref().and_then(|e| e.monitor.clone()) {
        Some(m) => {
            if !metrics.contains(&m) {
                return Err(bad(format!("early_stopping.monitor '{m}' is not a metric of the {task} task (metrics: {})", metrics.join(", "))));
            }
            m
        }
        None => primary.clone(),
    };
    let monitor_max = if monitor == primary { mode_max } else { matches!(monitor.as_str(), "test_acc" | "test_ndcg" | "test_r2") };
    if let Some(e) = &t.early_stopping {
        if e.patience == 0 {
            return Err(bad("early_stopping.patience must be at least 1"));
        }
        if e.min_delta < 0.0 {
            return Err(bad("early_stopping.min_delta must be 0 or more"));
        }
    }
    if let Some(s) = &t.scheduler {
        if s.name == "plateau" && t.early_stopping.is_none() && !metrics.contains(&monitor) {
            return Err(bad("plateau scheduling needs a monitored metric"));
        }
    }
    if !["latest", "all", "off", "none"].contains(&t.checkpoint.as_str()) {
        return Err(bad(format!("checkpoint mode must be 'latest', 'all', or 'off' (got '{}')", t.checkpoint)));
    }
    Ok(Plan { task, metrics, primary, monitor, monitor_max, target_cols, out_dim: od })
}

// ===========================================================================================================
// Script generation
// ===========================================================================================================

const HEADER: &str = r#"# Generated by Sentinel from component graph: @@ID@@
# Standalone PyTorch. Sentinel tracking is used only if the `sentinel` SDK is importable.
import csv
import json
import math
import os
import pathlib
import shutil
import torch
import torch.nn as nn
"#;

const H_FOCAL: &str = r#"def focal_loss(logits, target, gamma=2.0, weight=None):
    logp = torch.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1 - logpt.exp()) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()
"#;

const H_QUANTILE: &str = r#"def quantile_loss(pred, target, q=0.5):
    e = target - pred
    return torch.maximum(q * e, (q - 1) * e).mean()
"#;

const H_LOGCOSH: &str = r#"def log_cosh_loss(pred, target):
    x = pred - target
    return (x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)).mean()
"#;

const H_RANKING: &str = r#"def _pairs(s, y):
    return s.unsqueeze(1) - s.unsqueeze(0), (y.unsqueeze(1) > y.unsqueeze(0)).float()


def ranknet_loss(s, y):
    diff, mask = _pairs(s, y)
    if mask.sum() == 0:
        return s.sum() * 0.0
    return (torch.nn.functional.softplus(-diff) * mask).sum() / mask.sum()


def margin_rank_loss(s, y, margin=1.0):
    diff, mask = _pairs(s, y)
    if mask.sum() == 0:
        return s.sum() * 0.0
    return (torch.relu(margin - diff) * mask).sum() / mask.sum()


def listnet_loss(s, y):
    return -(torch.softmax(y, dim=0) * torch.log_softmax(s, dim=0)).sum()


def ndcg_at_k(s, y, k=10):
    k = min(k, len(s))
    order = s.argsort(descending=True)[:k]
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32, device=s.device))
    dcg = ((2.0 ** y[order] - 1.0) * disc).sum()
    ideal = y.sort(descending=True).values[:k]
    idcg = ((2.0 ** ideal - 1.0) * disc).sum()
    return None if idcg <= 0 else (dcg / idcg).item()
"#;

const CLASS_KEY: &str = r#"def _class_key(v):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)
"#;

const DEVICE_AMP: &str = r#"def _pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device 'cuda' was requested but CUDA is not available on this machine")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("device 'mps' was requested but Metal (MPS) is not available on this machine")
    return torch.device(name)


DEVICE = _pick_device(@@DEVICE@@)
AMP = @@AMP@@
if AMP == "auto":
    AMP = "fp16" if DEVICE.type == "cuda" else "off"
if AMP == "fp16" and DEVICE.type != "cuda":
    print("fp16 autocast needs CUDA; running in float32", flush=True)
    AMP = "off"
AMP_ON = AMP != "off"
AMP_DTYPE = torch.bfloat16 if AMP == "bf16" else torch.float16


def autocast_ctx():
    return torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=AMP_ON)


class _NoScaler:
    def scale(self, x):
        return x

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


scaler = torch.amp.GradScaler("cuda") if AMP == "fp16" else _NoScaler()
print(f"device: {DEVICE} · amp: {AMP}", flush=True)
"#;

fn pylist(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(", "))
}

fn num(v: f64) -> String {
    // Python-safe float literal
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 { format!("{v:.1}") } else { format!("{v}") }
}

/// Read the user's custom code: inline `source`, or `file` relative to the project directory.
fn custom_source(c: &crate::graph::CustomSpec) -> Result<(String, String), GraphError> {
    if let Some(src) = &c.source {
        return Ok((src.clone(), "custom code in the component".into()));
    }
    let file = c.file.as_deref().unwrap_or("");
    let path = std::path::Path::new(file);
    if path.is_absolute() || path.components().any(|p| matches!(p, std::path::Component::ParentDir)) {
        return Err(bad(format!("custom.file '{file}' must be a relative path inside the project (no `..`)")));
    }
    let text = std::fs::read_to_string(path).map_err(|e| bad(format!("custom.file '{file}' could not be read: {e} (paths are relative to the project directory)")))?;
    Ok((text, file.to_string()))
}

fn optimizer_code(t: &TrainingSpec) -> Result<String, GraphError> {
    let o = &t.optimizer;
    let f = |k: &str, d: f64| num(o.f64(k).unwrap_or(d));
    let betas = format!("betas=({}, {}), eps={}", f("beta1", 0.9), f("beta2", 0.999), o.f64("eps").map(|e| format!("{e:e}")).unwrap_or_else(|| "1e-8".into()));
    let (cls, extra) = match o.name.as_str() {
        "adam" => ("Adam", betas),
        "adamw" => ("AdamW", betas),
        "nadam" => ("NAdam", betas),
        "radam" => ("RAdam", betas),
        "adamax" => ("Adamax", betas),
        "sgd" => {
            let nesterov = o.bool("nesterov").or_else(|| o.str("nesterov").map(|s| s == "true")).unwrap_or(false);
            let momentum = o.f64("momentum").unwrap_or(0.0);
            if nesterov && momentum <= 0.0 {
                return Err(bad("optimizer sgd: nesterov needs momentum > 0"));
            }
            ("SGD", format!("momentum={}, nesterov={}, dampening={}", num(momentum), if nesterov { "True" } else { "False" }, f("dampening", 0.0)))
        }
        "rmsprop" => ("RMSprop", format!("alpha={}, momentum={}, eps={}", f("alpha", 0.99), f("momentum", 0.0), o.f64("eps").map(|e| format!("{e:e}")).unwrap_or_else(|| "1e-8".into()))),
        "adagrad" => ("Adagrad", format!("lr_decay={}, eps={}", f("lr_decay", 0.0), o.f64("eps").map(|e| format!("{e:e}")).unwrap_or_else(|| "1e-10".into()))),
        "adadelta" => ("Adadelta", format!("rho={}, eps={}", f("rho", 0.9), o.f64("eps").map(|e| format!("{e:e}")).unwrap_or_else(|| "1e-6".into()))),
        other => return Err(bad(format!("unsupported optimizer '{other}'"))),
    };
    Ok(format!(
        "LR = {}\nWEIGHT_DECAY = {}\n_params = [p for p in model.parameters() if p.requires_grad]\noptimizer = torch.optim.{cls}(_params, lr=LR, weight_decay=WEIGHT_DECAY, {extra})\n",
        num(t.lr),
        num(t.weight_decay)
    ))
}

/// Returns the scheduler setup and whether it steps every batch or on the monitored metric.
fn scheduler_code(t: &TrainingSpec, plan: &Plan) -> (String, bool, bool) {
    let Some(s) = &t.scheduler else { return ("scheduler = None\n".into(), false, false) };
    let sched = "torch.optim.lr_scheduler";
    let w = s.i64("warmup_epochs").unwrap_or(0);
    let mut code = String::new();
    match s.name.as_str() {
        "onecycle" => {
            code.push_str(&format!(
                "scheduler = {sched}.OneCycleLR(optimizer, max_lr={}, total_steps=EPOCHS * STEPS_PER_EPOCH, pct_start={})\n",
                s.f64("max_lr").map(num).unwrap_or_else(|| format!("{} * 10", num(t.lr))),
                num(s.f64("pct_start").unwrap_or(0.3))
            ));
            return (code, true, false);
        }
        "plateau" => {
            code.push_str(&format!(
                "scheduler = {sched}.ReduceLROnPlateau(optimizer, mode=\"{}\", factor={}, patience={}, min_lr={})\n",
                if plan.monitor_max { "max" } else { "min" },
                num(s.f64("factor").unwrap_or(0.5)),
                s.i64("patience").unwrap_or(10),
                num(s.f64("min_lr").unwrap_or(0.0))
            ));
            return (code, false, true);
        }
        _ => {}
    }
    let main = match s.name.as_str() {
        "constant" => format!("{sched}.ConstantLR(optimizer, factor=1.0, total_iters=1)"),
        "step" => {
            let ss = s.i64("step_size").filter(|v| *v > 0).map(|v| v.to_string()).unwrap_or_else(|| "max(1, EPOCHS // 3)".into());
            format!("{sched}.StepLR(optimizer, step_size={ss}, gamma={})", num(s.f64("gamma").unwrap_or(0.1)))
        }
        "multistep" => {
            let ms = match s.params.get("milestones").map(jv).and_then(|m| m.as_array().cloned()) {
                Some(a) => format!("[{}]", a.iter().filter_map(|x| x.as_u64()).map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
                None => "[max(1, EPOCHS // 2), max(2, EPOCHS * 3 // 4)]".into(),
            };
            format!("{sched}.MultiStepLR(optimizer, milestones={ms}, gamma={})", num(s.f64("gamma").unwrap_or(0.1)))
        }
        "exponential" => format!("{sched}.ExponentialLR(optimizer, gamma={})", num(s.f64("gamma").unwrap_or(0.95))),
        "cosine" => format!("{sched}.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - {w}), eta_min={})", num(s.f64("eta_min").unwrap_or(0.0))),
        "cosine_restarts" => format!(
            "{sched}.CosineAnnealingWarmRestarts(optimizer, T_0={}, T_mult={}, eta_min={})",
            s.i64("t_0").unwrap_or(10).max(1),
            s.i64("t_mult").unwrap_or(1).max(1),
            num(s.f64("eta_min").unwrap_or(0.0))
        ),
        _ => unreachable!("validated"),
    };
    if w > 0 {
        code.push_str(&format!(
            "_warm = {sched}.LinearLR(optimizer, start_factor=1e-3, total_iters={w})\n_main = {main}\nscheduler = {sched}.SequentialLR(optimizer, [_warm, _main], milestones=[{w}])\n"
        ));
    } else {
        code.push_str(&format!("scheduler = {main}\n"));
    }
    (code, false, false)
}

pub fn training_script(g: &ValidatedGraph, id: &str) -> Result<String, GraphError> {
    let t = g.spec.training.as_ref().ok_or_else(|| bad("graph has no `training` section"))?;
    if g.spec.inputs.len() != 1 {
        return Err(bad("training scripts need exactly one graph input"));
    }
    let plan = plan(g, t)?;
    let task = plan.task;
    let d = &t.data;
    let row_task = task != "ranking";
    let t0 = plan.target_cols.first().cloned().unwrap_or_default();
    let classes_task = matches!(task, "classification" | "binary") || (task == "custom" && t.custom.as_ref().is_some_and(|c| c.target_dtype == "long") && !plan.target_cols.is_empty());

    // ---- helpers the chosen losses need
    let mut helpers = String::new();
    let ln = t.loss.name.as_str();
    if ln == "focal" {
        helpers.push_str(H_FOCAL);
        helpers.push('\n');
    }
    if ln == "quantile" {
        helpers.push_str(H_QUANTILE);
        helpers.push('\n');
    }
    if ln == "log_cosh" {
        helpers.push_str(H_LOGCOSH);
        helpers.push('\n');
    }
    if task == "ranking" {
        helpers.push_str(H_RANKING);
        helpers.push('\n');
    }
    if classes_task || task == "classification" {
        helpers.push_str(CLASS_KEY);
        helpers.push('\n');
    }
    let mut custom_names = (String::new(), None::<String>);
    if task == "custom" {
        let c = t.custom.as_ref().unwrap();
        let (src, origin) = custom_source(c)?;
        for f in std::iter::once(&c.loss).chain(c.metric.iter()) {
            if !src.contains(&format!("def {f}(")) {
                return Err(bad(format!("custom code ({origin}) does not define `def {f}(`")));
            }
        }
        helpers.push_str(&format!("# ---- custom task code ({origin}) ----\n{}\n# ---- end of custom code ----\n\n", src.trim_end()));
        custom_names = (c.loss.clone(), c.metric.clone());
    }

    // ---- data
    let mut data = String::new();
    data.push_str(&format!("torch.manual_seed({})\nwith open({:?}, newline=\"\") as f:\n    rows = list(csv.DictReader(f, skipinitialspace=True))\n", d.csv, d.seed).replace(&format!("({})", d.seed), &format!("({})", d.seed)));
    // the seed line must come first; rebuild in the right order
    let mut data = format!("torch.manual_seed({})\nwith open({:?}, newline=\"\") as f:\n    rows = list(csv.DictReader(f, skipinitialspace=True))\n", d.seed, d.csv);
    data.push_str(&format!("FEATURES = {}\nTARGETS = {}\nGROUP = {}\n", pylist(&d.features), pylist(&plan.target_cols), d.group.as_ref().map(|g| format!("{g:?}")).unwrap_or_else(|| "None".into())));
    data.push_str("_used = FEATURES + TARGETS + ([GROUP] if GROUP else [])\n_total = len(rows)\nrows = [r for r in rows if all((r.get(c) or \"\").strip() != \"\" for c in _used)]\n");
    data.push_str(&format!("if len(rows) < _total:\n    print(f\"skipped {{_total - len(rows)}} rows with missing values\", flush=True)\nif not rows:\n    raise SystemExit(\"no usable rows in \" + {:?})\n", d.csv));
    data.push_str("X = torch.tensor([[float(r[c]) for c in FEATURES] for r in rows], dtype=torch.float32)");
    if let Some(r) = &d.reshape {
        data.push_str(&format!("\nX = X.reshape(X.shape[0], {})", r.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")));
    }
    data.push('\n');
    let class_map = "CLASSES = sorted({r[TARGETS[0]] for r in rows}, key=_class_key)\n_index = {c: i for i, c in enumerate(CLASSES)}\n";
    match task {
        "classification" => {
            data.push_str(class_map);
            data.push_str("Y = torch.tensor([_index[r[TARGETS[0]]] for r in rows], dtype=torch.long)\nprint(\"classes:\", CLASSES, flush=True)\n");
        }
        "binary" => {
            data.push_str(class_map);
            data.push_str("if len(CLASSES) != 2:\n    raise SystemExit(f\"binary classification needs exactly 2 distinct values in {TARGETS[0]!r}, found {len(CLASSES)}: {CLASSES}\")\n");
            data.push_str("Y = torch.tensor([[float(_index[r[TARGETS[0]]])] for r in rows], dtype=torch.float32)\nprint(\"classes (negative, positive):\", CLASSES, flush=True)\n");
        }
        "multilabel" | "regression" => data.push_str("Y = torch.tensor([[float(r[c]) for c in TARGETS] for r in rows], dtype=torch.float32)\n"),
        "ranking" => data.push_str("Y = torch.tensor([float(r[TARGETS[0]]) for r in rows], dtype=torch.float32)\n"),
        "reconstruction" | "vae" => data.push_str("Y = X  # unsupervised: the data is its own target\n"),
        _ => {
            if plan.target_cols.is_empty() {
                data.push_str("Y = X  # no target column: the loss sees the input\n");
            } else if classes_task {
                data.push_str(class_map);
                data.push_str("Y = torch.tensor([_index[r[TARGETS[0]]] for r in rows], dtype=torch.long)\nprint(\"classes:\", CLASSES, flush=True)\n");
            } else {
                data.push_str("Y = torch.tensor([[float(r[c]) for c in TARGETS] for r in rows], dtype=torch.float32)\n");
            }
        }
    }
    if row_task {
        data.push_str(&format!("\nperm = torch.randperm(len(X))\nn_test = int(len(X) * {})\nte, tr = perm[:n_test], perm[n_test:]\nxtr, ytr, xte, yte = X[tr], Y[tr], X[te], Y[te]\nif len(xte) == 0:\n    xte, yte = xtr, ytr\nN_UNITS = len(xtr)\n", num(d.test_split)));
    } else {
        data.push_str(&format!(
            "\n_by = {{}}\nfor _i, r in enumerate(rows):\n    _by.setdefault(r[GROUP], []).append(_i)\nGX = [X[v] for v in _by.values()]\nGY = [Y[v] for v in _by.values()]\nperm = torch.randperm(len(GX))\nn_test = int(len(GX) * {})\nte_g, tr_g = perm[:n_test].tolist(), perm[n_test:].tolist()\nif not te_g:\n    te_g = tr_g\nN_UNITS = len(tr_g)\nprint(f\"{{len(GX)}} queries ({{len(tr_g)}} train, {{len(te_g)}} test)\", flush=True)\n",
            num(d.test_split)
        ));
    }

    // ---- model, initialisation, freezing
    let mut init = String::new();
    if task == "classification" {
        init.push_str("with torch.no_grad():\n    _n_out = model(xtr[:1]).shape[-1]\nif _n_out != len(CLASSES):\n    raise SystemExit(f\"the model outputs {_n_out} values but the data has {len(CLASSES)} classes {CLASSES}; set the last linear block's `out` to {len(CLASSES)}\")\n");
    }
    if let Some(path) = &t.init_from {
        init.push_str(&format!(
            "_sd = torch.load({path:?}, map_location=\"cpu\")\n_res = model.load_state_dict(_sd, strict=False)\nprint(f\"init_from {{{path:?}}}: loaded, {{len(_res.missing_keys)}} missing and {{len(_res.unexpected_keys)}} unexpected keys\", flush=True)\n"
        ));
    }
    if !t.freeze.is_empty() {
        init.push_str(&format!("for _name in {}:\n    for _p in getattr(model, _name).parameters():\n        _p.requires_grad = False\nprint(\"frozen:\", {}, flush=True)\n", pylist(&t.freeze), pylist(&t.freeze)));
    }

    // ---- loss
    let lp = &t.loss;
    let mut loss_setup = String::new();
    let weights = |lp: &Named| -> String {
        if lp.str("class_weights") == Some("balanced") {
            "_counts = torch.bincount(ytr, minlength=len(CLASSES)).float()\n_w = (_counts.sum() / (len(CLASSES) * _counts.clamp(min=1))).to(DEVICE)\n".to_string()
        } else {
            "_w = None\n".to_string()
        }
    };
    match (task, ln) {
        ("classification", "cross_entropy") => {
            loss_setup.push_str(&weights(lp));
            loss_setup.push_str(&format!("loss_fn = nn.CrossEntropyLoss(weight=_w, label_smoothing={})\n\n\ndef compute_loss(out, y):\n    return loss_fn(out, y)\n", num(lp.f64("label_smoothing").unwrap_or(0.0))));
        }
        ("classification", "focal") => {
            loss_setup.push_str(&weights(lp));
            loss_setup.push_str(&format!("\n\ndef compute_loss(out, y):\n    return focal_loss(out, y, {}, _w)\n", num(lp.f64("gamma").unwrap_or(2.0))));
        }
        ("classification", "multi_margin") => loss_setup.push_str(&format!("loss_fn = nn.MultiMarginLoss(margin={})\n\n\ndef compute_loss(out, y):\n    return loss_fn(out, y)\n", num(lp.f64("margin").unwrap_or(1.0)))),
        ("binary" | "multilabel", _) => {
            let pw = match lp.params.get("pos_weight").map(jv).as_ref() {
                Some(serde_json::Value::String(s)) if s == "balanced" => "_pos = ytr.sum(0)\n_pw = ((len(ytr) - _pos) / _pos.clamp(min=1)).to(DEVICE)\n".to_string(),
                Some(v) if v.as_f64().is_some() => format!("_pw = torch.tensor({}, device=DEVICE)\n", num(v.as_f64().unwrap())),
                Some(serde_json::Value::Null) | None => "_pw = None\n".to_string(),
                Some(other) => return Err(bad(format!("loss bce_logits: pos_weight must be a number or \"balanced\", got {other:?}"))),
            };
            loss_setup.push_str(&pw);
            loss_setup.push_str("loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pw)\n\n\ndef compute_loss(out, y):\n    return loss_fn(out, y)\n");
        }
        ("regression", name) => {
            let expr = match name {
                "mse" => "nn.MSELoss()(out, y)".to_string(),
                "mae" => "nn.L1Loss()(out, y)".to_string(),
                "huber" => format!("nn.HuberLoss(delta={})(out, y)", num(lp.f64("delta").unwrap_or(1.0))),
                "smooth_l1" => format!("nn.SmoothL1Loss(beta={})(out, y)", num(lp.f64("beta").unwrap_or(1.0))),
                "quantile" => {
                    let q = lp.f64("q").unwrap_or(0.5);
                    if !(0.0 < q && q < 1.0) {
                        return Err(bad("loss quantile: q must be between 0 and 1"));
                    }
                    format!("quantile_loss(out, y, {})", num(q))
                }
                "poisson" => "nn.PoissonNLLLoss(log_input=True)(out, y)".to_string(),
                _ => "log_cosh_loss(out, y)".to_string(),
            };
            loss_setup.push_str(&format!("def compute_loss(out, y):\n    y = y.view_as(out)\n    return {expr}\n"));
        }
        ("reconstruction", name) => loss_setup.push_str(&format!("def compute_loss(out, y):\n    return {}(out, y)\n", if name == "mae" { "nn.L1Loss()" } else { "nn.MSELoss()" })),
        ("vae", _) => loss_setup.push_str(&format!("BETA = {}\n\n\ndef vae_loss(out, x):\n    recon = ((out[\"recon\"] - x) ** 2).mean()\n    kl = -0.5 * (1 + out[\"logvar\"] - out[\"mu\"] ** 2 - out[\"logvar\"].exp()).mean()\n    return recon + BETA * kl, recon, kl\n", num(t.beta))),
        ("ranking", name) => {
            let expr = match name {
                "ranknet" => "ranknet_loss(s, y)".to_string(),
                "margin_rank" => format!("margin_rank_loss(s, y, {})", num(lp.f64("margin").unwrap_or(1.0))),
                _ => "listnet_loss(s, y)".to_string(),
            };
            loss_setup.push_str(&format!("NDCG_K = {}\n\n\ndef rank_loss(s, y):\n    return {expr}\n", lp.i64("ndcg_k").unwrap_or(10).max(1)));
        }
        ("custom", _) => {
            loss_setup.push_str(&format!("compute_loss = {}\n", custom_names.0));
            if let Some(m) = &custom_names.1 {
                loss_setup.push_str(&format!("custom_metric = {m}\n"));
            }
        }
        (task, name) => return Err(bad(format!("no implementation for loss '{name}' on the {task} task"))),
    }

    // ---- evaluate()
    let mut eval = String::new();
    if row_task {
        eval.push_str("def evaluate():\n    model.train(False)\n    with torch.no_grad():\n        x = xte.to(DEVICE)\n        y = yte.to(DEVICE)\n        with autocast_ctx():\n            out = model(x)\n");
        if task == "vae" {
            eval.push_str("        out = {k: v.float() for k, v in out.items()}\n        return {\"test_recon\": ((out[\"recon\"] - x) ** 2).mean().item()}\n");
        } else {
            eval.push_str("        out = out.float()\n        m = {}\n");
            eval.push_str("        m[\"test_loss\"] = compute_loss(out, y).item()\n");
            match task {
                "classification" => eval.push_str("        m[\"test_acc\"] = (out.argmax(1) == y).float().mean().item()\n"),
                "binary" | "multilabel" => eval.push_str("        m[\"test_acc\"] = ((out > 0).float() == y).float().mean().item()\n"),
                "regression" => eval.push_str("        yv = y.view_as(out)\n        m[\"test_mae\"] = (out - yv).abs().mean().item()\n        m[\"test_r2\"] = 1 - ((out - yv) ** 2).sum().item() / max(1e-12, ((yv - yv.mean(0)) ** 2).sum().item())\n"),
                "custom" => {
                    if let Some(name) = &custom_names.1 {
                        eval.push_str(&format!("        m[\"test_{name}\"] = float(custom_metric(out, y))\n"));
                    }
                }
                _ => {}
            }
            eval.push_str("        return m\n");
        }
    } else {
        eval.push_str("def evaluate():\n    model.train(False)\n    losses, ndcgs = [], []\n    with torch.no_grad():\n        for g in te_g:\n            with autocast_ctx():\n                s = model(GX[g].to(DEVICE)).reshape(-1)\n            s = s.float()\n            y = GY[g].to(DEVICE)\n            losses.append(rank_loss(s, y).item())\n            v = ndcg_at_k(s, y, NDCG_K)\n            if v is not None:\n                ndcgs.append(v)\n    return {\"test_loss\": sum(losses) / len(losses), \"test_ndcg\": sum(ndcgs) / max(1, len(ndcgs))}\n");
    }

    // ---- one training batch (defines `loss`)
    let batch_code = if !row_task {
        "        _losses = []\n        for _g in idx.tolist():\n            xg, yg = GX[tr_g[_g]].to(DEVICE), GY[tr_g[_g]].to(DEVICE)\n            with autocast_ctx():\n                s = model(xg).reshape(-1)\n            _losses.append(rank_loss(s.float(), yg))\n        loss = torch.stack(_losses).mean()\n".to_string()
    } else if task == "vae" {
        "        xb = xtr[idx].to(DEVICE)\n        with autocast_ctx():\n            out = model(xb)\n        out = {k: v.float() for k, v in out.items()}\n        loss, recon, kl = vae_loss(out, xb)\n".to_string()
    } else {
        "        xb, yb = xtr[idx].to(DEVICE), ytr[idx].to(DEVICE)\n        with autocast_ctx():\n            out = model(xb)\n        loss = compute_loss(out.float(), yb)\n".to_string()
    };

    let (sched_code, per_batch, on_metric) = scheduler_code(t, &plan);
    let batch_default = if row_task { "N_UNITS".to_string() } else { "min(16, N_UNITS)".to_string() };
    let bs = t.batch_size.map(|b| b.to_string()).unwrap_or(batch_default);
    let es = t.early_stopping.as_ref();

    let mut lp_loop = String::new();
    lp_loop.push_str(&format!(
        "BATCH = {bs}\nACCUM = {}\nEPOCHS = {}\nPRIMARY = {:?}\nMONITOR = {:?}\nMONITOR_MAX = {}\nEARLY = {}\nPATIENCE = {}\nMIN_DELTA = {}\nSAVE_BEST = {}\nCLIP = {}\nCHECKPOINT_MODE = {:?}\nRESUME = {}\nN_BATCHES = math.ceil(N_UNITS / BATCH)\nSTEPS_PER_EPOCH = math.ceil(N_BATCHES / ACCUM)\n\n",
        t.accumulate,
        t.epochs,
        plan.primary,
        plan.monitor,
        if plan.monitor_max { "True" } else { "False" },
        if es.is_some() { "True" } else { "False" },
        es.map(|e| e.patience).unwrap_or(1),
        num(es.map(|e| e.min_delta).unwrap_or(0.0)),
        if t.save_best { "True" } else { "False" },
        t.clip_grad.map(num).unwrap_or_else(|| "None".into()),
        t.checkpoint,
        if t.resume { "True" } else { "False" },
    ));
    lp_loop.push_str(&optimizer_code(t)?);
    lp_loop.push_str(&sched_code);
    lp_loop.push('\n');
    lp_loop.push_str(
        r#"_run = sentinel.get_run() if sentinel else None
_art = pathlib.Path(_run.artifacts_dir) if _run is not None else pathlib.Path(os.environ.get("SENTINEL_ARTIFACT_DIR", "artifacts"))
_ckpt_dir = _art / "checkpoints"
_ckpt_dir.mkdir(parents=True, exist_ok=True)
_latest_ckpt = _ckpt_dir / "checkpoint_latest.pt"

START_EPOCH = 0
BEST, BEST_EPOCH, BAD, BEST_STATE, EPOCHS_RUN = None, -1, 0, None, 0
metrics = {}

if RESUME and _latest_ckpt.exists():
    try:
        _ck = torch.load(_latest_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(_ck["model_state_dict"])
        if "optimizer_state_dict" in _ck and _ck["optimizer_state_dict"] is not None and "optimizer" in locals():
            optimizer.load_state_dict(_ck["optimizer_state_dict"])
        if "scheduler_state_dict" in _ck and _ck["scheduler_state_dict"] is not None and "scheduler" in locals() and scheduler is not None:
            scheduler.load_state_dict(_ck["scheduler_state_dict"])
        START_EPOCH = _ck.get("epoch", -1) + 1
        BEST = _ck.get("best_metric", None)
        BEST_EPOCH = _ck.get("best_epoch", -1)
        if "metrics" in _ck and isinstance(_ck["metrics"], dict):
            metrics = _ck["metrics"]
        print(f"Resumed seamlessly from checkpoint: starting at epoch {START_EPOCH}", flush=True)
    except Exception as _e:
        print(f"Checkpoint resume failed, starting from scratch: {_e}", flush=True)

if START_EPOCH >= EPOCHS and not metrics:
    metrics = evaluate()

for epoch in range(START_EPOCH, EPOCHS):
    model.train(True)
    order = torch.randperm(N_UNITS)
    optimizer.zero_grad(set_to_none=True)
    for step, i in enumerate(range(0, N_UNITS, BATCH)):
        idx = order[i:i + BATCH]
"#,
    );
    lp_loop.push_str(&batch_code);
    lp_loop.push_str("        scaler.scale(loss / ACCUM).backward()\n        if (step + 1) % ACCUM == 0 or (step + 1) == N_BATCHES:\n            if CLIP is not None:\n                scaler.unscale_(optimizer)\n                torch.nn.utils.clip_grad_norm_(_params, CLIP)\n            scaler.step(optimizer)\n            scaler.update()\n            optimizer.zero_grad(set_to_none=True)\n");
    if per_batch {
        lp_loop.push_str("            scheduler.step()\n");
    }
    lp_loop.push_str("    EPOCHS_RUN = epoch + 1\n    metrics = evaluate()\n");
    if on_metric {
        lp_loop.push_str("    scheduler.step(metrics[MONITOR])\n");
    } else if t.scheduler.is_some() && !per_batch {
        lp_loop.push_str("    scheduler.step()\n");
    }
    lp_loop.push_str(
        r#"    _val = metrics[MONITOR]
    if BEST is None or ((_val > BEST + MIN_DELTA) if MONITOR_MAX else (_val < BEST - MIN_DELTA)):
        BEST, BEST_EPOCH, BAD = _val, epoch, 0
        if SAVE_BEST:
            BEST_STATE = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        BAD += 1
    if CHECKPOINT_MODE in ("latest", "all"):
        try:
            _payload = {
                "epoch": epoch,
                "step": (epoch + 1) * STEPS_PER_EPOCH,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if "scheduler" in locals() and scheduler is not None else None,
                "metrics": metrics,
                "best_metric": BEST,
                "best_epoch": BEST_EPOCH,
            }
            _tmp = _ckpt_dir / f".tmp_checkpoint_{epoch}.pt"
            torch.save(_payload, _tmp)
            if CHECKPOINT_MODE == "all":
                shutil.copy2(_tmp, _ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt")
            os.replace(_tmp, _latest_ckpt)
        except Exception as _ce:
            print(f"Checkpoint save warning: {_ce}", flush=True)
    STOP = EARLY and BAD >= PATIENCE
    if STOP:
        print(f"early stopping at epoch {epoch}: {MONITOR} has not improved for {PATIENCE} epochs (best {BEST:.4f} at epoch {BEST_EPOCH})", flush=True)
"#,
    );
    lp_loop.push_str("    if epoch % 10 == 0 or epoch == EPOCHS - 1 or STOP:\n        _lr = optimizer.param_groups[0][\"lr\"]\n        _parts = [f\"epoch {epoch:3d}\", f\"loss {loss.item():.4f}\", f\"lr {_lr:.6f}\"] + [f\"{k} {v:.4f}\" for k, v in metrics.items() if k != PRIMARY] + [f\"{PRIMARY} {metrics[PRIMARY]:.4f}\"]\n        print(\" \".join(_parts), flush=True)\n        if sentinel:\n");
    let extra_log = if task == "vae" { ", \"recon\": recon.item(), \"kl\": kl.item()" } else { "" };
    lp_loop.push_str(&format!("            sentinel.log({{\"train_loss\": loss.item(), \"lr\": _lr{extra_log}, **metrics}}, step=epoch)\n"));
    lp_loop.push_str("    if STOP:\n        break\nif BEST_STATE is not None:\n    model.load_state_dict({k: v.to(DEVICE) for k, v in BEST_STATE.items()})\n    metrics = evaluate()\n    print(f\"restored the best weights from epoch {BEST_EPOCH} ({MONITOR} {BEST:.4f})\", flush=True)\nelif not metrics:\n    metrics = evaluate()\n");

    // ---- export
    let classes_expr = if classes_task { "list(CLASSES)" } else if task == "multilabel" { "list(TARGETS)" } else { "None" };
    let target_json = if plan.target_cols.is_empty() { "None".to_string() } else { format!("{t0:?}") };
    let export = format!(
        r##"
# ---- export: weights, TorchScript, ONNX and metadata, next to the run's other artifacts ----------------------
_run = sentinel.get_run() if sentinel else None
_art = pathlib.Path(_run.artifacts_dir) if _run is not None else pathlib.Path(os.environ.get("SENTINEL_ARTIFACT_DIR", "artifacts"))
_art.mkdir(parents=True, exist_ok=True)
_plots_dir = _art / "plots"
_plots_dir.mkdir(parents=True, exist_ok=True)
_results_dir = _art / "results"
_results_dir.mkdir(parents=True, exist_ok=True)
model.cpu()
model.train(False)
_example = X[:1]
torch.save(model.state_dict(), _art / "model.pt")
_files = {{"weights": "model.pt"}}
try:
    torch.jit.trace(model, _example, strict=False, check_trace=False).save(str(_art / "model_script.pt"))
    _files["torchscript"] = "model_script.pt"
except Exception as _e:
    print("torchscript export skipped:", _e, flush=True)
try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the legacy exporter announces its own deprecation on every call
        torch.onnx.export(model, _example, str(_art / "model.onnx"), input_names=["input"], dynamic_axes={{"input": {{0: "batch"}}}}, dynamo=False)
    _files["onnx"] = "model.onnx"
except Exception as _e:
    print("onnx export skipped:", _e, flush=True)

_results_payload = {{
    "graph": {id:?},
    "task": {task:?},
    "best_epoch": BEST_EPOCH,
    "epochs_run": EPOCHS_RUN,
    "monitor": MONITOR,
    "metrics": metrics,
}}
with open(_art / "results.json", "w") as _rf:
    json.dump(_results_payload, _rf, indent=2)
with open(_results_dir / "summary.json", "w") as _rf:
    json.dump(_results_payload, _rf, indent=2)
_files["results"] = "results.json"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _metric_items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))]
    if _metric_items:
        _fig, _ax = _plt.subplots(figsize=(6, 3.2), dpi=100)
        _names = [k for k, _ in _metric_items]
        _vals = [float(v) for _, v in _metric_items]
        _bars = _ax.bar(_names, _vals, color="#38bdf8", alpha=0.85, width=0.45)
        _ax.set_title(f"{id} - Metrics", fontsize=10)
        _ax.grid(axis="y", linestyle="--", alpha=0.3)
        for _b, _val in zip(_bars, _vals):
            _ax.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 0.005, f"{{_val:.3f}}", ha="center", va="bottom", fontsize=8)
        _plt.tight_layout()
        _curve_file = _plots_dir / "training_curves.png"
        _fig.savefig(str(_curve_file), bbox_inches="tight")
        _plt.close(_fig)
        _files["training_curves"] = "plots/training_curves.png"
except Exception:
    pass

with open(_art / "meta.json", "w") as _f:
    json.dump({{
        "graph": {id:?}, "task": {task:?}, "features": FEATURES, "target": {target_json}, "targets": TARGETS, "group": GROUP,
        "classes": {classes_expr}, "input_shape": list(_example.shape[1:]), "loss": {loss:?}, "optimizer": {opt:?},
        "n_params": N_PARAMS, "n_trainable": N_TRAINABLE, "device": str(DEVICE), "amp": AMP, "epochs_run": EPOCHS_RUN,
        "best_epoch": BEST_EPOCH, "monitor": MONITOR, "torch": torch.__version__, "files": _files, **metrics,
    }}, _f, indent=2)
shutil.copyfile(__file__, _art / "train.py")
print("artifacts saved to", _art, flush=True)
if sentinel:
    sentinel.finish()
"##,
        id = id,
        task = task,
        loss = t.loss.name,
        opt = t.optimizer.name,
    );

    // ---- run configuration recorded with the run
    let model_params: serde_json::Map<String, serde_json::Value> = g.order.iter().map(|n| (n.clone(), serde_json::to_value(&g.params[n]).unwrap_or_default())).collect();
    let cfg = serde_json::to_string(&json!({
        "graph": id, "task": task, "loss": t.loss, "optimizer": t.optimizer, "lr": t.lr, "weight_decay": t.weight_decay, "epochs": t.epochs,
        "batch_size": t.batch_size, "beta": t.beta, "scheduler": t.scheduler, "clip_grad": t.clip_grad, "accumulate": t.accumulate,
        "amp": t.amp, "device": t.device, "early_stopping": t.early_stopping, "save_best": t.save_best, "freeze": t.freeze,
        "checkpoint": t.checkpoint, "resume": t.resume, "model": model_params,
    }))
    .unwrap();

    let mut s = String::new();
    s.push_str(&HEADER.replace("@@ID@@", id));
    s.push('\n');
    s.push_str(&helpers);
    s.push_str(&crate::codegen::generate_model(g));
    s.push_str("\n\n");
    s.push_str(&data);
    s.push('\n');
    s.push_str(&DEVICE_AMP.replace("@@DEVICE@@", &format!("{:?}", t.device)).replace("@@AMP@@", &format!("{:?}", t.amp)));
    s.push_str("\nmodel = Model()\n");
    s.push_str(&init);
    s.push_str("N_PARAMS = sum(p.numel() for p in model.parameters())\nN_TRAINABLE = sum(p.numel() for p in model.parameters() if p.requires_grad)\nprint(f\"model: {N_PARAMS:,} parameters ({N_PARAMS * 4 / 1048576:.2f} MB as float32)\" + (f\", {N_TRAINABLE:,} trainable\" if N_TRAINABLE != N_PARAMS else \"\"), flush=True)\nmodel.to(DEVICE)\n\n");
    s.push_str(&loss_setup);
    s.push_str("\n\n");
    s.push_str(&eval);
    s.push_str(&format!("\n\ntry:\n    import sentinel\n    sentinel.init(run_name={id:?}, config={{**json.loads({cfg:?}), \"n_params\": N_PARAMS, \"n_trainable\": N_TRAINABLE}})\nexcept Exception:\n    sentinel = None\n\n"));
    s.push_str(&lp_loop);
    s.push_str(&export);
    Ok(s)
}
