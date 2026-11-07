//! Autonomous Agent Diagnostics & Auto-Remediation Engine.
//!
//! Evaluates live telemetry and historical run metrics to produce actionable,
//! mathematically grounded prescriptions for autonomous coding agents and human ML engineers.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prescription {
    pub id: String,
    pub severity: String, // "critical" | "warning" | "info"
    pub category: String, // "convergence" | "gradient" | "capacity" | "hardware"
    pub title: String,
    pub root_cause: String,
    pub recommendation: String,
    pub cli_command: Option<String>,
    pub patch: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub healthy: bool,
    pub health_score: u32, // 0 - 100
    pub prescriptions: Vec<Prescription>,
    pub summary: String,
}

/// Diagnose a run directory and generate remediation prescriptions.
pub fn diagnose_run(run_dir: &Path, run_id: &str) -> DiagnosticReport {
    let mut prescriptions = Vec::new();
    let mut score: i32 = 100;

    // 1. Check metadata and metrics
    let meta = sentinel_core_meta(run_dir);
    let comp_name = meta.as_ref().and_then(|m| m["task"].as_str()).unwrap_or("model");

    // 2. Check for xAI explanation / representation collapse
    let xai_file = run_dir.join("artifacts").join("xai_explanation.json");
    if xai_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&xai_file) {
            if let Ok(xai) = serde_json::from_str::<Value>(&content) {
                if let Some(ranks) = xai["spectral_ranks"].as_object() {
                    for (layer, rank) in ranks {
                        let ratio = rank["collapse_ratio"].as_f64().unwrap_or(1.0);
                        let eff = rank["effective_rank"].as_f64().unwrap_or(0.0);
                        let max_r = rank["max_rank"].as_f64().unwrap_or(1.0);
                        if ratio < 0.25 {
                            score -= 20;
                            prescriptions.push(Prescription {
                                id: format!("COLLAPSE_{}", layer.replace('.', "_")),
                                severity: "warning".into(),
                                category: "capacity".into(),
                                title: format!("Representation Collapse in '{layer}'"),
                                root_cause: format!(
                                    "Effective spectral rank is only {:.2} out of {:.0} ({:.1}% capacity). The layer's singular values have degenerated into a lower-dimensional subspace.",
                                    eff, max_r, ratio * 100.0
                                ),
                                recommendation: "Inject weight decay or switch activation to LeakyReLU/GELU to retain representation rank.".into(),
                                cli_command: Some(format!("sentinel run {comp_name} --set weight_decay=1e-4")),
                                patch: Some(json!({ "optimizer.weight_decay": 0.0001 })),
                            });
                        }
                    }
                }
            }
        }
    }

    // 3. Check for live gradient flow file in workspace
    let flow_file = Path::new(".sentinel").join("live_flow.json");
    if flow_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&flow_file) {
            if let Ok(flow) = serde_json::from_str::<Value>(&content) {
                let vanishing_ratio = flow["vanishing_ratio"].as_f64().unwrap_or(1.0);
                let exploding = flow["exploding"].as_bool().unwrap_or(false);

                if exploding {
                    score -= 35;
                    prescriptions.push(Prescription {
                        id: "EXPLODING_GRADIENTS".into(),
                        severity: "critical".into(),
                        category: "gradient".into(),
                        title: "Exploding Gradient Flow Detected".into(),
                        root_cause: "Layer gradient norm exceeds numerical stability threshold (> 10.0). Risk of NaN weight updates.".into(),
                        recommendation: "Reduce learning rate by 5x and enforce gradient clipping.".into(),
                        cli_command: Some(format!("sentinel run {comp_name} --set lr=0.0002")),
                        patch: Some(json!({ "lr": 0.0002, "clip_grad_norm": 1.0 })),
                    });
                } else if vanishing_ratio < 1e-4 {
                    score -= 25;
                    prescriptions.push(Prescription {
                        id: "VANISHING_GRADIENTS".into(),
                        severity: "critical".into(),
                        category: "gradient".into(),
                        title: "Vanishing Gradient Flow (Deep Architecture Attenuation)".into(),
                        root_cause: format!(
                            "Gradient ratio between first and last layer is {:.2e}. Early layers are receiving virtually zero gradient signals.",
                            vanishing_ratio
                        ),
                        recommendation: "Insert residual skip connections or replace saturating activations with SiLU/GELU.".into(),
                        cli_command: Some(format!("sentinel run {comp_name} --set lr=0.002")),
                        patch: Some(json!({ "lr": 0.002 })),
                    });
                }

                if let Some(layers) = flow["layers"].as_array() {
                    for l in layers {
                        let dead = l["dead_pct"].as_f64().unwrap_or(0.0);
                        let name = l["name"].as_str().unwrap_or("layer");
                        if dead > 75.0 {
                            score -= 15;
                            prescriptions.push(Prescription {
                                id: format!("DEAD_NEURONS_{name}"),
                                severity: "warning".into(),
                                category: "convergence".into(),
                                title: format!("Dying Neurons ({:.0}%) in '{name}'", dead),
                                root_cause: "High percentage of post-activation tensors are clamped at zero, permanently killing backprop gradients.".into(),
                                recommendation: "Switch to LeakyReLU (alpha=0.01) or GELU to maintain nonzero gradient flow.".into(),
                                cli_command: Some(format!("sentinel run {comp_name} --set activation=gelu")),
                                patch: Some(json!({ "activation": "gelu" })),
                            });
                        }
                    }
                }
            }
        }
    }

    // Baseline prescription if healthy
    let final_score = score.clamp(0, 100) as u32;
    let healthy = prescriptions.is_empty();
    let summary = if healthy {
        format!("Run {run_id} is healthy with full representation rank and stable gradient flow.")
    } else {
        format!("Found {} issue(s) affecting model training health score ({}%).", prescriptions.len(), final_score)
    };

    DiagnosticReport {
        healthy,
        health_score: final_score,
        prescriptions,
        summary,
    }
}

fn sentinel_core_meta(run_dir: &Path) -> Option<Value> {
    let p = run_dir.join("artifacts").join("meta.json");
    if p.exists() {
        let content = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&content).ok()
    } else {
        None
    }
}
