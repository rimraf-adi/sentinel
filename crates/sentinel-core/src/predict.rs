//! Inference from a finished run's artifacts (`model_script.pt` + `meta.json`), run in a short-lived Python process
//! so the daemon never needs PyTorch itself.

use serde::Serialize;
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const GENERATE_PY: &str = r#"
import json, pathlib, sys
import torch

art = pathlib.Path(sys.argv[1])
count, seed = int(sys.argv[2]), int(sys.argv[3])
meta = json.load(open(art / "meta.json"))
if meta["task"] != "gan":
    raise SystemExit("this run is not a generative model (use predict)")
if "generator_script" not in meta.get("files", {}):
    raise SystemExit("this run has no TorchScript generator (see the training log for why)")
torch.manual_seed(seed)
g = torch.jit.load(str(art / meta["files"]["generator_script"]))
g.train(False)
with torch.no_grad():
    s = g(torch.randn(count, meta["latent"]))
print(json.dumps({"task": "gan", "features": meta["features"], "samples": s.tolist()}))
"#;

const PREDICT_PY: &str = r#"
import json, pathlib, sys
import torch

art = pathlib.Path(sys.argv[1])
meta = json.load(open(art / "meta.json"))
if meta["task"] == "gan":
    raise SystemExit("this run is a generative model: use generate, not predict")
rows = json.load(sys.stdin)["inputs"]
feats = meta["features"]
x = torch.tensor(rows, dtype=torch.float32)
if x.ndim != 2 or x.shape[1] != len(feats):
    raise SystemExit(f"expected rows of {len(feats)} values {feats}, got shape {list(x.shape)}")
x = x.reshape(x.shape[0], *meta["input_shape"])
if "torchscript" not in meta.get("files", {}):
    raise SystemExit("this run has no TorchScript export (see the training log for why)")
model = torch.jit.load(str(art / meta["files"]["torchscript"]))
model.train(False)
with torch.no_grad():
    out = model(x)
task = meta["task"]
classes = meta.get("classes")
if task == "classification":
    probs = torch.softmax(out, dim=1)
    preds = [{"label": classes[int(p.argmax())], "index": int(p.argmax()), "probabilities": {c: float(v) for c, v in zip(classes, p)}} for p in probs]
    res = {"task": task, "predictions": preds}
elif task == "binary":
    pos = torch.sigmoid(out.reshape(-1))
    preds = [{"label": classes[int(p > 0.5)], "index": int(p > 0.5), "probabilities": {classes[0]: 1 - float(p), classes[1]: float(p)}} for p in pos]
    res = {"task": task, "predictions": preds}
elif task == "multilabel":
    probs = torch.sigmoid(out)
    preds = []
    for p in probs:
        on = [c for c, v in zip(classes, p) if v > 0.5]
        preds.append({"label": "+".join(on) if on else "(none)", "index": -1, "probabilities": {c: float(v) for c, v in zip(classes, p)}, "positive": on})
    res = {"task": task, "predictions": preds}
elif task == "vae":
    res = {"task": task, "reconstruction": out["recon"].tolist(), "latent_mean": out["mu"].tolist()}
else:
    # regression, ranking (one score per row), reconstruction, custom: the raw output rows
    res = {"task": task, "values": out.reshape(x.shape[0], -1).tolist()}
print(json.dumps(res))
"#;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ArtifactFile {
    pub name: String,
    pub bytes: u64,
}

/// Files in a run's `artifacts/` directory (top level only).
pub fn list_artifacts(run_dir: &Path) -> Vec<ArtifactFile> {
    let mut v: Vec<ArtifactFile> = std::fs::read_dir(run_dir.join("artifacts"))
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| ArtifactFile { name: e.file_name().to_string_lossy().to_string(), bytes: e.metadata().map(|m| m.len()).unwrap_or(0) })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

pub fn read_meta(run_dir: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(run_dir.join("artifacts").join("meta.json")).ok()?).ok()
}

/// Run the model on `inputs` (one row of feature values per sample).
pub fn predict(run_dir: &Path, inputs: &[Vec<f64>]) -> Result<Value, String> {
    if inputs.is_empty() {
        return Err("provide at least one input row".into());
    }
    if inputs.len() > 10_000 {
        return Err("at most 10000 rows per request".into());
    }
    run_python(run_dir, PREDICT_PY, &[], json!({ "inputs": inputs }).to_string())
}

/// Draw `count` samples from a GAN run's generator.
pub fn generate(run_dir: &Path, count: usize, seed: u64) -> Result<Value, String> {
    if count == 0 || count > 5_000 {
        return Err("count must be between 1 and 5000".into());
    }
    run_python(run_dir, GENERATE_PY, &[count.to_string(), seed.to_string()], String::new())
}

const EXPLAIN_PY: &str = r#"
import json, pathlib, sys
import torch

art = pathlib.Path(sys.argv[1])
meta = json.load(open(art / "meta.json"))
rows = json.load(sys.stdin)["inputs"]
feats = meta.get("features", [])
x = torch.tensor(rows, dtype=torch.float32)
if "torchscript" not in meta.get("files", {}):
    raise SystemExit("this run has no TorchScript export")
model = torch.jit.load(str(art / meta["files"]["torchscript"]))
model.train(False)

target_sample = x[0:1]
baseline = torch.zeros_like(target_sample)
steps = 25
alphas = torch.linspace(0.0, 1.0, steps)
interp = torch.cat([baseline + alpha * (target_sample - baseline) for alpha in alphas], dim=0)
interp.requires_grad_(True)
out = model(interp)

if out.dim() > 1 and out.shape[-1] > 1:
    target_cls = int(out[-1].argmax().item())
    selected = out[:, target_cls].sum()
else:
    target_cls = 0
    selected = out.sum()

grads = torch.autograd.grad(selected, interp)[0]
avg_grads = grads.mean(dim=0, keepdim=True)
attributions = ((target_sample - baseline) * avg_grads).detach().squeeze(0).tolist()
sample_vals = target_sample.squeeze(0).tolist()

if not feats or len(feats) != len(attributions):
    feats = [f"f_{i}" for i in range(len(attributions))]

total_abs = sum(abs(a) for a in attributions) or 1e-9
items = []
for name, val, a in zip(feats, sample_vals, attributions):
    items.append({
        "feature": name,
        "value": round(float(val), 4),
        "attribution": round(float(a), 5),
        "importance_pct": round(abs(float(a)) / total_abs * 100.0, 2),
        "direction": "positive" if a > 0 else "negative" if a < 0 else "neutral"
    })
items.sort(key=lambda item: abs(item["attribution"]), reverse=True)

layer_ranks = {}
for name, param in model.named_parameters():
    if "weight" in name and param.dim() >= 2:
        w = param.detach().flatten(start_dim=1)
        try:
            _, s, _ = torch.linalg.svd(w.float(), full_matrices=False)
            tot = s.sum().item()
            if tot > 1e-9:
                p = s / tot
                p = p[p > 1e-12]
                h = -float((p * torch.log(p)).sum().item())
                eff = round(float(torch.exp(torch.tensor(h)).item()), 2)
                max_r = float(min(w.shape))
                layer_ranks[name] = {"effective_rank": eff, "max_rank": max_r, "collapse_ratio": round(eff / max(max_r, 1.0), 3)}
        except Exception:
            pass

res = {
    "target_class": target_cls,
    "attributions": items,
    "top_positive": [item for item in items if item["attribution"] > 0][:5],
    "top_negative": [item for item in items if item["attribution"] < 0][:5],
    "spectral_ranks": layer_ranks,
}
print(json.dumps(res))
"#;

/// Explain model decision on sample input using axiomatic Integrated Gradients and Spectral Rank.
pub fn explain(run_dir: &Path, inputs: &[Vec<f64>]) -> Result<Value, String> {
    // If an existing explanation artifact is present and no custom inputs given, return it
    let art_exp = run_dir.join("artifacts").join("xai_explanation.json");
    if inputs.is_empty() && art_exp.exists() {
        if let Ok(content) = std::fs::read_to_string(&art_exp) {
            if let Ok(val) = serde_json::from_str::<Value>(&content) {
                return Ok(val);
            }
        }
    }

    if inputs.is_empty() {
        return Err("provide at least one input row for explanation".into());
    }
    run_python(run_dir, EXPLAIN_PY, &[], json!({ "inputs": inputs }).to_string())
}


fn run_python(run_dir: &Path, script: &str, extra: &[String], stdin_payload: String) -> Result<Value, String> {
    let art = run_dir.join("artifacts");
    if !art.join("meta.json").exists() {
        return Err("this run has no exported model (meta.json missing); train it with a Sentinel graph component".into());
    }
    let mut child = Command::new(crate::platform::python_exe())
        .args(["-I", "-c", script])
        .arg(&art)
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", crate::platform::python_exe()))?;
    child.stdin.take().ok_or("no stdin")?.write_all(stdin_payload.as_bytes()).map_err(|e| e.to_string())?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    let out = rx.recv_timeout(Duration::from_secs(90)).map_err(|_| "prediction timed out after 90s".to_string())?.map_err(|e| e.to_string())?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("prediction failed").to_string());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("unreadable prediction output: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_files_and_missing_meta_is_a_clear_error() {
        let dir = std::env::temp_dir().join(format!("sentinel-pred-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("artifacts")).unwrap();
        std::fs::write(dir.join("artifacts/b.pt"), "12345").unwrap();
        std::fs::write(dir.join("artifacts/a.json"), "{}").unwrap();
        let files = list_artifacts(&dir);
        assert_eq!(files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["a.json", "b.pt"]);
        assert_eq!(files[1].bytes, 5);
        assert!(predict(&dir, &[vec![1.0]]).unwrap_err().contains("no exported model"));
        assert!(predict(&dir, &[]).unwrap_err().contains("at least one"));
        assert!(read_meta(&dir).is_none());
    }
}
