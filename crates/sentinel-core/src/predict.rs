//! Inference from a finished run's artifacts (`model_script.pt` + `meta.json`), run in a short-lived Python process
//! so the daemon never needs PyTorch itself.

use serde::Serialize;
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const PREDICT_PY: &str = r#"
import json, pathlib, sys
import torch

art = pathlib.Path(sys.argv[1])
meta = json.load(open(art / "meta.json"))
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
if task == "classification":
    probs = torch.softmax(out, dim=1)
    classes = meta["classes"]
    preds = [{"label": classes[int(p.argmax())], "index": int(p.argmax()), "probabilities": {c: float(v) for c, v in zip(classes, p)}} for p in probs]
    res = {"task": task, "predictions": preds}
elif task == "vae":
    res = {"task": task, "reconstruction": out["recon"].tolist(), "latent_mean": out["mu"].tolist()}
else:
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
    let art = run_dir.join("artifacts");
    if !art.join("meta.json").exists() {
        return Err("this run has no exported model (meta.json missing); train it with a Sentinel graph component".into());
    }
    let mut child = Command::new(crate::platform::python_exe())
        .args(["-I", "-c", PREDICT_PY])
        .arg(&art)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", crate::platform::python_exe()))?;
    let payload = json!({ "inputs": inputs }).to_string();
    child.stdin.take().ok_or("no stdin")?.write_all(payload.as_bytes()).map_err(|e| e.to_string())?;
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
