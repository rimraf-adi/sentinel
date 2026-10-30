//! `sentinel model list | predict | export`: use what training exported.

use crate::ModelCommands;
use sentinel_core::{predict, tracking};
use std::path::PathBuf;

fn root() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join("mlruns")
}

fn resolve(run: &str) -> anyhow::Result<(String, PathBuf)> {
    let id = tracking::resolve_run_id(&root(), run).map_err(|e| anyhow::anyhow!("{e} (see `sentinel model list`)"))?;
    let dir = tracking::run_dir(&root(), &id).ok_or_else(|| anyhow::anyhow!("run {id} not found"))?;
    Ok((id, dir))
}

pub fn run(cmd: ModelCommands) -> anyhow::Result<()> {
    match cmd {
        ModelCommands::List => {
            let mut any = false;
            println!("{:<10} {:<34} {:<15} {:<22} FILES", "RUN", "NAME", "TASK", "TEST METRIC");
            for r in tracking::list_runs(&root()) {
                let Some(dir) = tracking::run_dir(&root(), &r.run_id) else { continue };
                let Some(meta) = predict::read_meta(&dir) else { continue };
                any = true;
                let metric = meta
                    .as_object()
                    .and_then(|m| m.iter().find(|(k, _)| k.starts_with("test_")))
                    .map(|(k, v)| format!("{k} {:.4}", v.as_f64().unwrap_or(f64::NAN)))
                    .unwrap_or_default();
                let files = meta["files"].as_object().map(|f| f.keys().cloned().collect::<Vec<_>>().join(",")).unwrap_or_default();
                println!("{:<10} {:<34} {:<15} {:<22} {}", &r.run_id[..8], r.run_name.chars().take(34).collect::<String>(), meta["task"].as_str().unwrap_or("?"), metric, files);
            }
            if !any {
                println!("(no exported models yet: train a graph component with `sentinel run <component>`)");
            }
        }
        ModelCommands::Predict { run, input, json } => {
            let (id, dir) = resolve(&run)?;
            let rows: Vec<Vec<f64>> = input
                .iter()
                .map(|line| line.split(',').map(|v| v.trim().parse::<f64>().map_err(|_| anyhow::anyhow!("'{v}' is not a number in --input \"{line}\"", v = v.trim()))).collect())
                .collect::<Result<_, _>>()?;
            let result = predict::predict(&dir, &rows).map_err(|e| anyhow::anyhow!(e))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let meta = predict::read_meta(&dir).unwrap_or_default();
            let feats: Vec<String> = meta["features"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
            println!("run {} · {} · inputs: {}", &id[..8], result["task"].as_str().unwrap_or("?"), feats.join(", "));
            for (i, row) in rows.iter().enumerate() {
                let shown = row.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ");
                match result["task"].as_str() {
                    Some("classification") => {
                        let p = &result["predictions"][i];
                        let top = p["probabilities"][p["label"].as_str().unwrap_or("")].as_f64().unwrap_or(0.0);
                        println!("  [{shown}] -> {} ({:.1}%)", p["label"].as_str().unwrap_or("?"), top * 100.0);
                    }
                    Some("regression") => println!("  [{shown}] -> {}", result["values"][i]),
                    _ => println!("  [{shown}] -> reconstruction {} · latent {}", result["reconstruction"][i], result["latent_mean"][i]),
                }
            }
        }
        ModelCommands::Generate { run, count, seed, json } => {
            let (id, dir) = resolve(&run)?;
            let result = predict::generate(&dir, count, seed).map_err(|e| anyhow::anyhow!(e))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let feats: Vec<String> = result["features"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
            println!("run {} · {count} samples (seed {seed}) · {}", &id[..8], feats.join(", "));
            for row in result["samples"].as_array().into_iter().flatten() {
                let vals: Vec<String> = row.as_array().into_iter().flatten().map(|v| format!("{:>9.4}", v.as_f64().unwrap_or(f64::NAN))).collect();
                println!("  {}", vals.join("  "));
            }
        }
        ModelCommands::Export { run, format, output } => {
            let (id, dir) = resolve(&run)?;
            let meta = predict::read_meta(&dir).ok_or_else(|| anyhow::anyhow!("run {id} has no exported model"))?;
            let file = match format.as_str() {
                "torchscript" | "onnx" | "weights" | "generator" | "discriminator" => meta["files"][&format].as_str().map(String::from).ok_or_else(|| {
                    anyhow::anyhow!("this run has no {format} export (available: {})", meta["files"].as_object().map(|f| f.keys().cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default())
                })?,
                "meta" => "meta.json".to_string(),
                "script" => "train.py".to_string(),
                other => anyhow::bail!("unknown format '{other}' (torchscript | onnx | weights | meta | script)"),
            };
            let src = dir.join("artifacts").join(&file);
            let dest = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("{}-{file}", &id[..8])));
            std::fs::copy(&src, &dest)?;
            println!("{} -> {} ({} bytes)", src.display(), dest.display(), std::fs::metadata(&dest)?.len());
        }
    }
    Ok(())
}
