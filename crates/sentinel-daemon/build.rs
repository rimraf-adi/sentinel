//! Make sure `ui/dist` exists before `rust-embed` runs, so `cargo build` always works.
//!
//! * dist missing or older than `ui/src`: build it with npm (skipped if `SENTINEL_SKIP_UI` is set
//!   or npm is unavailable).
//! * still missing: write a placeholder page that explains how to build the UI.
//!
//! Release CI builds the UI once and sets `SENTINEL_SKIP_UI=1` for the per-target wheel builds.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

fn newest(dir: &Path) -> Option<SystemTime> {
    let mut best = None;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        let t = if p.is_dir() { newest(&p) } else { e.metadata().ok().and_then(|m| m.modified().ok()) };
        best = best.max(t);
    }
    best
}

fn main() {
    let ui: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ui");
    let dist = ui.join("dist");
    let index = dist.join("index.html");

    for p in ["src", "index.html", "package.json", "package-lock.json", "vite.config.ts"] {
        println!("cargo:rerun-if-changed={}", ui.join(p).display());
    }
    println!("cargo:rerun-if-env-changed=SENTINEL_SKIP_UI");

    let built = std::fs::metadata(&index).and_then(|m| m.modified()).ok();
    let src = newest(&ui.join("src")).max(std::fs::metadata(ui.join("index.html")).and_then(|m| m.modified()).ok());
    let stale = match (built, src) {
        (None, _) => true,
        (Some(b), Some(s)) => s > b,
        _ => false,
    };

    if stale && std::env::var_os("SENTINEL_SKIP_UI").is_none() && ui.join("package.json").exists() {
        let npm = if cfg!(windows) { "npm.cmd" } else { "npm" };
        let ok = |args: &[&str]| Command::new(npm).args(args).current_dir(&ui).status().map(|s| s.success()).unwrap_or(false);
        let installed = ui.join("node_modules").exists() || ok(&["ci"]);
        if installed && ok(&["run", "build"]) {
            println!("cargo:warning=built ui/dist with npm");
        } else {
            println!("cargo:warning=could not build the React UI (is Node installed?); see ui/README.md");
        }
    }

    if !index.exists() {
        std::fs::create_dir_all(&dist).expect("create ui/dist");
        std::fs::write(
            &index,
            "<!doctype html><meta charset=utf-8><title>Sentinel</title>\
             <body style=\"font-family:system-ui;background:#000;color:#ededed;padding:48px\">\
             <h1>Sentinel dashboard not built</h1>\
             <p>Run <code>cd ui &amp;&amp; npm ci &amp;&amp; npm run build</code>, then rebuild Sentinel. \
             The REST API at <code>/api</code> is available.</p>",
        )
        .expect("write placeholder");
    }
}
