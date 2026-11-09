//! The Python SDK is embedded in the binary and materialised into the project on demand, so generated
//! training scripts can `import sentinel` with no `pip install` and no path configuration.

use std::path::{Path, PathBuf};

pub const SDK_SOURCE: &str = include_str!("../../../packages/sentinel-sdk/src/sentinel/__init__.py");
pub const XAI_SOURCE: &str = include_str!("../../../packages/sentinel-sdk/src/sentinel/xai.py");
pub const HOOKS_SOURCE: &str = include_str!("../../../packages/sentinel-sdk/src/sentinel/hooks.py");

/// Write `.sentinel/sdk/sentinel/__init__.py`, `xai.py`, and `hooks.py` (only if changed) and return the directory to put on `PYTHONPATH`.
pub fn ensure_sdk(root: &Path) -> std::io::Result<PathBuf> {
    let sdk_dir = root.join(".sentinel").join("sdk");
    let pkg = sdk_dir.join("sentinel");
    let file = pkg.join("__init__.py");
    let xai_file = pkg.join("xai.py");
    let hooks_file = pkg.join("hooks.py");
    if std::fs::read_to_string(&file).map_or(true, |cur| cur != SDK_SOURCE) {
        std::fs::create_dir_all(&pkg)?;
        std::fs::write(&file, SDK_SOURCE)?;
    }
    if std::fs::read_to_string(&xai_file).map_or(true, |cur| cur != XAI_SOURCE) {
        std::fs::create_dir_all(&pkg)?;
        std::fs::write(&xai_file, XAI_SOURCE)?;
    }
    if std::fs::read_to_string(&hooks_file).map_or(true, |cur| cur != HOOKS_SOURCE) {
        std::fs::create_dir_all(&pkg)?;
        std::fs::write(&hooks_file, HOOKS_SOURCE)?;
    }
    Ok(sdk_dir)
}

/// `PYTHONPATH` value: an explicit `SENTINEL_SDK_PATH` override (for SDK development), then the embedded SDK,
/// then whatever the user already had.
pub fn python_path(root: &Path) -> std::io::Result<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("SENTINEL_SDK_PATH") {
        parts.push(p);
    }
    parts.push(ensure_sdk(root)?.display().to_string());
    if let Ok(p) = std::env::var("PYTHONPATH") {
        if !p.is_empty() {
            parts.push(p);
        }
    }
    Ok(parts.join(if cfg!(windows) { ";" } else { ":" }))
}

/// Write a generated script under `.sentinel/generated/` with a unique name (concurrent runs must not clash).
pub fn write_generated_script(root: &Path, id: &str, unique: &str, code: &str) -> std::io::Result<PathBuf> {
    let dir = root.join(".sentinel").join("generated");
    std::fs::create_dir_all(&dir)?;
    let stem: String = id.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let path = dir.join(format!("{stem}-{unique}.py"));
    std::fs::write(&path, code)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_a_usable_sdk_and_rewrites_only_on_change() {
        let root = std::env::temp_dir().join(format!("sentinel-sdk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = ensure_sdk(&root).unwrap();
        let f = dir.join("sentinel/__init__.py");
        let h = dir.join("sentinel/hooks.py");
        assert!(std::fs::read_to_string(&f).unwrap().contains("def init("));
        assert!(std::fs::read_to_string(&h).unwrap().contains("class HookRegistry"));
        std::fs::write(&f, "# stale").unwrap();
        ensure_sdk(&root).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), SDK_SOURCE);
        assert_eq!(std::fs::read_to_string(&h).unwrap(), HOOKS_SOURCE);
        assert!(python_path(&root).unwrap().contains(".sentinel/sdk"));
    }

    #[test]
    fn script_names_are_unique_and_safe() {
        let root = std::env::temp_dir().join(format!("sentinel-gen-{}", std::process::id()));
        let a = write_generated_script(&root, "tabular/ann@v1", "aaaa", "x").unwrap();
        let b = write_generated_script(&root, "tabular/ann@v1", "bbbb", "y").unwrap();
        assert_ne!(a, b);
        assert!(a.file_name().unwrap().to_str().unwrap().starts_with("tabular_ann_v1-"));
    }
}
