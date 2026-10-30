//! Workspace detection for the dashboard header and `/api/repo/status`.

use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceInfo {
    pub is_sentinel_repo: bool,
    pub workspace_name: String,
    pub workspace_path: String,
    pub full_workspace_path: String,
    pub git_branch: String,
    pub git_clean: bool,
}

impl WorkspaceInfo {
    pub fn detect() -> Self {
        let cwd = std::env::current_dir().unwrap_or_default();
        let dir = cwd.canonicalize().unwrap_or(cwd);
        let (git_branch, git_clean) = detect_git(&dir);
        Self {
            is_sentinel_repo: dir.join("sentinel.lock").exists(),
            workspace_name: project_name(&dir),
            workspace_path: shorten_path(&dir),
            full_workspace_path: dir.to_string_lossy().to_string(),
            git_branch,
            git_clean,
        }
    }
}

/// `project_name` from sentinel.yaml, else the directory name.
fn project_name(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("sentinel.yaml"))
        .ok()
        .and_then(|c| {
            c.lines().find_map(|l| {
                l.trim()
                    .strip_prefix("project_name:")
                    .map(|v| v.trim().trim_matches(|c| c == '"' || c == '\'').to_string())
            })
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            dir.file_name().and_then(|n| n.to_str()).unwrap_or("workspace").to_string()
        })
}

pub fn shorten_path(path: &Path) -> String {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        if let Ok(suffix) = path.strip_prefix(&home) {
            return format!("~/{}", suffix.display());
        }
    }
    path.to_string_lossy().to_string()
}

fn detect_git(dir: &Path) -> (String, bool) {
    let branch = std::fs::read_to_string(dir.join(".git/HEAD"))
        .ok()
        .and_then(|c| {
            let t = c.trim();
            t.strip_prefix("ref: refs/heads/")
                .map(str::to_string)
                .or_else(|| t.get(..7).map(str::to_string))
        })
        .unwrap_or_else(|| "main".to_string());
    let clean = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(dir)
        .output()
        .map(|o| o.stdout.is_empty())
        .unwrap_or(true);
    (branch, clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortens_home_paths() {
        std::env::set_var("HOME", "/home/u");
        assert_eq!(shorten_path(Path::new("/home/u/proj")), "~/proj");
        assert_eq!(shorten_path(Path::new("/opt/x")), "/opt/x");
    }
}
