//! UI state models, idle detector, and state transitions.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DashboardMode {
    #[default]
    ReadOnly,
    Editing,
    Idle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    pub mode: DashboardMode,
    pub active_tab: String,
    pub is_connected: bool,
    pub is_sentinel_repo: bool,
    pub workspace_name: String,
    pub workspace_path: String,
    pub full_workspace_path: String,
    pub git_branch: String,
    pub git_clean: bool,
    pub idle_timeout_secs: u64,
    #[serde(skip)]
    pub last_interaction: Option<Instant>,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self::new()
    }
}

impl DashboardState {
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_default();
        let canonical = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());

        let is_sentinel = canonical.join("sentinel.lock").exists()
            || std::path::Path::new("sentinel.lock").exists();

        // 1. Resolve project name: check sentinel.yaml first, then fallback to directory name
        let name = if let Ok(content) = std::fs::read_to_string(canonical.join("sentinel.yaml")) {
            content
                .lines()
                .find_map(|line| {
                    let trimmed = line.trim();
                    if trimmed.starts_with("project_name:") {
                        let val = trimmed.trim_start_matches("project_name:").trim();
                        Some(val.trim_matches('"').trim_matches('\'').to_string())
                    } else {
                        None
                    }
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| {
                    canonical
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("workspace")
                        .to_string()
                })
        } else {
            canonical
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("workspace")
                .to_string()
        };

        // 2. Shorten path with ~
        let short_path = shorten_path(&canonical);
        let full_path = canonical.to_string_lossy().to_string();

        // 3. Detect git branch & status
        let (git_branch, git_clean) = detect_git(&canonical);

        Self {
            mode: DashboardMode::ReadOnly,
            active_tab: "overview".to_string(),
            is_connected: false,
            is_sentinel_repo: is_sentinel,
            workspace_name: name,
            workspace_path: short_path,
            full_workspace_path: full_path,
            git_branch,
            git_clean,
            idle_timeout_secs: 15,
            last_interaction: Some(Instant::now()),
        }
    }

    pub fn with_sentinel_repo(mut self, is_sentinel: bool) -> Self {
        self.is_sentinel_repo = is_sentinel;
        self
    }

    /// User interacted (clicked edit, entered input).
    pub fn on_user_interaction(&mut self) {
        self.last_interaction = Some(Instant::now());
        if self.mode == DashboardMode::Idle {
            self.mode = DashboardMode::ReadOnly;
        }
    }

    /// Explicitly enter edit mode.
    pub fn enter_edit(&mut self) {
        self.mode = DashboardMode::Editing;
        self.last_interaction = Some(Instant::now());
    }

    /// Revert back to read-only.
    pub fn exit_edit(&mut self) {
        self.mode = DashboardMode::ReadOnly;
        self.last_interaction = Some(Instant::now());
    }

    /// Switch active dashboard view tab.
    pub fn switch_tab(&mut self, tab: impl Into<String>) {
        self.active_tab = tab.into();
        self.on_user_interaction();
    }

    /// Check if idle timeout has expired. If so, drop down to Idle state.
    pub fn check_idle_timeout(&mut self, now: Instant) -> bool {
        if let Some(last) = self.last_interaction {
            if now.duration_since(last) >= Duration::from_secs(self.idle_timeout_secs) {
                if self.mode != DashboardMode::Idle {
                    self.mode = DashboardMode::Idle;
                    return true;
                }
            }
        }
        false
    }

    /// Polling interval dynamically slows down during idle to save CPU & RAM.
    pub fn get_telemetry_polling_interval(&self) -> Duration {
        match self.mode {
            DashboardMode::Editing => Duration::from_millis(500),
            DashboardMode::ReadOnly => Duration::from_millis(1000),
            DashboardMode::Idle => Duration::from_millis(5000), // Sleepy mode: 5s
        }
    }
}

/// Shorten an absolute path using ~ if inside home directory
pub fn shorten_path(path: &std::path::Path) -> String {
    if let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) {
        if let Ok(suffix) = path.strip_prefix(&home) {
            return format!("~/{}", suffix.display());
        }
    }
    path.to_string_lossy().to_string()
}

/// Dynamically detect current git branch and working directory clean status
pub fn detect_git(cwd: &std::path::Path) -> (String, bool) {
    let head_path = cwd.join(".git/HEAD");
    let branch = if head_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&head_path) {
            let trimmed = content.trim();
            if let Some(b) = trimmed.strip_prefix("ref: refs/heads/") {
                b.to_string()
            } else if trimmed.len() >= 7 {
                trimmed[..7].to_string()
            } else {
                "main".to_string()
            }
        } else {
            "main".to_string()
        }
    } else {
        "main".to_string()
    };

    let clean = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(cwd)
        .output()
        .map(|o| o.stdout.is_empty())
        .unwrap_or(true);

    (branch, clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_transitions() {
        let mut state = DashboardState::new();
        assert_eq!(state.mode, DashboardMode::ReadOnly);
        assert_eq!(state.get_telemetry_polling_interval(), Duration::from_millis(1000));

        // Enter edit
        state.enter_edit();
        assert_eq!(state.mode, DashboardMode::Editing);
        assert_eq!(state.get_telemetry_polling_interval(), Duration::from_millis(500));

        // Revert to read-only
        state.exit_edit();
        assert_eq!(state.mode, DashboardMode::ReadOnly);
    }

    #[test]
    fn test_idle_timeout_drops_to_idle() {
        let mut state = DashboardState::new();
        state.idle_timeout_secs = 2;

        let start = Instant::now();
        state.last_interaction = Some(start);

        // Before timeout
        assert!(!state.check_idle_timeout(start + Duration::from_secs(1)));
        assert_eq!(state.mode, DashboardMode::ReadOnly);

        // After timeout
        assert!(state.check_idle_timeout(start + Duration::from_secs(3)));
        assert_eq!(state.mode, DashboardMode::Idle);
        assert_eq!(state.get_telemetry_polling_interval(), Duration::from_millis(5000));

        // User interacts -> wakes up to read-only
        state.on_user_interaction();
        assert_eq!(state.mode, DashboardMode::ReadOnly);
    }
}
