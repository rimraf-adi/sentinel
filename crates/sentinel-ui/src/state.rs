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
        Self {
            mode: DashboardMode::ReadOnly,
            active_tab: "overview".to_string(),
            is_connected: false,
            idle_timeout_secs: 15,
            last_interaction: Some(Instant::now()),
        }
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
