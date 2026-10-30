//! UI state models and transitions.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DashboardMode {
    #[default]
    ReadOnly,
    Editing,
    Idle,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UiState {
    pub mode: DashboardMode,
    pub active_tab: String,
    pub is_connected: bool,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            mode: DashboardMode::ReadOnly,
            active_tab: "overview".to_string(),
            is_connected: false,
        }
    }

    pub fn enter_edit(&mut self) {
        self.mode = DashboardMode::Editing;
    }

    pub fn enter_idle(&mut self) {
        self.mode = DashboardMode::Idle;
    }

    pub fn reset_read_only(&mut self) {
        self.mode = DashboardMode::ReadOnly;
    }
}
