//! Sentinel Web UI: Pure Rust Leptos / Dioxus WASM interface.

pub mod state;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiMode {
    ReadOnly,
    Editing,
    Idle,
}
