//! Sentinel Web UI: Pure Rust WASM & HTML dashboard interface.

pub mod state;
pub mod view;

pub use state::{DashboardMode, DashboardState};
pub use view::DashboardView;
