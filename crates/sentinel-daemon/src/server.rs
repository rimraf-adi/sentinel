//! Axum HTTP & WebSocket Server for Sentinel Daemon.

use crate::scheduler::{DynamicGpuPool, HardwareMonitor};
use crate::supervisor::ProcessSupervisor;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use sentinel_core::models::{HardwareStats, JobRecord, JobSpec};
use sentinel_core::templates::TemplateRegistry;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub supervisor: ProcessSupervisor,
    pub gpu_pool: DynamicGpuPool,
    pub hardware_monitor: Arc<Mutex<HardwareMonitor>>,
    pub templates: Arc<TemplateRegistry>,
}

pub struct Server;

impl Server {
    /// Create router with all Sentinel Daemon REST & WebSocket endpoints.
    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/api/health", get(health_check))
            .route("/api/jobs", get(list_jobs).post(submit_job))
            .route("/api/jobs/:id", get(get_job))
            .route("/api/jobs/:id/stop", post(stop_job))
            .route("/api/jobs/:id/logs", get(get_job_logs))
            .route("/api/hardware", get(get_hardware))
            .route("/api/templates", get(list_templates))
            .route("/ws/live", get(ws_handler))
            .layer(CorsLayer::permissive())
            .with_state(state)
    }

    /// Server-side downsampling for metric time series (LTTB / nth-point downsampling).
    pub fn downsample_series(points: &[(u64, f64)], max_points: usize) -> Vec<(u64, f64)> {
        if points.len() <= max_points || max_points == 0 {
            return points.to_vec();
        }

        let step = (points.len() as f64) / (max_points as f64);
        let mut downsampled = Vec::with_capacity(max_points);

        for i in 0..max_points {
            let idx = (i as f64 * step).floor() as usize;
            if idx < points.len() {
                downsampled.push(points[idx]);
            }
        }

        // Always include the last point
        if let Some(&last) = points.last() {
            if downsampled.last() != Some(&last) {
                downsampled.push(last);
            }
        }

        downsampled
    }
}

async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "service": "sentinel-daemon",
        "version": "0.1.0"
    }))
}

async fn list_jobs(State(state): State<AppState>) -> impl IntoResponse {
    let jobs = state.supervisor.list_jobs().await;
    Json(jobs)
}

async fn submit_job(
    State(state): State<AppState>,
    Json(spec): Json<JobSpec>,
) -> Result<Json<JobRecord>, (StatusCode, String)> {
    let gpus = spec.gpu_ids.len() as u32;
    // Check if GPUs need allocation
    let allocated_gpus = state.gpu_pool.allocate(gpus).await.ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Insufficient GPU resources available in pool".to_string(),
        )
    })?;

    let mut final_spec = spec;
    final_spec.gpu_ids = allocated_gpus;

    match state.supervisor.start_job(final_spec).await {
        Ok(record) => Ok(Json(record)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

async fn get_job(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<Json<JobRecord>, StatusCode> {
    state
        .supervisor
        .get_job(id)
        .await
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn stop_job(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    state
        .supervisor
        .stop_job(id)
        .await
        .map(|_| Json(json!({"status": "cancelled", "job_id": id})))
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))
}

async fn get_job_logs(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> impl IntoResponse {
    let logs = state.supervisor.get_logs(id).await;
    Json(json!({ "job_id": id, "logs": logs }))
}

async fn get_hardware(State(state): State<AppState>) -> Json<HardwareStats> {
    let stats = state.hardware_monitor.lock().await.collect_stats("local");
    Json(stats)
}

async fn list_templates(State(state): State<AppState>) -> impl IntoResponse {
    let templates: Vec<_> = state.templates.list().into_iter().cloned().collect();
    Json(templates)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let mut interval = tokio::time::interval(Duration::from_millis(500));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let stats = state.hardware_monitor.lock().await.collect_stats("local");
                let msg = json!({
                    "type": "telemetry",
                    "data": stats
                });
                if socket.send(Message::Text(msg.to_string())).await.is_err() {
                    break;
                }
            }
            msg = socket.recv() => {
                if msg.is_none() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_app_state() -> AppState {
        AppState {
            supervisor: ProcessSupervisor::new(),
            gpu_pool: DynamicGpuPool::new(2),
            hardware_monitor: Arc::new(Mutex::new(HardwareMonitor::new())),
            templates: Arc::new(TemplateRegistry::new()),
        }
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = test_app_state();
        let app = Server::router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_hardware_endpoint() {
        let state = test_app_state();
        let app = Server::router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/hardware")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_downsampling_utility() {
        let raw: Vec<(u64, f64)> = (0..1000).map(|i| (i, (i as f64).sin())).collect();
        let downsampled = Server::downsample_series(&raw, 100);

        assert!(downsampled.len() <= 101);
        assert_eq!(downsampled.first().unwrap().0, 0);
        assert_eq!(downsampled.last().unwrap().0, 999);
    }
}
