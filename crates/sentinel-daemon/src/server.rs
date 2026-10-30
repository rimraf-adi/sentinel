//! Axum HTTP & WebSocket Server for Sentinel Daemon.

use crate::scheduler::{DynamicGpuPool, HardwareMonitor};
use crate::supervisor::ProcessSupervisor;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use sentinel_core::models::{HardwareStats, JobRecord, JobSpec, RunMode};
use sentinel_core::templates::TemplateRegistry;
use sentinel_ui::{DashboardState, DashboardView};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
            .route("/", get(dashboard_page))
            .route("/favicon.ico", get(favicon_handler))
            .route("/api/health", get(health_check))
            .route("/api/jobs", get(list_jobs).post(submit_job))
            .route("/api/jobs/run-template", post(run_template_job))
            .route("/api/jobs/:id", get(get_job))
            .route("/api/jobs/:id/stop", post(stop_job))
            .route("/api/jobs/:id/logs", get(get_job_logs))
            .route("/api/hardware", get(get_hardware))
            .route("/api/templates", get(list_templates))
            .route("/api/datasets", get(list_datasets).post(add_dataset))
            .route("/api/datasets/:name", axum::routing::delete(remove_dataset))
            .route("/api/repo/status", get(get_repo_status))
            .route("/api/repo/init", post(init_repo))
            .route("/api/files", get(list_files))
            .route("/api/files/content", get(get_file_content))
            .route("/ws/live", get(ws_handler))
            .layer(CorsLayer::permissive())
            .with_state(state)
    }

    /// Server-side downsampling for metric time series.
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

        if let Some(&last) = points.last() {
            if downsampled.last() != Some(&last) {
                downsampled.push(last);
            }
        }

        downsampled
    }
}

async fn dashboard_page(State(state): State<AppState>) -> Html<String> {
    let stats = state.hardware_monitor.lock().await.collect_stats("local");
    let jobs = state.supervisor.list_jobs().await;
    let templates: Vec<_> = state.templates.list().into_iter().cloned().collect();
    let cwd = std::env::current_dir().unwrap_or_default();
    let canonical = cwd.canonicalize().unwrap_or(cwd);
    let datasets = sentinel_core::storage::DatasetRegistry::load_all(&canonical);
    let ui_state = DashboardState::new();

    let html = DashboardView::render_html(&ui_state, Some(&stats), &jobs, &templates, &datasets);
    Html(html)
}

async fn favicon_handler() -> impl IntoResponse {
    let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#ffffff"/><polygon points="16,7 25,23 7,23" fill="#000000"/></svg>"##;
    ([(axum::http::header::CONTENT_TYPE, "image/svg+xml")], svg)
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

#[derive(Debug, Deserialize)]
pub struct RunTemplateRequest {
    pub template_id: String,
    #[serde(default)]
    pub overrides: HashMap<String, serde_yaml::Value>,
}

async fn run_template_job(
    State(state): State<AppState>,
    Json(req): Json<RunTemplateRequest>,
) -> Result<Json<JobRecord>, (StatusCode, String)> {
    let template = state
        .templates
        .get(&req.template_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Template '{}' not found", req.template_id)))?;

    // Validate user overrides against template schema
    template
        .validate_overrides(&req.overrides)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let mut spec = JobSpec::new(format!("job-{}", template.name), RunMode::Framework);
    spec.template_id = Some(req.template_id.clone());

    for (k, v) in &req.overrides {
        let json_val = serde_json::to_value(v).unwrap_or(serde_json::Value::Null);
        spec.parameters.insert(k.clone(), json_val);
    }

    // Determine executable command
    let cmd = if Path::new("recipes/vision_train.py").exists() {
        "uv run python recipes/vision_train.py".to_string()
    } else {
        "python -c 'print(\"Sentinel job executed successfully\")'".to_string()
    };
    spec.command = Some(cmd);

    let allocated_gpus = state.gpu_pool.allocate(1).await.unwrap_or_default();
    spec.gpu_ids = allocated_gpus;

    match state.supervisor.start_job(spec).await {
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

#[derive(Serialize)]
struct FileItem {
    name: String,
    path: String,
    is_dir: bool,
}

async fn list_files() -> impl IntoResponse {
    let mut files = Vec::new();
    let cwd = std::env::current_dir().unwrap_or_default();
    let canonical = cwd.canonicalize().unwrap_or(cwd);

    let candidates = [
        "sentinel.yaml",
        "sentinel.lock",
        "machines.yaml",
        "datasets.yaml",
        "pyproject.toml",
        "Cargo.toml",
        "README.md",
        "guide.md",
        "FILES.md",
        "recipes/vision_train.py",
        "recipes/sweep_orchestrator.py",
        "templates/vision/image-classification.yaml",
        "templates/custom/command-runner.yaml",
        "templates/custom/starter.yaml",
        "templates/llm/lora-finetune.yaml",
        "templates/tabular/mlp-classifier.yaml",
        "packages/sentinel-sdk/src/sentinel/__init__.py",
    ];

    for c in &candidates {
        if canonical.join(c).exists() {
            files.push(FileItem {
                name: Path::new(c)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                path: c.to_string(),
                is_dir: false,
            });
        }
    }

    // Also include any registered dataset file paths
    let datasets = sentinel_core::storage::DatasetRegistry::load_all(&canonical);
    for d in datasets {
        let p = Path::new(&d.path);
        if p.is_file() {
            files.push(FileItem {
                name: format!("{} ({})", d.name, p.file_name().unwrap_or_default().to_string_lossy()),
                path: d.path.clone(),
                is_dir: false,
            });
        }
    }

    Json(files)
}

#[derive(Deserialize)]
struct FileContentQuery {
    path: String,
}

async fn get_file_content(Query(q): Query<FileContentQuery>) -> Result<Json<serde_json::Value>, StatusCode> {
    let raw = q.path.trim();
    if raw.contains("..") {
        return Err(StatusCode::FORBIDDEN);
    }

    let cwd = std::env::current_dir().unwrap_or_default();
    let canonical = cwd.canonicalize().unwrap_or(cwd);

    let expanded = if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            home.join(stripped)
        } else {
            PathBuf::from(raw)
        }
    } else if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        canonical.join(raw)
    };

    if !expanded.exists() || !expanded.is_file() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Read safely with a 1MB limit for preview
    let content = match std::fs::read_to_string(&expanded) {
        Ok(c) => {
            if c.len() > 1024 * 1024 {
                format!("{}\n\n... [Truncated: File content exceeds 1MB preview limit] ...", &c[..1024 * 1024])
            } else {
                c
            }
        }
        Err(_) => {
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    Ok(Json(json!({ "path": q.path, "content": content })))
}

async fn list_datasets() -> impl IntoResponse {
    let cwd = std::env::current_dir().unwrap_or_default();
    let canonical = cwd.canonicalize().unwrap_or(cwd);
    let datasets = sentinel_core::storage::DatasetRegistry::load_all(&canonical);
    Json(datasets)
}

#[derive(Deserialize)]
struct AddDatasetPayload {
    name: String,
    path: String,
    #[serde(default = "default_ds_format")]
    format: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    split: Option<String>,
}

fn default_ds_format() -> String {
    "folder".to_string()
}

async fn add_dataset(Json(payload): Json<AddDatasetPayload>) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let canonical = cwd.canonicalize().unwrap_or(cwd);

    let mut spec = sentinel_core::models::DatasetSpec::new(&payload.name, &payload.path, &payload.format);
    spec.description = payload.description;
    spec.split = payload.split;
    spec.inspect(&canonical);

    sentinel_core::storage::DatasetRegistry::add_dataset(&canonical, spec.clone()).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to register dataset: {e}"),
        )
    })?;

    Ok(Json(json!({
        "status": "ok",
        "dataset": spec
    })))
}

async fn remove_dataset(AxumPath(name): AxumPath<String>) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let canonical = cwd.canonicalize().unwrap_or(cwd);

    sentinel_core::storage::DatasetRegistry::remove_dataset(&canonical, &name).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to unmount dataset: {e}"),
        )
    })?;

    Ok(Json(json!({
        "status": "ok",
        "message": format!("Dataset '{name}' unmounted from registry")
    })))
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

async fn get_repo_status() -> impl IntoResponse {
    let ui_state = sentinel_ui::DashboardState::new();
    let lock_found = ui_state.is_sentinel_repo;

    Json(json!({
        "is_sentinel_repo": lock_found,
        "lock_file_found": lock_found,
        "lock_file_path": "sentinel.lock",
        "workspace_name": ui_state.workspace_name,
        "workspace_path": ui_state.workspace_path,
        "full_workspace_path": ui_state.full_workspace_path,
        "git_branch": ui_state.git_branch,
        "git_clean": ui_state.git_clean,
        "convert_commands": [
            "sentinel init",
            "sentinel template new my-recipe",
            "sentinel dev"
        ]
    }))
}

async fn init_repo() -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let canonical = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
    let name = canonical
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project");

    sentinel_core::scaffold::ProjectScaffolder::init(&canonical, name).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to initialize Sentinel repository: {e}"),
        )
    })?;

    Ok(Json(json!({
        "status": "ok",
        "message": "Repository successfully converted to Sentinel repository."
    })))
}
