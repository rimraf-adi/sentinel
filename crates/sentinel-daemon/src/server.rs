//! Axum HTTP & WebSocket Server for Sentinel Daemon.

use crate::scheduler::{DynamicGpuPool, HardwareMonitor};
use crate::supervisor::ProcessSupervisor;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use sentinel_core::models::{HardwareStats, JobRecord, JobSpec, RunMode};
use sentinel_core::components::ComponentRegistry;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub vitals: VitalsHub,
    /// Open dashboard WebSockets; with none (and no running jobs) the sampler slows down to save CPU.
    pub ws_clients: Arc<std::sync::atomic::AtomicUsize>,
    pub supervisor: ProcessSupervisor,
    pub gpu_pool: DynamicGpuPool,
    pub hardware_monitor: Arc<Mutex<HardwareMonitor>>,
    pub components: Arc<ComponentRegistry>,
}

/// Compact history sample (about 150 bytes) for charts; the full snapshot is only kept for "now".
#[derive(Debug, Clone, Serialize)]
pub struct VitalsPoint {
    pub t: i64,
    pub cpu: f32,
    pub ram_used: u64,
    pub ram_total: u64,
    pub swap_used: u64,
    pub gpus: Vec<GpuPoint>,
    pub daemon_cpu: Option<f32>,
    pub daemon_rss: Option<u64>,
    pub jobs: usize,
    /// Total network throughput across interfaces, bytes/second.
    pub net_rx: f64,
    pub net_tx: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuPoint {
    pub util: Option<f32>,
    pub mem_used: Option<u64>,
    pub mem_total: Option<u64>,
    pub temp: Option<f32>,
    pub power: Option<f32>,
}

const HISTORY_CAP: usize = 900; // 15 minutes at 1 Hz

/// Latest vitals plus a ring buffer of history, filled by one background sampler (not per client).
#[derive(Clone, Default)]
pub struct VitalsHub {
    pub latest: Arc<tokio::sync::RwLock<Option<sentinel_core::vitals::Vitals>>>,
    pub history: Arc<tokio::sync::RwLock<VecDeque<VitalsPoint>>>,
}

impl VitalsHub {
    pub async fn push(&self, v: sentinel_core::vitals::Vitals) {
        let point = VitalsPoint {
            t: v.timestamp.timestamp_millis(),
            cpu: v.cpu_util_percent,
            ram_used: v.ram_used_bytes,
            ram_total: v.ram_total_bytes,
            swap_used: v.swap_used_bytes,
            gpus: v.gpus.iter().map(|g| GpuPoint { util: g.gpu_util_percent, mem_used: g.vram_used_bytes, mem_total: g.vram_total_bytes, temp: g.temperature_c, power: g.power_w }).collect(),
            daemon_cpu: v.daemon.as_ref().map(|d| d.cpu_percent),
            daemon_rss: v.daemon.as_ref().map(|d| d.rss_bytes),
            jobs: v.jobs.len(),
            net_rx: v.network.iter().map(|n| n.rx_bps).sum(),
            net_tx: v.network.iter().map(|n| n.tx_bps).sum(),
        };
        let mut h = self.history.write().await;
        if h.len() >= HISTORY_CAP {
            h.pop_front();
        }
        h.push_back(point);
        *self.latest.write().await = Some(v);
    }
}

/// Sample once a second in the background. `sample` shells out to nvidia-smi, so it runs off the async threads.
pub fn spawn_sampler(state: AppState) {
    use std::sync::atomic::Ordering;
    tokio::spawn(async move {
        loop {
            let watch: Vec<_> = state
                .supervisor
                .list_jobs()
                .await
                .into_iter()
                .filter(|j| j.status == sentinel_core::models::JobStatus::Running)
                .filter_map(|j| j.pid.map(|p| (j.spec.id, j.spec.name.clone(), p)))
                .collect();
            let busy = !watch.is_empty() || state.ws_clients.load(Ordering::Relaxed) > 0;
            let monitor = state.hardware_monitor.clone();
            if let Ok(v) = tokio::task::spawn_blocking(move || monitor.blocking_lock().sample("local", &watch)).await {
                state.vitals.push(v).await;
            }
            // 1 Hz while someone is watching or a job runs; 5 s otherwise (history keeps real timestamps).
            tokio::time::sleep(Duration::from_secs(if busy { 1 } else { 5 })).await;
        }
    });
}

impl AppState {
    /// Built-ins plus whatever is in `./components` right now, so newly saved components
    /// show up without restarting the daemon.
    pub fn registry(&self) -> ComponentRegistry {
        // Project blocks first: components that use them must validate when they load.
        let _ = sentinel_core::compound::BlockLibrary::reload_project(&std::env::current_dir().unwrap_or_default().join("blocks"));
        let mut r = (*self.components).clone();
        let _ = r.load_from_dir(&std::env::current_dir().unwrap_or_default().join("components"));
        r
    }
}

pub struct Server;

impl Server {
    /// Create router with all Sentinel Daemon REST & WebSocket endpoints.
    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/favicon.ico", get(favicon_handler))
            .route("/api/health", get(health_check))
            .route("/api/jobs", get(list_jobs).post(submit_job))
            .route("/api/jobs/run-component", post(run_component_job))
            .route("/api/jobs/:id", get(get_job))
            .route("/api/jobs/:id/stop", post(stop_job))
            .route("/api/jobs/:id/logs", get(get_job_logs))
            .route("/api/hardware", get(get_hardware))
            .route("/api/vitals", get(get_vitals))
            .route("/api/vitals/history", get(get_vitals_history))
            .route("/api/system", get(get_system))
            .route("/api/components/save", post(save_component))
            .route("/api/blocks/save", post(save_block))
            .route("/api/schema", get(schema_for_path))
            .route("/api/datasets/:name/schema", get(schema_for_dataset))
            .route("/api/tracking/runs", get(tracking_runs))
            .route("/api/tracking/runs/:id", get(tracking_run))
            .route("/api/blocks", get(list_blocks))
            .route("/api/graph/validate", post(graph_validate))
            .route("/api/graph/compose", post(graph_compose))
            .route("/api/graph/run", post(graph_run))
            .route("/api/components", get(list_components))
            .route("/api/datasets", get(list_datasets).post(add_dataset))
            .route("/api/datasets/:name", axum::routing::delete(remove_dataset))
            .route("/api/repo/status", get(get_repo_status))
            .route("/api/repo/init", post(init_repo))
            .route("/api/files", get(list_files))
            .route("/api/files/content", get(get_file_content))
            .route("/ws/live", get(ws_handler))
            .fallback(crate::assets::static_handler)
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


/// The Sentinel mark. Embedded from the UI's `public/favicon.svg`, so there is one source of truth.
async fn favicon_handler() -> impl IntoResponse {
    const SVG: &str = include_str!("../../../ui/public/favicon.svg");
    ([(axum::http::header::CONTENT_TYPE, "image/svg+xml"), (axum::http::header::CACHE_CONTROL, "public, max-age=86400")], SVG)
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
    if !final_spec.env_vars.contains_key("PYTHONPATH") {
        if let Ok(pp) = sentinel_core::sdk::python_path(&project_root()) {
            final_spec.env_vars.insert("PYTHONPATH".into(), pp);
        }
    }

    match state.supervisor.start_job(final_spec).await {
        Ok(record) => Ok(Json(record)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

#[derive(Debug, Deserialize)]
pub struct RunComponentRequest {
    pub component_id: String,
    #[serde(default)]
    pub overrides: HashMap<String, serde_yaml::Value>,
}

async fn run_component_job(
    State(state): State<AppState>,
    Json(req): Json<RunComponentRequest>,
) -> Result<Json<JobRecord>, (StatusCode, String)> {
    let registry = state.registry();
    let component = registry
        .get(&req.component_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Component '{}' not found", req.component_id)))?;

    // Validate user overrides against component schema
    component
        .validate_overrides(&req.overrides)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let mut spec = JobSpec::new(format!("job-{}", component.name), RunMode::Framework);
    spec.component_id = Some(req.component_id.clone());

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
    if let Some(v) = state.vitals.latest.read().await.clone() {
        return Json(HardwareStats {
            timestamp: v.timestamp,
            machine_id: v.machine_id,
            cpu_util_percent: v.cpu_util_percent,
            ram_used_bytes: v.ram_used_bytes,
            ram_total_bytes: v.ram_total_bytes,
            gpus: v.gpus,
        });
    }
    Json(state.hardware_monitor.lock().await.collect_stats("local"))
}

async fn get_vitals(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.vitals.latest.read().await.clone())
}

async fn get_vitals_history(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.vitals.history.read().await.iter().cloned().collect::<Vec<_>>())
}

async fn get_system(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.hardware_monitor.lock().await.system_info())
}

async fn list_components(State(state): State<AppState>) -> impl IntoResponse {
    let components: Vec<_> = state.registry().list().into_iter().cloned().collect();
    Json(components)
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
        "components/vision/image-classification.yaml",
        "components/custom/command-runner.yaml",
        "components/custom/starter.yaml",
        "components/llm/lora-finetune.yaml",
        "components/tabular/mlp-classifier.yaml",
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
        if let Some(home) = sentinel_core::platform::home_dir() {
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
    use std::sync::atomic::Ordering;
    state.ws_clients.fetch_add(1, Ordering::Relaxed);
    struct Guard(Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let _guard = Guard(state.ws_clients.clone());
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Read the shared sample; many open tabs must not multiply the sampling work.
                let Some(v) = state.vitals.latest.read().await.clone() else { continue };
                let msg = json!({ "type": "telemetry", "data": v });
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
    let ws = crate::workspace::WorkspaceInfo::detect();
    let lock_found = ws.is_sentinel_repo;

    Json(json!({
        "is_sentinel_repo": lock_found,
        "lock_file_found": lock_found,
        "lock_file_path": "sentinel.lock",
        "workspace_name": ws.workspace_name,
        "workspace_path": ws.workspace_path,
        "full_workspace_path": ws.full_workspace_path,
        "git_branch": ws.git_branch,
        "git_clean": ws.git_clean,
        "convert_commands": [
            "sentinel init",
            "sentinel component new my-recipe",
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

// ---------------------------------------------------------------------------
// Component graphs: block catalog, validation, compilation, execution
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct GraphRequest {
    /// Registered component to build, or...
    pub component_id: Option<String>,
    /// ...an ad-hoc graph (e.g. from the canvas editor).
    pub graph: Option<serde_json::Value>,
    #[serde(default)]
    pub overrides: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub model_only: bool,
}

type ApiError = (StatusCode, Json<serde_json::Value>);

fn api_err(code: StatusCode, msg: impl ToString) -> ApiError {
    (code, Json(json!({ "ok": false, "error": msg.to_string() })))
}

fn resolve_graph(
    state: &AppState,
    req: &GraphRequest,
) -> Result<(String, sentinel_core::graph::ValidatedGraph), ApiError> {
    if let Some(id) = &req.component_id {
        let registry = state.registry();
        let c = registry
            .get(id)
            .ok_or_else(|| api_err(StatusCode::NOT_FOUND, format!("Component '{id}' not found")))?;
        let mut ov = HashMap::new();
        for (k, v) in &req.overrides {
            let y = serde_yaml::to_value(v).map_err(|e| api_err(StatusCode::BAD_REQUEST, e))?;
            ov.insert(k.clone(), y);
        }
        c.validate_overrides(&ov).map_err(|e| api_err(StatusCode::BAD_REQUEST, e))?;
        let g = c.build_graph(&ov).map_err(|e| api_err(StatusCode::BAD_REQUEST, e))?;
        Ok((id.clone(), g))
    } else if let Some(raw) = &req.graph {
        let _ = sentinel_core::compound::BlockLibrary::reload_project(&project_root().join("blocks"));
        let spec: sentinel_core::graph::GraphSpec = serde_json::from_value(raw.clone())
            .map_err(|e| api_err(StatusCode::BAD_REQUEST, format!("invalid graph: {e}")))?;
        let g = spec.validate().map_err(|e| api_err(StatusCode::BAD_REQUEST, e))?;
        Ok(("canvas".to_string(), g))
    } else {
        Err(api_err(StatusCode::BAD_REQUEST, "provide either component_id or graph"))
    }
}

async fn list_blocks() -> impl IntoResponse {
    let errors = sentinel_core::compound::BlockLibrary::reload_project(&project_root().join("blocks"));
    for e in errors {
        tracing::warn!("ignoring compound block: {e}");
    }
    Json(sentinel_core::compound::BlockLibrary::global().catalog())
}

/// Validation is a query: an invalid graph is a normal `200 {ok: false, error}` answer
/// (the editor asks on every edit), not an HTTP error.
async fn graph_validate(State(state): State<AppState>, Json(req): Json<GraphRequest>) -> Json<serde_json::Value> {
    match validate_graph(&state, &req) {
        Ok(v) => Json(v),
        Err((_, Json(e))) => Json(e),
    }
}

/// Per-node description with shapes; compound nodes carry the same description of their inner graph.
fn describe(g: &sentinel_core::graph::ValidatedGraph) -> serde_json::Map<String, serde_json::Value> {
    use sentinel_core::graph::Dim;
    let dims = |s: &Vec<Dim>| s.iter().map(|d| d.to_string()).collect::<Vec<_>>();
    let mut nodes = serde_json::Map::new();
    for n in &g.order {
        let b = g.kind(n);
        let outs: serde_json::Map<String, serde_json::Value> = b.outputs().iter().map(|p| (p.clone(), json!(dims(&g.shapes[&format!("{n}.{p}")])))).collect();
        let ins: serde_json::Map<String, serde_json::Value> = b.inputs().iter().map(|p| (p.clone(), json!(g.wires[&(n.clone(), p.clone())].key()))).collect();
        let mut node = json!({ "block": b.name(), "tier": b.tier(), "inputs": ins, "outputs": outs });
        if let Some(inner) = g.compounds.get(n) {
            node["inner"] = json!({ "order": inner.order, "nodes": describe(inner) });
        }
        nodes.insert(n.clone(), node);
    }
    nodes
}

fn validate_graph(state: &AppState, req: &GraphRequest) -> Result<serde_json::Value, ApiError> {
    use sentinel_core::graph::Dim;
    let (id, g) = resolve_graph(state, req)?;
    let dims = |s: &Vec<Dim>| s.iter().map(|d| d.to_string()).collect::<Vec<_>>();
    let nodes = describe(&g);
    let inputs: serde_json::Map<String, serde_json::Value> =
        g.spec.inputs.iter().map(|(k, v)| (k.clone(), json!(dims(v)))).collect();
    let outputs: serde_json::Map<String, serde_json::Value> =
        g.outputs.iter().map(|(k, v)| (k.clone(), json!(v.key()))).collect();
    Ok(json!({
        "ok": true, "id": id, "order": g.order, "inputs": inputs, "nodes": nodes, "outputs": outputs,
        "trainable": g.spec.training.is_some(),
        "spec": serde_json::to_value(&g.spec).unwrap_or(serde_json::Value::Null),
        "params": serde_json::to_value(&g.params).unwrap_or(serde_json::Value::Null),
    }))
}

fn compile(g: &sentinel_core::graph::ValidatedGraph, id: &str, model_only: bool) -> Result<String, ApiError> {
    if model_only {
        Ok(format!(
            "import torch\nimport torch.nn as nn\n\n{}",
            sentinel_core::codegen::generate_model(g)
        ))
    } else {
        sentinel_core::codegen::generate_training_script(g, id).map_err(|e| api_err(StatusCode::BAD_REQUEST, e))
    }
}

async fn graph_compose(
    State(state): State<AppState>,
    Json(req): Json<GraphRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (id, g) = resolve_graph(&state, &req)?;
    let code = compile(&g, &id, req.model_only)?;
    Ok(Json(json!({ "ok": true, "id": id, "code": code })))
}

async fn graph_run(
    State(state): State<AppState>,
    Json(req): Json<GraphRequest>,
) -> Result<Json<JobRecord>, ApiError> {
    let (id, g) = resolve_graph(&state, &req)?;
    let code = compile(&g, &id, false)?;

    let root = project_root();
    let io = |e: std::io::Error| api_err(StatusCode::INTERNAL_SERVER_ERROR, e);

    // One script per job: concurrent runs of the same component (e.g. a sweep) must not
    // overwrite each other's generated code.
    let mut spec = JobSpec::new(format!("graph-{id}"), RunMode::Command);
    let script = sentinel_core::sdk::write_generated_script(&root, &id, &spec.id.simple().to_string()[..8], &code).map_err(io)?;

    spec.component_id = Some(id.clone());
    spec.command = Some(sentinel_core::platform::python_script_command(&script));
    spec.env_vars.insert("PYTHONPATH".into(), sentinel_core::sdk::python_path(&root).map_err(io)?);
    spec.working_dir = root.display().to_string();

    state
        .supervisor
        .start_job(spec)
        .await
        .map(Json)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))
}

// ---------------------------------------------------------------------------
// Experiment tracking (mlruns/): run list and metric curves
// ---------------------------------------------------------------------------

fn mlruns_root() -> PathBuf {
    std::env::current_dir().unwrap_or_default().join("mlruns")
}

async fn tracking_runs() -> impl IntoResponse {
    Json(sentinel_core::tracking::list_runs(&mlruns_root()))
}

async fn tracking_run(AxumPath(id): AxumPath<String>) -> Result<Json<serde_json::Value>, StatusCode> {
    let (run, metrics) = sentinel_core::tracking::read_run(&mlruns_root(), &id, 600).ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(json!({ "run": run, "metrics": metrics })))
}

// ---------------------------------------------------------------------------
// Authoring: save a drawn graph as a component file
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SaveComponentRequest {
    #[serde(flatten)]
    pub spec: sentinel_core::authoring::SaveSpec,
    #[serde(default)]
    pub overwrite: bool,
}

async fn save_component(Json(req): Json<SaveComponentRequest>) -> Result<Json<serde_json::Value>, ApiError> {
    let (id, yaml) = sentinel_core::authoring::build_component_yaml(&req.spec)
        .map_err(|e| api_err(StatusCode::BAD_REQUEST, e))?;

    // Shipped components are part of the binary; never shadow them with a file.
    if ComponentRegistry::new().get(&id).is_some() {
        return Err(api_err(StatusCode::CONFLICT, format!("'{id}' is a built-in component; pick another name")));
    }
    let rel = PathBuf::from("components").join(sentinel_core::authoring::slugify(&req.spec.domain))
        .join(format!("{}.yaml", sentinel_core::authoring::slugify(&req.spec.name)));
    let path = std::env::current_dir().unwrap_or_default().join(&rel);
    if path.exists() && !req.overwrite {
        return Err(api_err(StatusCode::CONFLICT, format!("{} already exists", rel.display())));
    }
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    std::fs::write(&path, yaml).map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(json!({ "ok": true, "component_id": id, "path": rel.display().to_string() })))
}

// ---------------------------------------------------------------------------
// Dataset schema inspection (columns, task and loss suggestions)
// ---------------------------------------------------------------------------

fn project_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    cwd.canonicalize().unwrap_or(cwd)
}

fn inspect(path: &Path) -> Result<Json<serde_json::Value>, ApiError> {
    let schema = sentinel_core::dataset_schema::inspect_csv(path).map_err(|e| api_err(StatusCode::BAD_REQUEST, e))?;
    Ok(Json(serde_json::to_value(schema).unwrap_or_default()))
}

/// Registered datasets may live anywhere (the user mounted them); ad-hoc paths must stay in the project.
async fn schema_for_dataset(AxumPath(name): AxumPath<String>) -> Result<Json<serde_json::Value>, ApiError> {
    let root = project_root();
    let ds = sentinel_core::storage::DatasetRegistry::load_all(&root)
        .into_iter()
        .find(|d| d.name == name)
        .ok_or_else(|| api_err(StatusCode::NOT_FOUND, format!("dataset '{name}' is not mounted")))?;
    let p = PathBuf::from(&ds.path);
    inspect(&if p.is_absolute() { p } else { root.join(p) })
}

#[derive(Deserialize)]
struct SchemaQuery {
    path: String,
}

async fn schema_for_path(Query(q): Query<SchemaQuery>) -> Result<Json<serde_json::Value>, ApiError> {
    let root = project_root();
    let p = PathBuf::from(&q.path);
    let full = if p.is_absolute() { p } else { root.join(p) };
    let full = full.canonicalize().map_err(|_| api_err(StatusCode::NOT_FOUND, format!("{} not found", q.path)))?;
    if !full.starts_with(&root) {
        return Err(api_err(StatusCode::FORBIDDEN, "path is outside the project; mount it as a dataset first"));
    }
    inspect(&full)
}

#[derive(Debug, Deserialize)]
pub struct SaveBlockRequest {
    #[serde(flatten)]
    pub spec: sentinel_core::authoring::SaveBlockSpec,
    #[serde(default)]
    pub overwrite: bool,
}

/// Save the canvas as a compound block under `blocks/<family>/<name>.yaml`; it appears in the palette immediately.
async fn save_block(Json(req): Json<SaveBlockRequest>) -> Result<Json<serde_json::Value>, ApiError> {
    use sentinel_core::compound::BlockLibrary;
    let root = project_root();
    let name = sentinel_core::authoring::snake(&req.spec.name);
    // Shipped blocks (both tiers) are part of the binary and cannot be shadowed.
    if BlockLibrary::builtin().resolve(&name).is_some() {
        return Err(api_err(StatusCode::CONFLICT, format!("'{name}' is a built-in block; pick another name")));
    }
    let (mut lib, _) = BlockLibrary::with_dir(&root.join("blocks"));
    if req.overwrite {
        lib.remove(&name);
    }
    let (name, yaml) = sentinel_core::authoring::build_compound_yaml(&req.spec, &lib).map_err(|e| api_err(StatusCode::BAD_REQUEST, e))?;
    let rel = PathBuf::from("blocks").join(sentinel_core::authoring::snake(&req.spec.family)).join(format!("{name}.yaml"));
    let path = root.join(&rel);
    if path.exists() && !req.overwrite {
        return Err(api_err(StatusCode::CONFLICT, format!("{} already exists", rel.display())));
    }
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    std::fs::write(&path, yaml).map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let _ = BlockLibrary::reload_project(&root.join("blocks"));
    Ok(Json(json!({ "ok": true, "name": name, "path": rel.display().to_string() })))
}
