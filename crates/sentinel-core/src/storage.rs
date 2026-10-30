//! Storage drivers (MLflow native layout).

#[derive(Debug, Default)]
pub struct MlflowStorage {
    pub root_dir: std::path::PathBuf,
}
