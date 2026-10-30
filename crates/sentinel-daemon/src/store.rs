//! On-disk job history (`.sentinel/jobs/<id>.json` + `<id>.log`) so runs survive a daemon restart.

use sentinel_core::models::{JobRecord, JobStatus};
use std::io::Write;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct JobStore {
    dir: PathBuf,
}

impl JobStore {
    pub fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn json(&self, id: Uuid) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
    fn log(&self, id: Uuid) -> PathBuf {
        self.dir.join(format!("{id}.log"))
    }

    /// Best-effort, atomic (write + rename). A failed save must never take a job down.
    pub fn save(&self, rec: &JobRecord) {
        let tmp = self.dir.join(format!("{}.json.tmp", rec.spec.id));
        let res = serde_json::to_vec_pretty(rec)
            .map_err(std::io::Error::other)
            .and_then(|b| std::fs::write(&tmp, b))
            .and_then(|_| std::fs::rename(&tmp, self.json(rec.spec.id)));
        if let Err(e) = res {
            tracing::warn!("could not persist job {}: {e}", rec.spec.id);
        }
    }

    pub fn append_log(&self, id: Uuid, line: &str) {
        let res = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log(id))
            .and_then(|mut f| writeln!(f, "{line}"));
        if let Err(e) = res {
            tracing::warn!("could not persist log line for {id}: {e}");
        }
    }

    /// Every stored job with its logs. Jobs still marked running belonged to a process that no longer
    /// exists (the daemon restarted), so they are closed out as failed instead of showing as live forever.
    pub fn load_all(&self) -> Vec<(JobRecord, Vec<String>)> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return out };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let Ok(bytes) = std::fs::read(&p) else { continue };
            let Ok(mut rec) = serde_json::from_slice::<JobRecord>(&bytes) else {
                tracing::warn!("skipping unreadable job file {}", p.display());
                continue;
            };
            if matches!(rec.status, JobStatus::Running | JobStatus::Pending) {
                rec.status = JobStatus::Failed;
                rec.finished_at = Some(chrono::Utc::now());
                rec.error_message = Some("daemon restarted while this job was running".into());
                self.save(&rec);
            }
            let logs = std::fs::read_to_string(self.log(rec.spec.id))
                .map(|t| t.lines().map(str::to_string).collect())
                .unwrap_or_default();
            out.push((rec, logs));
        }
        out.sort_by_key(|(r, _)| r.started_at);
        out
    }
}
