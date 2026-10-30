//! Read-only access to MLflow-compatible run tracking (`mlruns/<experiment>/<run_id>/...`)
//! as written by the Sentinel SDK, for run lists, params, and metric curves.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricPoint {
    pub step: i64,
    pub value: f64,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RunSummary {
    pub run_id: String,
    pub experiment_id: String,
    pub run_name: String,
    /// running | finished | failed | killed | unknown
    pub status: String,
    pub start_time: Option<i64>,
    pub end_time: Option<i64>,
    pub params: BTreeMap<String, String>,
    /// Last logged value per metric.
    pub latest: BTreeMap<String, f64>,
}

#[derive(Deserialize, Default)]
struct Meta {
    #[serde(default)]
    run_name: String,
    #[serde(default)]
    status: i64,
    #[serde(default)]
    start_time: Option<i64>,
    #[serde(default)]
    end_time: Option<i64>,
}

fn status_name(code: i64) -> &'static str {
    match code {
        1 => "running",
        3 => "finished",
        4 => "failed",
        5 => "killed",
        _ => "unknown",
    }
}

/// Run ids are hex uuids; refuse anything else so ids can't escape the tracking directory.
pub fn valid_run_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric())
}

fn experiments(root: &Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(root) else { return vec![] };
    let mut out: Vec<_> = rd
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (!name.starts_with('.')).then(|| (name, e.path()))
        })
        .collect();
    out.sort();
    out
}

fn read_kv_dir(dir: &Path) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.path().is_file() {
                if let Ok(v) = std::fs::read_to_string(e.path()) {
                    m.insert(e.file_name().to_string_lossy().to_string(), v.trim().to_string());
                }
            }
        }
    }
    m
}

fn parse_metric_file(text: &str) -> Vec<MetricPoint> {
    let mut pts: Vec<MetricPoint> = text
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let timestamp = it.next()?.parse().ok()?;
            let value: f64 = it.next()?.parse().ok()?;
            let step = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            value.is_finite().then_some(MetricPoint { step, value, timestamp })
        })
        .collect();
    pts.sort_by_key(|p| (p.step, p.timestamp));
    pts
}

fn summarize(exp: &str, run_dir: &Path) -> Option<RunSummary> {
    let meta: Meta = serde_yaml::from_str(&std::fs::read_to_string(run_dir.join("meta.yaml")).ok()?).ok()?;
    let run_id = run_dir.file_name()?.to_string_lossy().to_string();
    let mut latest = BTreeMap::new();
    for (k, _) in read_kv_dir(&run_dir.join("metrics")) {
        if let Ok(t) = std::fs::read_to_string(run_dir.join("metrics").join(&k)) {
            if let Some(p) = parse_metric_file(&t).last() {
                latest.insert(k, p.value);
            }
        }
    }
    Some(RunSummary {
        run_name: if meta.run_name.is_empty() { run_id.chars().take(8).collect() } else { meta.run_name },
        run_id,
        experiment_id: exp.to_string(),
        status: status_name(meta.status).to_string(),
        start_time: meta.start_time,
        end_time: meta.end_time,
        params: read_kv_dir(&run_dir.join("params")),
        latest,
    })
}

/// All tracked runs, newest first.
pub fn list_runs(root: &Path) -> Vec<RunSummary> {
    let mut runs = Vec::new();
    for (exp, dir) in experiments(root) {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten().filter(|e| e.path().is_dir()) {
                if let Some(r) = summarize(&exp, &e.path()) {
                    runs.push(r);
                }
            }
        }
    }
    runs.sort_by(|a, b| b.start_time.cmp(&a.start_time));
    runs
}

/// One run plus every metric series, thinned to at most `max_points` per series.
pub fn read_run(root: &Path, run_id: &str, max_points: usize) -> Option<(RunSummary, BTreeMap<String, Vec<MetricPoint>>)> {
    if !valid_run_id(run_id) {
        return None;
    }
    for (exp, dir) in experiments(root) {
        let run_dir = dir.join(run_id);
        if !run_dir.is_dir() {
            continue;
        }
        let summary = summarize(&exp, &run_dir)?;
        let mut series = BTreeMap::new();
        for (k, text) in read_kv_dir_raw(&run_dir.join("metrics")) {
            series.insert(k, thin(parse_metric_file(&text), max_points));
        }
        return Some((summary, series));
    }
    None
}

fn read_kv_dir_raw(dir: &Path) -> Vec<(String, String)> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .filter_map(|e| Some((e.file_name().to_string_lossy().to_string(), std::fs::read_to_string(e.path()).ok()?)))
                .collect()
        })
        .unwrap_or_default()
}

/// Evenly thin a series, always keeping the first and last point.
fn thin(pts: Vec<MetricPoint>, max: usize) -> Vec<MetricPoint> {
    if max < 2 || pts.len() <= max {
        return pts;
    }
    let last = pts.len() - 1;
    (0..max).map(|i| pts[i * last / (max - 1)].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_run(root: &Path, id: &str, name: &str, status: i64, start: i64, metric: &str) {
        let d = root.join("0").join(id);
        std::fs::create_dir_all(d.join("metrics")).unwrap();
        std::fs::create_dir_all(d.join("params")).unwrap();
        std::fs::write(d.join("meta.yaml"), format!("run_name: {name}\nstatus: {status}\nstart_time: {start}\nend_time: {}\n", start + 5)).unwrap();
        std::fs::write(d.join("params/lr"), "0.01").unwrap();
        std::fs::write(d.join("metrics/loss"), metric).unwrap();
    }

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("sentinel-tracking-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn lists_newest_first_with_latest_metric_and_params() {
        let r = tmp("list");
        make_run(&r, "aaa", "old", 3, 100, "1 0.9 0\n2 0.5 1\n");
        make_run(&r, "bbb", "new", 1, 200, "1 0.7 0\n");
        let runs = list_runs(&r);
        assert_eq!(runs.iter().map(|x| x.run_name.as_str()).collect::<Vec<_>>(), ["new", "old"]);
        assert_eq!(runs[1].status, "finished");
        assert_eq!(runs[0].status, "running");
        assert_eq!(runs[1].latest["loss"], 0.5);
        assert_eq!(runs[1].params["lr"], "0.01");
    }

    #[test]
    fn skips_garbage_lines_and_non_finite_values() {
        let r = tmp("garbage");
        make_run(&r, "ccc", "x", 3, 1, "1 0.5 0\nnot a line\n2 NaN 1\n3 inf 2\n4 0.25 3\n");
        let (_, s) = read_run(&r, "ccc", 100).unwrap();
        assert_eq!(s["loss"].iter().map(|p| p.step).collect::<Vec<_>>(), [0, 3]);
    }

    #[test]
    fn thinning_keeps_endpoints() {
        let pts: Vec<_> = (0..1000).map(|i| MetricPoint { step: i, value: i as f64, timestamp: i }).collect();
        let t = thin(pts, 50);
        assert_eq!(t.len(), 50);
        assert_eq!((t[0].step, t[49].step), (0, 999));
    }

    #[test]
    fn rejects_path_traversal_ids() {
        let r = tmp("trav");
        make_run(&r, "ddd", "x", 3, 1, "1 1 0\n");
        for bad in ["..", "../0/ddd", "a/b", "", "ddd/../ddd"] {
            assert!(read_run(&r, bad, 10).is_none(), "{bad}");
        }
        assert!(read_run(&r, "ddd", 10).is_some());
    }

    #[test]
    fn missing_root_is_empty() {
        assert!(list_runs(Path::new("/definitely/not/here")).is_empty());
    }
}
