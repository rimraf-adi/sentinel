//! GPU discovery and telemetry without NVML: `nvidia-smi` is parsed for NVIDIA GPUs, and Apple Silicon is
//! identified through `system_profiler`. Anything a platform cannot report stays `None`.

use crate::models::GpuStats;
use std::collections::HashMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const GPU_QUERY: &str = "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,driver_version,utilization.memory,fan.speed,clocks.sm,clocks.mem,power.limit,pstate";
const APPS_QUERY: &str = "--query-compute-apps=pid,used_memory";

/// Run a command, returning stdout if it exits 0 within `timeout`.
fn run(cmd: &str, args: &[&str], timeout: Duration) -> Result<String, String> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{cmd}: {e}"))?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                let mut err = String::new();
                if let Some(mut o) = child.stdout.take() {
                    let _ = o.read_to_string(&mut out);
                }
                if let Some(mut e) = child.stderr.take() {
                    let _ = e.read_to_string(&mut err);
                }
                return if status.success() { Ok(out) } else { Err(format!("{cmd} failed: {}", first_line(&err, &out))) };
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{cmd} timed out after {}s", timeout.as_secs()));
            }
            Err(e) => return Err(format!("{cmd}: {e}")),
        }
    }
}

fn first_line(a: &str, b: &str) -> String {
    a.lines().chain(b.lines()).map(str::trim).find(|l| !l.is_empty()).unwrap_or("no output").to_string()
}

/// `[N/A]`, `[Not Supported]`, `[Insufficient Permissions]`, empty: all mean "not reported".
fn num<T: std::str::FromStr>(s: &str) -> Option<T> {
    let s = s.trim();
    if s.is_empty() || s.starts_with('[') { None } else { s.parse().ok() }
}

const MIB: u64 = 1024 * 1024;

/// Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`.
pub fn parse_nvidia_smi(out: &str) -> Vec<GpuStats> {
    out.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() < 8 {
                return None;
            }
            // The first 8 columns are always present; the rest only when the query asked for them.
            let at = |i: usize| f.get(i).copied().unwrap_or("");
            Some(GpuStats {
                index: num(f[0])?,
                name: f[1].to_string(),
                vendor: "nvidia".into(),
                vram_used_bytes: num::<u64>(f[2]).map(|m| m * MIB),
                vram_total_bytes: num::<u64>(f[3]).map(|m| m * MIB),
                gpu_util_percent: num(f[4]),
                temperature_c: num(f[5]),
                power_w: num(f[6]),
                driver: Some(f[7].to_string()).filter(|d| !d.starts_with('[') && !d.is_empty()),
                mem_util_percent: num(at(8)),
                fan_percent: num(at(9)),
                clock_graphics_mhz: num(at(10)),
                clock_memory_mhz: num(at(11)),
                power_limit_w: num(at(12)),
                pstate: Some(at(13).to_string()).filter(|p| p.starts_with('P')),
            })
        })
        .collect()
}

/// Parse `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` into pid -> bytes.
pub fn parse_compute_apps(out: &str) -> HashMap<u32, u64> {
    let mut m = HashMap::new();
    for line in out.lines() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() >= 2 {
            if let Some(pid) = num::<u32>(f[0]) {
                *m.entry(pid).or_insert(0) += num::<u64>(f[1]).unwrap_or(0) * MIB;
            }
        }
    }
    m
}

/// Run nvidia-smi from PATH or the OS-specific install locations. A driver error from a binary that exists
/// is reported in preference to "not found".
fn run_nvidia(args: &[&str]) -> Result<String, String> {
    let mut not_found = None;
    for exe in crate::platform::nvidia_smi_candidates() {
        match run(&exe, args, Duration::from_secs(3)) {
            Ok(o) => return Ok(o),
            Err(e) if e.contains("No such file") || e.contains("cannot find") || e.contains("not found") || e.contains("os error 2") => {
                not_found.get_or_insert(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(not_found.map_or_else(|| "nvidia-smi not found".into(), |e| format!("nvidia-smi not found ({e})")))
}

pub fn query_nvidia() -> Result<Vec<GpuStats>, String> {
    run_nvidia(&[GPU_QUERY, "--format=csv,noheader,nounits"]).map(|o| parse_nvidia_smi(&o))
}

pub fn query_compute_apps() -> Result<HashMap<u32, u64>, String> {
    run_nvidia(&[APPS_QUERY, "--format=csv,noheader,nounits"]).map(|o| parse_compute_apps(&o))
}

/// Name GPUs that have no telemetry tool: `lspci` on Linux, PowerShell on Windows. Names only, never metrics.
pub fn detect_other() -> Vec<GpuStats> {
    match crate::platform::Os::current() {
        crate::platform::Os::Linux => run("lspci", &["-mm"], Duration::from_secs(3)).map(|o| parse_lspci(&o)).unwrap_or_default(),
        crate::platform::Os::Windows => run(
            "powershell",
            &["-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            Duration::from_secs(8),
        )
        .map(|o| parse_windows_names(&o))
        .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn unknown_metrics(index: u32, name: String, vendor: &str) -> GpuStats {
    GpuStats { index, name, vendor: vendor.into(), ..Default::default() }
}

fn vendor_of(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("nvidia") {
        "nvidia"
    } else if n.contains("amd") || n.contains("advanced micro") || n.contains("radeon") || n.contains("ati ") {
        "amd"
    } else if n.contains("intel") {
        "intel"
    } else {
        "other"
    }
}

/// Parse `lspci -mm` machine-readable lines: `00:02.0 "VGA compatible controller" "Intel Corporation" "UHD 630" ...`.
pub fn parse_lspci(out: &str) -> Vec<GpuStats> {
    let mut gpus = Vec::new();
    for line in out.lines() {
        let q: Vec<&str> = line.split('"').skip(1).step_by(2).collect();
        if q.len() < 3 {
            continue;
        }
        let class = q[0].to_lowercase();
        if class.contains("vga") || class.contains("3d controller") || class.contains("display controller") {
            let name = format!("{} {}", q[1], q[2]);
            gpus.push(unknown_metrics(gpus.len() as u32, name.clone(), vendor_of(&name)));
        }
    }
    gpus
}

pub fn parse_windows_names(out: &str) -> Vec<GpuStats> {
    out.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .enumerate()
        .map(|(i, n)| unknown_metrics(i as u32, n.to_string(), vendor_of(n)))
        .collect()
}

/// Identify the Apple GPU (name and core count only; macOS exposes no unprivileged live metrics).
pub fn detect_apple() -> Option<GpuStats> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = run("system_profiler", &["SPDisplaysDataType", "-json"], Duration::from_secs(5)).ok()?;
    parse_system_profiler(&out)
}

pub fn parse_system_profiler(json: &str) -> Option<GpuStats> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let d = v.get("SPDisplaysDataType")?.as_array()?.first()?;
    let model = d.get("sppci_model").or_else(|| d.get("_name"))?.as_str()?;
    if !model.to_lowercase().contains("apple") {
        return None;
    }
    let cores = d.get("sppci_cores").and_then(|c| c.as_str()).map(|c| format!(" ({c}-core GPU)")).unwrap_or_default();
    Some(GpuStats { name: format!("{model}{cores}"), vendor: "apple".into(), ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_GPUS: &str = "0, NVIDIA GeForce RTX 4090, 1234, 24564, 7, 38, 35.42, 550.54.14\n\
                            1, NVIDIA A100-SXM4-40GB, 0, 40960, 0, 30, [N/A], 535.104.05\n";

    #[test]
    fn parses_nvidia_smi_csv() {
        let g = parse_nvidia_smi(TWO_GPUS);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(g[0].vram_used_bytes, Some(1234 * MIB));
        assert_eq!(g[0].vram_total_bytes, Some(24564 * MIB));
        assert_eq!((g[0].gpu_util_percent, g[0].temperature_c, g[0].power_w), (Some(7.0), Some(38.0), Some(35.42)));
        assert_eq!(g[0].driver.as_deref(), Some("550.54.14"));
        assert_eq!(g[1].power_w, None, "[N/A] must stay unknown, not become 0");
        assert_eq!(g[1].index, 1);
    }

    #[test]
    fn parses_the_extended_columns() {
        let g = parse_nvidia_smi("0, NVIDIA RTX A6000, 100, 49140, 55, 61, 210.5, 550.1, 31, 45, 1740, 8000, 300.00, P2\n\
                                  1, Tesla V100, 0, 16384, 0, 33, 40, 535.1, 0, [N/A], 135, 877, 250.00, P8");
        assert_eq!(g[0].mem_util_percent, Some(31.0));
        assert_eq!(g[0].fan_percent, Some(45.0));
        assert_eq!((g[0].clock_graphics_mhz, g[0].clock_memory_mhz), (Some(1740), Some(8000)));
        assert_eq!(g[0].power_limit_w, Some(300.0));
        assert_eq!(g[0].pstate.as_deref(), Some("P2"));
        assert_eq!(g[1].fan_percent, None, "passively cooled datacenter GPUs report no fan");
        assert_eq!(g[1].pstate.as_deref(), Some("P8"));
        // older 8-column output still parses, the extras are simply unknown
        let old = parse_nvidia_smi("0, GTX 1080, 500, 8192, 10, 50, 60.0, 470.1");
        assert_eq!((old[0].fan_percent, old[0].pstate.clone()), (None, None));
    }

    #[test]
    fn unsupported_fields_are_none_not_zero() {
        let g = parse_nvidia_smi("0, Tesla T4, [Not Supported], 15360, [Insufficient Permissions], [N/A], [N/A], [N/A]");
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].vram_used_bytes, g[0].gpu_util_percent, g[0].temperature_c, g[0].driver.clone()), (None, None, None, None));
        assert_eq!(g[0].vram_total_bytes, Some(15360 * MIB));
    }

    #[test]
    fn garbage_and_error_text_yield_no_gpus() {
        assert!(parse_nvidia_smi("").is_empty());
        assert!(parse_nvidia_smi("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.").is_empty());
        assert!(parse_nvidia_smi("x, y, z").is_empty());
    }

    #[test]
    fn compute_apps_sum_per_pid() {
        let m = parse_compute_apps("4242, 1500\n4242, 500\n77, 100\n");
        assert_eq!(m[&4242], 2000 * MIB);
        assert_eq!(m[&77], 100 * MIB);
        assert!(parse_compute_apps("No devices were found").is_empty());
        assert!(parse_compute_apps("").is_empty());
    }

    #[test]
    fn apple_gpu_has_name_but_no_invented_metrics() {
        let j = r#"{"SPDisplaysDataType":[{"_name":"Apple M2 Pro","sppci_model":"Apple M2 Pro","sppci_cores":"19"}]}"#;
        let g = parse_system_profiler(j).unwrap();
        assert_eq!(g.name, "Apple M2 Pro (19-core GPU)");
        assert_eq!(g.vendor, "apple");
        assert!(g.gpu_util_percent.is_none() && g.vram_total_bytes.is_none() && g.temperature_c.is_none());
        assert!(parse_system_profiler(r#"{"SPDisplaysDataType":[{"sppci_model":"AMD Radeon Pro"}]}"#).is_none());
        assert!(parse_system_profiler("not json").is_none());
    }

    #[test]
    fn lspci_lists_display_adapters_only() {
        let out = "00:00.0 \"Host bridge\" \"Intel Corporation\" \"Device 9b61\" -r0c\n\
                   00:02.0 \"VGA compatible controller\" \"Intel Corporation\" \"UHD Graphics 630\" -r02\n\
                   01:00.0 \"3D controller\" \"NVIDIA Corporation\" \"GA102 [GeForce RTX 3090]\" -ra1\n\
                   03:00.0 \"Display controller\" \"Advanced Micro Devices, Inc. [AMD/ATI]\" \"Navi 21\" -r00\n";
        let g = parse_lspci(out);
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].vendor, "intel");
        assert_eq!(g[1].vendor, "nvidia");
        assert_eq!(g[2].vendor, "amd");
        assert!(g.iter().all(|x| x.gpu_util_percent.is_none() && x.vram_total_bytes.is_none()));
        assert!(parse_lspci("garbage").is_empty());
    }

    #[test]
    fn windows_powershell_names_are_parsed() {
        let g = parse_windows_names("NVIDIA GeForce RTX 4070\r\nIntel(R) UHD Graphics\r\n\r\n");
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].vendor.as_str(), g[1].vendor.as_str()), ("nvidia", "intel"));
        assert_eq!(g[1].index, 1);
    }

    #[test]
    fn missing_binary_is_a_clear_error() {
        let e = run("definitely-not-a-real-binary-xyz", &[], Duration::from_secs(1)).unwrap_err();
        assert!(e.contains("definitely-not-a-real-binary-xyz"));
    }

    #[test]
    fn slow_commands_time_out() {
        let (cmd, args): (&str, &[&str]) = if cfg!(windows) { ("ping", &["-n", "6", "127.0.0.1"]) } else { ("sleep", &["5"]) };
        let e = run(cmd, args, Duration::from_millis(150)).unwrap_err();
        assert!(e.contains("timed out"), "{e}");
    }
}
