//! System vitals: host CPU/RAM/swap/load, real GPU telemetry, per-job process trees, and Sentinel's own footprint.

use crate::gpu;
use crate::models::GpuStats;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use sysinfo::{Components, Disks, Networks, Pid, ProcessesToUpdate, System};
use uuid::Uuid;

/// A process (tree) footprint. `cpu_percent` is in cores: 100 = one fully busy core.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcStat {
    pub pid: u32,
    pub cpu_percent: f32,
    pub rss_bytes: u64,
    /// Number of processes in the tree (a job is `sh -c` plus its python child, and any workers).
    pub processes: u32,
    /// Disk read/write throughput of the tree, bytes per second.
    #[serde(default)]
    pub io_read_bps: f64,
    #[serde(default)]
    pub io_write_bps: f64,
}

/// One network interface. Rates are bytes/second since the previous sample.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NetStat {
    pub name: String,
    pub rx_bps: f64,
    pub tx_bps: f64,
    pub rx_total: u64,
    pub tx_total: u64,
    pub rx_errors: u64,
    pub tx_errors: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiskStat {
    pub name: String,
    pub mount: String,
    pub fs: String,
    /// `ssd` | `hdd` | `unknown`
    pub kind: String,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub removable: bool,
}

/// A hardware temperature sensor. Many platforms expose none (e.g. most VMs, Apple Silicon without root).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sensor {
    pub label: String,
    pub temp_c: f32,
    pub max_c: Option<f32>,
    pub critical_c: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopProc {
    pub pid: u32,
    pub name: String,
    pub cpu_percent: f32,
    pub rss_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobVitals {
    pub job_id: Uuid,
    pub name: String,
    pub stat: ProcStat,
    /// GPU memory held by this job's processes (NVIDIA only).
    pub gpu_mem_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Vitals {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub machine_id: String,
    pub cpu_util_percent: f32,
    pub per_core: Vec<f32>,
    pub cpu_count: usize,
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
    pub load_avg: [f64; 3],
    pub uptime_secs: u64,
    pub gpus: Vec<GpuStats>,
    /// The Sentinel process itself (daemon or CLI).
    pub daemon: Option<ProcStat>,
    pub jobs: Vec<JobVitals>,
    /// Per-core clock in MHz; empty when the OS does not report it.
    #[serde(default)]
    pub cpu_freq_mhz: Vec<u64>,
    #[serde(default)]
    pub network: Vec<NetStat>,
    #[serde(default)]
    pub disks: Vec<DiskStat>,
    #[serde(default)]
    pub sensors: Vec<Sensor>,
}

/// Static facts about the machine, for the System page and `sentinel status`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemInfo {
    pub os: String,
    pub os_version: Option<String>,
    pub kernel: Option<String>,
    pub host: Option<String>,
    pub arch: String,
    pub cpu_brand: String,
    pub cpu_count: usize,
    pub ram_total_bytes: u64,
    pub version: String,
    /// Why GPU telemetry is unavailable (nvidia-smi missing, driver error, ...), if it is.
    pub gpu_note: Option<String>,
}

/// A job to watch: id, display name, root pid.
pub type Watch = (Uuid, String, u32);

const GPU_REFRESH: Duration = Duration::from_secs(1);

pub struct VitalsMonitor {
    sys: System,
    networks: Networks,
    disks: Disks,
    components: Components,
    last_net: Instant,
    last_proc: Instant,
    last_disks: Instant,
    /// Temperatures move slowly and reading every sensor is costly (IOKit on Apple Silicon): cache for 5 s.
    sensor_cache: (Instant, Vec<Sensor>),
    /// Full process scans discover new children; in between, only known job processes are refreshed.
    last_full_scan: Option<Instant>,
    known_pids: Vec<Pid>,
    nvidia: bool,
    /// GPUs we can only name (Apple, or anything without nvidia-smi): shown without invented metrics.
    named_only: Vec<GpuStats>,
    gpu_error: Option<String>,
    gpu_cache: (Option<Instant>, Vec<GpuStats>, HashMap<u32, u64>),
}

impl Default for VitalsMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl VitalsMonitor {
    pub fn new() -> Self {
        let mut sys = System::new();
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        let (nvidia, gpu_error) = match gpu::query_nvidia() {
            Ok(g) if !g.is_empty() => (true, None),
            Ok(_) => (false, None),
            Err(e) => (false, Some(e)),
        };
        // Discovery commands are slow (system_profiler, lspci, PowerShell), so identify these GPUs once.
        let named_only = if nvidia { Vec::new() } else { gpu::detect_apple().into_iter().chain(gpu::detect_other()).collect() };
        let now = Instant::now();
        Self {
            sys,
            networks: Networks::new_with_refreshed_list(),
            disks: Disks::new_with_refreshed_list(),
            components: Components::new_with_refreshed_list(),
            last_net: now,
            last_proc: now,
            last_disks: now,
            sensor_cache: (now - Duration::from_secs(60), Vec::new()),
            last_full_scan: None,
            known_pids: Vec::new(),
            nvidia,
            named_only,
            gpu_error,
            gpu_cache: (None, Vec::new(), HashMap::new()),
        }
    }

    pub fn system_info(&mut self) -> SystemInfo {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        SystemInfo {
            os: crate::platform::Os::current().name().into(),
            os_version: System::long_os_version(),
            kernel: System::kernel_version(),
            host: System::host_name(),
            arch: std::env::consts::ARCH.into(),
            cpu_brand: self.sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default(),
            cpu_count: self.sys.cpus().len(),
            ram_total_bytes: self.sys.total_memory(),
            version: env!("CARGO_PKG_VERSION").into(),
            gpu_note: self.gpu_error.clone().filter(|_| self.named_only.is_empty()),
        }
    }

    /// Why NVIDIA telemetry is unavailable, if nvidia-smi was tried and failed (not found, driver error, ...).
    pub fn gpu_error(&self) -> Option<&str> {
        self.gpu_error.as_deref()
    }

    pub fn has_nvidia(&self) -> bool {
        self.nvidia
    }

    /// GPUs and per-pid GPU memory, refreshed at most once a second (nvidia-smi is a subprocess).
    fn gpus(&mut self) -> (Vec<GpuStats>, HashMap<u32, u64>) {
        if !self.nvidia {
            return (self.named_only.clone(), HashMap::new());
        }
        let stale = self.gpu_cache.0.map_or(true, |t| t.elapsed() >= GPU_REFRESH);
        if stale {
            let gpus = gpu::query_nvidia().unwrap_or_else(|e| {
                self.gpu_error = Some(e);
                self.gpu_cache.1.clone()
            });
            let apps = gpu::query_compute_apps().unwrap_or_default();
            self.gpu_cache = (Some(Instant::now()), gpus, apps);
        }
        (self.gpu_cache.1.clone(), self.gpu_cache.2.clone())
    }

    /// Take a snapshot. Only Sentinel's own process is refreshed unless there are jobs to watch.
    pub fn sample(&mut self, machine_id: &str, watch: &[Watch]) -> Vitals {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let me = sysinfo::get_current_pid().ok();
        if watch.is_empty() {
            self.known_pids.clear();
            self.last_full_scan = None;
            if let Some(me) = me {
                self.sys.refresh_processes(ProcessesToUpdate::Some(&[me]));
            }
        } else if self.last_full_scan.map_or(true, |t| t.elapsed() >= Duration::from_secs(3)) {
            self.sys.refresh_processes(ProcessesToUpdate::All);
            self.last_full_scan = Some(Instant::now());
        } else {
            let pids: Vec<Pid> = me.into_iter().chain(self.known_pids.iter().copied()).collect();
            self.sys.refresh_processes(ProcessesToUpdate::Some(&pids));
        }
        let now = Instant::now();
        let proc_dt = now.duration_since(self.last_proc).as_secs_f64().max(0.001);
        self.last_proc = now;
        let (gpus, gpu_apps) = self.gpus();
        let network = self.network_stats(now);
        let sensors = self.sensors();
        let disks = self.disk_stats(now);

        // children index for tree walks
        let mut children: HashMap<Pid, Vec<Pid>> = HashMap::new();
        if !watch.is_empty() {
            for (pid, p) in self.sys.processes() {
                if let Some(parent) = p.parent() {
                    children.entry(parent).or_default().push(*pid);
                }
            }
        }
        let tree = |root: Pid, sys: &System| -> (ProcStat, HashSet<u32>) {
            let mut stack = vec![root];
            let mut seen = HashSet::new();
            let (mut cpu, mut rss, mut n) = (0.0f32, 0u64, 0u32);
            let (mut rd, mut wr) = (0u64, 0u64);
            while let Some(pid) = stack.pop() {
                if !seen.insert(pid.as_u32()) {
                    continue;
                }
                if let Some(p) = sys.process(pid) {
                    cpu += p.cpu_usage();
                    rss += p.memory();
                    let io = p.disk_usage();
                    rd += io.read_bytes;
                    wr += io.written_bytes;
                    n += 1;
                }
                if let Some(kids) = children.get(&pid) {
                    stack.extend(kids.iter().copied());
                }
            }
            (ProcStat { pid: root.as_u32(), cpu_percent: cpu, rss_bytes: rss, processes: n, io_read_bps: rd as f64 / proc_dt, io_write_bps: wr as f64 / proc_dt }, seen)
        };

        let daemon = me.and_then(|me| {
            self.sys.process(me).map(|p| {
                let io = p.disk_usage();
                ProcStat { pid: me.as_u32(), cpu_percent: p.cpu_usage(), rss_bytes: p.memory(), processes: 1, io_read_bps: io.read_bytes as f64 / proc_dt, io_write_bps: io.written_bytes as f64 / proc_dt }
            })
        });
        let mut known: Vec<Pid> = Vec::new();
        let jobs = watch
            .iter()
            .filter(|(_, _, pid)| self.sys.process(Pid::from_u32(*pid)).is_some())
            .map(|(id, name, pid)| {
                let (stat, pids) = tree(Pid::from_u32(*pid), &self.sys);
                known.extend(pids.iter().map(|p| Pid::from_u32(*p)));
                let gpu_mem = if self.nvidia { Some(pids.iter().filter_map(|p| gpu_apps.get(p)).sum()) } else { None };
                JobVitals { job_id: *id, name: name.clone(), stat, gpu_mem_bytes: gpu_mem }
            })
            .collect();

        self.known_pids = known;
        let load = System::load_average();
        Vitals {
            timestamp: chrono::Utc::now(),
            machine_id: machine_id.to_string(),
            cpu_util_percent: self.sys.global_cpu_usage(),
            per_core: self.sys.cpus().iter().map(|c| c.cpu_usage()).collect(),
            cpu_count: self.sys.cpus().len(),
            ram_used_bytes: self.sys.used_memory(),
            ram_total_bytes: self.sys.total_memory(),
            swap_used_bytes: self.sys.used_swap(),
            swap_total_bytes: self.sys.total_swap(),
            load_avg: [load.one, load.five, load.fifteen],
            uptime_secs: System::uptime(),
            gpus,
            daemon,
            jobs,
            cpu_freq_mhz: {
                let f: Vec<u64> = self.sys.cpus().iter().map(|c| c.frequency()).collect();
                if f.iter().all(|&x| x == 0) { Vec::new() } else { f }
            },
            network,
            disks,
            sensors,
        }
    }

    fn network_stats(&mut self, now: Instant) -> Vec<NetStat> {
        let dt = now.duration_since(self.last_net).as_secs_f64().max(0.001);
        self.last_net = now;
        self.networks.refresh();
        let mut v: Vec<NetStat> = self
            .networks
            .iter()
            .filter(|(name, _)| !is_loopback(name))
            .map(|(name, d)| NetStat {
                name: name.clone(),
                rx_bps: d.received() as f64 / dt,
                tx_bps: d.transmitted() as f64 / dt,
                rx_total: d.total_received(),
                tx_total: d.total_transmitted(),
                rx_errors: d.total_errors_on_received(),
                tx_errors: d.total_errors_on_transmitted(),
            })
            // Virtual interfaces that never carried traffic are noise.
            .filter(|n| n.rx_total + n.tx_total > 0)
            .collect();
        v.sort_by(|a, b| (b.rx_total + b.tx_total).cmp(&(a.rx_total + a.tx_total)));
        v
    }

    /// Free space changes slowly, so disks are re-read every 10 s.
    fn disk_stats(&mut self, now: Instant) -> Vec<DiskStat> {
        if now.duration_since(self.last_disks) >= Duration::from_secs(10) {
            self.disks.refresh();
            self.last_disks = now;
        }
        let mut out: Vec<DiskStat> = Vec::new();
        for d in self.disks.iter() {
            let fs = d.file_system().to_string_lossy().to_string();
            let mount = d.mount_point().display().to_string();
            if d.total_space() == 0 || is_pseudo_fs(&fs) || is_noise_mount(&mount) {
                continue;
            }
            let name = d.name().to_string_lossy().to_string();
            // Several mounts of one volume (APFS, btrfs subvolumes): keep the shortest mount point.
            if let Some(e) = out.iter_mut().find(|e| e.name == name && e.total_bytes == d.total_space()) {
                if mount.len() < e.mount.len() {
                    e.mount = mount;
                }
                continue;
            }
            out.push(DiskStat {
                name,
                mount,
                fs,
                kind: match d.kind() {
                    sysinfo::DiskKind::SSD => "ssd",
                    sysinfo::DiskKind::HDD => "hdd",
                    _ => "unknown",
                }
                .into(),
                total_bytes: d.total_space(),
                available_bytes: d.available_space(),
                removable: d.is_removable(),
            });
        }
        out
    }

    fn sensors(&mut self) -> Vec<Sensor> {
        if self.sensor_cache.0.elapsed() < Duration::from_secs(5) {
            return self.sensor_cache.1.clone();
        }
        self.components.refresh();
        let fresh = self.read_sensors();
        self.sensor_cache = (Instant::now(), fresh.clone());
        fresh
    }

    fn read_sensors(&self) -> Vec<Sensor> {
        self.components
            .iter()
            .filter(|c| c.temperature().is_finite() && c.temperature() > 0.0)
            .take(32)
            .map(|c| Sensor {
                label: c.label().to_string(),
                temp_c: c.temperature(),
                max_c: Some(c.max()).filter(|m| m.is_finite() && *m > 0.0),
                critical_c: c.critical().filter(|m| m.is_finite() && *m > 0.0),
            })
            .collect()
    }

    /// The busiest processes system-wide (refreshes every process, so use it only for display).
    pub fn top_processes(&mut self, n: usize) -> Vec<TopProc> {
        self.sys.refresh_processes(ProcessesToUpdate::All);
        let mut v: Vec<TopProc> = self
            .sys
            .processes()
            .iter()
            .map(|(pid, p)| TopProc { pid: pid.as_u32(), name: p.name().to_string_lossy().to_string(), cpu_percent: p.cpu_usage(), rss_bytes: p.memory() })
            .collect();
        v.sort_by(|a, b| b.cpu_percent.partial_cmp(&a.cpu_percent).unwrap_or(std::cmp::Ordering::Equal).then(b.rss_bytes.cmp(&a.rss_bytes)));
        v.truncate(n);
        v
    }
}

fn is_loopback(name: &str) -> bool {
    let n = name.to_lowercase();
    n == "lo" || n == "lo0" || n.starts_with("loopback")
}

fn is_pseudo_fs(fs: &str) -> bool {
    matches!(fs, "tmpfs" | "devtmpfs" | "squashfs" | "overlay" | "ramfs" | "efivarfs" | "devfs" | "autofs" | "proc" | "sysfs" | "cgroup" | "cgroup2" | "fuse.lxcfs")
}

fn is_noise_mount(mount: &str) -> bool {
    // macOS exposes a dozen helper volumes; only `/` and the Data volume matter.
    (mount.starts_with("/System/Volumes/") && mount != "/System/Volumes/Data")
        || mount.starts_with("/boot/efi")
        || mount.starts_with("/snap/")
        || mount.starts_with("/var/lib/docker")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::process::{Command, Stdio};

    #[test]
    fn host_numbers_are_real_and_consistent() {
        let mut m = VitalsMonitor::new();
        m.sample("t", &[]);
        std::thread::sleep(Duration::from_millis(250));
        let v = m.sample("t", &[]);
        assert!(v.ram_total_bytes > 0 && v.ram_used_bytes <= v.ram_total_bytes);
        assert!(v.cpu_count >= 1 && v.per_core.len() == v.cpu_count);
        assert!(v.per_core.iter().all(|c| (0.0..=100.5).contains(c)));
        assert!(v.uptime_secs > 0);
        let me = v.daemon.expect("own process stats");
        assert_eq!(me.pid, std::process::id());
        assert!(me.rss_bytes > 1_000_000, "a running process has more than 1 MB resident");
    }

    #[test]
    fn network_disks_and_io_are_real() {
        let mut m = VitalsMonitor::new();
        m.sample("t", &[]);
        std::thread::sleep(Duration::from_millis(300));
        let v = m.sample("t", &[]);
        assert!(!v.disks.is_empty(), "a machine running tests has at least one disk");
        for d in &v.disks {
            assert!(d.available_bytes <= d.total_bytes && d.total_bytes > 0);
        }
        assert!(v.network.iter().all(|n| n.rx_bps >= 0.0 && n.tx_bps >= 0.0 && !is_loopback(&n.name)));
        let me = v.daemon.unwrap();
        assert!(me.io_read_bps >= 0.0 && me.io_write_bps >= 0.0);
    }

    #[test]
    fn top_processes_are_sorted_and_include_us() {
        let mut m = VitalsMonitor::new();
        m.top_processes(5);
        std::thread::sleep(Duration::from_millis(250));
        let t = m.top_processes(500);
        assert!(!t.is_empty());
        assert!(t.windows(2).all(|w| w[0].cpu_percent >= w[1].cpu_percent));
        assert!(t.iter().any(|p| p.pid == std::process::id()));
    }

    #[test]
    fn noise_filters() {
        assert!(is_loopback("lo0") && is_loopback("Loopback Pseudo-Interface 1") && !is_loopback("eth0"));
        assert!(is_pseudo_fs("tmpfs") && !is_pseudo_fs("apfs") && !is_pseudo_fs("ntfs"));
        assert!(is_noise_mount("/System/Volumes/VM") && !is_noise_mount("/System/Volumes/Data") && !is_noise_mount("/"));
    }

    #[test]
    fn no_gpu_is_reported_as_none_not_invented() {
        let m = VitalsMonitor::new();
        // Whatever is reported must carry no invented metrics unless nvidia-smi supplied them.
        let mut m = m;
        let v = m.sample("t", &[]);
        if !m.has_nvidia() {
            assert!(v.gpus.iter().all(|g| g.gpu_util_percent.is_none() && g.vram_used_bytes.is_none() && g.temperature_c.is_none()));
        }
    }

    #[cfg(unix)]
    #[test]
    fn job_tree_sums_the_shell_and_its_children() {
        // sh -> sleep: two processes, so the tree must count both and attribute memory to the job.
        let mut child = Command::new("sh").arg("-c").arg("sleep 5; true").stdout(Stdio::null()).spawn().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        let mut m = VitalsMonitor::new();
        m.sample("t", &[]);
        let v = m.sample("t", &[(Uuid::new_v4(), "j".into(), child.id())]);
        let _ = child.kill();
        let _ = child.wait();
        assert_eq!(v.jobs.len(), 1);
        assert!(v.jobs[0].stat.processes >= 1);
        assert!(v.jobs[0].stat.rss_bytes > 0);
        assert!(v.jobs[0].gpu_mem_bytes.is_none() || m.has_nvidia());
    }

    #[test]
    fn dead_jobs_are_dropped_not_fabricated() {
        let mut m = VitalsMonitor::new();
        let v = m.sample("t", &[(Uuid::new_v4(), "gone".into(), 4_000_000)]);
        assert!(v.jobs.is_empty());
    }
}
