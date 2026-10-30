//! `sentinel status | top | gpus | doctor`: vitals and environment checks that work on Linux, macOS and Windows.

use sentinel_core::gpu;
use sentinel_core::models::GpuStats;
use sentinel_core::platform::{self, Os};
use sentinel_core::vitals::{TopProc, Vitals, VitalsMonitor};
use std::io::{IsTerminal, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::time::Duration;

/* ---------- formatting ---------- */

struct Style {
    color: bool,
    unicode: bool,
}

impl Style {
    fn detect() -> Self {
        let tty = std::io::stdout().is_terminal();
        let color = tty && std::env::var_os("NO_COLOR").is_none();
        // Legacy Windows consoles (no Windows Terminal) render block characters badly.
        let unicode = !(cfg!(windows) && std::env::var_os("WT_SESSION").is_none());
        Style { color, unicode }
    }
    fn paint(&self, code: &str, s: &str) -> String {
        if self.color { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
    }
    fn bar(&self, pct: f32, width: usize) -> String {
        let pct = pct.clamp(0.0, 100.0);
        let filled = ((pct / 100.0) * width as f32).round() as usize;
        let (on, off) = if self.unicode { ('█', '░') } else { ('#', '-') };
        let b: String = (0..width).map(|i| if i < filled { on } else { off }).collect();
        let code = if pct >= 90.0 { "31" } else if pct >= 70.0 { "33" } else { "32" };
        self.paint(code, &b)
    }
    fn dim(&self, s: &str) -> String {
        self.paint("2", s)
    }
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1024f64.powi(3))
}
fn mb_or_gb(b: u64) -> String {
    if b >= 1024 * 1024 * 1024 { gb(b) } else { format!("{:.0} MB", b as f64 / 1024f64 / 1024f64) }
}
fn opt<T, F: Fn(T) -> String>(v: Option<T>, f: F) -> String {
    v.map(f).unwrap_or_else(|| "n/a".into())
}

fn gpu_lines(st: &Style, gpus: &[GpuStats], note: Option<&str>) -> Vec<String> {
    if gpus.is_empty() {
        let why = note.map(|n| format!(" ({n})")).unwrap_or_default();
        return vec![format!("  gpu      none detected{}", st.dim(&why))];
    }
    gpus.iter()
        .map(|g| {
            let util = g.gpu_util_percent.map(|u| format!("{} {:>3.0}%", st.bar(u, 10), u)).unwrap_or_else(|| "util n/a".into());
            let mem = match (g.vram_used_bytes, g.vram_total_bytes) {
                (Some(u), Some(t)) => format!("{}/{}", gb(u), gb(t)),
                (None, Some(t)) => format!("{} total", gb(t)),
                _ => "mem n/a".into(),
            };
            let power = match (g.power_w, g.power_limit_w) {
                (Some(p), Some(l)) => Some(format!("{p:.0}/{l:.0} W")),
                (Some(p), None) => Some(format!("{p:.0} W")),
                _ => None,
            };
            let clocks = match (g.clock_graphics_mhz, g.clock_memory_mhz) {
                (Some(c), Some(m)) => Some(format!("{c}/{m} MHz")),
                (Some(c), None) => Some(format!("{c} MHz")),
                _ => None,
            };
            let extra = [
                g.temperature_c.map(|t| format!("{t:.0}°C")),
                power,
                clocks,
                g.fan_percent.map(|f| format!("fan {f:.0}%")),
                g.pstate.clone(),
                g.mem_util_percent.map(|m| format!("memctl {m:.0}%")),
                g.driver.clone().map(|d| format!("driver {d}")),
            ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" · ");
            let tail = if g.vendor == "apple" { st.dim("  (macOS exposes no live GPU metrics without root)") } else { String::new() };
            format!("  gpu {:<4} {}  {}  {}  {}{}", g.index, g.name, util, mem, extra, tail)
        })
        .collect()
}

fn rate(bps: f64) -> String {
    let (v, u) = if bps >= 1e9 { (bps / 1e9, "GB/s") } else if bps >= 1e6 { (bps / 1e6, "MB/s") } else if bps >= 1e3 { (bps / 1e3, "KB/s") } else { (bps, "B/s") };
    format!("{v:.1} {u}")
}

fn render(st: &Style, v: &Vitals, sys: &sentinel_core::vitals::SystemInfo, daemon: Option<&DaemonView>, note: Option<&str>, procs: &[TopProc]) -> String {
    let mut o = String::new();
    let ram_pct = v.ram_used_bytes as f32 / v.ram_total_bytes.max(1) as f32 * 100.0;
    o += &format!("  system   {} {} · {} · {} · {} cores · {}\n", sys.os, sys.os_version.clone().unwrap_or_default().trim(), sys.arch, sys.cpu_brand, sys.cpu_count, gb(sys.ram_total_bytes));
    o += &format!("  cpu      {} {:>5.1}%   load {:.2} {:.2} {:.2}\n", st.bar(v.cpu_util_percent, 24), v.cpu_util_percent, v.load_avg[0], v.load_avg[1], v.load_avg[2]);
    let cores: String = v.per_core.iter().map(|c| format!("{:>3.0}", c)).collect::<Vec<_>>().join(" ");
    o += &format!("  cores    {}\n", st.dim(&cores));
    if !v.cpu_freq_mhz.is_empty() {
        let avg = v.cpu_freq_mhz.iter().sum::<u64>() / v.cpu_freq_mhz.len() as u64;
        let max = v.cpu_freq_mhz.iter().max().copied().unwrap_or(0);
        o += &format!("  clock    avg {avg} MHz · max {max} MHz\n");
    }
    o += &format!("  memory   {} {}/{} ({:.0}%)   swap {}/{}\n", st.bar(ram_pct, 24), gb(v.ram_used_bytes), gb(v.ram_total_bytes), ram_pct, gb(v.swap_used_bytes), gb(v.swap_total_bytes));
    for l in gpu_lines(st, &v.gpus, note) {
        o += &l;
        o.push('\n');
    }
    if v.network.is_empty() {
        o += &format!("  network  {}\n", st.dim("no active interfaces"));
    } else {
        let (rx, tx): (f64, f64) = v.network.iter().fold((0.0, 0.0), |a, n| (a.0 + n.rx_bps, a.1 + n.tx_bps));
        o += &format!("  network  ↓ {:<10} ↑ {:<10}", rate(rx), rate(tx));
        let top: Vec<String> = v.network.iter().take(3).map(|n| format!("{} ↓{} ↑{}", n.name, rate(n.rx_bps), rate(n.tx_bps))).collect();
        o += &format!("  {}\n", st.dim(&top.join(" · ")));
    }
    for d in &v.disks {
        let used = d.total_bytes.saturating_sub(d.available_bytes);
        let pct = used as f32 / d.total_bytes.max(1) as f32 * 100.0;
        o += &format!("  disk     {} {}/{} ({:.0}%)  {} {} {}\n", st.bar(pct, 12), gb(used), gb(d.total_bytes), pct, d.mount, st.dim(&d.fs), st.dim(&d.kind));
    }
    if !v.sensors.is_empty() {
        let s: Vec<String> = v.sensors.iter().take(6).map(|s| format!("{} {:.0}°C", s.label, s.temp_c)).collect();
        o += &format!("  temps    {}\n", s.join(" · "));
    }
    if let Some(d) = &v.daemon {
        o += &format!("  sentinel pid {} · {:.1}% cpu · {} rss · io r {} w {}\n", d.pid, d.cpu_percent, mb_or_gb(d.rss_bytes), rate(d.io_read_bps), rate(d.io_write_bps));
    }
    match daemon {
        Some(d) => o += &format!("  daemon   {}\n", st.paint("32", &format!("running on :{} · {} jobs, {} running", d.port, d.total, v.jobs.len()))),
        None => o += &format!("  daemon   {}\n", st.dim("not running (start it with `sentinel dev`)")),
    }
    if !v.jobs.is_empty() {
        o += "\n  JOB                          PID      CPU%     RSS       GPU MEM   DISK R/W\n";
        for j in &v.jobs {
            o += &format!("  {:<28} {:<8} {:<8.0} {:<9} {:<9} {} / {}\n", j.name.chars().take(28).collect::<String>(), j.stat.pid, j.stat.cpu_percent, mb_or_gb(j.stat.rss_bytes), opt(j.gpu_mem_bytes, mb_or_gb), rate(j.stat.io_read_bps), rate(j.stat.io_write_bps));
        }
    }
    if !procs.is_empty() {
        o += "\n  TOP PROCESSES                PID      CPU%     RSS\n";
        for p in procs {
            o += &format!("  {:<28} {:<8} {:<8.1} {}\n", p.name.chars().take(28).collect::<String>(), p.pid, p.cpu_percent, mb_or_gb(p.rss_bytes));
        }
    }
    o
}

/* ---------- talking to a running daemon (std only: it is plain HTTP on loopback) ---------- */

fn http_get(port: u16, path: &str) -> Option<String> {
    let mut s = TcpStream::connect_timeout(&format!("127.0.0.1:{port}").parse().ok()?, Duration::from_millis(400)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    // HTTP/1.0 so the server answers with a plain Content-Length body (no chunked encoding).
    write!(s, "GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n").ok()?;
    let mut raw = String::new();
    s.read_to_string(&mut raw).ok()?;
    let (head, body) = raw.split_once("\r\n\r\n")?;
    head.starts_with("HTTP/1.0 200").then(|| body.to_string()).or_else(|| head.starts_with("HTTP/1.1 200").then(|| body.to_string()))
}

struct DaemonView {
    port: u16,
    total: usize,
}

fn daemon_vitals(port: u16) -> Option<(Vitals, DaemonView)> {
    let v: Vitals = serde_json::from_str(&http_get(port, "/api/vitals")?).ok()?;
    let total = http_get(port, "/api/jobs").and_then(|b| serde_json::from_str::<Vec<serde_json::Value>>(&b).ok()).map_or(0, |j| j.len());
    Some((v, DaemonView { port, total }))
}

/// Prefer the daemon's view (it knows about jobs); otherwise sample locally.
fn snapshot(monitor: &mut VitalsMonitor, port: u16) -> (Vitals, Option<DaemonView>) {
    if let Some((v, d)) = daemon_vitals(port) {
        return (v, Some(d));
    }
    monitor.sample("local", &[]);
    std::thread::sleep(Duration::from_millis(300)); // CPU usage needs two samples to mean anything
    (monitor.sample("local", &[]), None)
}

/* ---------- commands ---------- */

pub fn gpus(json: bool) -> anyhow::Result<()> {
    let mut m = VitalsMonitor::new();
    let list = m.sample("local", &[]).gpus;
    if json {
        println!("{}", serde_json::to_string_pretty(&list)?);
        return Ok(());
    }
    let st = Style::detect();
    let info = m.system_info();
    for l in gpu_lines(&st, &list, info.gpu_note.as_deref()) {
        println!("{l}");
    }
    if list.is_empty() {
        println!("  {}", install_hint());
    }
    Ok(())
}

pub fn status(json: bool, port: u16) -> anyhow::Result<()> {
    let mut m = VitalsMonitor::new();
    let (v, d) = snapshot(&mut m, port);
    if json {
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    let st = Style::detect();
    let info = m.system_info();
    println!("{}", st.paint("1", "sentinel status"));
    print!("{}", render(&st, &v, &info, d.as_ref(), info.gpu_note.as_deref(), &[]));
    Ok(())
}

pub async fn top(interval: f32, port: u16, procs: usize) -> anyhow::Result<()> {
    let st = Style::detect();
    let mut m = VitalsMonitor::new();
    let info = m.system_info();
    let interval = Duration::from_secs_f32(interval.clamp(0.25, 30.0));
    let interactive = std::io::stdout().is_terminal();
    loop {
        let (v, d) = tokio::task::block_in_place(|| snapshot(&mut m, port));
        if interactive {
            print!("\x1b[2J\x1b[H");
        }
        println!("{}  {}", st.paint("1", "sentinel top"), st.dim(&format!("every {:.1}s · Ctrl-C to quit", interval.as_secs_f32())));
        let top = tokio::task::block_in_place(|| m.top_processes(procs));
        print!("{}", render(&st, &v, &info, d.as_ref(), info.gpu_note.as_deref(), &top));
        std::io::stdout().flush().ok();
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = tokio::signal::ctrl_c() => { println!(); return Ok(()); }
        }
        if !interactive {
            println!("---");
        }
    }
}

fn install_hint() -> &'static str {
    match Os::current() {
        Os::Windows => "NVIDIA GPU? Install the driver from nvidia.com/drivers (it provides nvidia-smi).",
        Os::Linux => "NVIDIA GPU? Install the driver (e.g. `sudo ubuntu-drivers install`) so `nvidia-smi` works.",
        Os::MacOs => "Apple Silicon uses Metal (PyTorch `mps` device); NVIDIA GPUs are not supported on macOS.",
        Os::Other => "GPU telemetry needs nvidia-smi.",
    }
}

enum Level {
    Ok,
    Warn,
    Fail,
}

fn out(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).output().ok()?;
    o.status.success().then(|| format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)).trim().to_string())
}

pub fn doctor(port: u16) -> anyhow::Result<()> {
    let st = Style::detect();
    let mut failed = false;
    let mut line = |lvl: Level, name: &str, detail: String, fix: Option<String>| {
        let (sym, code) = match lvl {
            Level::Ok => ("✔", "32"),
            Level::Warn => ("⚠", "33"),
            Level::Fail => {
                failed = true;
                ("✖", "31")
            }
        };
        let sym = if st.unicode { sym } else { match sym { "✔" => "ok", "⚠" => "!!", _ => "XX" } };
        println!("  {} {:<14} {}", st.paint(code, sym), name, detail);
        if let Some(f) = fix {
            println!("      {}", st.dim(&format!("→ {f}")));
        }
    };

    println!("{}", st.paint("1", "sentinel doctor"));
    let os = Os::current();
    line(Level::Ok, "platform", format!("{} / {}", os.name(), std::env::consts::ARCH), None);

    // Python
    let py = platform::python_exe();
    let py_fix = match os {
        Os::Windows => "install Python 3.10+ from python.org (tick \"Add to PATH\"), or set SENTINEL_PYTHON",
        Os::MacOs => "brew install python@3.12, or set SENTINEL_PYTHON",
        _ => "install python3 (e.g. `sudo apt install python3 python3-pip`), or set SENTINEL_PYTHON",
    };
    match out(&py, &["--version"]) {
        Some(v) => {
            let ok = v.split_whitespace().nth(1).and_then(|n| n.split('.').nth(1)).and_then(|m| m.parse::<u32>().ok()).map_or(true, |minor| minor >= 9);
            line(if ok { Level::Ok } else { Level::Fail }, "python", format!("{py}: {v}"), (!ok).then(|| "Sentinel needs Python 3.9 or newer".to_string()));
        }
        None => line(Level::Fail, "python", format!("`{py}` not found"), Some(py_fix.into())),
    }

    // PyTorch (and which accelerator it can use)
    let probe = "import torch;print(torch.__version__);print(torch.cuda.is_available());print(bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()))";
    match out(&py, &["-c", probe]) {
        Some(o) => {
            let l: Vec<&str> = o.lines().collect();
            let accel = if l.get(1) == Some(&"True") { "CUDA available" } else if l.get(2) == Some(&"True") { "Apple MPS available" } else { "CPU only" };
            line(Level::Ok, "pytorch", format!("{} ({accel})", l.first().unwrap_or(&"?")), None);
        }
        None => line(Level::Warn, "pytorch", "not importable".into(), Some("pip install torch (GPU builds: pytorch.org/get-started)".into())),
    }
    line(
        if out("uv", &["--version"]).is_some() { Level::Ok } else { Level::Warn },
        "uv",
        out("uv", &["--version"]).unwrap_or_else(|| "not found (optional; used by sweeps)".into()),
        out("uv", &["--version"]).is_none().then(|| "https://docs.astral.sh/uv/getting-started/installation/".to_string()),
    );

    // GPU
    match gpu::query_nvidia() {
        Ok(g) if !g.is_empty() => {
            let names = g.iter().map(|x| x.name.clone()).collect::<Vec<_>>().join(", ");
            line(Level::Ok, "nvidia-smi", format!("{} GPU(s): {names} · driver {}", g.len(), g[0].driver.clone().unwrap_or_default()), None);
        }
        Ok(_) => line(Level::Warn, "nvidia-smi", "ran but listed no GPUs".into(), None),
        Err(e) => {
            let other = gpu::detect_apple().into_iter().chain(gpu::detect_other()).collect::<Vec<_>>();
            if let Some(n) = other.iter().find(|g| g.vendor == "nvidia") {
                line(Level::Warn, "nvidia-smi", format!("{} found but nvidia-smi failed: {e}", n.name), Some(install_hint().into()));
            } else if let Some(g) = other.first() {
                line(Level::Ok, "gpu", format!("{} (no NVIDIA telemetry; {})", g.name, if g.vendor == "apple" { "Metal/MPS" } else { "CPU training" }), None);
            } else {
                line(Level::Ok, "gpu", "no GPU detected: training runs on CPU".into(), None);
            }
        }
    }

    // Project
    line(
        if std::path::Path::new("sentinel.lock").exists() { Level::Ok } else { Level::Warn },
        "project",
        if std::path::Path::new("sentinel.lock").exists() { "sentinel.lock found".into() } else { "not a Sentinel project here".into() },
        (!std::path::Path::new("sentinel.lock").exists()).then(|| "run `sentinel init`".to_string()),
    );
    match std::fs::create_dir_all(".sentinel").and_then(|_| std::fs::write(".sentinel/.doctor", "ok")) {
        Ok(_) => {
            let _ = std::fs::remove_file(".sentinel/.doctor");
            line(Level::Ok, "write access", ".sentinel/ is writable".into(), None);
        }
        Err(e) => line(Level::Fail, "write access", format!("cannot write .sentinel/: {e}"), Some("run from a directory you can write to".into())),
    }
    let mut reg = sentinel_core::components::ComponentRegistry::new();
    let custom = reg.load_from_dir(std::path::Path::new("components")).unwrap_or(0);
    line(Level::Ok, "components", format!("{} available ({custom} from ./components)", reg.list().len()), None);

    // Daemon port
    if http_get(port, "/api/health").is_some_and(|b| b.contains("sentinel-daemon")) {
        line(Level::Ok, "daemon", format!("Sentinel is running on :{port}"), None);
    } else if TcpListener::bind(("127.0.0.1", port)).is_ok() {
        line(Level::Ok, "port", format!(":{port} is free"), None);
    } else {
        line(Level::Warn, "port", format!(":{port} is in use by something else"), Some(format!("use `sentinel dev --port <other>`")));
    }

    println!();
    if failed {
        println!("  {}", st.paint("31", "Some checks failed."));
        std::process::exit(1);
    }
    println!("  {}", st.paint("32", "Ready."));
    Ok(())
}
