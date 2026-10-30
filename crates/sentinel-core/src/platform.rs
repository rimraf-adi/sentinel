//! One place that knows how Linux, macOS and Windows differ, so callers never hand-pick commands.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Linux,
    MacOs,
    Windows,
    Other,
}

impl Os {
    pub const fn current() -> Os {
        if cfg!(target_os = "linux") {
            Os::Linux
        } else if cfg!(target_os = "macos") {
            Os::MacOs
        } else if cfg!(target_os = "windows") {
            Os::Windows
        } else {
            Os::Other
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Os::Linux => "linux",
            Os::MacOs => "macos",
            Os::Windows => "windows",
            Os::Other => "other",
        }
    }
}

/// The Python interpreter to launch: `python3` on Unix, `python` on Windows (where `python3` is usually the
/// Microsoft Store stub). `SENTINEL_PYTHON` overrides both.
pub fn python_exe() -> String {
    if let Ok(p) = std::env::var("SENTINEL_PYTHON") {
        if !p.trim().is_empty() {
            return p;
        }
    }
    if cfg!(windows) { "python".into() } else { "python3".into() }
}

/// Quote one argument for the platform shell (`sh` single quotes, `cmd` double quotes).
pub fn shell_quote(arg: &str) -> String {
    if cfg!(windows) {
        format!("\"{}\"", arg.replace('"', "\\\""))
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

/// Command line that runs a Python script with the right interpreter and quoting for this OS.
pub fn python_script_command(script: &Path) -> String {
    format!("{} {}", python_exe(), shell_quote(&script.display().to_string()))
}

/// Program and arguments that run `command` through the platform shell.
pub fn shell_invocation(command: &str) -> (&'static str, Vec<String>) {
    if cfg!(windows) {
        ("cmd", vec!["/C".into(), command.into()])
    } else {
        ("sh", vec!["-c".into(), command.into()])
    }
}

/// The user's home directory (`HOME`, or `USERPROFILE` on Windows).
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
}

/// Candidate locations for `nvidia-smi`: PATH first, then where each OS installs it when PATH is incomplete.
pub fn nvidia_smi_candidates() -> Vec<String> {
    let mut v = vec!["nvidia-smi".to_string()];
    match Os::current() {
        Os::Windows => {
            v.push(r"C:\Windows\System32\nvidia-smi.exe".into());
            v.push(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe".into());
        }
        Os::Linux => {
            v.push("/usr/bin/nvidia-smi".into());
            v.push("/usr/local/nvidia/bin/nvidia-smi".into());
            v.push("/usr/lib/wsl/lib/nvidia-smi".into()); // WSL2 exposes the Windows driver here
        }
        _ => {}
    }
    v
}

/// Ask the OS to stop a process tree: `taskkill /T /F` on Windows, SIGTERM elsewhere.
pub fn terminate_process(pid: u32) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    #[cfg(unix)]
    {
        // SAFETY: plain signal delivery to a pid we spawned; failure (already exited) is ignored.
        unsafe {
            libc_kill(pid as i32);
        }
    }
}

#[cfg(unix)]
unsafe fn libc_kill(pid: i32) {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    const SIGTERM: i32 = 15;
    kill(pid, SIGTERM);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_the_host_os() {
        let os = Os::current();
        assert_ne!(os, Os::Other);
        assert_eq!(os == Os::Windows, cfg!(windows));
        assert_eq!(os.name().is_empty(), false);
    }

    #[test]
    fn quoting_protects_spaces_and_quotes() {
        let q = shell_quote("my dir/it's.py");
        if cfg!(windows) {
            assert!(q.starts_with('"') && q.ends_with('"'));
        } else {
            assert_eq!(q, r"'my dir/it'\''s.py'");
        }
    }

    #[test]
    fn script_command_uses_the_right_interpreter() {
        let c = python_script_command(Path::new("a b/run.py"));
        assert!(c.starts_with(&python_exe()) && c.contains("run.py"));
        if !cfg!(windows) {
            assert!(c.starts_with("python3 ") || std::env::var("SENTINEL_PYTHON").is_ok());
        }
    }

    #[test]
    fn shell_invocation_matches_os() {
        let (prog, args) = shell_invocation("echo hi");
        assert_eq!(prog, if cfg!(windows) { "cmd" } else { "sh" });
        assert_eq!(args[1], "echo hi");
    }

    #[test]
    fn nvidia_candidates_start_with_path_lookup() {
        let c = nvidia_smi_candidates();
        assert_eq!(c[0], "nvidia-smi");
        assert!(Os::current() != Os::Windows || c.iter().any(|p| p.ends_with("nvidia-smi.exe")));
    }

    #[test]
    fn terminating_a_real_process_works_on_this_os() {
        let mut child = if cfg!(windows) {
            std::process::Command::new("cmd").args(["/C", "ping -n 30 127.0.0.1 >nul"]).spawn().unwrap()
        } else {
            std::process::Command::new("sleep").arg("30").spawn().unwrap()
        };
        terminate_process(child.id());
        let status = child.wait().unwrap();
        assert!(!status.success(), "terminated process must not exit cleanly");
    }
}
