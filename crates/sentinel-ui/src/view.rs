//! HTML & WebAssembly Dashboard View Generator.
//! Ultra-lightweight: zero external framework overhead, inline vanilla JS idle controller.

use crate::state::DashboardState;
use sentinel_core::models::{HardwareStats, JobRecord};
use sentinel_core::templates::Template;

pub struct DashboardView;

impl DashboardView {
    /// Render complete standalone single-page dashboard HTML.
    pub fn render_html(
        state: &DashboardState,
        stats: Option<&HardwareStats>,
        jobs: &[JobRecord],
        templates: &[Template],
    ) -> String {
        let mode_badge = match state.mode {
            crate::state::DashboardMode::ReadOnly => {
                r#"<span class="badge badge-readonly">🔒 READ-ONLY (Safe Mode)</span>"#
            }
            crate::state::DashboardMode::Editing => {
                r#"<span class="badge badge-editing">✏️ EDITING (Active Mode)</span>"#
            }
            crate::state::DashboardMode::Idle => {
                r#"<span class="badge badge-idle">🌙 IDLE (Low-Power Mode)</span>"#
            }
        };

        let cpu_percent = stats.map(|s| s.cpu_util_percent).unwrap_or(0.0);
        let ram_gb_used = stats
            .map(|s| s.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);
        let ram_gb_total = stats
            .map(|s| s.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(1.0);

        let gpus_html = if let Some(s) = stats {
            s.gpus
                .iter()
                .map(|gpu| {
                    let vram_used_gb = gpu.vram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    let vram_total_gb = gpu.vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    format!(
                        r#"<div class="card stat-card">
                            <div class="stat-title">GPU {}: {}</div>
                            <div class="stat-value">{:.1}% <span class="subtext">({:.1}°C)</span></div>
                            <div class="progress-bar"><div class="progress-fill" style="width: {:.1}%"></div></div>
                            <div class="stat-footer">VRAM: {:.2} GB / {:.2} GB</div>
                        </div>"#,
                        gpu.index, gpu.name, gpu.gpu_util_percent, gpu.temperature_c, gpu.gpu_util_percent, vram_used_gb, vram_total_gb
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            r#"<div class="stat-card">No GPU Telemetry Available</div>"#.to_string()
        };

        let jobs_html = if jobs.is_empty() {
            r#"<tr><td colspan="5" class="empty-row">No jobs submitted yet. Run a template or script to start.</td></tr>"#
                .to_string()
        } else {
            jobs.iter()
                .map(|j| {
                    let status_class = match j.status {
                        sentinel_core::models::JobStatus::Completed => "status-completed",
                        sentinel_core::models::JobStatus::Running => "status-running",
                        sentinel_core::models::JobStatus::Failed => "status-failed",
                        _ => "status-pending",
                    };
                    format!(
                        r#"<tr>
                            <td class="font-mono">{}</td>
                            <td>{}</td>
                            <td><span class="status-pill {}">{:?}</span></td>
                            <td>PID: {}</td>
                            <td><button class="btn-sm" onclick="stopJob('{}')">Stop</button></td>
                        </tr>"#,
                        &j.spec.id.to_string()[..8],
                        j.spec.name,
                        status_class,
                        j.status,
                        j.pid.unwrap_or(0),
                        j.spec.id
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let templates_html = templates
            .iter()
            .map(|t| {
                format!(
                    r#"<div class="template-card">
                        <div class="template-header">
                            <span class="template-domain">[{}]</span>
                            <strong>{}</strong>
                        </div>
                        <div class="template-id font-mono">{}</div>
                        <p class="template-desc">{}</p>
                        <div class="template-actions">
                            <button class="btn-sm" onclick="inspectTemplate('{}')">Inspect</button>
                            <button class="btn-sm btn-primary" onclick="launchTemplate('{}')">Run</button>
                        </div>
                    </div>"#,
                    t.domain, t.name, t.template_id, t.description, t.template_id, t.template_id
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel — ML Training Supervisor</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --primary: #238636;
            --accent: #58a6ff;
            --warning: #d29922;
            --danger: #f85149;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .logo {{ font-size: 20px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }}
        .badge {{ padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
        .badge-readonly {{ background: #1f242c; color: var(--text-muted); border: 1px solid var(--border); }}
        .badge-editing {{ background: #238636; color: #fff; }}
        .badge-idle {{ background: #382c16; color: var(--warning); border: 1px solid var(--warning); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
        .stat-title {{ font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 8px; }}
        .subtext {{ font-size: 14px; font-weight: 400; color: var(--text-muted); }}
        .progress-bar {{ background: var(--border); height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 8px; }}
        .progress-fill {{ background: var(--accent); height: 100%; }}
        .stat-footer {{ font-size: 12px; color: var(--text-muted); }}
        .section-title {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #fff; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }}
        th {{ color: var(--text-muted); font-weight: 600; }}
        .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
        .status-pill {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
        .status-completed {{ background: #1f3724; color: #3fb950; }}
        .status-running {{ background: #162a45; color: #58a6ff; }}
        .status-failed {{ background: #391c1d; color: #f85149; }}
        .status-pending {{ background: #262c36; color: var(--text-muted); }}
        .btn-sm {{ background: var(--surface); color: var(--text); border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
        .btn-sm:hover {{ background: #21262d; }}
        .btn-primary {{ background: var(--primary); color: #fff; border: none; }}
        .btn-primary:hover {{ background: #2ea043; }}
        .template-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
        .template-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
        .template-header {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }}
        .template-domain {{ font-size: 11px; text-transform: uppercase; color: var(--accent); font-weight: 600; }}
        .template-id {{ font-size: 11px; color: var(--text-muted); margin-bottom: 8px; }}
        .template-desc {{ font-size: 13px; color: var(--text); margin-bottom: 12px; }}
        .template-actions {{ display: flex; gap: 8px; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🛡️ Sentinel ML Supervisor</div>
        <div id="mode-badge-container">{mode_badge}</div>
    </div>

    <div class="section-title">Hardware Telemetry</div>
    <div class="grid">
        <div class="card stat-card">
            <div class="stat-title">Host CPU Utilization</div>
            <div class="stat-value">{cpu_percent:.1}%</div>
            <div class="progress-bar"><div class="progress-fill" style="width: {cpu_percent:.1}%"></div></div>
            <div class="stat-footer">System Cores Active</div>
        </div>
        <div class="card stat-card">
            <div class="stat-title">System RAM Used</div>
            <div class="stat-value">{ram_gb_used:.2} GB <span class="subtext">/ {ram_gb_total:.1} GB</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width: {ram_used_pct:.1}%"></div></div>
            <div class="stat-footer">Free Memory Monitored</div>
        </div>
        {gpus_html}
    </div>

    <div class="section-title">Job Supervisor</div>
    <div class="card" style="margin-bottom: 24px; overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th>Job ID</th>
                    <th>Name</th>
                    <th>Status</th>
                    <th>PID</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {jobs_html}
            </tbody>
        </table>
    </div>

    <div class="section-title">Template Registry</div>
    <div class="template-grid">
        {templates_html}
    </div>

    <script>
        // Idle detection: drop to low-power read-only after 15s of inactivity
        let lastInteraction = Date.now();
        ['click', 'keydown', 'mousemove', 'scroll'].forEach(evt => {{
            window.addEventListener(evt, () => {{
                lastInteraction = Date.now();
            }}, {{ passive: true }});
        }});

        function checkIdle() {{
            const idle = Date.now() - lastInteraction > 15000;
            const badge = document.getElementById('mode-badge-container');
            if (idle && badge) {{
                badge.innerHTML = '<span class="badge badge-idle">🌙 IDLE (Low-Power Mode)</span>';
            }}
        }}
        setInterval(checkIdle, 2000);

        function stopJob(jobId) {{
            fetch('/api/jobs/' + jobId + '/stop', {{ method: 'POST' }})
                .then(r => r.json())
                .then(() => location.reload())
                .catch(e => console.error(e));
        }}

        function inspectTemplate(templateId) {{
            alert('Template: ' + templateId + ' (Read-Only Mode)');
        }}

        function launchTemplate(templateId) {{
            alert('Launching template: ' + templateId);
        }}
    </script>
</body>
</html>"#,
            mode_badge = mode_badge,
            cpu_percent = cpu_percent,
            ram_gb_used = ram_gb_used,
            ram_gb_total = ram_gb_total,
            ram_used_pct = if ram_gb_total > 0.0 {
                (ram_gb_used / ram_gb_total) * 100.0
            } else {
                0.0
            },
            gpus_html = gpus_html,
            jobs_html = jobs_html,
            templates_html = templates_html
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_html_contains_critical_sections() {
        let state = DashboardState::new();
        let html = DashboardView::render_html(&state, None, &[], &[]);

        assert!(html.contains("Sentinel ML Supervisor"));
        assert!(html.contains("READ-ONLY (Safe Mode)"));
        assert!(html.contains("Hardware Telemetry"));
        assert!(html.contains("Job Supervisor"));
        assert!(html.contains("Template Registry"));
        assert!(html.contains("checkIdle"));
    }
}
