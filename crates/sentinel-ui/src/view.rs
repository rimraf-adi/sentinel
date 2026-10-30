//! HTML & WebAssembly Dashboard View Generator.
//! IDE / Code view is strictly READ-ONLY. Everything else (configs, templates, jobs, sweeps) is EDITABLE.

use crate::state::DashboardState;
use sentinel_core::models::{HardwareStats, JobRecord};
use sentinel_core::templates::Template;

pub struct DashboardView;

impl DashboardView {
    /// Render complete standalone single-page dashboard HTML.
    pub fn render_html(
        _state: &DashboardState,
        stats: Option<&HardwareStats>,
        jobs: &[JobRecord],
        templates: &[Template],
    ) -> String {
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
            r#"<tr id="empty-jobs-row"><td colspan="6" class="empty-row">No jobs submitted yet. Select a template to launch an experiment.</td></tr>"#
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
                            <td><strong>{}</strong></td>
                            <td><span class="status-pill {}">{:?}</span></td>
                            <td>PID: {}</td>
                            <td>{}</td>
                            <td>
                                <button class="btn-sm" onclick="viewLogs('{}')">Logs</button>
                                <button class="btn-sm btn-danger" onclick="stopJob('{}')">Stop</button>
                            </td>
                        </tr>"#,
                        &j.spec.id.to_string()[..8],
                        j.spec.name,
                        status_class,
                        j.status,
                        j.pid.unwrap_or(0),
                        j.spec.template_id.as_deref().unwrap_or("Custom"),
                        j.spec.id,
                        j.spec.id
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Templates rendered with editable form previews and interactive launch modals
        let templates_json = serde_json::to_string(templates).unwrap_or_else(|_| "[]".to_string());

        let templates_cards_html = templates
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
                        <div class="param-summary">
                            <span>{} editable params</span> • <span>{} locked 🔒</span>
                        </div>
                        <div class="template-actions">
                            <button class="btn-sm btn-primary" onclick="openTemplateEditor('{}')">⚙️ Configure & Launch</button>
                        </div>
                    </div>"#,
                    t.domain, t.name, t.template_id, t.description, t.parameters.len(), t.locked.len(), t.template_id
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
            --surface-hover: #1f242c;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --primary: #238636;
            --primary-hover: #2ea043;
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
            margin-bottom: 20px;
        }}
        .logo {{ font-size: 20px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }}
        .nav-tabs {{ display: flex; gap: 8px; }}
        .nav-tab {{
            background: transparent;
            color: var(--text-muted);
            border: 1px solid transparent;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .nav-tab:hover {{ background: var(--surface); color: var(--text); }}
        .nav-tab.active {{ background: var(--surface); color: #fff; border-color: var(--border); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .badge {{ padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
        .badge-readonly {{ background: #1f242c; color: var(--text-muted); border: 1px solid var(--border); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
        .stat-title {{ font-size: 13px; color: var(--text-muted); margin-bottom: 6px; }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 8px; }}
        .subtext {{ font-size: 14px; font-weight: 400; color: var(--text-muted); }}
        .progress-bar {{ background: var(--border); height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 8px; }}
        .progress-fill {{ background: var(--accent); height: 100%; }}
        .stat-footer {{ font-size: 12px; color: var(--text-muted); }}
        .section-title {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #fff; display: flex; justify-content: space-between; align-items: center; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }}
        th {{ color: var(--text-muted); font-weight: 600; }}
        .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
        .status-pill {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
        .status-completed {{ background: #1f3724; color: #3fb950; }}
        .status-running {{ background: #162a45; color: #58a6ff; }}
        .status-failed {{ background: #391c1d; color: #f85149; }}
        .status-pending {{ background: #262c36; color: var(--text-muted); }}
        .btn-sm {{ background: var(--surface); color: var(--text); border: 1px solid var(--border); padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500; }}
        .btn-sm:hover {{ background: var(--surface-hover); }}
        .btn-primary {{ background: var(--primary); color: #fff; border: none; }}
        .btn-primary:hover {{ background: var(--primary-hover); }}
        .btn-danger {{ background: #391c1d; color: var(--danger); border-color: #532223; }}
        .btn-danger:hover {{ background: #4e2527; }}
        .template-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
        .template-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
        .template-header {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }}
        .template-domain {{ font-size: 11px; text-transform: uppercase; color: var(--accent); font-weight: 600; }}
        .template-id {{ font-size: 11px; color: var(--text-muted); margin-bottom: 8px; }}
        .template-desc {{ font-size: 13px; color: var(--text); margin-bottom: 12px; }}
        .param-summary {{ font-size: 12px; color: var(--text-muted); margin-bottom: 14px; }}
        .template-actions {{ display: flex; gap: 8px; }}

        /* Modal & Form Styles (Editable) */
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 100; justify-content: center; align-items: center; }}
        .modal-content {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; width: 550px; max-width: 90%; max-height: 85vh; overflow-y: auto; padding: 24px; }}
        .modal-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }}
        .form-group {{ margin-bottom: 14px; }}
        .form-label {{ display: block; font-size: 12px; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
        .form-desc {{ font-size: 11px; color: var(--text-muted); margin-bottom: 6px; }}
        .form-input {{ width: 100%; background: var(--bg); border: 1px solid var(--border); color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
        .form-input:focus {{ border-color: var(--accent); outline: none; }}
        .locked-param {{ background: #191f28; border: 1px solid var(--border); padding: 8px 12px; border-radius: 6px; font-size: 12px; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; }}

        /* IDE View (STRICTLY READ-ONLY) */
        .ide-container {{ display: flex; height: 600px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--bg); }}
        .ide-sidebar {{ width: 240px; border-right: 1px solid var(--border); background: var(--surface); padding: 12px; overflow-y: auto; }}
        .ide-sidebar-title {{ font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }}
        .file-item {{ padding: 6px 10px; border-radius: 4px; font-size: 13px; font-family: ui-monospace, monospace; cursor: pointer; color: var(--text); margin-bottom: 2px; }}
        .file-item:hover {{ background: var(--surface-hover); }}
        .file-item.active {{ background: #1f2d3d; color: var(--accent); }}
        .ide-main {{ flex: 1; display: flex; flex-direction: column; background: var(--bg); }}
        .ide-toolbar {{ padding: 8px 16px; border-bottom: 1px solid var(--border); background: var(--surface); display: flex; justify-content: space-between; align-items: center; }}
        .ide-banner {{ background: #16202c; border: 1px solid #203348; color: var(--accent); padding: 8px 14px; border-radius: 6px; font-size: 12px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
        .code-area {{ flex: 1; width: 100%; background: transparent; border: none; color: #e6edf3; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; padding: 16px; resize: none; outline: none; line-height: 1.5; white-space: pre; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🛡️ Sentinel ML Supervisor</div>
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('overview')">📊 Control Room</button>
            <button class="nav-tab" onclick="switchTab('templates')">⚙️ Templates (Editable)</button>
            <button class="nav-tab" onclick="switchTab('ide')">💻 Code View (Read-Only)</button>
        </div>
    </div>

    <!-- TAB 1: OVERVIEW & CONTROL ROOM -->
    <div id="tab-overview" class="tab-content active">
        <div class="section-title">Hardware Telemetry (Live NVML / System)</div>
        <div class="grid">
            <div class="card stat-card">
                <div class="stat-title">Host CPU Utilization</div>
                <div class="stat-value">{cpu_percent:.1}%</div>
                <div class="progress-bar"><div class="progress-fill" style="width: {cpu_percent:.1}%"></div></div>
                <div class="stat-footer">System Cores Monitored</div>
            </div>
            <div class="card stat-card">
                <div class="stat-title">System RAM Used</div>
                <div class="stat-value">{ram_gb_used:.2} GB <span class="subtext">/ {ram_gb_total:.1} GB</span></div>
                <div class="progress-bar"><div class="progress-fill" style="width: {ram_used_pct:.1}%"></div></div>
                <div class="stat-footer">Allocated Host Memory</div>
            </div>
            {gpus_html}
        </div>

        <div class="section-title">
            <span>Process Supervisor</span>
            <button class="btn-sm btn-primary" onclick="switchTab('templates')">+ New Run</button>
        </div>
        <div class="card" style="margin-bottom: 24px; overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Job ID</th>
                        <th>Name</th>
                        <th>Status</th>
                        <th>PID</th>
                        <th>Template</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="jobs-table-body">
                    {jobs_html}
                </tbody>
            </table>
        </div>
    </div>

    <!-- TAB 2: TEMPLATES & CONFIGS (EDITABLE) -->
    <div id="tab-templates" class="tab-content">
        <div class="section-title">
            <span>Template Catalog & Recipe Configurator</span>
            <span class="subtext">Editable hyperparameters & launch parameters</span>
        </div>
        <div class="template-grid">
            {templates_cards_html}
        </div>
    </div>

    <!-- TAB 3: IDE & CODE VIEW (STRICTLY READ-ONLY) -->
    <div id="tab-ide" class="tab-content">
        <div class="ide-banner">
            🔒 <strong>Read-Only Code View</strong> — Ultra-low latency and zero RAM bloat. Code cannot be edited in the browser; use your local IDE for heavy development.
        </div>
        <div class="ide-container">
            <div class="ide-sidebar">
                <div class="ide-sidebar-title">Project Files</div>
                <div id="file-list">
                    <div class="file-item active" onclick="loadFile('recipes/vision_train.py')">recipes/vision_train.py</div>
                    <div class="file-item" onclick="loadFile('recipes/sweep_orchestrator.py')">recipes/sweep_orchestrator.py</div>
                    <div class="file-item" onclick="loadFile('templates/vision/image-classification.yaml')">templates/vision/image-classification.yaml</div>
                    <div class="file-item" onclick="loadFile('sentinel.yaml')">sentinel.yaml</div>
                    <div class="file-item" onclick="loadFile('pyproject.toml')">pyproject.toml</div>
                    <div class="file-item" onclick="loadFile('sentinel.md')">sentinel.md</div>
                </div>
            </div>
            <div class="ide-main">
                <div class="ide-toolbar">
                    <span id="active-file-name" class="font-mono">recipes/vision_train.py (Read-Only)</span>
                    <button class="btn-sm" onclick="copyCode()">📋 Copy Code</button>
                </div>
                <textarea id="code-viewer" class="code-area" readonly spellcheck="false"></textarea>
            </div>
        </div>
    </div>

    <!-- INTERACTIVE TEMPLATE CONFIGURATOR MODAL (EDITABLE) -->
    <div id="editor-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div>
                    <h3 id="modal-template-name">Configure Recipe</h3>
                    <div id="modal-template-id" class="font-mono subtext"></div>
                </div>
                <button class="btn-sm" onclick="closeModal()">✕</button>
            </div>
            <div id="modal-params-container">
                <!-- Dynamically populated with editable inputs -->
            </div>
            <div style="margin-top: 20px; display: flex; justify-content: flex-end; gap: 8px;">
                <button class="btn-sm" onclick="closeModal()">Cancel</button>
                <button class="btn-sm btn-primary" onclick="launchEditedRun()">🚀 Launch Experiment</button>
            </div>
        </div>
    </div>

    <!-- LIVE LOG DRAWER MODAL -->
    <div id="logs-modal" class="modal">
        <div class="modal-content" style="width: 750px;">
            <div class="modal-header">
                <h3 id="logs-modal-title">Job Execution Logs</h3>
                <button class="btn-sm" onclick="closeLogsModal()">✕</button>
            </div>
            <textarea id="logs-viewer" class="code-area" style="height: 400px; background: #000; color: #58a6ff;" readonly></textarea>
        </div>
    </div>

    <script>
        const templates = {templates_json};
        let activeTemplate = null;

        function switchTab(tabId) {{
            document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');

            if (tabId === 'ide') {{
                loadFile('recipes/vision_train.py');
            }}
        }}

        function openTemplateEditor(templateId) {{
            const t = templates.find(x => x.template_id === templateId);
            if (!t) return;
            activeTemplate = t;

            document.getElementById('modal-template-name').textContent = t.name;
            document.getElementById('modal-template-id').textContent = t.template_id;

            const container = document.getElementById('modal-params-container');
            container.innerHTML = '';

            // Render editable parameters
            Object.entries(t.parameters).forEach(([key, spec]) => {{
                const group = document.createElement('div');
                group.className = 'form-group';

                const label = document.createElement('label');
                label.className = 'form-label';
                label.textContent = key;

                const desc = document.createElement('div');
                desc.className = 'form-desc';
                desc.textContent = spec.help || '';

                let input;
                if (spec.options && spec.options.length > 0) {{
                    input = document.createElement('select');
                    input.className = 'form-input';
                    spec.options.forEach(opt => {{
                        const option = document.createElement('option');
                        option.value = opt;
                        option.textContent = opt;
                        if (opt === spec.default) option.selected = true;
                        input.appendChild(option);
                    }});
                }} else {{
                    input = document.createElement('input');
                    input.className = 'form-input';
                    input.type = spec.type === 'int' || spec.type === 'float' ? 'number' : 'text';
                    if (spec.type === 'float') input.step = '0.0001';
                    input.value = spec.default;
                    if (spec.range) {{
                        input.min = spec.range[0];
                        input.max = spec.range[1];
                    }}
                }}
                input.id = 'input-' + key;

                group.appendChild(label);
                group.appendChild(desc);
                group.appendChild(input);
                container.appendChild(group);
            }});

            // Render locked parameters (display only)
            if (Object.keys(t.locked).length > 0) {{
                const lockedTitle = document.createElement('div');
                lockedTitle.style = 'font-size: 12px; font-weight: 700; color: #8b949e; text-transform: uppercase; margin: 16px 0 8px;';
                lockedTitle.textContent = 'Locked Framework Settings 🔒';
                container.appendChild(lockedTitle);

                Object.entries(t.locked).forEach(([k, v]) => {{
                    const lockedDiv = document.createElement('div');
                    lockedDiv.className = 'locked-param';
                    lockedDiv.innerHTML = '<span>' + k + '</span><strong>' + v + '</strong>';
                    container.appendChild(lockedDiv);
                }});
            }}

            document.getElementById('editor-modal').style.display = 'flex';
        }}

        function closeModal() {{
            document.getElementById('editor-modal').style.display = 'none';
        }}

        function launchEditedRun() {{
            if (!activeTemplate) return;
            const overrides = {{}};

            Object.entries(activeTemplate.parameters).forEach(([k, spec]) => {{
                const input = document.getElementById('input-' + k);
                if (input) {{
                    let val = input.value;
                    if (spec.type === 'int') val = parseInt(val, 10);
                    else if (spec.type === 'float') val = parseFloat(val);
                    overrides[k] = val;
                }}
            }});

            fetch('/api/jobs/run-template', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    template_id: activeTemplate.template_id,
                    overrides: overrides
                }})
            }})
            .then(r => r.json())
            .then(job => {{
                closeModal();
                alert('🚀 Experiment started! Job ID: ' + job.spec.id);
                location.reload();
            }})
            .catch(e => alert('Failed to launch: ' + e));
        }}

        function stopJob(jobId) {{
            fetch('/api/jobs/' + jobId + '/stop', {{ method: 'POST' }})
                .then(r => r.json())
                .then(() => location.reload())
                .catch(e => console.error(e));
        }}

        function viewLogs(jobId) {{
            fetch('/api/jobs/' + jobId + '/logs')
                .then(r => r.json())
                .then(data => {{
                    document.getElementById('logs-modal-title').textContent = 'Logs for Job ' + jobId.substring(0, 8);
                    document.getElementById('logs-viewer').value = data.logs.join('\n') || 'No logs captured yet.';
                    document.getElementById('logs-modal').style.display = 'flex';
                }})
                .catch(e => alert('Error fetching logs: ' + e));
        }}

        function closeLogsModal() {{
            document.getElementById('logs-modal').style.display = 'none';
        }}

        function loadFile(path) {{
            document.querySelectorAll('.file-item').forEach(el => {{
                el.classList.toggle('active', el.textContent === path);
            }});
            document.getElementById('active-file-name').textContent = path + ' (Read-Only)';

            fetch('/api/files/content?path=' + encodeURIComponent(path))
                .then(r => r.json())
                .then(data => {{
                    document.getElementById('code-viewer').value = data.content || '';
                }})
                .catch(e => {{
                    document.getElementById('code-viewer').value = 'Failed to load file content: ' + e;
                }});
        }}

        function copyCode() {{
            const code = document.getElementById('code-viewer').value;
            navigator.clipboard.writeText(code).then(() => alert('Copied to clipboard!'));
        }}
    </script>
</body>
</html>"#,
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
            templates_cards_html = templates_cards_html,
            templates_json = templates_json
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
        assert!(html.contains("Control Room"));
        assert!(html.contains("Code View (Read-Only)"));
        assert!(html.contains("Templates (Editable)"));
        assert!(html.contains("Configure Recipe"));
    }
}
