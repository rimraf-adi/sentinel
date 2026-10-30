//! HTML & WebAssembly Dashboard View Generator.
//! IDE / Code view is strictly READ-ONLY. Everything else (configs, templates, jobs, sweeps) is EDITABLE.
//! Minimalist, clean black-and-white theme inspired by Vercel (Geist Design System).

use crate::state::DashboardState;
use sentinel_core::models::{HardwareStats, JobRecord};
use sentinel_core::templates::Template;

pub struct DashboardView;

impl DashboardView {
    /// Render complete standalone single-page dashboard HTML with Vercel black & white aesthetic.
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
        let ram_used_pct = if ram_gb_total > 0.0 {
            (ram_gb_used / ram_gb_total) * 100.0
        } else {
            0.0
        };

        let gpus_html = if let Some(s) = stats {
            if s.gpus.is_empty() {
                r##"<div class="card stat-card"><div class="stat-title">GPU Telemetry</div><div class="stat-value">None Detected</div><div class="stat-footer">Running on CPU host</div></div>"##.to_string()
            } else {
                s.gpus
                    .iter()
                    .map(|gpu| {
                        let vram_used_gb = gpu.vram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                        let vram_total_gb = gpu.vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                        format!(
                            r##"<div class="card stat-card">
                                <div class="stat-title">GPU {}: {}</div>
                                <div class="stat-value">{:.1}% <span class="subtext">({:.1}°C)</span></div>
                                <div class="progress-bar"><div class="progress-fill" style="width: {:.1}%"></div></div>
                                <div class="stat-footer">VRAM: {:.2} GB / {:.2} GB</div>
                            </div>"##,
                            gpu.index, gpu.name, gpu.gpu_util_percent, gpu.temperature_c, gpu.gpu_util_percent, vram_used_gb, vram_total_gb
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        } else {
            r##"<div class="card stat-card"><div class="stat-title">GPU Telemetry</div><div class="stat-value">Local Host</div><div class="stat-footer">Apple Silicon / CPU Worker Pool</div></div>"##.to_string()
        };

        // Render Recent Run card (empty state if jobs is empty)
        let recent_run_html = if let Some(latest) = jobs.last() {
            let status_pill = match latest.status {
                sentinel_core::models::JobStatus::Running => {
                    r##"<span class="pill-badge pill-running"><span class="dot-running"></span> Running</span>"##
                }
                sentinel_core::models::JobStatus::Completed => {
                    r##"<span class="pill-badge pill-completed"><span class="dot-completed"></span> Completed</span>"##
                }
                sentinel_core::models::JobStatus::Failed => {
                    r##"<span class="pill-badge pill-failed"><span class="dot-failed"></span> Failed</span>"##
                }
                _ => r##"<span class="pill-badge pill-idle"><span class="dot-idle"></span> Pending</span>"##,
            };

            let run_id_short = if latest.spec.id.to_string().len() >= 8 {
                &latest.spec.id.to_string()[..8]
            } else {
                "latest"
            };

            let template_name = latest
                .spec
                .template_id
                .as_deref()
                .unwrap_or("custom-recipe");

            format!(
                r##"<div class="recent-run-card">
                    <div class="recent-run-header">
                        <div class="recent-run-title-group">
                            <div class="recent-run-icon">
                                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                            </div>
                            <div>
                                <div class="recent-run-label">Recent run</div>
                                <div class="recent-run-id font-mono">run-{}</div>
                            </div>
                        </div>
                        <div class="recent-run-actions">
                            {}
                            <button class="btn-ghost" onclick="switchTab('runs')">All runs &rarr;</button>
                        </div>
                    </div>
                    <div class="recent-run-body">
                        <div>
                            <div class="recent-run-name">{}</div>
                            <div class="recent-run-subtitle font-mono">{}</div>
                        </div>
                        <div class="recent-run-metric">
                            <div class="metric-label">Status</div>
                            <div class="metric-value font-mono">{:?}</div>
                        </div>
                    </div>
                    <div class="recent-run-meta">
                        <div>
                            <span class="meta-label">Started</span>
                            <span class="meta-val">Recently</span>
                        </div>
                        <div>
                            <span class="meta-label">PID</span>
                            <span class="meta-val font-mono">{}</span>
                        </div>
                        <div>
                            <span class="meta-label">Worker</span>
                            <span class="meta-val">local &middot; Slot 0</span>
                        </div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-labels">
                            <span>Progress</span>
                            <span>Active</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 100%;"></div>
                        </div>
                    </div>
                </div>"##,
                run_id_short,
                status_pill,
                latest.spec.name,
                template_name,
                latest.status,
                latest.pid.unwrap_or(0)
            )
        } else {
            // Empty state for Recent Run when no jobs exist
            r##"<div class="recent-run-card">
                <div class="recent-run-header">
                    <div class="recent-run-title-group">
                        <div class="recent-run-icon">
                            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                        </div>
                        <div>
                            <div class="recent-run-label">Recent run</div>
                            <div class="recent-run-id font-mono">&mdash;</div>
                        </div>
                    </div>
                    <div class="recent-run-actions">
                        <span class="pill-badge pill-idle"><span class="dot-idle"></span> Idle</span>
                        <button class="btn-ghost" onclick="switchTab('runs')">All runs &rarr;</button>
                    </div>
                </div>
                <div class="empty-state-box">
                    <div class="empty-state-title">No training runs recorded</div>
                    <div class="empty-state-desc">Launch your first model run from a curated recipe template or start a hyperparameter sweep.</div>
                    <div style="margin-top: 16px;">
                        <button class="btn btn-primary btn-sm" onclick="openTemplateSelector()">+ New run</button>
                    </div>
                </div>
            </div>"##.to_string()
        };

        // Projects Section: Since data is not there, render clean empty state
        let projects_html = r##"<div class="projects-empty-card">
            <div class="projects-empty-icon">
                <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>
                </svg>
            </div>
            <div class="projects-empty-title">No projects registered yet</div>
            <div class="projects-empty-desc">Each ML project repository stays in its own directory. When connected, repositories will appear here with active git branch and latest run status.</div>
        </div>"##;

        let jobs_rows_html = if jobs.is_empty() {
            r##"<tr><td colspan="6" class="empty-row">No runs recorded yet. Click <strong>+ New run</strong> to start an experiment.</td></tr>"##.to_string()
        } else {
            jobs.iter()
                .map(|j| {
                    let status_pill = match j.status {
                        sentinel_core::models::JobStatus::Completed => {
                            r##"<span class="pill-badge pill-completed"><span class="dot-completed"></span> Completed</span>"##
                        }
                        sentinel_core::models::JobStatus::Running => {
                            r##"<span class="pill-badge pill-running"><span class="dot-running"></span> Running</span>"##
                        }
                        sentinel_core::models::JobStatus::Failed => {
                            r##"<span class="pill-badge pill-failed"><span class="dot-failed"></span> Failed</span>"##
                        }
                        _ => r##"<span class="pill-badge pill-idle"><span class="dot-idle"></span> Pending</span>"##,
                    };
                    let short_id = if j.spec.id.to_string().len() >= 8 {
                        &j.spec.id.to_string()[..8]
                    } else {
                        "job"
                    };
                    format!(
                        r##"<tr>
                            <td class="font-mono">run-{}</td>
                            <td><strong>{}</strong></td>
                            <td>{}</td>
                            <td class="font-mono">PID: {}</td>
                            <td>{}</td>
                            <td>
                                <button class="btn-sm btn-outline" onclick="viewLogs('{}')">Logs</button>
                                <button class="btn-sm btn-danger" onclick="stopJob('{}')">Stop</button>
                            </td>
                        </tr>"##,
                        short_id,
                        j.spec.name,
                        status_pill,
                        j.pid.unwrap_or(0),
                        j.spec.template_id.as_deref().unwrap_or("Custom"),
                        j.spec.id,
                        j.spec.id
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let templates_json = serde_json::to_string(templates).unwrap_or_else(|_| "[]".to_string());

        let templates_cards_html = templates
            .iter()
            .map(|t| {
                format!(
                    r##"<div class="template-card">
                        <div class="template-header">
                            <span class="template-domain">{}</span>
                            <span class="template-id font-mono">{}</span>
                        </div>
                        <h4 class="template-name">{}</h4>
                        <p class="template-desc">{}</p>
                        <div class="param-summary">
                            <span>{} editable params</span> &middot; <span>{} locked 🔒</span>
                        </div>
                        <div class="template-actions">
                            <button class="btn btn-primary btn-sm" onclick="openTemplateEditor('{}')">⚙️ Configure & Launch</button>
                        </div>
                    </div>"##,
                    t.domain.to_uppercase(),
                    t.template_id,
                    t.name,
                    t.description,
                    t.parameters.len(),
                    t.locked.len(),
                    t.template_id
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r###"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel &mdash; Training Control</title>
    <style>
        :root {{
            --bg: #000000;
            --sidebar-bg: #000000;
            --surface: #0a0a0a;
            --surface-hover: #121212;
            --border: #222222;
            --border-subtle: #171717;
            --border-focus: #ffffff;
            --text: #ededed;
            --text-secondary: #a1a1a1;
            --text-muted: #666666;
            --btn-primary-bg: #ffffff;
            --btn-primary-text: #000000;
            --btn-primary-hover: #cccccc;
            --danger: #ff453a;
            --danger-bg: #1c0e0e;
            --sidebar-w: 250px;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
            line-height: 1.5;
            display: flex;
            min-height: 100vh;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }}

        /* SIDEBAR NAVIGATION (VERCEL BLACK STYLE) */
        .sidebar {{
            width: var(--sidebar-w);
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 16px 14px;
            flex-shrink: 0;
            user-select: none;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding: 4px 6px;
        }}
        .brand-icon {{
            width: 28px;
            height: 28px;
            background: #ffffff;
            color: #000000;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
        }}
        .brand-icon svg {{
            fill: #000000;
        }}
        .brand-title {{
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: -0.01em;
            line-height: 1.2;
        }}
        .brand-subtitle {{
            font-size: 9px;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        /* WORKSPACE CARD */
        .workspace-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 20px;
        }}
        .workspace-card-top {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .workspace-monogram {{
            width: 26px;
            height: 26px;
            background: #171717;
            border: 1px solid #2e2e2e;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            color: #ffffff;
        }}
        .workspace-meta {{
            flex: 1;
            overflow: hidden;
        }}
        .workspace-name {{
            font-size: 13px;
            font-weight: 600;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .workspace-path {{
            font-size: 11px;
            color: var(--text-muted);
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .workspace-card-bottom {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            padding-top: 8px;
            border-top: 1px solid var(--border-subtle);
        }}
        .branch-pill {{
            display: flex;
            align-items: center;
            gap: 4px;
            color: var(--text-secondary);
        }}
        .clean-pill {{
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 5px;
            color: #ededed;
            font-weight: 500;
        }}
        .dot {{ width: 5px; height: 5px; border-radius: 50%; background: #ffffff; }}

        /* SIDEBAR SECTIONS */
        .nav-section {{
            margin-bottom: 20px;
        }}
        .nav-section-title {{
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0 8px 6px;
        }}
        .nav-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 7px 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s ease;
            margin-bottom: 2px;
            border: 1px solid transparent;
        }}
        .nav-item:hover {{
            background: var(--surface-hover);
            color: #ffffff;
        }}
        .nav-item.active {{
            background: #171717;
            color: #ffffff;
            border-color: #2e2e2e;
            font-weight: 600;
        }}
        .nav-badge {{
            margin-left: auto;
            background: #171717;
            border: 1px solid #2e2e2e;
            color: var(--text-muted);
            padding: 1px 6px;
            border-radius: 10px;
            font-size: 10px;
            font-weight: 600;
        }}
        .badge-ro {{
            margin-left: auto;
            background: #171717;
            border: 1px solid #2e2e2e;
            color: #ededed;
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}

        /* MAIN CONTENT AREA */
        .main-wrapper {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            min-width: 0;
            background: var(--bg);
        }}
        .top-bar {{
            height: 52px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 28px;
            background: var(--bg);
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .breadcrumbs {{
            font-size: 13px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .bc-current {{
            color: #ffffff;
            font-weight: 500;
        }}
        .top-status {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .status-pill {{
            background: #0a0a0a;
            border: 1px solid var(--border);
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .dot-white {{ width: 5px; height: 5px; border-radius: 50%; background: #ffffff; }}
        .user-avatar {{
            width: 26px;
            height: 26px;
            background: #171717;
            border: 1px solid #2e2e2e;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            color: #ffffff;
        }}

        .content {{
            padding: 28px 32px 48px;
            max-width: 1200px;
            width: 100%;
        }}

        /* PAGE HEADER */
        .page-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 24px;
        }}
        .category-eyebrow {{
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .page-title {{
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.02em;
            margin-bottom: 4px;
        }}
        .page-subtitle {{
            font-size: 13px;
            color: var(--text-secondary);
        }}
        .page-actions {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        /* VERCEL BUTTONS (CLEAN B&W) */
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-size: 13px;
            font-weight: 500;
            padding: 7px 14px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
            border: 1px solid transparent;
            text-decoration: none;
        }}
        .btn-primary {{
            background: var(--btn-primary-bg);
            color: var(--btn-primary-text);
            border-color: #ffffff;
        }}
        .btn-primary:hover {{
            background: var(--btn-primary-hover);
            border-color: #cccccc;
        }}
        .btn-secondary {{
            background: #000000;
            color: #ffffff;
            border-color: var(--border);
        }}
        .btn-secondary:hover {{
            background: #171717;
            border-color: #444444;
        }}
        .btn-ghost {{
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
        }}
        .btn-ghost:hover {{
            background: #171717;
            color: #ffffff;
            border-color: #444444;
        }}
        .btn-sm {{
            padding: 5px 12px;
            font-size: 12px;
            border-radius: 6px;
        }}
        .btn-outline {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
            cursor: pointer;
        }}
        .btn-outline:hover {{ background: #171717; border-color: #444444; }}
        .btn-danger {{
            background: var(--danger-bg);
            border: 1px solid #4a1515;
            color: var(--danger);
            cursor: pointer;
        }}
        .btn-danger:hover {{ background: #2e0f0f; }}

        /* RECENT RUN CARD */
        .recent-run-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 30px;
        }}
        .recent-run-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }}
        .recent-run-title-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .recent-run-icon {{
            width: 28px;
            height: 28px;
            border-radius: 6px;
            background: #171717;
            border: 1px solid #2e2e2e;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .recent-run-label {{
            font-size: 13px;
            font-weight: 600;
            color: #ffffff;
        }}
        .recent-run-id {{
            font-size: 11px;
            color: var(--text-muted);
        }}
        .recent-run-actions {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .recent-run-body {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 18px;
        }}
        .recent-run-name {{
            font-size: 18px;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: -0.01em;
            margin-bottom: 4px;
        }}
        .recent-run-subtitle {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .recent-run-metric {{
            text-align: right;
        }}
        .metric-label {{
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 2px;
            letter-spacing: 0.05em;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
        }}
        .recent-run-meta {{
            display: flex;
            align-items: center;
            gap: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 16px;
            font-size: 12px;
        }}
        .meta-label {{
            color: var(--text-muted);
            display: block;
            margin-bottom: 2px;
        }}
        .meta-val {{
            color: #ffffff;
            font-weight: 500;
        }}
        .progress-container {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .progress-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
        }}
        .progress-bar {{
            background: #171717;
            height: 4px;
            border-radius: 2px;
            overflow: hidden;
            width: 100%;
        }}
        .progress-fill {{
            background: #ffffff;
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }}

        /* EMPTY STATES */
        .empty-state-box {{
            padding: 16px 0 6px;
            text-align: left;
        }}
        .empty-state-title {{
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 4px;
        }}
        .empty-state-desc {{
            font-size: 12px;
            color: var(--text-muted);
            max-width: 480px;
        }}
        .projects-empty-card {{
            background: var(--surface);
            border: 1px dashed var(--border);
            border-radius: 8px;
            padding: 36px 20px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .projects-empty-icon {{
            width: 40px;
            height: 40px;
            border-radius: 8px;
            background: #121212;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 12px;
        }}
        .projects-empty-title {{
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 4px;
        }}
        .projects-empty-desc {{
            font-size: 12px;
            color: var(--text-muted);
            max-width: 440px;
            line-height: 1.5;
        }}

        /* SECTION HEADER */
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 14px;
        }}
        .section-eyebrow {{
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .section-title {{
            font-size: 17px;
            font-weight: 600;
            color: #ffffff;
        }}
        .section-subtitle {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .badge-count {{
            background: #121212;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }}

        /* STATUS PILLS */
        .pill-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            background: #121212;
            border: 1px solid var(--border);
            color: var(--text-secondary);
        }}
        .pill-running {{ color: #ffffff; border-color: #444444; }}
        .pill-completed {{ color: #ededed; }}
        .pill-failed {{ color: var(--danger); border-color: #4a1515; }}
        .dot-running {{ width: 5px; height: 5px; border-radius: 50%; background: #ffffff; box-shadow: 0 0 4px #ffffff; }}
        .dot-completed {{ width: 5px; height: 5px; border-radius: 50%; background: #ededed; }}
        .dot-failed {{ width: 5px; height: 5px; border-radius: 50%; background: var(--danger); }}
        .dot-idle {{ width: 5px; height: 5px; border-radius: 50%; background: #555555; }}

        /* HARDWARE TELEMETRY GRID */
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .stat-title {{ font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }}
        .stat-value {{ font-size: 20px; font-weight: 600; color: #ffffff; margin-bottom: 8px; }}
        .subtext {{ font-size: 12px; font-weight: 400; color: var(--text-muted); }}
        .stat-footer {{ font-size: 11px; color: var(--text-muted); margin-top: 8px; }}

        /* TABLES */
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }}
        th {{ color: var(--text-muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
        .empty-row {{ text-align: center; color: var(--text-muted); padding: 32px; font-size: 13px; }}

        /* TEMPLATES & RECIPES GRID */
        .template-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 14px;
        }}
        .template-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
        }}
        .template-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }}
        .template-domain {{
            font-size: 10px;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: 0.06em;
            background: #171717;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #2e2e2e;
        }}
        .template-id {{
            font-size: 11px;
            color: var(--text-muted);
        }}
        .template-name {{
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 4px;
        }}
        .template-desc {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 14px;
            flex: 1;
            line-height: 1.4;
        }}
        .param-summary {{
            font-size: 11px;
            color: var(--text-muted);
            margin-bottom: 14px;
            padding-top: 8px;
            border-top: 1px solid var(--border-subtle);
        }}

        /* STRICTLY READ-ONLY IDE CODE VIEW */
        .ide-banner {{
            background: #0a0a0a;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 12px;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .ide-container {{
            display: flex;
            height: 600px;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: #000000;
        }}
        .ide-sidebar {{
            width: 240px;
            border-right: 1px solid var(--border);
            background: #050505;
            padding: 12px;
            overflow-y: auto;
            flex-shrink: 0;
        }}
        .ide-sidebar-title {{
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
            padding: 0 4px;
        }}
        .file-item {{
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            cursor: pointer;
            color: var(--text-secondary);
            margin-bottom: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .file-item:hover {{ background: #171717; color: #ffffff; }}
        .file-item.active {{ background: #1f1f1f; color: #ffffff; font-weight: 600; border: 1px solid #333333; }}
        .ide-main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #000000;
            min-width: 0;
        }}
        .ide-toolbar {{
            padding: 8px 14px;
            border-bottom: 1px solid var(--border);
            background: #050505;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .code-area {{
            flex: 1;
            width: 100%;
            background: #000000;
            border: none;
            color: #ededed;
            font-family: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
            font-size: 13px;
            padding: 16px;
            resize: none;
            outline: none;
            line-height: 1.55;
            white-space: pre;
            overflow-y: auto;
        }}

        /* MODALS */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(4px);
            z-index: 100;
            justify-content: center;
            align-items: center;
        }}
        .modal-content {{
            background: #0a0a0a;
            border: 1px solid var(--border);
            border-radius: 8px;
            width: 560px;
            max-width: 92%;
            max-height: 85vh;
            overflow-y: auto;
            padding: 22px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.8);
        }}
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }}
        .form-group {{ margin-bottom: 14px; }}
        .form-label {{ display: block; font-size: 12px; font-weight: 500; color: #ffffff; margin-bottom: 4px; }}
        .form-desc {{ font-size: 11px; color: var(--text-muted); margin-bottom: 6px; }}
        .form-input {{
            width: 100%;
            background: #000000;
            border: 1px solid var(--border);
            color: #ffffff;
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 13px;
        }}
        .form-input:focus {{ border-color: #ffffff; outline: none; }}
        .locked-param {{
            background: #121212;
            border: 1px solid var(--border-subtle);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}

        .tab-pane {{ display: none; }}
        .tab-pane.active {{ display: block; }}
        .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    </style>
</head>
<body>

    <!-- LEFT SIDEBAR -->
    <aside class="sidebar">
        <div class="brand">
            <div class="brand-icon">
                <svg viewBox="0 0 24 24" width="16" height="16">
                    <polygon points="12 2 22 20 2 20"/>
                </svg>
            </div>
            <div class="brand-info">
                <div class="brand-title">sentinel</div>
                <div class="brand-subtitle">TRAINING CONTROL</div>
            </div>
        </div>

        <!-- WORKSPACE CARD -->
        <div class="workspace-card">
            <div class="workspace-card-top">
                <div class="workspace-monogram">UX</div>
                <div class="workspace-meta">
                    <div class="workspace-name">
                        <span>uxML</span>
                        <span style="font-size: 10px; color: var(--text-muted);">&or;</span>
                    </div>
                    <div class="workspace-path">~/100xcode/uxML</div>
                </div>
            </div>
            <div class="workspace-card-bottom">
                <div class="branch-pill">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
                    <span>main</span>
                </div>
                <div class="clean-pill"><span class="dot"></span> clean</div>
            </div>
        </div>

        <!-- WORKSPACE NAV GROUP -->
        <div class="nav-section">
            <div class="nav-section-title">Workspace</div>
            <div class="nav-item active" id="nav-overview" onclick="switchTab('overview')">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>
                <span>Overview</span>
            </div>
            <div class="nav-item" id="nav-runs" onclick="switchTab('runs')">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                <span>Runs</span>
                <span class="nav-badge" id="sidebar-runs-badge">{jobs_count}</span>
            </div>
            <div class="nav-item" id="nav-sweeps" onclick="switchTab('sweeps')">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                <span>Sweeps</span>
                <span class="nav-badge">0</span>
            </div>
        </div>

        <!-- PROJECT NAV GROUP -->
        <div class="nav-section">
            <div class="nav-section-title">Project</div>
            <div class="nav-item" id="nav-templates" onclick="switchTab('templates')">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                <span>Templates</span>
            </div>
            <div class="nav-item" id="nav-datasets" onclick="switchTab('datasets')">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
                <span>Datasets</span>
            </div>
            <div class="nav-item" id="nav-machines" onclick="switchTab('machines')">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
                <span>Machines</span>
            </div>
            <div class="nav-item" id="nav-ide" onclick="switchTab('ide')">
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                <span>Code (Read-Only)</span>
                <span class="badge-ro">RO</span>
            </div>
        </div>
    </aside>

    <!-- MAIN VIEW -->
    <div class="main-wrapper">
        <!-- TOP BREADCRUMB / STATUS BAR -->
        <header class="top-bar">
            <div class="breadcrumbs">
                <span>uxML</span>
                <span>/</span>
                <span class="bc-current" id="top-breadcrumb">Overview</span>
            </div>
            <div class="top-status">
                <div class="status-pill"><span class="dot-white"></span> UI prototype</div>
                <div class="status-pill"><span class="dot-white"></span> daemon online</div>
                <div class="user-avatar">AD</div>
            </div>
        </header>

        <!-- PAGE CONTENT -->
        <main class="content">

            <!-- TAB 1: OVERVIEW -->
            <section id="pane-overview" class="tab-pane active">
                <div class="page-header">
                    <div>
                        <div class="category-eyebrow">HOME / SUMMARY</div>
                        <h1 class="page-title">Workspace overview</h1>
                        <p class="page-subtitle">Latest run status and project repos.</p>
                    </div>
                    <div class="page-actions">
                        <button class="btn btn-secondary" onclick="openNewSweepModal()">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                            <span>New sweep</span>
                        </button>
                        <button class="btn btn-primary" onclick="openTemplateSelector()">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                            <span>New run</span>
                        </button>
                    </div>
                </div>

                <!-- RECENT RUN CARD -->
                {recent_run_html}

                <!-- PROJECTS SECTION -->
                <div class="section-header">
                    <div>
                        <div class="section-eyebrow">GIT REPOS</div>
                        <h2 class="section-title">Projects</h2>
                        <div class="section-subtitle">Each project stays in its own repo.</div>
                    </div>
                    <span class="badge-count">0 projects</span>
                </div>

                <!-- EMPTY STATE PROJECTS CARD (as requested: keep empty since no projects right now) -->
                {projects_html}
            </section>

            <!-- TAB 2: RUNS SUPERVISOR -->
            <section id="pane-runs" class="tab-pane">
                <div class="page-header">
                    <div>
                        <div class="category-eyebrow">EXPERIMENT TRACKING</div>
                        <h1 class="page-title">Training Runs</h1>
                        <p class="page-subtitle">Process supervisor, live stdout streaming, and process controls.</p>
                    </div>
                    <button class="btn btn-primary" onclick="openTemplateSelector()">+ New run</button>
                </div>
                <div class="card" style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Run ID</th>
                                <th>Experiment Name</th>
                                <th>Status</th>
                                <th>Worker PID</th>
                                <th>Recipe Template</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="runs-table-body">
                            {jobs_rows_html}
                        </tbody>
                    </table>
                </div>
            </section>

            <!-- TAB 3: SWEEPS -->
            <section id="pane-sweeps" class="tab-pane">
                <div class="page-header">
                    <div>
                        <div class="category-eyebrow">HYPERPARAMETER TUNING</div>
                        <h1 class="page-title">Hyperparameter Sweeps</h1>
                        <p class="page-subtitle">Optuna-backed ASHA multi-trial automated exploration.</p>
                    </div>
                    <button class="btn btn-primary" onclick="openNewSweepModal()">Configure Sweep</button>
                </div>
                <div class="card" style="padding: 24px; text-align: center;">
                    <div style="font-size: 14px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">No active sweep campaigns</div>
                    <div style="font-size: 12px; color: var(--text-muted); max-width: 480px; margin: 0 auto 16px;">
                        Configure hyperparameter bounds (learning rates, batch sizes, dropouts) to run concurrent ASHA trials with early-stopping.
                    </div>
                    <button class="btn btn-secondary btn-sm" onclick="openNewSweepModal()">Configure New Sweep</button>
                </div>
            </section>

            <!-- TAB 4: TEMPLATES & RECIPES (EDITABLE) -->
            <section id="pane-templates" class="tab-pane">
                <div class="page-header">
                    <div>
                        <div class="category-eyebrow">RECIPE REGISTRY</div>
                        <h1 class="page-title">Template Catalog</h1>
                        <p class="page-subtitle">Fully editable hyperparameters with locked framework guarantees.</p>
                    </div>
                    <button class="btn btn-secondary" onclick="openCustomTemplateModal()">+ Custom Template</button>
                </div>
                <div class="template-grid">
                    {templates_cards_html}
                </div>
            </section>

            <!-- TAB 5: DATASETS -->
            <section id="pane-datasets" class="tab-pane">
                <div class="page-header">
                    <div>
                        <div class="category-eyebrow">DATA MANAGEMENT</div>
                        <h1 class="page-title">Datasets</h1>
                        <p class="page-subtitle">Local storage mounts and cloud object store links.</p>
                    </div>
                </div>
                <div class="projects-empty-card">
                    <div class="projects-empty-icon">
                        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.8">
                            <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
                        </svg>
                    </div>
                    <div class="projects-empty-title">No datasets mounted</div>
                    <div class="projects-empty-desc">Connect datasets via local symlink or remote S3/GCS buckets to make them accessible to training recipes.</div>
                </div>
            </section>

            <!-- TAB 6: MACHINES & HARDWARE -->
            <section id="pane-machines" class="tab-pane">
                <div class="page-header">
                    <div>
                        <div class="category-eyebrow">INFRASTRUCTURE</div>
                        <h1 class="page-title">Machines & Telemetry</h1>
                        <p class="page-subtitle">Live NVML GPU metrics, host utilization, and worker pool slots.</p>
                    </div>
                </div>
                <div class="grid">
                    <div class="card stat-card">
                        <div class="stat-title">Host CPU Utilization</div>
                        <div class="stat-value">{cpu_percent:.1}%</div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {cpu_percent:.1}%"></div></div>
                        <div class="stat-footer">System Cores Monitored</div>
                    </div>
                    <div class="card stat-card">
                        <div class="stat-title">Host Memory Allocated</div>
                        <div class="stat-value">{ram_gb_used:.2} GB <span class="subtext">/ {ram_gb_total:.1} GB</span></div>
                        <div class="progress-bar"><div class="progress-fill" style="width: {ram_used_pct:.1}%"></div></div>
                        <div class="stat-footer">System RAM Utilization</div>
                    </div>
                    {gpus_html}
                </div>
            </section>

            <!-- TAB 7: CODE VIEW (STRICTLY READ-ONLY) -->
            <section id="pane-ide" class="tab-pane">
                <div class="ide-banner">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                    <span><strong>Read-Only Code View</strong> &mdash; Zero memory bloat and ultra-low latency. Code cannot be edited in the browser; use your local IDE for development.</span>
                </div>
                <div class="ide-container">
                    <aside class="ide-sidebar">
                        <div class="ide-sidebar-title">Project Files</div>
                        <div id="file-list">
                            <div class="file-item active" onclick="loadFile('recipes/vision_train.py')">recipes/vision_train.py</div>
                            <div class="file-item" onclick="loadFile('recipes/sweep_orchestrator.py')">recipes/sweep_orchestrator.py</div>
                            <div class="file-item" onclick="loadFile('templates/vision/image-classification.yaml')">templates/vision/image-classification.yaml</div>
                            <div class="file-item" onclick="loadFile('templates/llm/lora-finetune.yaml')">templates/llm/lora-finetune.yaml</div>
                            <div class="file-item" onclick="loadFile('templates/tabular/mlp-classifier.yaml')">templates/tabular/mlp-classifier.yaml</div>
                            <div class="file-item" onclick="loadFile('pyproject.toml')">pyproject.toml</div>
                            <div class="file-item" onclick="loadFile('sentinel.md')">sentinel.md</div>
                            <div class="file-item" onclick="loadFile('FILES.md')">FILES.md</div>
                        </div>
                    </aside>
                    <div class="ide-main">
                        <div class="ide-toolbar">
                            <span id="active-file-name" class="font-mono" style="font-size: 12px; color: var(--text-secondary);">recipes/vision_train.py (Read-Only)</span>
                            <button class="btn btn-secondary btn-sm" onclick="copyCode()">📋 Copy Code</button>
                        </div>
                        <textarea id="code-viewer" class="code-area" readonly spellcheck="false"></textarea>
                    </div>
                </div>
            </section>

        </main>
    </div>

    <!-- INTERACTIVE TEMPLATE CONFIGURATOR MODAL (EDITABLE) -->
    <div id="editor-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div>
                    <h3 id="modal-template-name" style="font-size: 15px; font-weight: 600; color: #ffffff;">Configure Recipe</h3>
                    <div id="modal-template-id" class="font-mono" style="font-size: 11px; color: var(--text-muted);"></div>
                </div>
                <button class="btn-ghost" onclick="closeModal()">✕</button>
            </div>
            <div id="modal-params-container">
                <!-- Dynamically populated with editable inputs -->
            </div>
            <div style="margin-top: 20px; display: flex; justify-content: flex-end; gap: 8px;">
                <button class="btn btn-secondary btn-sm" onclick="closeModal()">Cancel</button>
                <button class="btn btn-primary btn-sm" onclick="launchEditedRun()">Launch Experiment</button>
            </div>
        </div>
    </div>

    <!-- CUSTOM TEMPLATE MODAL -->
    <div id="custom-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div>
                    <h3 style="font-size: 15px; font-weight: 600; color: #ffffff;">Custom Recipe Template</h3>
                    <div class="font-mono" style="font-size: 11px; color: var(--text-muted);">Launch custom script with full control</div>
                </div>
                <button class="btn-ghost" onclick="closeCustomModal()">✕</button>
            </div>
            <div class="form-group">
                <label class="form-label">Job Name</label>
                <input class="form-input" id="custom-job-name" value="custom-model-train" placeholder="e.g. bert-finetune" />
            </div>
            <div class="form-group">
                <label class="form-label">Execution Command (strictly uv or python)</label>
                <input class="form-input font-mono" id="custom-job-cmd" value="uv run python recipes/vision_train.py --epochs 1" />
            </div>
            <div style="margin-top: 20px; display: flex; justify-content: flex-end; gap: 8px;">
                <button class="btn btn-secondary btn-sm" onclick="closeCustomModal()">Cancel</button>
                <button class="btn btn-primary btn-sm" onclick="launchCustomRun()">Submit Custom Job</button>
            </div>
        </div>
    </div>

    <!-- NEW SWEEP MODAL -->
    <div id="sweep-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div>
                    <h3 style="font-size: 15px; font-weight: 600; color: #ffffff;">Configure Hyperparameter Sweep</h3>
                    <div class="font-mono" style="font-size: 11px; color: var(--text-muted);">Optuna ASHA multi-trial engine</div>
                </div>
                <button class="btn-ghost" onclick="closeSweepModal()">✕</button>
            </div>
            <div class="form-group">
                <label class="form-label">Sweep Name</label>
                <input class="form-input" id="sweep-name" value="cifar10-lr-sweep" />
            </div>
            <div class="form-group">
                <label class="form-label">Total Trials</label>
                <input class="form-input" id="sweep-trials" type="number" value="10" min="1" max="50" />
            </div>
            <div class="form-group">
                <label class="form-label">Max Epochs per Trial</label>
                <input class="form-input" id="sweep-max-epochs" type="number" value="8" min="1" max="50" />
            </div>
            <div style="margin-top: 20px; display: flex; justify-content: flex-end; gap: 8px;">
                <button class="btn btn-secondary btn-sm" onclick="closeSweepModal()">Cancel</button>
                <button class="btn btn-primary btn-sm" onclick="launchSweep()">Start Sweep</button>
            </div>
        </div>
    </div>

    <!-- LIVE LOG DRAWER MODAL -->
    <div id="logs-modal" class="modal">
        <div class="modal-content" style="width: 780px;">
            <div class="modal-header">
                <h3 id="logs-modal-title" style="font-size: 14px; font-weight: 600; color: #ffffff;">Job Execution Logs</h3>
                <button class="btn-ghost" onclick="closeLogsModal()">✕</button>
            </div>
            <textarea id="logs-viewer" class="code-area" style="height: 400px; background: #000000; color: #ededed; border: 1px solid var(--border); border-radius: 6px;" readonly></textarea>
        </div>
    </div>

    <script>
        const templates = {templates_json};
        let activeTemplate = null;

        function switchTab(tabId) {{
            // Update sidebar nav items
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            const navTarget = document.getElementById('nav-' + tabId);
            if (navTarget) navTarget.classList.add('active');

            // Update content tab panes
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            const paneTarget = document.getElementById('pane-' + tabId);
            if (paneTarget) paneTarget.classList.add('active');

            // Update top breadcrumb
            const titles = {{
                'overview': 'Overview',
                'runs': 'Runs',
                'sweeps': 'Sweeps',
                'templates': 'Templates',
                'datasets': 'Datasets',
                'machines': 'Machines',
                'ide': 'Code (Read-Only)'
            }};
            document.getElementById('top-breadcrumb').textContent = titles[tabId] || 'Overview';

            if (tabId === 'ide') {{
                loadFile('recipes/vision_train.py');
            }}
        }}

        function openTemplateSelector() {{
            switchTab('templates');
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
                lockedTitle.style = 'font-size: 10px; font-weight: 600; color: #666666; text-transform: uppercase; margin: 16px 0 8px; letter-spacing: 0.06em;';
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

        function openCustomTemplateModal() {{
            document.getElementById('custom-modal').style.display = 'flex';
        }}

        function closeCustomModal() {{
            document.getElementById('custom-modal').style.display = 'none';
        }}

        function openNewSweepModal() {{
            document.getElementById('sweep-modal').style.display = 'flex';
        }}

        function closeSweepModal() {{
            document.getElementById('sweep-modal').style.display = 'none';
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
                alert('Experiment started! Job ID: ' + job.spec.id);
                location.reload();
            }})
            .catch(e => alert('Failed to launch: ' + e));
        }}

        function launchCustomRun() {{
            const name = document.getElementById('custom-job-name').value;
            const cmd = document.getElementById('custom-job-cmd').value;
            fetch('/api/jobs', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    name: name,
                    command: cmd,
                    run_mode: 'Standalone'
                }})
            }})
            .then(r => r.json())
            .then(job => {{
                closeCustomModal();
                alert('Custom job started! Job ID: ' + job.spec.id);
                location.reload();
            }})
            .catch(e => alert('Failed to submit job: ' + e));
        }}

        function launchSweep() {{
            const trials = document.getElementById('sweep-trials').value;
            const maxEpochs = document.getElementById('sweep-max-epochs').value;
            fetch('/api/jobs', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    name: 'asha-sweep-run',
                    command: 'uv run python recipes/sweep_orchestrator.py --trials ' + trials + ' --max-epochs ' + maxEpochs,
                    run_mode: 'Standalone'
                }})
            }})
            .then(r => r.json())
            .then(job => {{
                closeSweepModal();
                alert('Sweep started! Job ID: ' + job.spec.id);
                switchTab('runs');
                location.reload();
            }})
            .catch(e => alert('Failed to start sweep: ' + e));
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
            navigator.clipboard.writeText(code).then(() => alert('Copied code to clipboard!'));
        }}
    </script>
</body>
</html>"###,
            jobs_count = jobs.len(),
            recent_run_html = recent_run_html,
            projects_html = projects_html,
            jobs_rows_html = jobs_rows_html,
            templates_cards_html = templates_cards_html,
            templates_json = templates_json,
            cpu_percent = cpu_percent,
            ram_gb_used = ram_gb_used,
            ram_gb_total = ram_gb_total,
            ram_used_pct = ram_used_pct,
            gpus_html = gpus_html
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

        assert!(html.contains("sentinel"));
        assert!(html.contains("Workspace overview"));
        assert!(html.contains("Code (Read-Only)"));
        assert!(html.contains("Templates"));
        assert!(html.contains("Configure Recipe"));
        assert!(html.contains("No projects registered yet"));
        assert!(html.contains("No training runs recorded"));
    }
}
