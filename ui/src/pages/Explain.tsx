import { useEffect, useRef, useState } from 'react';
import { api, type DiagnosticReport, type ExplainResult, type LossLandscape, type TrackedRun } from '../api';
import { usePoll } from '../hooks';
import { Icon } from '../icons';
import { EmptyState, Skeleton } from '../ui';

export function Explain() {
  const runs = usePoll(api.trackedRuns, 5000);
  const [selectedRun, setSelectedRun] = useState<string>('live');
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticReport | null>(null);
  const [landscape, setLandscape] = useState<LossLandscape | null>(null);
  const [loading, setLoading] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, number>>({});
  const [computing, setComputing] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Fetch explanation, diagnostics, and loss landscape
  useEffect(() => {
    let active = true;
    const fetchAll = async () => {
      const isLive = selectedRun === 'live';
      const targetId = isLive ? undefined : selectedRun;

      // 1. Fetch prescriptions
      api.prescriptions(targetId).then((diag) => {
        if (active && diag) setDiagnostics(diag);
      });

      // 2. Fetch loss landscape
      api.lossLandscape().then((ls) => {
        if (active && ls) setLandscape(ls);
      });

      // 3. Fetch explanation
      if (isLive) {
        const res = await api.liveExplain();
        if (active && res) {
          setExplanation(res);
          const initial: Record<string, number> = {};
          res.attributions.forEach((a) => {
            initial[a.feature] = a.value;
          });
          setInputValues((prev) => (Object.keys(prev).length === 0 ? initial : prev));
        } else if (active && runs.data && runs.data.length > 0) {
          const first = runs.data[0];
          if (first) setSelectedRun(first.run_id);
        }
      } else {
        setLoading(true);
        const res = await api.runExplain(selectedRun);
        if (active && res) {
          setExplanation(res);
          const initial: Record<string, number> = {};
          res.attributions.forEach((a) => {
            initial[a.feature] = a.value;
          });
          setInputValues(initial);
        }
        if (active) setLoading(false);
      }
    };

    fetchAll();
    const iv = setInterval(() => {
      if (selectedRun === 'live') fetchAll();
    }, 4000);

    return () => {
      active = false;
      clearInterval(iv);
    };
  }, [selectedRun]);

  // Draw 2D Topological Loss Landscape Contour
  useEffect(() => {
    if (!landscape || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { surface, grid_size, min_loss, max_loss } = landscape;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cellW = w / grid_size;
    const cellH = h / grid_size;
    const lossRange = Math.max(0.001, max_loss - min_loss);

    // Heatmap cells
    for (let j = 0; j < grid_size; j++) {
      for (let i = 0; i < grid_size; i++) {
        const val = surface[j]?.[i] ?? min_loss;
        const norm = (val - min_loss) / lossRange; // 0 (min) to 1 (max)

        // Viridis/Plasma-inspired gradient: dark violet -> teal -> yellow
        let r = 0, g = 0, b = 0;
        if (norm < 0.35) {
          const t = norm / 0.35;
          r = Math.floor(20 + 20 * t);
          g = Math.floor(15 + 80 * t);
          b = Math.floor(60 + 140 * t);
        } else if (norm < 0.75) {
          const t = (norm - 0.35) / 0.4;
          r = Math.floor(40 + 170 * t);
          g = Math.floor(95 + 110 * t);
          b = Math.floor(200 - 100 * t);
        } else {
          const t = (norm - 0.75) / 0.25;
          r = Math.floor(210 + 45 * t);
          g = Math.floor(205 + 45 * t);
          b = Math.floor(100 - 60 * t);
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(i * cellW, j * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Contour isolines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.lineWidth = 1;
    for (let k = 1; k < 6; k++) {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, (w / 2.5) * (k / 6), 0, Math.PI * 2);
      ctx.stroke();
    }

    // Center Minimum Marker (W*)
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = '10px monospace';
    ctx.fillText('W* (Center)', w / 2 + 8, h / 2 + 4);
  }, [landscape]);

  const handleRecalculate = async () => {
    if (!explanation) return;
    setComputing(true);
    try {
      const row = explanation.attributions.map((a) => inputValues[a.feature] ?? a.value);
      const res = await api.explain(selectedRun === 'live' ? undefined : selectedRun, [row]);
      if (res) setExplanation(res);
    } catch (e) {
      console.error('Explanation recalculation failed:', e);
    } finally {
      setComputing(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const runsList = runs.data ?? [];

  return (
    <div className="flow-page" style={{ padding: '24px 32px', maxWidth: 1240, margin: '0 auto' }}>
      {/* Header */}
      <div className="row sp" style={{ marginBottom: 20, flexWrap: 'wrap', gap: 16 }}>
        <div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--accent)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>
            AGENT-FIRST OBSERVABILITY · INTERPRETABILITY &amp; REMEDIATION
          </div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span>Explainable AI (xAI) Diagnostics</span>
            <span className="tag" style={{ background: 'rgba(56, 189, 248, 0.1)', color: 'var(--accent)', borderColor: 'rgba(56, 189, 248, 0.25)', fontSize: 11 }}>
              Axiomatic Path Integral
            </span>
          </h2>
          <p style={{ margin: '6px 0 0', color: 'var(--text-muted)', fontSize: 13 }}>
            Axiomatic Integrated Gradients attributions, SVD representation rank, autonomous remediation prescriptions, and 2D loss flatness.
          </p>
        </div>

        <div className="row" style={{ gap: 12, alignItems: 'center' }}>
          <label className="mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>Target Run:</label>
          <select
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            style={{
              background: 'var(--surface-sunken)',
              color: 'var(--text)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              padding: '6px 12px',
              fontSize: 12,
              fontFamily: 'monospace',
            }}
          >
            <option value="live">● Live Training (.sentinel/xai)</option>
            {runsList.map((r: TrackedRun) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_name || r.run_id.slice(0, 8)} ({r.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Skeleton h={120} />
          <Skeleton h={300} />
        </div>
      )}

      {/* Autonomous Agent Remediation Plane */}
      {diagnostics && (
        <div
          style={{
            background: 'var(--surface)',
            border: `1px solid ${diagnostics.healthy ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            borderRadius: 8,
            padding: 18,
            marginBottom: 24,
          }}
        >
          <div className="row sp" style={{ marginBottom: 12 }}>
            <div className="row" style={{ gap: 10, alignItems: 'center' }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: diagnostics.healthy ? 'var(--ok)' : 'var(--danger)',
                  boxShadow: `0 0 10px ${diagnostics.healthy ? 'var(--ok)' : 'var(--danger)'}`,
                }}
              />
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Autonomous Agent Remediation Plane</h3>
              <span className="mono tag" style={{ fontSize: 11, background: 'var(--surface-sunken)', color: 'var(--text)' }}>
                Health Score: <b style={{ color: diagnostics.healthy ? 'var(--ok)' : 'var(--danger)' }}>{diagnostics.health_score}%</b>
              </span>
            </div>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {diagnostics.prescriptions.length} Prescription(s) Available
            </span>
          </div>

          <p style={{ margin: '0 0 14px', fontSize: 13, color: 'var(--text-muted)' }}>{diagnostics.summary}</p>

          <div style={{ display: 'grid', gap: 10 }}>
            {diagnostics.prescriptions.map((p) => {
              const isCrit = p.severity === 'critical';
              const sevColor = isCrit ? 'var(--danger)' : 'var(--warning)';
              return (
                <div
                  key={p.id}
                  style={{
                    background: 'var(--surface-sunken)',
                    border: `1px solid ${isCrit ? 'rgba(239, 68, 68, 0.4)' : 'var(--border)'}`,
                    borderRadius: 6,
                    padding: '12px 16px',
                  }}
                >
                  <div className="row sp" style={{ marginBottom: 6 }}>
                    <div className="row" style={{ gap: 8, alignItems: 'center' }}>
                      <span className="mono" style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color: sevColor, background: `${sevColor}18`, padding: '2px 6px', borderRadius: 4 }}>
                        {p.severity}
                      </span>
                      <b style={{ fontSize: 13, color: 'var(--text)' }}>{p.title}</b>
                    </div>
                    {p.cli_command && (
                      <button
                        className="btn sm"
                        onClick={() => copyToClipboard(p.cli_command!, p.id)}
                        style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}
                      >
                        <Icon name="code" size={12} />
                        <span>{copiedId === p.id ? 'Copied!' : 'Copy Agent Action'}</span>
                      </button>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
                    <b>Root Cause:</b> {p.root_cause}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--ok)' }}>
                    <b>Remediation:</b> {p.recommendation}
                  </div>
                  {p.cli_command && (
                    <div className="mono" style={{ marginTop: 6, fontSize: 11, background: 'rgba(0,0,0,0.3)', padding: '4px 8px', borderRadius: 4, color: 'var(--accent)' }}>
                      $ {p.cli_command}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Grid: 2D Loss Landscape & SVD Spectral Ranks */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 380px) 1fr', gap: 24, marginBottom: 24 }}>
        {/* Filter-Normalized 2D Loss Landscape */}
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: 20,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div className="row sp" style={{ marginBottom: 12 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>2D Loss Landscape Flatness</h3>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                Li et al. (2018) Filter-Normalized Contour
              </div>
            </div>
            {landscape && (
              <span className="mono tag" style={{ fontSize: 10, background: 'rgba(16, 185, 129, 0.1)', color: 'var(--ok)' }}>
                {landscape.flatness_grade}
              </span>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'center', margin: 'auto 0' }}>
            <canvas
              ref={canvasRef}
              width={280}
              height={280}
              style={{
                borderRadius: 6,
                border: '1px solid var(--border)',
                background: '#090d16',
                boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              }}
            />
          </div>

          {landscape && (
            <div className="row sp mono" style={{ marginTop: 14, fontSize: 11, color: 'var(--text-muted)' }}>
              <span>Flatness: <b style={{ color: 'var(--text)' }}>{landscape.flatness_score}</b></span>
              <span>Center Loss: <b style={{ color: 'var(--ok)' }}>{landscape.center_loss}</b></span>
            </div>
          )}
        </div>

        {/* SVD Spectral Rank Representation Collapse Diagnostics */}
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: 20,
          }}
        >
          <div className="row sp" style={{ marginBottom: 14 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Layer Spectral Ranks &amp; Representation Collapse</h3>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                Dimensionality of learned representations via SVD entropy: exp(-&#8721; p_i ln p_i)
              </div>
            </div>
            {explanation && (
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Target Class: <b style={{ color: 'var(--accent)' }}>{explanation.target_class}</b>
              </span>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            {explanation &&
              Object.entries(explanation.spectral_ranks).map(([name, rank]) => {
                const ratio = rank.collapse_ratio;
                const status = ratio < 0.2 ? 'collapsed' : ratio < 0.45 ? 'reduced' : 'healthy';
                const statusColor = status === 'collapsed' ? 'var(--danger)' : status === 'reduced' ? 'var(--warning)' : 'var(--ok)';
                return (
                  <div
                    key={name}
                    style={{
                      background: 'var(--surface-sunken)',
                      border: `1px solid ${status === 'collapsed' ? 'rgba(239, 68, 68, 0.4)' : 'var(--border)'}`,
                      borderRadius: 6,
                      padding: 12,
                    }}
                  >
                    <div className="row sp" style={{ marginBottom: 6 }}>
                      <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }} title={name}>
                        {name.length > 18 ? '…' + name.slice(-16) : name}
                      </span>
                      <span className="mono" style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', color: statusColor, padding: '2px 5px', borderRadius: 4, background: `${statusColor}18` }}>
                        {status}
                      </span>
                    </div>

                    <div className="row sp mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                      <span>Rank: <b style={{ color: 'var(--text)' }}>{rank.effective_rank}</b> / {rank.max_rank}</span>
                      <span>{(ratio * 100).toFixed(1)}%</span>
                    </div>

                    <div style={{ height: 4, background: 'rgba(255, 255, 255, 0.08)', borderRadius: 2, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${Math.min(100, Math.max(5, ratio * 100))}%`, background: statusColor, transition: 'width 0.3s ease' }} />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      </div>

      {!loading && !explanation && (
        <EmptyState
          icon={<Icon name="xai" size={32} />}
          title="No xAI Explanations Found"
          hint="Train a model using Sentinel or add 'sentinel.log_explanation()' to your training loop for instant axiomatic interpretability."
        />
      )}

      {/* Integrated Gradients Feature Attributions */}
      {explanation && (
        <div style={{ display: 'grid', gap: 24 }}>
          <div
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: 20,
            }}
          >
            <div className="row sp" style={{ marginBottom: 16 }}>
              <div>
                <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Integrated Gradients: Axiomatic Feature Attributions</h3>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                  Path integral attributions satisfying completeness: &#8721; Attr_i &#8776; f(x) - f(x_0)
                </div>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <span className="tag" style={{ background: 'rgba(16, 185, 129, 0.1)', color: 'var(--ok)', borderColor: 'rgba(16, 185, 129, 0.25)', fontSize: 11 }}>
                  ● Positive Evidence
                </span>
                <span className="tag" style={{ background: 'rgba(244, 63, 94, 0.1)', color: 'var(--danger)', borderColor: 'rgba(244, 63, 94, 0.25)', fontSize: 11 }}>
                  ● Negative Evidence
                </span>
              </div>
            </div>

            <div style={{ display: 'grid', gap: 10 }}>
              {explanation.attributions.map((attr) => {
                const isPositive = attr.attribution >= 0;
                const barColor = isPositive ? 'var(--ok)' : 'var(--danger)';
                const pct = Math.min(100, Math.max(2, attr.importance_pct));

                return (
                  <div
                    key={attr.feature}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '160px 100px 1fr 90px',
                      alignItems: 'center',
                      gap: 16,
                      padding: '8px 12px',
                      background: 'var(--surface-sunken)',
                      borderRadius: 6,
                      fontSize: 12,
                    }}
                  >
                    <span className="mono" style={{ fontWeight: 600, color: 'var(--text)' }} title={attr.feature}>
                      {attr.feature}
                    </span>
                    <span className="mono" style={{ color: 'var(--text-muted)' }}>
                      val: <b style={{ color: 'var(--text)' }}>{attr.value}</b>
                    </span>

                    <div style={{ position: 'relative', height: 16, background: 'rgba(255, 255, 255, 0.05)', borderRadius: 3, overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${pct}%`,
                          background: barColor,
                          borderRadius: 3,
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </div>

                    <span className="mono" style={{ textAlign: 'right', fontWeight: 600, color: barColor }}>
                      {isPositive ? '+' : ''}{attr.attribution.toFixed(4)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Interactive What-If Perturbation Tester */}
          <div
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: 20,
            }}
          >
            <div className="row sp" style={{ marginBottom: 14 }}>
              <div>
                <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Interactive What-If Perturbation Tester</h3>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                  Modify feature values below to simulate counterfactuals and recompute Integrated Gradients path integrals.
                </div>
              </div>
              <button
                className="btn primary sm"
                onClick={handleRecalculate}
                disabled={computing}
                style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              >
                <Icon name="bolt" size={14} />
                <span>{computing ? 'Calculating Path Integral…' : 'Explain Prediction'}</span>
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
              {explanation.attributions.map((attr) => (
                <div
                  key={'input_' + attr.feature}
                  style={{
                    background: 'var(--surface-sunken)',
                    border: '1px solid var(--border)',
                    borderRadius: 6,
                    padding: '8px 12px',
                  }}
                >
                  <label className="mono" style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                    {attr.feature}
                  </label>
                  <input
                    type="number"
                    step="0.05"
                    value={inputValues[attr.feature] ?? attr.value}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value) || 0;
                      setInputValues((prev) => ({ ...prev, [attr.feature]: v }));
                    }}
                    style={{
                      width: '100%',
                      boxSizing: 'border-box',
                      background: 'var(--surface)',
                      border: '1px solid var(--border)',
                      borderRadius: 4,
                      padding: '4px 8px',
                      color: 'var(--text)',
                      fontFamily: 'monospace',
                      fontSize: 12,
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
