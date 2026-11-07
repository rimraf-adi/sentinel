import { useEffect, useState } from 'react';
import { api, type ExplainResult, type TrackedRun } from '../api';
import { usePoll } from '../hooks';
import { Icon } from '../icons';
import { EmptyState, Skeleton } from '../ui';

export function Explain() {
  const runs = usePoll(api.trackedRuns, 5000);
  const [selectedRun, setSelectedRun] = useState<string>('live');
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, number>>({});
  const [computing, setComputing] = useState(false);

  // Poll live explanation if "live" selected
  useEffect(() => {
    let active = true;
    const fetchExplain = async () => {
      if (selectedRun === 'live') {
        const res = await api.liveExplain();
        if (active && res) {
          setExplanation(res);
          const initial: Record<string, number> = {};
          res.attributions.forEach((a) => {
            initial[a.feature] = a.value;
          });
          setInputValues((prev) => (Object.keys(prev).length === 0 ? initial : prev));
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

    fetchExplain();
    const iv = setInterval(() => {
      if (selectedRun === 'live') fetchExplain();
    }, 3000);

    return () => {
      active = false;
      clearInterval(iv);
    };
  }, [selectedRun]);

  const handleRecalculate = async () => {
    if (!explanation) return;
    setComputing(true);
    try {
      const row = explanation.attributions.map((a) => inputValues[a.feature] ?? a.value);
      const res = await api.explain(selectedRun === 'live' ? undefined : selectedRun, [row]);
      if (res) {
        setExplanation(res);
      }
    } catch (e) {
      console.error('Explanation recalculation failed:', e);
    } finally {
      setComputing(false);
    }
  };

  const runsList = runs.data ?? [];

  return (
    <div className="flow-page" style={{ padding: '24px 32px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div className="row sp" style={{ marginBottom: 20, flexWrap: 'wrap', gap: 16 }}>
        <div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--accent)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>
            AGENT-FIRST OBSERVABILITY · INTERPRETABILITY LAYER
          </div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span>Explainable AI (xAI) Diagnostics</span>
            <span className="tag" style={{ background: 'rgba(56, 189, 248, 0.1)', color: 'var(--accent)', borderColor: 'rgba(56, 189, 248, 0.25)', fontSize: 11 }}>
              Axiomatic Path Integral
            </span>
          </h2>
          <p style={{ margin: '6px 0 0', color: 'var(--text-muted)', fontSize: 13 }}>
            Axiomatic Integrated Gradients feature attribution and SVD spectral rank representation collapse detection.
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
          <Skeleton h={140} />
          <Skeleton h={300} />
        </div>
      )}

      {!loading && !explanation && (
        <EmptyState
          icon={<Icon name="xai" size={32} />}
          title="No xAI Explanations Found"
          hint="Train a model using Sentinel or add 'sentinel.log_explanation()' to your training loop for instant axiomatic interpretability."
        />
      )}

      {!loading && explanation && (
        <div style={{ display: 'grid', gap: 24 }}>
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
                  Dimensionality of learned representations via Singular Value Decomposition entropy: exp(-&#8721; p_i ln p_i)
                </div>
              </div>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Target Class: <b style={{ color: 'var(--accent)' }}>{explanation.target_class}</b>
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
              {Object.entries(explanation.spectral_ranks).map(([name, rank]) => {
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
                        {name.length > 22 ? '…' + name.slice(-20) : name}
                      </span>
                      <span
                        className="mono"
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          textTransform: 'uppercase',
                          color: statusColor,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: `${statusColor}18`,
                        }}
                      >
                        {status}
                      </span>
                    </div>

                    <div className="row sp mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                      <span>Rank: <b style={{ color: 'var(--text)' }}>{rank.effective_rank}</b> / {rank.max_rank}</span>
                      <span>{(ratio * 100).toFixed(1)}%</span>
                    </div>

                    <div style={{ height: 4, background: 'rgba(255, 255, 255, 0.08)', borderRadius: 2, overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.min(100, Math.max(5, ratio * 100))}%`,
                          background: statusColor,
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
              {Object.keys(explanation.spectral_ranks).length === 0 && (
                <div style={{ gridColumn: '1 / -1', color: 'var(--text-muted)', fontSize: 12, fontStyle: 'italic' }}>
                  No 2D weight matrices detected in current architecture.
                </div>
              )}
            </div>
          </div>

          {/* Integrated Gradients Feature Attributions */}
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

                    {/* Bidirectional Bar */}
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
