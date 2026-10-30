import { useEffect, useMemo, useState } from 'react';
import { api, type GenerateResult, type ModelMeta, type PredictResult, type MetricPoint, type TrackedRun } from '../api';
import { usePoll } from '../hooks';
import { Icon } from '../icons';
import { LineChart, type Series } from '../LineChart';
import { EmptyState, useToast } from '../ui';

const COLORS = ['#60a5fa', '#f59e0b', '#4ade80', '#f472b6', '#a78bfa', '#fb7185', '#2dd4bf', '#facc15'];
const fmtNum = (v: number | undefined) => (v === undefined ? '—' : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : (+v.toPrecision(4)).toString());
const when = (ms: number | null) => (ms ? new Date(ms).toLocaleString() : '—');
const dur = (r: TrackedRun) => (r.start_time && r.end_time ? `${((r.end_time - r.start_time) / 1000).toFixed(1)}s` : '—');

const bytes = (n: number) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

/** Generated samples: a scatter plot for 2-D data, otherwise a table of the first rows. */
function Scatter({ result }: { result: GenerateResult }) {
  const rows = result.samples;
  if (result.features.length === 2) {
    const xs = rows.map((r) => r[0]), ys = rows.map((r) => r[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const W = 300, H = 220, m = 28, sx = (v: number) => m + ((v - x0) / Math.max(1e-9, x1 - x0)) * (W - 2 * m), sy = (v: number) => H - m - ((v - y0) / Math.max(1e-9, y1 - y0)) * (H - 2 * m);
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${rows.length} generated samples`} style={{ maxWidth: 360, border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg)' }}>
        {rows.map((r, i) => <circle key={i} cx={sx(r[0])} cy={sy(r[1])} r="2.2" fill="#7c5cff" fillOpacity=".6" />)}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="#8c8c99">{result.features[0]} ({x0.toFixed(1)} … {x1.toFixed(1)})</text>
        <text x={8} y={H / 2} fontSize="10" fill="#8c8c99" transform={`rotate(-90 8 ${H / 2})`} textAnchor="middle">{result.features[1]} ({y0.toFixed(1)} … {y1.toFixed(1)})</text>
      </svg>
    );
  }
  return <div className="mono" style={{ fontSize: 11 }}>{rows.slice(0, 8).map((r, i) => <div key={i}>{r.map((v) => v.toFixed(3)).join('  ')}</div>)}</div>;
}

/** Download a run's exports and try its model on new data. */
function ModelPanel({ run }: { run: TrackedRun }) {
  const toast = useToast();
  const [info, setInfo] = useState<{ files: { name: string; bytes: number }[]; meta: ModelMeta | null } | null>(null);
  const [vals, setVals] = useState<string[]>([]);
  const [res, setRes] = useState<PredictResult | null>(null);
  const [gen, setGen] = useState<GenerateResult | null>(null);
  const [count, setCount] = useState(200);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setRes(null);
    setGen(null);
    api.artifacts(run.run_id).then((i) => { setInfo(i); setVals((i.meta?.features ?? []).map(() => '0')); }).catch(() => setInfo({ files: [], meta: null }));
  }, [run.run_id]);
  if (!info) return null;
  const meta = info.meta;
  const sample = async () => {
    setBusy(true);
    try { setGen(await api.generate(run.run_id, count, Math.floor(Math.random() * 1e6))); } catch (e) { toast((e as Error).message, true); } finally { setBusy(false); }
  };
  const go = async () => {
    setBusy(true);
    try { setRes(await api.predict(run.run_id, [vals.map(Number)])); } catch (e) { toast((e as Error).message, true); } finally { setBusy(false); }
  };
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="row sp"><b>Model · {run.run_name} <span className="mono" style={{ color: 'var(--text-muted)', fontWeight: 400 }}>{run.run_id.slice(0, 8)}</span></b>
        {meta && <span className="tag">{meta.task}</span>}</div>
      {!meta ? <p className="hint" style={{ marginTop: 8 }}>This run exported no model. Train a graph component (Components or Builder) to get weights, TorchScript and ONNX.</p> : (
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', marginTop: 12 }}>
          <div>
            <div className="stat-title">Exports</div>
            <div className="chips" style={{ margin: '8px 0' }}>
              {info.files.map((f) => <a key={f.name} className="chip on" href={`/api/tracking/runs/${run.run_id}/artifacts/${f.name}`} download title="Download">{f.name} · {bytes(f.bytes)}</a>)}
            </div>
            <p className="hint">TorchScript loads with <span className="mono">torch.jit.load</span> and needs no Sentinel code. <span className="mono">sentinel model predict {run.run_id.slice(0, 8)} -i …</span> does the same from the terminal.</p>
          </div>
          {meta.task === 'gan' ? (
          <div>
            <div className="stat-title">Generate</div>
            <div className="row" style={{ margin: '8px 0' }}>
              <label className="field" style={{ margin: 0 }}><span style={{ display: 'block', fontSize: 11, color: 'var(--text-secondary)', marginBottom: 2 }}>samples</span>
                <input type="number" min={1} max={5000} value={count} onChange={(e) => setCount(Math.max(1, Math.min(5000, +e.target.value || 1)))} aria-label="samples" style={{ width: 100 }} /></label>
              <button className="btn primary sm" style={{ alignSelf: 'flex-end' }} onClick={sample} disabled={busy}>{busy ? 'Sampling…' : 'Generate'}</button>
            </div>
            {gen && <Scatter result={gen} />}
          </div>
          ) : (
          <div>
            <div className="stat-title">Try it</div>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))', margin: '8px 0' }}>
              {meta.features.map((f, i) => (
                <label key={f} className="field" style={{ margin: 0 }}><span style={{ display: 'block', fontSize: 11, color: 'var(--text-secondary)', marginBottom: 2 }}>{f}</span>
                  <input type="number" step="any" value={vals[i] ?? ''} onChange={(e) => setVals((v) => v.map((x, j) => (j === i ? e.target.value : x)))} aria-label={f} />
                </label>
              ))}
            </div>
            <button className="btn primary sm" onClick={go} disabled={busy || vals.some((v) => v === '' || Number.isNaN(Number(v)))}>{busy ? 'Running…' : 'Predict'}</button>
            {res?.task === 'classification' && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 16, fontWeight: 600 }}>{res.predictions[0].label}</div>
                {Object.entries(res.predictions[0].probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([c, p]) => (
                  <div key={c} className="row" style={{ gap: 8, margin: '4px 0' }}><span className="mono" style={{ width: 90, fontSize: 11 }}>{c}</span>
                    <div className="bar" style={{ flex: 1, margin: 0 }}><div style={{ width: `${p * 100}%` }} /></div><span className="mono" style={{ fontSize: 11, width: 44, textAlign: 'right' }}>{(p * 100).toFixed(1)}%</span></div>
                ))}
              </div>
            )}
            {res?.task === 'regression' && <div className="stat-value" style={{ marginTop: 10 }}>{res.values[0].map((v) => +v.toPrecision(5)).join(', ')}</div>}
            {res?.task === 'vae' && <div className="mono" style={{ fontSize: 12, marginTop: 10 }}>reconstruction {JSON.stringify(res.reconstruction[0].map((v) => +v.toPrecision(4)))}<br />latent mean {JSON.stringify(res.latent_mean[0].map((v) => +v.toPrecision(4)))}</div>}
          </div>
          )}
        </div>
      )}
    </div>
  );
}

export function Experiments({ go }: { go: (t: string) => void }) {
  const runs = usePoll(api.trackedRuns, 2000);
  const [picked, setPicked] = useState<string[]>([]);
  const [data, setData] = useState<Record<string, Record<string, MetricPoint[]>>>({});
  const list = runs.data ?? [];

  // default to the newest run
  useEffect(() => { if (!picked.length && list.length) setPicked([list[0].run_id]); }, [list, picked.length]);

  const active = useMemo(() => picked.join(','), [picked]);
  const anyRunning = list.some((r) => picked.includes(r.run_id) && r.status === 'running');
  useEffect(() => {
    let live = true;
    const pull = () => Promise.all(picked.map((id) => api.trackedRun(id).then((r) => [id, r.metrics] as const).catch(() => null)))
      .then((rs) => live && setData(Object.fromEntries(rs.filter((x): x is NonNullable<typeof x> => !!x))));
    pull();
    const t = anyRunning ? setInterval(pull, 1500) : undefined;
    return () => { live = false; if (t) clearInterval(t); };
  }, [active, anyRunning]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-8)));
  const color = (id: string) => COLORS[picked.indexOf(id) % COLORS.length];
  const metricNames = [...new Set(picked.flatMap((id) => Object.keys(data[id] ?? {})))].sort();
  const paramKeys = [...new Set(list.flatMap((r) => Object.keys(r.params)))].sort();
  const metricKeys = [...new Set(list.flatMap((r) => Object.keys(r.latest)))].sort();

  return (
    <>
      <div className="page-header"><div><div className="eyebrow">TRACKING</div><h1 className="page-title">Experiments</h1>
        <p className="page-sub">Metric curves from <span className="mono">mlruns/</span>. Select up to 8 runs to compare.</p></div></div>
      {metricNames.length > 0 && (
        <div className="grid cards charts" style={{ marginBottom: 16 }}>
          {metricNames.map((m) => (
            <LineChart key={m} title={m} series={picked.filter((id) => data[id]?.[m]).map<Series>((id) => {
              const r = list.find((x) => x.run_id === id);
              return { id, label: `${r?.run_name ?? 'run'} · ${id.slice(0, 6)}`, color: color(id), points: data[id][m] };
            })} />
          ))}
        </div>
      )}
      {picked.length > 0 && list.find((r) => r.run_id === picked[0]) && <ModelPanel run={list.find((r) => r.run_id === picked[0])!} />}
      <div className="card" style={{ padding: 0, overflowX: 'auto' }}>
        <table>
          <thead><tr><th /><th>Run</th><th>Status</th><th>Started</th><th>Time</th>
            {metricKeys.map((k) => <th key={k}>{k}</th>)}{paramKeys.map((k) => <th key={k} style={{ color: '#888' }}>{k}</th>)}</tr></thead>
          <tbody>
            {list.length === 0 && <tr><td colSpan={5}><EmptyState icon={<Icon name="experiments" size={20} />} title="No tracked runs yet" hint="Components log metrics with the Sentinel SDK automatically.">
              <button className="btn primary" onClick={() => go('components')}>Train something</button></EmptyState></td></tr>}
            {list.map((r) => (
              <tr key={r.run_id} onClick={() => toggle(r.run_id)} style={{ cursor: 'pointer', background: picked.includes(r.run_id) ? '#101010' : undefined }}>
                <td><input type="checkbox" checked={picked.includes(r.run_id)} readOnly style={{ accentColor: picked.includes(r.run_id) ? color(r.run_id) : undefined }} /></td>
                <td><b>{r.run_name}</b><div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{r.run_id.slice(0, 8)}</div></td>
                <td><span className="pill"><span className={'dot ' + (r.status === 'finished' ? 'ok' : r.status === 'running' ? 'info pulse' : r.status === 'failed' ? 'err' : 'warn')} />{r.status}</span></td>
                <td style={{ color: 'var(--text-secondary)' }}>{when(r.start_time)}</td><td className="mono">{dur(r)}</td>
                {metricKeys.map((k) => <td key={k} className="mono">{fmtNum(r.latest[k])}</td>)}
                {paramKeys.map((k) => <td key={k} className="mono" style={{ color: '#888' }}>{r.params[k] ?? '—'}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
