import { useEffect, useMemo, useState } from 'react';
import { api, type MetricPoint, type TrackedRun } from '../api';
import { usePoll } from '../hooks';
import { Icon } from '../icons';
import { LineChart, type Series } from '../LineChart';
import { EmptyState } from '../ui';

const COLORS = ['#60a5fa', '#f59e0b', '#4ade80', '#f472b6', '#a78bfa', '#fb7185', '#2dd4bf', '#facc15'];
const fmtNum = (v: number | undefined) => (v === undefined ? '—' : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : (+v.toPrecision(4)).toString());
const when = (ms: number | null) => (ms ? new Date(ms).toLocaleString() : '—');
const dur = (r: TrackedRun) => (r.start_time && r.end_time ? `${((r.end_time - r.start_time) / 1000).toFixed(1)}s` : '—');

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
