import { useMemo, useState } from 'react';
import type { MetricPoint } from './api';

export interface Series { id: string; label: string; color: string; points: MetricPoint[] }

const W = 520, H = 220, M = { l: 46, r: 12, t: 10, b: 26 };

function ticks(min: number, max: number, n = 5): number[] {
  if (min === max) return [min];
  const raw = (max - min) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
const fmt = (v: number) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : +v.toPrecision(3) + '');

/** Dependency-free multi-series line chart with hover readout. */
export function LineChart({ title, series }: { title: string; series: Series[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const all = series.flatMap((s) => s.points);
  const dom = useMemo(() => {
    if (!all.length) return null;
    const xs = all.map((p) => p.step), ys = all.map((p) => p.value);
    let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
    const pad = (y1 - y0) * 0.06;
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: y0 - pad, y1: y1 + pad };
  }, [all.length, series]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!dom) return <div className="card"><b>{title}</b><div className="empty">No data yet.</div></div>;

  const sx = (x: number) => M.l + ((x - dom.x0) / Math.max(1e-9, dom.x1 - dom.x0)) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - ((y - dom.y0) / (dom.y1 - dom.y0)) * (H - M.t - M.b);
  const steps = [...new Set(all.map((p) => p.step))].sort((a, b) => a - b);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = dom.x0 + (((e.clientX - r.left) / r.width) * W - M.l) / (W - M.l - M.r) * (dom.x1 - dom.x0);
    setHover(steps.reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a), steps[0]));
  };

  return (
    <div className="card">
      <div className="row sp"><b>{title}</b>
        {hover !== null && <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>step {hover}</span>}</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label={title}>
        {ticks(dom.y0, dom.y1).map((t) => (
          <g key={t}><line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} stroke="#1c1c1c" />
            <text x={M.l - 6} y={sy(t) + 3} textAnchor="end" fontSize="10" fill="#777">{fmt(t)}</text></g>
        ))}
        {ticks(dom.x0, dom.x1, 6).map((t) => (
          <text key={t} x={sx(t)} y={H - 8} textAnchor="middle" fontSize="10" fill="#777">{t}</text>
        ))}
        {series.map((s) => (
          <polyline key={s.id} fill="none" stroke={s.color} strokeWidth="1.6" strokeLinejoin="round"
            points={s.points.map((p) => `${sx(p.step)},${sy(p.value)}`).join(' ')} />
        ))}
        {hover !== null && <line x1={sx(hover)} x2={sx(hover)} y1={M.t} y2={H - M.b} stroke="#444" />}
        {hover !== null && series.map((s) => {
          const p = s.points.reduce<MetricPoint | null>((a, b) => (a && Math.abs(a.step - hover) <= Math.abs(b.step - hover) ? a : b), null);
          return p ? <circle key={s.id} cx={sx(p.step)} cy={sy(p.value)} r="3.5" fill={s.color} /> : null;
        })}
      </svg>
      <div className="legend">
        {series.map((s) => {
          const p = hover === null ? s.points[s.points.length - 1] : s.points.reduce<MetricPoint | null>((a, b) => (a && Math.abs(a.step - hover) <= Math.abs(b.step - hover) ? a : b), null);
          return <span key={s.id}><i style={{ background: s.color }} />{s.label}{p && <b className="mono"> {fmt(p.value)}</b>}</span>;
        })}
      </div>
    </div>
  );
}
