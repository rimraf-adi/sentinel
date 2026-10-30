/** Tiny area chart for stat cards. Purely decorative: the number beside it carries the information. */
export function Sparkline({ values, color = '#7c5cff', max }: { values: number[]; color?: string; max?: number }) {
  if (values.length < 2) return <div className="skeleton" style={{ height: 34, marginTop: 8 }} aria-hidden />;
  const W = 200, H = 34;
  const hi = max ?? Math.max(...values, 1e-9), lo = max !== undefined ? 0 : Math.min(...values);
  const span = Math.max(1e-9, hi - lo);
  const pts = values.map((v, i) => [(i / (values.length - 1)) * W, H - 3 - ((v - lo) / span) * (H - 6)] as const);
  const line = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const id = `sp-${color.replace('#', '')}`;
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      <defs><linearGradient id={id} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={color} stopOpacity=".35" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient></defs>
      <polygon points={`0,${H} ${line} ${W},${H}`} fill={`url(#${id})`} />
      <polyline points={line} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
