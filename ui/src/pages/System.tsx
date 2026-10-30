import { useMemo, useState } from 'react';
import { api, type GpuStats, type Vitals, type VitalsPoint } from '../api';
import { usePoll } from '../hooks';
import { LineChart, type Series } from '../LineChart';

const gb = (b: number) => (b / 1024 ** 3).toFixed(1);
const mb = (b: number) => (b >= 1024 ** 3 ? `${gb(b)} GB` : `${(b / 1024 ** 2).toFixed(0)} MB`);
const rate = (bps: number) => (bps >= 1e9 ? `${(bps / 1e9).toFixed(1)} GB/s` : bps >= 1e6 ? `${(bps / 1e6).toFixed(1)} MB/s` : bps >= 1e3 ? `${(bps / 1e3).toFixed(1)} KB/s` : `${bps.toFixed(0)} B/s`);
const pct = (v: number) => `${v.toFixed(0)}%`;
const na = (v: number | null | undefined, f: (x: number) => string) => (v === null || v === undefined ? 'n/a' : f(v));
const uptime = (s: number) => `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
const C = { a: '#60a5fa', b: '#f59e0b', c: '#4ade80', d: '#f472b6' };

function Bar({ value, warn = 70, bad = 90 }: { value: number; warn?: number; bad?: number }) {
  const color = value >= bad ? 'var(--danger)' : value >= warn ? 'var(--warn)' : '#fff';
  return <div className="bar"><div style={{ width: `${Math.min(100, Math.max(0, value))}%`, background: color }} /></div>;
}

function Gpu({ g }: { g: GpuStats }) {
  const mem = g.vram_used_bytes !== null && g.vram_total_bytes ? (g.vram_used_bytes / g.vram_total_bytes) * 100 : null;
  return (
    <div className="card">
      <div className="row sp"><b>GPU {g.index} · {g.name}</b><span className="tag">{g.vendor}</span></div>
      {g.gpu_util_percent !== null && <><div className="stat-title" style={{ marginTop: 10 }}>Utilization</div><Bar value={g.gpu_util_percent} /></>}
      <div className="kv">
        <span>Utilization</span><b>{na(g.gpu_util_percent, pct)}</b>
        <span>Memory</span><b>{g.vram_total_bytes !== null ? `${na(g.vram_used_bytes, mb)} / ${mb(g.vram_total_bytes)}${mem !== null ? ` (${mem.toFixed(0)}%)` : ''}` : 'n/a'}</b>
        <span>Memory controller</span><b>{na(g.mem_util_percent, pct)}</b>
        <span>Temperature</span><b>{na(g.temperature_c, (t) => `${t.toFixed(0)} °C`)}</b>
        <span>Power</span><b>{na(g.power_w, (p) => `${p.toFixed(0)} W${g.power_limit_w ? ` / ${g.power_limit_w.toFixed(0)} W` : ''}`)}</b>
        <span>Clocks (gfx / mem)</span><b>{g.clock_graphics_mhz !== null ? `${g.clock_graphics_mhz} / ${g.clock_memory_mhz ?? 'n/a'} MHz` : 'n/a'}</b>
        <span>Fan</span><b>{na(g.fan_percent, pct)}</b>
        <span>P-state</span><b>{g.pstate ?? 'n/a'}</b>
        <span>Driver</span><b>{g.driver ?? 'n/a'}</b>
      </div>
      {g.vendor === 'apple' && <p className="hint">macOS exposes live Apple GPU metrics only to root (<span className="mono">powermetrics</span>), so they are shown as n/a rather than guessed.</p>}
    </div>
  );
}

function series(points: VitalsPoint[], id: string, label: string, color: string, pick: (p: VitalsPoint) => number | null | undefined): Series {
  const now = points.length ? points[points.length - 1].t : 0;
  return {
    id, label, color,
    points: points.flatMap((p) => { const v = pick(p); return v === null || v === undefined ? [] : [{ step: Math.round((p.t - now) / 1000), value: v, timestamp: p.t }]; }),
  };
}

export function System({ hw }: { hw: Vitals | null }) {
  const info = usePoll(api.system, 60000);
  const hist = usePoll(api.vitalsHistory, 2000);
  const pts = hist.data ?? [];
  const gpuCount = hw?.gpus.length ?? 0;
  const hasGpuMetrics = pts.some((p) => p.gpus.some((g) => g.util !== null || g.mem_used !== null));
  const hasTemp = pts.some((p) => p.gpus.some((g) => g.temp !== null));
  const hasPower = pts.some((p) => p.gpus.some((g) => g.power !== null));
  const gpuIdx = useMemo(() => Array.from({ length: gpuCount }, (_, i) => i), [gpuCount]);
  const [allSensors, setAllSensors] = useState(false);
  const sys = info.data;
  const ramPct = hw ? (hw.ram_used_bytes / hw.ram_total_bytes) * 100 : 0;

  return (
    <>
      <div className="page-header">
        <div><div className="eyebrow">SYSTEM</div><h1 className="page-title">System vitals</h1>
          <p className="page-sub">{sys ? `${sys.host ?? 'this machine'} · ${sys.os} ${sys.os_version?.trim() ?? ''} · ${sys.arch} · ${sys.cpu_brand} · ${sys.cpu_count} cores · ${gb(sys.ram_total_bytes)} GB` : 'Loading…'}
            {hw ? ` · up ${uptime(hw.uptime_secs)}` : ''}</p></div>
        <span className="pill"><span className={'dot ' + (hw ? 'ok' : 'err')} />{hw ? 'live · 1 Hz' : 'waiting for telemetry'}</span>
      </div>

      {hw && (
        <div className="grid stats">
          <div className="card"><div className="stat-title">CPU</div><div className="stat-value">{hw.cpu_util_percent.toFixed(1)}%</div><Bar value={hw.cpu_util_percent} />
            <div className="stat-foot">load {hw.load_avg.map((l) => l.toFixed(2)).join(' · ')}{hw.cpu_freq_mhz.length ? ` · ${Math.round(hw.cpu_freq_mhz.reduce((a, b) => a + b, 0) / hw.cpu_freq_mhz.length)} MHz avg` : ''}</div></div>
          <div className="card"><div className="stat-title">Memory</div><div className="stat-value">{gb(hw.ram_used_bytes)} <small>/ {gb(hw.ram_total_bytes)} GB</small></div><Bar value={ramPct} warn={80} bad={92} />
            <div className="stat-foot">swap {gb(hw.swap_used_bytes)} / {gb(hw.swap_total_bytes)} GB</div></div>
          <div className="card"><div className="stat-title">Network</div>
            <div className="stat-value" style={{ fontSize: 18 }}>↓ {rate(hw.network.reduce((a, n) => a + n.rx_bps, 0))}</div>
            <div className="stat-value" style={{ fontSize: 18 }}>↑ {rate(hw.network.reduce((a, n) => a + n.tx_bps, 0))}</div>
            <div className="stat-foot">{hw.network.length} active interface{hw.network.length === 1 ? '' : 's'}</div></div>
          {hw.daemon && <div className="card"><div className="stat-title">Sentinel itself</div><div className="stat-value">{mb(hw.daemon.rss_bytes)}</div>
            <div className="stat-foot">{hw.daemon.cpu_percent.toFixed(1)}% cpu · pid {hw.daemon.pid} · io r {rate(hw.daemon.io_read_bps)} w {rate(hw.daemon.io_write_bps)}</div></div>}
        </div>
      )}

      <div className="grid cards charts gap-t">
        <LineChart title="CPU %" xLabel="s" format={pct} series={[series(pts, 'cpu', 'cpu', C.a, (p) => p.cpu)]} />
        <LineChart title="Memory" xLabel="s" format={(v) => `${gb(v)} GB`} series={[series(pts, 'ram', 'RAM used', C.b, (p) => p.ram_used), series(pts, 'swap', 'swap used', C.d, (p) => p.swap_used)]} />
        <LineChart title="Network" xLabel="s" format={rate} series={[series(pts, 'rx', 'download', C.c, (p) => p.net_rx), series(pts, 'tx', 'upload', C.a, (p) => p.net_tx)]} />
        <LineChart title="Sentinel process" xLabel="s" format={mb} series={[series(pts, 'rss', 'sentinel memory (RSS)', C.b, (p) => p.daemon_rss)]} />
        {hasGpuMetrics && <LineChart title="GPU utilization" xLabel="s" format={pct} series={gpuIdx.map((i) => series(pts, `gu${i}`, `GPU ${i}`, Object.values(C)[i % 4], (p) => p.gpus[i]?.util))} />}
        {hasGpuMetrics && <LineChart title="GPU memory" xLabel="s" format={mb} series={gpuIdx.map((i) => series(pts, `gm${i}`, `GPU ${i}`, Object.values(C)[i % 4], (p) => p.gpus[i]?.mem_used))} />}
        {hasTemp && <LineChart title="GPU temperature" xLabel="s" format={(v) => `${v.toFixed(0)} °C`} series={gpuIdx.map((i) => series(pts, `gt${i}`, `GPU ${i}`, Object.values(C)[i % 4], (p) => p.gpus[i]?.temp))} />}
        {hasPower && <LineChart title="GPU power" xLabel="s" format={(v) => `${v.toFixed(0)} W`} series={gpuIdx.map((i) => series(pts, `gp${i}`, `GPU ${i}`, Object.values(C)[i % 4], (p) => p.gpus[i]?.power))} />}
      </div>

      {hw && (
        <>
          <div className="nav-title" style={{ padding: 0, margin: '22px 0 8px' }}>GPUs</div>
          {hw.gpus.length === 0
            ? <div className="card empty">No GPU detected{sys?.gpu_note ? ` (${sys.gpu_note})` : ''}. Training runs on the CPU.<br /><span className="mono" style={{ fontSize: 11 }}>sentinel doctor</span> explains how to enable GPU support on this OS.</div>
            : <div className="grid cards">{hw.gpus.map((g) => <Gpu key={g.index} g={g} />)}</div>}

          <div className="nav-title" style={{ padding: 0, margin: '22px 0 8px' }}>CPU cores</div>
          <div className="card"><div className="cores">{hw.per_core.map((c, i) => (
            <div key={i}><div className="stat-foot">core {i}{hw.cpu_freq_mhz[i] ? ` · ${hw.cpu_freq_mhz[i]} MHz` : ''}</div><Bar value={c} /><div className="mono" style={{ fontSize: 11 }}>{c.toFixed(0)}%</div></div>
          ))}</div></div>

          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', marginTop: 14 }}>
            <div className="card" style={{ padding: 0 }}>
              <div className="card-h">Disks</div>
              <table><thead><tr><th>Mount</th><th>Type</th><th>Used</th><th /></tr></thead><tbody>
                {hw.disks.length === 0 && <tr><td colSpan={4} className="empty">No disks reported.</td></tr>}
                {hw.disks.map((d) => { const used = d.total_bytes - d.available_bytes, p = (used / d.total_bytes) * 100; return (
                  <tr key={d.name + d.mount}><td className="mono">{d.mount}</td><td>{d.fs} · {d.kind}{d.removable ? ' · removable' : ''}</td>
                    <td className="mono">{gb(used)} / {gb(d.total_bytes)} GB</td><td style={{ width: 90 }}><Bar value={p} warn={80} bad={92} /></td></tr>
                ); })}
              </tbody></table>
            </div>
            <div className="card" style={{ padding: 0 }}>
              <div className="card-h">Network interfaces</div>
              <table><thead><tr><th>Interface</th><th>↓</th><th>↑</th><th>Total ↓ / ↑</th><th>Errors</th></tr></thead><tbody>
                {hw.network.length === 0 && <tr><td colSpan={5} className="empty">No active interfaces.</td></tr>}
                {hw.network.map((n) => (
                  <tr key={n.name}><td className="mono">{n.name}</td><td className="mono">{rate(n.rx_bps)}</td><td className="mono">{rate(n.tx_bps)}</td>
                    <td className="mono">{mb(n.rx_total)} / {mb(n.tx_total)}</td><td className="mono" style={{ color: n.rx_errors + n.tx_errors ? 'var(--danger)' : undefined }}>{n.rx_errors + n.tx_errors}</td></tr>
                ))}
              </tbody></table>
            </div>
            <div className="card" style={{ padding: 0 }}>
              <div className="card-h row sp">Temperature sensors
                {hw.sensors.length > 8 && <button className="btn sm" onClick={() => setAllSensors(!allSensors)}>{allSensors ? 'Show hottest 8' : `Show all ${hw.sensors.length}`}</button>}</div>
              <table><tbody>
                {hw.sensors.length === 0 && <tr><td className="empty">This platform exposes no temperature sensors to unprivileged processes.</td></tr>}
                {[...hw.sensors].sort((a, b) => b.temp_c - a.temp_c).slice(0, allSensors ? undefined : 8).map((s, i) => (
                  <tr key={s.label + i}><td>{s.label}</td><td className="mono" style={{ color: s.critical_c && s.temp_c >= s.critical_c * 0.9 ? 'var(--danger)' : undefined }}>{s.temp_c.toFixed(0)} °C</td>
                    <td className="mono" style={{ color: 'var(--text-muted)' }}>{s.max_c ? `max ${s.max_c.toFixed(0)}` : ''}</td></tr>
                ))}
              </tbody></table>
            </div>
          </div>

          <div className="nav-title" style={{ padding: 0, margin: '22px 0 8px' }}>Running jobs</div>
          <div className="card" style={{ padding: 0 }}>
            <table><thead><tr><th>Job</th><th>PID</th><th>Processes</th><th>CPU (cores)</th><th>Memory</th><th>GPU memory</th><th>Disk read / write</th></tr></thead><tbody>
              {hw.jobs.length === 0 && <tr><td colSpan={7} className="empty">No running jobs. Start one from Components or the Builder.</td></tr>}
              {hw.jobs.map((j) => (
                <tr key={j.job_id}><td><b>{j.name}</b></td><td className="mono">{j.stat.pid}</td><td className="mono">{j.stat.processes}</td>
                  <td className="mono">{(j.stat.cpu_percent / 100).toFixed(2)}</td><td className="mono">{mb(j.stat.rss_bytes)}</td>
                  <td className="mono">{na(j.gpu_mem_bytes, mb)}</td><td className="mono">{rate(j.stat.io_read_bps)} / {rate(j.stat.io_write_bps)}</td></tr>
              ))}
            </tbody></table>
          </div>
        </>
      )}
    </>
  );
}
