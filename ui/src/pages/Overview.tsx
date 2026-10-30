import type { Hardware, JobRecord, RepoStatus } from '../api';
import { api } from '../api';
import { StatusPill, fmtGB, fmtTime, useToast } from '../ui';

export function Overview({ repo, jobs, hw, onRepoChanged, go }: { repo: RepoStatus | null; jobs: JobRecord[]; hw: Hardware | null; onRepoChanged: () => void; go: (t: string) => void }) {
  const toast = useToast();
  const ramPct = hw ? (hw.ram_used_bytes / hw.ram_total_bytes) * 100 : 0;
  const recent = jobs.slice(0, 5);
  const init = async () => {
    try { await api.initRepo(); toast('Sentinel repository initialized'); onRepoChanged(); } catch (e) { toast((e as Error).message, true); }
  };
  return (
    <>
      <div className="page-header">
        <div><div className="eyebrow">WORKSPACE</div><h1 className="page-title">Workspace overview</h1>
          <p className="page-sub">{repo ? repo.full_workspace_path : 'Loading…'}</p></div>
        <button className="btn primary" onClick={() => go('builder')}>+ New component</button>
      </div>
      {repo && !repo.is_sentinel_repo && (
        <div className="alert">
          <span>This directory is not a Sentinel repository yet (no <span className="mono">sentinel.lock</span>).</span>
          <button className="btn sm" onClick={init}>Initialize</button>
        </div>
      )}
      <div className="grid stats">
        <div className="card"><div className="stat-title">Host CPU</div><div className="stat-value">{hw ? hw.cpu_util_percent.toFixed(1) : '—'}%</div>
          <div className="bar"><div style={{ width: `${hw?.cpu_util_percent ?? 0}%` }} /></div><div className="stat-foot">System cores monitored</div></div>
        <div className="card"><div className="stat-title">Host memory</div>
          <div className="stat-value">{hw ? fmtGB(hw.ram_used_bytes) : '—'} GB <small>/ {hw ? fmtGB(hw.ram_total_bytes) : '—'} GB</small></div>
          <div className="bar"><div style={{ width: `${ramPct}%` }} /></div><div className="stat-foot">RAM utilization</div></div>
        {hw?.gpus.map((g) => (
          <div className="card" key={g.index}><div className="stat-title">GPU {g.index}</div><div className="stat-value" style={{ fontSize: 16 }}>{g.name}</div>
            {g.gpu_util_percent !== null && <div className="bar"><div style={{ width: `${g.gpu_util_percent}%` }} /></div>}
            <div className="stat-foot">
              {g.gpu_util_percent !== null ? `${g.gpu_util_percent.toFixed(0)}%` : 'util n/a'}
              {g.vram_total_bytes !== null && ` · ${g.vram_used_bytes !== null ? fmtGB(g.vram_used_bytes) + '/' : ''}${fmtGB(g.vram_total_bytes)} GB`}
              {g.temperature_c !== null && ` · ${g.temperature_c.toFixed(0)}°C`}
              {g.vendor === 'apple' && ' · live metrics need root on macOS'}
            </div></div>
        ))}
      </div>
      <div className="card gap-t" style={{ padding: 0 }}>
        <div className="row sp" style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
          <b>Recent runs</b><button className="btn sm" onClick={() => go('runs')}>All runs →</button>
        </div>
        <table><tbody>
          {recent.length === 0 && <tr><td className="empty">No runs yet. Compose a component in the Builder and press Run.</td></tr>}
          {recent.map((j) => (
            <tr key={j.spec.id}><td>{j.spec.name}</td><td><StatusPill status={j.status} /></td><td style={{ color: 'var(--text-muted)' }}>{fmtTime(j.started_at)}</td></tr>
          ))}
        </tbody></table>
      </div>
    </>
  );
}
