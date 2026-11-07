import type { JobRecord, RepoStatus, Vitals } from '../api';
import { api } from '../api';
import { usePoll } from '../hooks';
import { Icon } from '../icons';
import { Sparkline } from '../Sparkline';
import { EmptyState, Skeleton, StatusPill, fmtGB, fmtTime, useToast } from '../ui';

const rate = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB/s` : b >= 1e3 ? `${(b / 1e3).toFixed(0)} KB/s` : `${b.toFixed(0)} B/s`);
const metric = (v: number | undefined) => (v === undefined ? '' : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(1) : (+v.toPrecision(3)).toString());

export function Overview({ repo, jobs, hw, onRepoChanged, go }: {
  repo: RepoStatus | null; jobs: JobRecord[]; hw: Vitals | null; onRepoChanged: () => void; go: (t: string) => void;
}) {
  const toast = useToast();
  const hist = usePoll(api.vitalsHistory, 3000);
  const exps = usePoll(api.trackedRuns, 3000);
  const pts = hist.data ?? [];
  const running = jobs.filter((j) => j.status === 'running');
  const ramPct = hw ? (hw.ram_used_bytes / hw.ram_total_bytes) * 100 : 0;
  const gpu = hw?.gpus.find((g) => g.gpu_util_percent !== null);
  const init = async () => { try { await api.initRepo(); toast('Sentinel repository initialized'); onRepoChanged(); } catch (e) { toast((e as Error).message, true); } };


  return (
    <>
      <section className="hero">
        <div className="row sp" style={{ alignItems: 'flex-start' }}>
          <div>
            <div className="eyebrow">WORKSPACE</div>
            <h1>Welcome to <span style={{ color: 'var(--accent)' }}>{repo?.workspace_name ?? 'Sentinel'}</span></h1>
            <p className="mono" style={{ fontSize: 12 }}>{repo?.full_workspace_path ?? <Skeleton h={14} w={320} />}</p>
          </div>
          <div className="row">
            <button className="btn" onClick={() => go('runs')}><Icon name="runs" size={14} /> Runs{running.length > 0 && ` · ${running.length} live`}</button>
            <button className="btn primary" onClick={() => go('visualizer')}><Icon name="visualizer" size={14} /> DAG Visualizer</button>
          </div>
        </div>
      </section>

      {running.length > 0 && (
        <div className="card alert hover" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 16px', marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="dot ok pulse" />
            <span style={{ fontSize: 13 }}>Training job active: <b className="mono" style={{ color: '#fff' }}>{running[0].spec.name}</b></span>
          </div>
          <button className="btn sm" onClick={() => go('runs')}>View Runs &amp; Logs →</button>
        </div>
      )}

      {repo && !repo.is_sentinel_repo && (
        <div className="alert">
          <span>This directory is not a Sentinel repository yet (no <span className="mono">sentinel.lock</span>).</span>
          <button className="btn sm" onClick={init}>Initialize</button>
        </div>
      )}


      <div className="grid stats">
        <div className="card hover"><div className="stat-title">CPU</div>
          {hw ? <div className="stat-value">{hw.cpu_util_percent.toFixed(0)}<small>%</small></div> : <Skeleton h={30} w={80} style={{ margin: '4px 0' }} />}
          <Sparkline values={pts.map((p) => p.cpu)} max={100} color="#60a5fa" /></div>
        <div className="card hover"><div className="stat-title">Memory</div>
          {hw ? <div className="stat-value">{fmtGB(hw.ram_used_bytes)} <small>/ {fmtGB(hw.ram_total_bytes)} GB</small></div> : <Skeleton h={30} w={120} style={{ margin: '4px 0' }} />}
          <Sparkline values={pts.map((p) => (p.ram_used / p.ram_total) * 100)} max={100} color={ramPct > 85 ? '#fbbf24' : '#5b9bff'} /></div>
        <div className="card hover"><div className="stat-title">Network</div>
          {hw ? <div className="stat-value" style={{ fontSize: 20 }}>↓ {rate(hw.network.reduce((a, n) => a + n.rx_bps, 0))}</div> : <Skeleton h={30} w={120} style={{ margin: '4px 0' }} />}
          <Sparkline values={pts.map((p) => p.net_rx)} color="#34d399" /></div>
        <div className="card hover"><div className="stat-title">{gpu ? `GPU · ${gpu.name}` : 'GPU'}</div>
          {hw ? (gpu ? <div className="stat-value">{gpu.gpu_util_percent!.toFixed(0)}<small>%</small></div> : <div className="stat-value" style={{ fontSize: 15, lineHeight: 1.4 }}>{hw.gpus[0]?.name ?? 'None detected'}</div>) : <Skeleton h={30} w={120} style={{ margin: '4px 0' }} />}
          {gpu ? <Sparkline values={pts.map((p) => p.gpus[0]?.util ?? 0)} max={100} color="#f472b6" /> : <div className="stat-foot" style={{ marginTop: 8 }}>{hw?.gpus.length ? 'live metrics unavailable on this GPU' : 'training runs on the CPU'}</div>}</div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(420px, 100%), 1fr))', marginTop: 16 }}>
        <div className="card" style={{ padding: 0 }}>
          <div className="row sp" style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}><b>Latest experiments</b><button className="btn sm" onClick={() => go('experiments')}>Compare →</button></div>
          {!exps.data ? <div style={{ padding: 16 }}><Skeleton h={14} style={{ marginBottom: 12 }} /><Skeleton h={14} w="80%" /></div>
            : exps.data.length === 0 ? <EmptyState icon={<Icon name="experiments" size={20} />} title="No experiments yet" hint="Train a component and its metric curves land here.">
                <button className="btn primary" onClick={() => go('components')}>Pick a template</button></EmptyState>
            : <table><tbody>{exps.data.slice(0, 5).map((r) => (
              <tr key={r.run_id} style={{ cursor: 'pointer' }} onClick={() => go('experiments')}>
                <td><b>{r.run_name}</b><div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{r.run_id.slice(0, 8)}</div></td>
                <td><span className="pill"><span className={'dot ' + (r.status === 'finished' ? 'ok' : r.status === 'running' ? 'info pulse' : 'err')} />{r.status}</span></td>
                <td><div className="chips">{Object.entries(r.latest).slice(0, 2).map(([k, v]) => <span className="chip" key={k}>{k} {metric(v)}</span>)}</div></td>
              </tr>))}</tbody></table>}
        </div>
        <div className="card" style={{ padding: 0 }}>
          <div className="row sp" style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}><b>Recent jobs</b><button className="btn sm" onClick={() => go('runs')}>All runs →</button></div>
          {jobs.length === 0 ? <EmptyState icon={<Icon name="runs" size={20} />} title="No runs yet" hint="Start a training run from Templates, an agent, or the CLI." />
            : <table><tbody>{[...jobs].reverse().slice(0, 5).map((j) => (
              <tr key={j.spec.id}><td>{j.spec.name}</td><td><StatusPill status={j.status} /></td><td style={{ color: 'var(--text-muted)' }}>{fmtTime(j.started_at)}</td></tr>))}</tbody></table>}
        </div>
      </div>
    </>
  );
}
