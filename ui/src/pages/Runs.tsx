import { useEffect, useState } from 'react';
import { api, type JobRecord } from '../api';
import { Icon } from '../icons';
import { EmptyState, Modal, StatusPill, fmtTime, useToast } from '../ui';
import { Flow } from './Flow';

function JobInspector({ job, onClose, go }: { job: JobRecord; onClose: () => void; go: (t: string) => void }) {
  const [tab, setTab] = useState<'flow' | 'logs'>(job.status === 'running' ? 'flow' : 'logs');
  const [lines, setLines] = useState<string[]>([]);

  useEffect(() => {
    let live = true;
    const pull = () => api.logs(job.spec.id).then((r) => live && setLines(r.logs)).catch(() => {});
    pull();
    const t = setInterval(pull, job.status === 'running' ? 1000 : 60000);
    return () => { live = false; clearInterval(t); };
  }, [job.spec.id, job.status]);

  return (
    <Modal wide title={`Live Telemetry · ${job.spec.name}`} sub={job.spec.id} onClose={onClose}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {/* Tab switch between Live Flow & Logs */}
        <div className="row sp" style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
          <div className="chips" role="group">
            <button 
              className={'chip' + (tab === 'flow' ? ' on' : '')} 
              onClick={() => setTab('flow')}
            >
              <span className={'dot ' + (job.status === 'running' ? 'ok pulse' : 'info')} style={{ marginRight: 6 }} />
              Live Gradient Flow &amp; Probing
            </button>
            <button 
              className={'chip' + (tab === 'logs' ? ' on' : '')} 
              onClick={() => setTab('logs')}
            >
              Execution Logs ({lines.length} lines)
            </button>
          </div>

          <div className="row" style={{ gap: 8 }}>
            <button className="btn sm" onClick={() => { onClose(); go('flow'); }}>
              Open Fullscreen Flow ↗
            </button>
          </div>
        </div>

        {tab === 'flow' ? (
          <div style={{ height: 480, display: 'flex', flexDirection: 'column' }}>
            <Flow compact embeddedJobId={job.spec.id} />
          </div>
        ) : (
          <pre className="code" style={{ maxHeight: 480, overflowY: 'auto' }}>
            {lines.length ? lines.join('\n') : 'No logs captured yet.'}
          </pre>
        )}
      </div>
    </Modal>
  );
}

export function Runs({ jobs, refresh, go }: { jobs: JobRecord[]; refresh: () => void; go: (t: string) => void }) {
  const toast = useToast();
  const [open, setOpen] = useState<string | null>(null);
  const [custom, setCustom] = useState(false);
  const [name, setName] = useState('custom-run');
  const [cmd, setCmd] = useState('python train.py');
  const job = jobs.find((j) => j.spec.id === open);

  const stop = async (id: string) => { 
    try { 
      await api.stop(id); 
      refresh(); 
    } catch (e) { 
      toast((e as Error).message, true); 
    } 
  };

  const submit = async () => {
    try { 
      await api.submit(name, cmd); 
      setCustom(false); 
      toast('Run started'); 
      refresh(); 
    } catch (e) { 
      toast((e as Error).message, true); 
    }
  };

  return (
    <>
      <div className="page-header">
        <div>
          <div className="eyebrow">EXECUTION</div>
          <h1 className="page-title">Runs</h1>
          <p className="page-sub">Supervised training jobs with real-time gradient &amp; activation flow inspection.</p>
        </div>
        <button className="btn primary" onClick={() => setCustom(true)}>+ New run</button>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Component</th>
              <th>Started</th>
              <th>Exit</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {jobs.length === 0 && (
              <tr>
                <td colSpan={6}>
                  <EmptyState 
                    icon={<Icon name="runs" size={20} />} 
                    title="No runs yet" 
                    hint="Start a component, or run any training script under Sentinel supervision."
                  >
                    <button className="btn primary" onClick={() => go('components')}>Pick a template</button>
                  </EmptyState>
                </td>
              </tr>
            )}
            {jobs.map((j) => (
              <tr key={j.spec.id}>
                <td>
                  <b>{j.spec.name}</b>
                  <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{j.spec.id.slice(0, 8)}</div>
                </td>
                <td><StatusPill status={j.status} /></td>
                <td className="mono" style={{ fontSize: 11 }}>{j.spec.component_id ?? '—'}</td>
                <td style={{ color: 'var(--text-secondary)' }}>{fmtTime(j.started_at)}</td>
                <td className="mono">{j.exit_code ?? '—'}</td>
                <td>
                  <div className="row" style={{ justifyContent: 'flex-end', gap: 6 }}>
                    <button 
                      className={'btn sm' + (j.status === 'running' ? ' primary' : '')} 
                      onClick={() => setOpen(j.spec.id)} 
                      title="Inspect Live Gradient &amp; Activation Flow"
                    >
                      {j.status === 'running' ? (
                        <>
                          <span className="dot ok pulse" style={{ marginRight: 4 }} />
                          Live Flow &amp; Probing
                        </>
                      ) : (
                        'Flow & Logs'
                      )}
                    </button>
                    {j.status === 'running' && (
                      <button className="btn sm danger" onClick={() => stop(j.spec.id)}>
                        Stop
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {job && <JobInspector job={job} onClose={() => setOpen(null)} go={go} />}

      {custom && (
        <Modal title="New command run" sub="Run any shell command under Sentinel supervision" onClose={() => setCustom(false)}>
          <div className="field"><label>Name</label><input value={name} onChange={(e) => setName(e.target.value)} /></div>
          <div className="field"><label>Command</label><input className="mono" value={cmd} onChange={(e) => setCmd(e.target.value)} /></div>
          <div className="modal-foot">
            <button className="btn" onClick={() => setCustom(false)}>Cancel</button>
            <button className="btn primary" onClick={submit}>Start</button>
          </div>
        </Modal>
      )}
    </>
  );
}
