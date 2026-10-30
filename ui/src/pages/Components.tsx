import { useState } from 'react';
import { api, type Component } from '../api';
import { usePoll } from '../hooks';
import { Field, Modal, useToast } from '../ui';

function Configure({ c, onClose, refresh, go }: { c: Component; onClose: () => void; refresh: () => void; go: (t: string) => void }) {
  const toast = useToast();
  const [vals, setVals] = useState<Record<string, string>>(
    Object.fromEntries(Object.entries(c.parameters).map(([k, p]) => [k, String(p.default ?? '')])),
  );
  const coerce = () => {
    const o: Record<string, unknown> = {};
    for (const [k, p] of Object.entries(c.parameters)) {
      const v = vals[k];
      o[k] = p.type === 'int' ? parseInt(v, 10) : p.type === 'float' ? parseFloat(v) : p.type === 'bool' ? v === 'true' : v;
    }
    return o;
  };
  const run = async () => {
    try {
      if (c.graph) await api.runGraph({ component_id: c.component_id, overrides: coerce() });
      else await api.runComponent(c.component_id, coerce());
      toast('Experiment started'); refresh(); onClose(); go('runs');
    } catch (e) { toast((e as Error).message, true); }
  };
  return (
    <Modal title={c.name} sub={c.component_id} onClose={onClose}>
      <p className="hint">{c.description}</p>
      {Object.entries(c.parameters).map(([k, p]) => (
        <Field key={k} label={k} hint={[p.help, p.range ? `range ${p.range[0]}–${p.range[1]}` : ''].filter(Boolean).join(' · ')}>
          {p.type === 'choice' ? (
            <select value={vals[k]} onChange={(e) => setVals({ ...vals, [k]: e.target.value })}>{p.options.map((o) => <option key={o}>{o}</option>)}</select>
          ) : (
            <input value={vals[k]} onChange={(e) => setVals({ ...vals, [k]: e.target.value })} />
          )}
        </Field>
      ))}
      {Object.keys(c.locked).length > 0 && <p className="hint">Locked: {Object.keys(c.locked).join(', ')}</p>}
      <div className="modal-foot"><button className="btn" onClick={onClose}>Cancel</button><button className="btn primary" onClick={run}>Launch experiment</button></div>
    </Modal>
  );
}

export function Components({ refresh, go, openInBuilder }: { refresh: () => void; go: (t: string) => void; openInBuilder: (id: string) => void }) {
  const { data } = usePoll(api.components, 15000);
  const [sel, setSel] = useState<Component | null>(null);
  const domains = [...new Set((data ?? []).map((c) => c.domain))].sort();
  return (
    <>
      <div className="page-header"><div><div className="eyebrow">COMPONENT REGISTRY</div><h1 className="page-title">Components</h1>
        <p className="page-sub">Readymade architectures with editable hyperparameters. Open any graph component in the Builder.</p></div></div>
      {domains.map((d) => (
        <section key={d} style={{ marginBottom: 22 }}>
          <div className="nav-title" style={{ padding: 0, marginBottom: 8 }}>{d}</div>
          <div className="grid cards">
            {data!.filter((c) => c.domain === d).map((c) => (
              <div className="card" key={c.component_id}>
                <div className="row sp"><b>{c.name}</b>{c.graph ? <span className="tag">GRAPH</span> : null}</div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{c.component_id}</div>
                <p style={{ color: 'var(--text-secondary)', margin: '8px 0 12px' }}>{c.description}</p>
                <div style={{ color: 'var(--text-muted)', fontSize: 11, marginBottom: 10 }}>{Object.keys(c.parameters).length} hyperparameters</div>
                <div className="row">
                  <button className="btn sm primary" onClick={() => setSel(c)}>Configure</button>
                  {c.graph ? <button className="btn sm" onClick={() => openInBuilder(c.component_id)}>Open in Builder</button> : null}
                </div>
              </div>
            ))}
          </div>
        </section>
      ))}
      {sel && <Configure c={sel} onClose={() => setSel(null)} refresh={refresh} go={go} />}
    </>
  );
}
