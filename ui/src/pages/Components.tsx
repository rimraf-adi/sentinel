import { useState } from 'react';
import { api, type Component } from '../api';
import { usePoll } from '../hooks';
import { DatasetPicker, SchemaEditor, taskOf, type Picked } from '../DatasetPicker';
import { Icon } from '../icons';
import { EmptyState, Field, Modal, Skeleton, fmtBytes, fmtCount, useToast } from '../ui';

function Configure({ c, onClose, refresh, go }: { c: Component; onClose: () => void; refresh: () => void; go: (t: string) => void }) {
  const toast = useToast();
  const [vals, setVals] = useState<Record<string, string>>(
    Object.fromEntries(Object.entries(c.parameters).map(([k, p]) => [k, String(p.default ?? '')])),
  );
  const [picked, setPicked] = useState<Picked | null>(null);
  const [cols, setCols] = useState<{ features: string[]; target: string }>({ features: [], target: '' });

  /** Fill every dataset-related hyperparameter this component declares. */
  const fill = (pk: Picked, features: string[], target: string) => {
    const t = taskOf(pk.schema, target);
    const want: Record<string, string> = {
      csv: pk.path, feature_cols: features.join(','), target, features: String(features.length),
      classes: String(t.outputs), loss: t.loss, seq_len: String(features.length), step_features: '1',
    };
    setVals((v) => ({ ...v, ...Object.fromEntries(Object.entries(want).filter(([k]) => k in c.parameters)) }));
    setCols({ features, target });
  };
  const onPick = (pk: Picked) => {
    setPicked(pk);
    const sg = pk.schema.suggestion;
    const last = pk.schema.columns[pk.schema.columns.length - 1]?.name ?? '';
    fill(pk, sg?.features ?? [], sg?.target ?? last);
  };
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
      if (c.graph || c.gan) await api.runGraph({ component_id: c.component_id, overrides: coerce() });
      else await api.runComponent(c.component_id, coerce());
      toast('Experiment started'); refresh(); onClose(); go('runs');
    } catch (e) { toast((e as Error).message, true); }
  };
  return (
    <Modal title={c.name} sub={c.component_id} onClose={onClose}>
      <p className="hint">{c.description}</p>
      {'csv' in c.parameters && (
        <div className="card" style={{ marginBottom: 12, background: 'var(--bg)' }}>
          <DatasetPicker onPick={onPick} />
          {picked && <SchemaEditor schema={picked.schema} features={cols.features} target={cols.target} onChange={(f, t) => fill(picked, f, t)} />}
        </div>
      )}
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

const DOMAIN_COLOR: Record<string, string> = { tabular: '#5b9bff', sequence: '#fbbf24', vision: '#f472b6', graph: '#22d3ee', llm: '#34d399', custom: '#94a3b8' };
const colorOf = (d: string) => DOMAIN_COLOR[d] ?? '#94a3b8';

export function Components({ refresh, go, openInVisualizer }: { refresh: () => void; go: (t: string) => void; openInVisualizer: (id: string) => void }) {
  const { data } = usePoll(api.components, 15000);
  const [sel, setSel] = useState<Component | null>(null);
  const [q, setQ] = useState('');
  const [domain, setDomain] = useState<string | null>(null);
  const domains = [...new Set((data ?? []).map((c) => c.domain))].sort();
  const shown = (data ?? []).filter((c) => (!domain || c.domain === domain)
    && `${c.name} ${c.component_id} ${c.description}`.toLowerCase().includes(q.toLowerCase()));
  return (
    <>
      <div className="page-header">
        <div>
          <div className="eyebrow">COMPONENT REGISTRY</div>
          <h1 className="page-title">Components</h1>
          <p className="page-sub">Declarative model architectures and experiment recipes. Inspect live topologies in the DAG Visualizer.</p>
        </div>
      </div>
      <div className="filterbar">
        <input className="input" type="search" placeholder="Search components…" aria-label="Search components" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="chips" role="group" aria-label="Filter by domain">
          <button className={'chip' + (domain === null ? ' on' : '')} onClick={() => setDomain(null)}>all</button>
          {domains.map((d) => <button key={d} className={'chip' + (domain === d ? ' on' : '')} onClick={() => setDomain(domain === d ? null : d)}>{d}</button>)}
        </div>
      </div>
      <div className="grid cards">
        {!data && Array.from({ length: 6 }, (_, i) => <div className="card" key={i}><Skeleton h={34} w={34} /><Skeleton h={14} w="60%" style={{ margin: '12px 0 8px' }} /><Skeleton h={10} /><Skeleton h={10} w="80%" style={{ marginTop: 6 }} /></div>)}
        {shown.map((c) => (
          <div className="card hover" key={c.component_id} style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="row sp" style={{ alignItems: 'flex-start' }}>
              <div className="icon-tile" style={{ background: colorOf(c.domain) + '22', color: colorOf(c.domain) }}>{c.domain.slice(0, 2).toUpperCase()}</div>
              <div className="chips">
                {c.graph ? <span className="chip on">graph</span> : null}
                {c.gan ? <span className="chip on">gan</span> : null}
                <span className="chip">{c.domain}</span>
                {c.origin === 'user' ? <span className="chip" style={{ color: 'var(--accent)' }}>template</span> : c.origin === 'project' ? <span className="chip">custom</span> : null}
              </div>
            </div>
            <b style={{ marginTop: 12, fontSize: 14 }}>{c.name}</b>
            <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>{c.component_id}</div>
            <p style={{ color: 'var(--text-secondary)', margin: '8px 0 14px', flex: 1 }}>{c.description}</p>
            <div className="row sp">
              <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{Object.keys(c.parameters).length} hyperparameters{c.stats ? ` · ${fmtCount(c.stats.params)} params (${fmtBytes(c.stats.bytes_fp32)})` : ''}</span>
              <div className="row">
                {c.graph ? <button className="btn sm" onClick={() => openInVisualizer(c.component_id)}>View DAG</button> : null}
                <button className="btn sm primary" onClick={() => setSel(c)}>Configure</button>
              </div>
            </div>
          </div>
        ))}
      </div>
      {data && shown.length === 0 && <EmptyState icon={<Icon name="search" size={20} />} title="No components match" hint="Try a different search or clear the domain filter."><button className="btn" onClick={() => { setQ(''); setDomain(null); }}>Clear filters</button></EmptyState>}
      {sel && <Configure c={sel} onClose={() => setSel(null)} refresh={refresh} go={go} />}
    </>
  );
}
