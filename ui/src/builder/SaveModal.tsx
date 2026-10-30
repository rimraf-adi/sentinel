import { useMemo, useState } from 'react';
import { api, type BlockInfo, type GraphSpec, type ValidateOk } from '../api';
import { Field, Modal, useToast } from '../ui';

interface Row { node: string; param: string; value: unknown; on: boolean; name: string }

/** Save the current graph as `components/<domain>/<name>.yaml`, choosing which block parameters become hyperparameters. */
export function SaveModal({ spec, result, blocks, domains, families, onClose, onSaved, onSavedBlock }: {
  spec: GraphSpec; result: ValidateOk; blocks: Record<string, BlockInfo>; domains: string[]; families: string[]; onClose: () => void; onSaved: (id: string) => void; onSavedBlock: () => void;
}) {
  const toast = useToast();
  const [mode, setMode] = useState<'component' | 'block'>('component');
  const [name, setName] = useState('');
  const [domain, setDomain] = useState('custom');
  const [family, setFamily] = useState('core');
  const [description, setDescription] = useState('');
  const [overwrite, setOverwrite] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const initial = useMemo<Row[]>(() => result.order.flatMap((node) => {
    const info = blocks[spec.nodes[node]?.block.split('@')[0]];
    return (info?.params ?? [])
      .map((p) => ({ node, param: p.name, value: result.params[node]?.[p.name] }))
      .filter((r) => ['number', 'string', 'boolean'].includes(typeof r.value))
      .map((r) => ({ ...r, on: false, name: `${node}_${r.param}` }));
  }), [result, blocks, spec]);
  const [rows, setRows] = useState(initial);
  const set = (i: number, patch: Partial<Row>) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));

  const save = async () => {
    const expose = rows.filter((x) => x.on).map(({ node, param, name: n }) => ({ node, param, name: n }));
    try {
      if (mode === 'block') {
        const r = await api.saveBlock({ name, family, description, graph: spec, overwrite, expose });
        toast(`Saved compound block ${r.name} → ${r.path}`);
        onSavedBlock();
      } else {
        const r = await api.saveComponent({ name, domain, description, graph: spec, overwrite, expose });
        toast(`Saved ${r.component_id} → ${r.path}`);
        onSaved(r.component_id);
      }
      onClose();
    } catch (e) {
      const m = (e as Error).message;
      setError(m);
      if (/already exists/.test(m)) setOverwrite(true);
    }
  };

  return (
    <Modal wide title={mode === 'block' ? 'Save as compound block' : 'Save as model component'} sub={mode === 'block' ? 'A reusable subgraph for the palette: graph inputs and outputs become its ports' : 'Writes a reusable, sweepable model into this project'} onClose={onClose}>
      <div className="tier-tabs" role="tablist" aria-label="Save as" style={{ marginBottom: 12 }}>
        <button role="tab" aria-selected={mode === 'component'} className={mode === 'component' ? 'on' : ''} onClick={() => { setMode('component'); setError(null); setOverwrite(false); }}>Model component</button>
        <button role="tab" aria-selected={mode === 'block'} className={mode === 'block' ? 'on' : ''} onClick={() => { setMode('block'); setError(null); setOverwrite(false); }}>Compound block</button>
      </div>
      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <Field label="Name"><input value={name} onChange={(e) => { setName(e.target.value); setOverwrite(false); setError(null); }} placeholder="My classifier" autoFocus /></Field>
        {mode === 'component'
          ? <Field label="Domain"><input list="sv-domains" value={domain} onChange={(e) => setDomain(e.target.value)} />
              <datalist id="sv-domains">{domains.map((d) => <option key={d} value={d} />)}</datalist></Field>
          : <Field label="Family" hint="where it appears in the palette"><input list="sv-families" value={family} onChange={(e) => setFamily(e.target.value)} />
              <datalist id="sv-families">{families.map((d) => <option key={d} value={d} />)}</datalist></Field>}
      </div>
      <Field label="Description"><input value={description} onChange={(e) => setDescription(e.target.value)} /></Field>
      <div className="nav-title" style={{ padding: 0, margin: '12px 0 6px' }}>Hyperparameters</div>
      <p className="hint">Tick the block parameters you want to expose. Give several parameters the same name to tie them together (needed for e.g. both sides of a residual add). {mode === 'component' ? 'Learning rate and epochs are added automatically when the graph has a training section.' : 'A compound block carries no training recipe.'}</p>
      <div style={{ maxHeight: 260, overflow: 'auto', border: '1px solid var(--border)', borderRadius: 6 }}>
        <table>
          <thead><tr><th /><th>Block parameter</th><th>Default</th><th>Hyperparameter name</th></tr></thead>
          <tbody>
            {rows.length === 0 && <tr><td colSpan={4} className="empty">No scalar parameters.</td></tr>}
            {rows.map((r, i) => (
              <tr key={r.node + r.param}>
                <td><input type="checkbox" checked={r.on} onChange={(e) => set(i, { on: e.target.checked })} /></td>
                <td className="mono">{r.node}.{r.param}</td><td className="mono">{String(r.value)}</td>
                <td><input className="mono" disabled={!r.on} value={r.name} onChange={(e) => set(i, { name: e.target.value })} style={{ width: '100%', background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 5, padding: '3px 6px' }} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {error && <p className="err-text gap-t">{error}</p>}
      <div className="modal-foot">
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={!name.trim()} onClick={save}>{overwrite ? 'Overwrite' : 'Save'}</button>
      </div>
    </Modal>
  );
}
