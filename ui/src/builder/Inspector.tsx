import { useEffect, useState } from 'react';
import type { BlockInfo, NodeDesc, Training, ValidateOk } from '../api';
import { DatasetPicker, SchemaEditor, taskOf, type Picked } from '../DatasetPicker';
import { Field, fmtBytes, fmtCount } from '../ui';
import { IDENT } from './convert';
import type { BNode } from './types';

/** Text input that commits on blur/Enter so partial edits (e.g. JSON) don't thrash validation. */
function Commit({ value, onCommit, mono }: { value: string; onCommit: (v: string) => void; mono?: boolean }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return <input className={mono ? 'mono' : ''} value={v} onChange={(e) => setV(e.target.value)} onBlur={() => v !== value && onCommit(v)} onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />;
}

const parse = (s: string): unknown => { try { return JSON.parse(s); } catch { return s; } };
const show = (v: unknown) => (v === undefined || v === null ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v));
/** What a compound block is made of, recursively, with the shape on every inner node's first output. */
function Inner({ nodes, order, depth }: { nodes: Record<string, NodeDesc>; order: string[]; depth: number }) {
  return (
    <>
      {order.map((id) => {
        const n = nodes[id];
        const shape = Object.values(n.outputs)[0];
        return (
          <div key={id}>
            <div className="inner-row" style={{ paddingLeft: depth * 14 }}>
              <span className={'tier-dot ' + n.tier} title={n.tier} />
              <span>{id}</span><span className="muted">{n.block}</span>
              {n.params > 0 && <span className="mono pc">{fmtCount(n.params)}</span>}
              {shape && <span className="mono shape">[{shape.join(', ')}]</span>}
            </div>
            {n.inner && <Inner nodes={n.inner.nodes} order={n.inner.order} depth={depth + 1} />}
          </div>
        );
      })}
    </>
  );
}

const sel = (opts: string[], cur: string, on: (v: string) => void) => (
  <select value={cur} onChange={(e) => on(e.target.value)}>{opts.map((o) => <option key={o} value={o}>{o || '(none)'}</option>)}</select>
);

export interface InspectorProps {
  node: BNode | null;
  result: ValidateOk | null;
  blocks: Record<string, BlockInfo>;
  siblings: BNode[];
  training: Training | null;
  setTraining: (t: Training | null) => void;
  patchData: (id: string, patch: Record<string, unknown>) => void;
  rename: (id: string, to: string) => boolean;
  remove: () => void;
  applyDataset: (d: { csv: string; features: string[]; target: string; loss: string; outputs: number }) => void;
}

export function Inspector(p: InspectorProps) {
  const { node, blocks } = p;
  const [picked, setPicked] = useState<Picked | null>(null);
  const choose = (pk: Picked, features: string[], target: string) => {
    const t = taskOf(pk.schema, target);
    p.applyDataset({ csv: pk.path, features, target, loss: t.loss, outputs: t.outputs });
  };
  const del = <button className="btn sm danger" onClick={p.remove}>Delete</button>;

  if (node?.type === 'input') {
    return (
      <>
        <h4>Graph input</h4>
        <Field label="Name"><Commit value={node.data.name} onCommit={(v) => IDENT.test(v) && p.patchData(node.id, { name: v })} /></Field>
        <Field label="Shape" hint='JSON, symbols like "B" allowed'><Commit mono value={JSON.stringify(node.data.shape)} onCommit={(v) => { const x = parse(v); if (Array.isArray(x)) p.patchData(node.id, { shape: x }); }} /></Field>
        {del}
      </>
    );
  }
  if (node?.type === 'output') {
    return (
      <>
        <h4>Graph output</h4>
        <Field label="Name"><Commit value={node.data.name} onCommit={(v) => IDENT.test(v) && p.patchData(node.id, { name: v })} /></Field>
        <p className="hint">Wire a block's output port to this node.</p>
        {del}
      </>
    );
  }
  if (node?.type === 'block') {
    const b = blocks[node.data.block];
    const sharable = p.siblings.filter((s) => s.type === 'block' && s.id !== node.id && s.data.block === node.data.block && !s.data.share).map((s) => s.id);
    const setParam = (k: string, v: unknown) => p.patchData(node.id, { params: { ...node.data.params, [k]: v } });
    return (
      <>
        <h4>{node.data.block} <span className={'tier-tag ' + (b?.tier ?? '')}>{b?.tier}</span></h4>
        <p className="hint">{b?.description}</p>
        {p.result?.nodes[node.id] && (
          <div className="kv" style={{ marginBottom: 10 }}>
            <span>Parameters</span><b>{p.result.nodes[node.id].shared ? 'shared (counted once)' : `${p.result.nodes[node.id].params.toLocaleString()} · ${fmtBytes(p.result.nodes[node.id].params * 4)}`}</b>
            <span>MACs / sample</span><b>{p.result.nodes[node.id].macs === null ? 'n/a' : fmtCount(p.result.nodes[node.id].macs!)}</b>
          </div>
        )}
        {b?.tier === 'compound' && (
          <div className="inner">
            <div className="stat-title">Built from</div>
            <div className="chips" style={{ margin: '6px 0 8px' }}>{b.uses.map((u) => <span className="chip" key={u}>{u}</span>)}</div>
            {p.result?.nodes[node.id]?.inner
              ? <Inner nodes={p.result.nodes[node.id].inner!.nodes} order={p.result.nodes[node.id].inner!.order} depth={0} />
              : <p className="hint">Connect inputs and fix any errors to see the inner graph with live shapes.</p>}
          </div>
        )}
        <Field label="Node id"><Commit value={node.id} onCommit={(v) => p.rename(node.id, v)} /></Field>
        {b?.params.map((bp) => {
          const v = node.data.params[bp.name] !== undefined ? node.data.params[bp.name] : bp.default;
          const label = bp.name + (bp.required ? ' *' : '');
          if (typeof bp.default === 'boolean')
            return <Field key={bp.name} label={label} hint={bp.help}><input type="checkbox" checked={!!v} onChange={(e) => setParam(bp.name, e.target.checked)} /></Field>;
          if (bp.options.length > 0)
            return <Field key={bp.name} label={label} hint={bp.help}>{sel(bp.options, String(v), (x) => setParam(bp.name, x))}</Field>;
          // an empty optional number means "use the default" (null)
          return <Field key={bp.name} label={label} hint={bp.help}><Commit value={show(v)} onCommit={(s) => setParam(bp.name, s.trim() === '' && bp.default === null ? null : parse(s))} /></Field>;
        })}
        {sharable.length > 0 && <Field label="Share weights with">{sel(['', ...sharable], node.data.share ?? '', (x) => p.patchData(node.id, { share: x || null }))}</Field>}
        {del}
      </>
    );
  }
  const t = p.training;
  const set = (patch: Partial<Training>) => t && p.setTraining({ ...t, ...patch });
  return (
    <>
      <h4>Training</h4>
      {!t ? (
        <>
          <p className="hint">Add a training section to run this graph on a CSV.</p>
          <button className="btn sm" onClick={() => p.setTraining({ loss: 'cross_entropy', optimizer: 'adam', lr: 0.01, epochs: 200, data: { csv: 'data/demo.csv', features: ['x1', 'x2'], target: 'label' } })}>Add training</button>
        </>
      ) : (
        <>
          <DatasetPicker onPick={(pk) => { setPicked(pk); const sg = pk.schema.suggestion; choose(pk, sg?.features ?? [], sg?.target ?? pk.schema.columns[pk.schema.columns.length - 1]?.name ?? ''); }} />
          {picked && <SchemaEditor schema={picked.schema} features={t.data.features} target={t.data.target} onChange={(f, tg) => choose(picked, f, tg)} />}
          <p className="hint gap-t">Choosing a dataset also sets the input width and the final linear block's size.</p>
          <Field label="Loss">{sel(['cross_entropy', 'mse', 'bce_logits'], t.loss, (v) => set({ loss: v }))}</Field>
          <Field label="Optimizer">{sel(['adam', 'adamw', 'sgd'], t.optimizer, (v) => set({ optimizer: v }))}</Field>
          <Field label="Learning rate"><Commit value={String(t.lr)} onCommit={(v) => set({ lr: parseFloat(v) })} /></Field>
          <Field label="Epochs"><Commit value={String(t.epochs)} onCommit={(v) => set({ epochs: parseInt(v, 10) })} /></Field>
          <Field label="Batch size" hint="empty = full batch"><Commit value={show(t.batch_size)} onCommit={(v) => set({ batch_size: v.trim() ? parseInt(v, 10) : null })} /></Field>
          <Field label="CSV path"><Commit mono value={t.data.csv} onCommit={(v) => set({ data: { ...t.data, csv: v } })} /></Field>
          <Field label="Feature columns" hint="comma separated"><Commit value={t.data.features.join(', ')} onCommit={(v) => set({ data: { ...t.data, features: v.split(',').map((s) => s.trim()).filter(Boolean) } })} /></Field>
          <Field label="Target column"><Commit value={t.data.target} onCommit={(v) => set({ data: { ...t.data, target: v } })} /></Field>
          <button className="btn sm danger" onClick={() => p.setTraining(null)}>Remove training</button>
        </>
      )}
      <p className="hint gap-t">Drag blocks from the palette, wire output ports to input ports. Click a wire and press Delete to remove it. Ctrl/Cmd+Z undoes.</p>
    </>
  );
}
