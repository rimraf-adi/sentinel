import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider, applyEdgeChanges, applyNodeChanges, useReactFlow,
  type Connection, type Edge, type EdgeChange, type NodeChange,
} from '@xyflow/react';
import { api, type BlockInfo, type Component, type ValidateOk } from '../api';
import { Icon } from '../icons';
import { Modal, fmtBytes, fmtCount, useToast } from '../ui';
import { BlockNodeView, BuilderCtx, InputNodeView, OutputNodeView } from './BlockNode';
import { IDENT, emptyGraph, fromSpec, toSpec, wouldCycle } from './convert';
import { Inspector } from './Inspector';
import { SaveModal } from './SaveModal';
import type { BNode, Graph } from './types';

const FAM_COLOR: Record<string, string> = { core: '#94a3b8', shape: '#2dd4bf', sequence: '#fbbf24', attention: '#c4a7e7', vision: '#f0a0c0', generative: '#fb923c', graph: '#67d4e8' };
const famColor = (f?: string) => (f && FAM_COLOR[f]) || '#94a3b8';
const nodeTypes = { block: BlockNodeView, input: InputNodeView, output: OutputNodeView };
const FAMILY_ORDER = ['core', 'shape', 'sequence', 'attention', 'vision', 'generative', 'graph'];
const edgeId = (e: { source: string; sourceHandle?: string | null; target: string; targetHandle?: string | null }) =>
  `${e.source}.${e.sourceHandle}->${e.target}.${e.targetHandle}`;

/** Free spot to the right of the rightmost node, so a new block never hides a port. */
const nextPos = (nodes: BNode[]) => {
  const right = nodes.reduce((a, n) => (n.position.x >= a.position.x ? n : a), nodes[0] ?? { position: { x: -270, y: 80 } });
  return { x: right.position.x + 270, y: right.position.y };
};

type Status = { kind: 'ok' | 'err' | 'idle'; text: string };

interface Props { initialComponent: string | null; onConsumed: () => void; refreshJobs: () => void; go: (t: string) => void }

function Inner({ initialComponent, onConsumed, refreshJobs, go }: Props) {
  const toast = useToast();
  const rf = useReactFlow();
  const [g, setGraph] = useState<Graph>(emptyGraph);
  const gRef = useRef(g);
  gRef.current = g;
  const [blocks, setBlocks] = useState<Record<string, BlockInfo>>({});
  const [ready, setReady] = useState<Component[]>([]);
  const [result, setResult] = useState<ValidateOk | null>(null);
  const [status, setStatus] = useState<Status>({ kind: 'idle', text: 'empty graph' });
  const [errorNode, setErrorNode] = useState<string | null>(null);
  const [code, setCode] = useState<{ title: string; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [find, setFind] = useState('');
  const [tier, setTier] = useState<'fundamental' | 'compound'>('fundamental');

  /* ---------- history ---------- */
  const past = useRef<Graph[]>([]);
  const future = useRef<Graph[]>([]);
  const lastPush = useRef(0);
  const snapshot = useCallback((coalesce = false) => {
    const now = Date.now();
    if (coalesce && now - lastPush.current < 700) return;
    lastPush.current = now;
    past.current.push(gRef.current);
    if (past.current.length > 60) past.current.shift();
    future.current = [];
  }, []);
  const apply = useCallback((fn: (g: Graph) => Graph, coalesce = false) => { snapshot(coalesce); setGraph(fn); }, [snapshot]);
  const undo = useCallback(() => { const p = past.current.pop(); if (p) { future.current.push(gRef.current); setGraph(p); } }, []);
  const redo = useCallback(() => { const f = future.current.pop(); if (f) { past.current.push(gRef.current); setGraph(f); } }, []);

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'z') return;
      if (/INPUT|SELECT|TEXTAREA/.test((document.activeElement?.tagName ?? ''))) return;
      e.preventDefault();
      e.shiftKey ? redo() : undo();
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [undo, redo]);

  const refreshBlocks = useCallback(() => { api.blocks().then((l) => setBlocks(Object.fromEntries(l.map((b) => [b.name, b])))).catch((e: Error) => toast(e.message, true)); }, [toast]);
  const refreshReady = useCallback(() => { api.components().then((c) => setReady(c.filter((x) => x.graph))).catch(() => {}); }, []);

  /* ---------- load catalogs ---------- */
  useEffect(() => {
    refreshBlocks();
    refreshReady();
  }, [toast]); // eslint-disable-line react-hooks/exhaustive-deps

  const load = useCallback(async (componentId: string) => {
    const r = await api.validate({ component_id: componentId });
    if (!r.ok) { toast(r.error, true); return; }
    apply(() => fromSpec(r.spec, blocks));
    setTimeout(() => rf.fitView({ padding: 0.2, maxZoom: 1 }), 50);
  }, [apply, blocks, rf, toast]);

  useEffect(() => {
    if (!initialComponent) return;
    if (initialComponent === '__new__') { apply(() => emptyGraph()); onConsumed(); return; }
    if (Object.keys(blocks).length) { load(initialComponent); onConsumed(); }
  }, [initialComponent, blocks, load, onConsumed, apply]);

  /* ---------- live validation ---------- */
  const specKey = useMemo(() => JSON.stringify(toSpec(g)), [g]);
  const seq = useRef(0);
  useEffect(() => {
    const id = ++seq.current;
    if (!g.nodes.some((n) => n.type === 'block')) { setResult(null); setErrorNode(null); setStatus({ kind: 'idle', text: 'empty graph' }); return; }
    const t = setTimeout(async () => {
      const r = await api.validate({ graph: toSpec(gRef.current) });
      if (id !== seq.current) return;
      if (r.ok) {
        setResult(r); setErrorNode(null);
        setStatus({ kind: 'ok', text: `valid ✔ ${r.order.length} blocks${r.trainable ? '' : ' · add training to run'}` });
      } else {
        setResult(null);
        setStatus({ kind: 'err', text: r.error });
        const m = r.error.match(/node '([^']+)'|error at ([A-Za-z0-9_]+)\.|port ([A-Za-z0-9_]+)\./);
        setErrorNode(m ? (m[1] ?? m[2] ?? m[3]) : null);
      }
    }, 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specKey]);

  /* ---------- editing ---------- */
  const onNodesChange = useCallback((c: NodeChange[]) => setGraph((p) => ({ ...p, nodes: applyNodeChanges(c, p.nodes as never) as unknown as BNode[] })), []);
  const onEdgesChange = useCallback((c: EdgeChange[]) => setGraph((p) => ({ ...p, edges: applyEdgeChanges(c, p.edges) })), []);
  const isValidConnection = useCallback((c: Edge | Connection) => !!c.source && !!c.target && !wouldCycle(gRef.current.edges, c.source, c.target), []);
  const onConnect = useCallback((c: Connection) => {
    apply((p) => ({
      ...p,
      edges: [...p.edges.filter((e) => !(e.target === c.target && e.targetHandle === c.targetHandle)), { id: edgeId(c), source: c.source, sourceHandle: c.sourceHandle, target: c.target, targetHandle: c.targetHandle }],
    }));
  }, [apply]);

  const addBlock = useCallback((block: string, position: { x: number; y: number }) => {
    const base = block.replace(/\W/g, '');
    apply((p) => {
      let n = 1; while (p.nodes.some((x) => x.id === base + n)) n++;
      const info = blocks[block];
      const params: Record<string, unknown> = {};
      info?.params.forEach((bp) => { if (bp.required) params[bp.name] = bp.name === 'shape' ? [2, 1] : 16; });
      return { ...p, nodes: [...p.nodes.map((x) => ({ ...x, selected: false })), { id: base + n, type: 'block', position, selected: true, data: { block, params, share: null } } as BNode] };
    });
  }, [apply, blocks]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const b = e.dataTransfer.getData('application/sentinel-block');
    if (b) addBlock(b, rf.screenToFlowPosition({ x: e.clientX - 90, y: e.clientY - 14 }));
  }, [addBlock, rf]);

  const addIO = (kind: 'input' | 'output') => {
    apply((p) => {
      const names = new Set(p.nodes.filter((n) => n.type === kind).map((n) => (n.data as { name: string }).name));
      let i = 1; const base = kind === 'input' ? 'x' : 'out';
      let name = kind === 'input' && !names.has('x') ? 'x' : base + i;
      while (names.has(name)) name = base + ++i;
      const maxX = Math.max(0, ...p.nodes.map((n) => n.position.x));
      const node = kind === 'input'
        ? ({ id: `in:${name}`, type: 'input', position: { x: 0, y: 160 * p.nodes.filter((n) => n.type === 'input').length }, data: { name, shape: ['B', 4] }, selected: true } as BNode)
        : ({ id: `out:${name}`, type: 'output', position: { x: maxX + 270, y: 160 * p.nodes.filter((n) => n.type === 'output').length }, data: { name }, selected: true } as BNode);
      return { ...p, nodes: [...p.nodes.map((x) => ({ ...x, selected: false })), node] };
    });
  };

  const retarget = (p: Graph, from: string, to: string): Graph => ({
    ...p,
    nodes: p.nodes.map((n) => (n.id === from ? ({ ...n, id: to } as BNode) : n)),
    edges: p.edges.map((e) => {
      const ne = { ...e, source: e.source === from ? to : e.source, target: e.target === from ? to : e.target };
      return { ...ne, id: edgeId(ne) };
    }),
  });

  const rename = useCallback((id: string, to: string): boolean => {
    const p = gRef.current;
    if (!IDENT.test(to) || (to !== id && (p.nodes.some((n) => n.id === to) || p.nodes.some((n) => n.type === 'input' && n.data.name === to)))) {
      toast('Node id must be a unique identifier', true); return false;
    }
    if (to === id) return true;
    apply((q) => {
      const r = retarget(q, id, to);
      return { ...r, nodes: r.nodes.map((n) => (n.type === 'block' && n.data.share === id ? { ...n, data: { ...n.data, share: to } } : n)) as BNode[] };
    });
    return true;
  }, [apply, toast]);

  const patchData = useCallback((id: string, patch: Record<string, unknown>) => {
    const node = gRef.current.nodes.find((n) => n.id === id);
    if (!node) return;
    if ((node.type === 'input' || node.type === 'output') && typeof patch.name === 'string' && patch.name !== node.data.name) {
      const name = patch.name;
      const clash = gRef.current.nodes.some((n) => n.type === node.type && n.data.name === name)
        || (node.type === 'input' && gRef.current.nodes.some((n) => n.id === name));
      if (!IDENT.test(name) || clash) { toast('Name must be a unique identifier', true); return; }
      apply((q) => {
        const r = retarget(q, id, `${node.type === 'input' ? 'in' : 'out'}:${name}`);
        return { ...r, nodes: r.nodes.map((n) => (n.id === `${node.type === 'input' ? 'in' : 'out'}:${name}` ? ({ ...n, data: { ...n.data, ...patch } } as BNode) : n)) };
      });
      return;
    }
    apply((q) => ({ ...q, nodes: q.nodes.map((n) => (n.id === id ? ({ ...n, data: { ...n.data, ...patch } } as BNode) : n)) }), true);
  }, [apply, toast]);

  /** Apply a dataset choice: training data, loss, input width, and the output layer's size. */
  const applyDataset = useCallback((d: { csv: string; features: string[]; target: string; loss: string; outputs: number }) => {
    apply((p) => {
      const base = p.training ?? { loss: d.loss, optimizer: 'adam', lr: 0.01, epochs: 200, data: { csv: d.csv, features: d.features, target: d.target } };
      // keep a loss the user chose when it still fits the dataset's task family
      const family = d.loss === 'cross_entropy' ? ['cross_entropy', 'focal', 'multi_margin'] : ['mse', 'mae', 'huber', 'smooth_l1', 'quantile', 'poisson', 'log_cosh'];
      const cur = typeof base.loss === 'string' ? base.loss : base.loss.name;
      const training = { ...base, loss: family.includes(cur) ? base.loss : d.loss, task: undefined, data: { ...base.data, csv: d.csv, features: d.features, target: d.target, targets: undefined } };
      const inputs = p.nodes.filter((n) => n.type === 'input');
      const outEdge = p.edges.find((e) => e.target.startsWith('out:'));
      const nodes = p.nodes.map((n) => {
        if (inputs.length === 1 && n.id === inputs[0].id && n.type === 'input' && n.data.shape.length === 2)
          return { ...n, data: { ...n.data, shape: [n.data.shape[0], d.features.length] } } as BNode;
        if (outEdge && n.id === outEdge.source && n.type === 'block' && n.data.block === 'linear')
          return { ...n, data: { ...n.data, params: { ...n.data.params, out: d.outputs } } } as BNode;
        return n;
      });
      return { ...p, training, nodes };
    }, true);
  }, [apply]);

  const removeSelected = useCallback(() => {
    apply((p) => {
      const gone = new Set(p.nodes.filter((n) => n.selected).map((n) => n.id));
      return {
        ...p,
        nodes: p.nodes.filter((n) => !gone.has(n.id)),
        edges: p.edges.filter((e) => !gone.has(e.source) && !gone.has(e.target)),
      };
    });
  }, [apply]);

  /* ---------- actions ---------- */
  const hasBlocks = g.nodes.some((n) => n.type === 'block');
  const compose = async () => {
    try {
      const r = await api.compose({ graph: toSpec(g), model_only: !g.training });
      setCode({ title: g.training ? 'Generated PyTorch — standalone training script' : 'Generated PyTorch — model only (add training for a full script)', text: r.code });
    } catch (e) { toast((e as Error).message, true); }
  };
  const run = async () => {
    if (!g.training) {
      apply((p) => ({ ...p, training: { loss: 'cross_entropy', optimizer: 'adam', lr: 0.01, epochs: 200, data: { csv: 'data/demo.csv', features: ['x1', 'x2'], target: 'label' } } }));
      toast('Training section added — review it in the inspector, then press Run');
      return;
    }
    try { await api.runGraph({ graph: toSpec(g) }); toast('Run started'); refreshJobs(); go('runs'); }
    catch (e) { toast((e as Error).message, true); }
  };

  const selected = g.nodes.find((n) => n.selected) ?? null;
  const ctx = useMemo(() => ({ blocks, result, errorNode }), [blocks, result, errorNode]);

  return (
    <BuilderCtx.Provider value={ctx}>
      <div className="page-header" style={{ marginBottom: 12 }}>
        <div><div className="eyebrow">COMPONENT GRAPH</div><h1 className="page-title">Builder</h1>
          <p className="page-sub">Drag blocks onto the canvas and wire output ports to input ports. Shapes are checked live.</p></div>
      </div>
      <div className="bd-bar">
        <select value="" onChange={(e) => e.target.value && load(e.target.value)}>
          <option value="">Load readymade component…</option>
          {ready.map((c) => <option key={c.component_id} value={c.component_id}>{c.name} ({c.component_id})</option>)}
        </select>
        <span className="bd-sep" aria-hidden />
        <button className="btn" onClick={() => apply(() => emptyGraph())}>New</button>
        <button className="btn" onClick={() => addIO('input')}>+ Input</button>
        <button className="btn" onClick={() => addIO('output')}>+ Output</button>
        <span className="bd-sep" aria-hidden />
        <button className="btn" onClick={undo} disabled={!past.current.length}>Undo</button>
        <button className="btn" onClick={redo} disabled={!future.current.length}>Redo</button>
        <span className="bd-sep" aria-hidden />
        <button className="btn" onClick={compose} disabled={!hasBlocks}>View PyTorch</button>
        <button className="btn" onClick={() => setSaving(true)} disabled={!result} title={result ? 'Save as a reusable component' : 'Fix validation errors first'}>Save as component</button>
        <button className="btn primary" onClick={run} disabled={!hasBlocks}>Run</button>
        <div className={'bd-status ' + status.kind} title={status.text}>{status.text}</div>
      </div>
      {result?.stats && (
        <div className="bd-stats" role="status" aria-label="Model size">
          <span><b>{fmtCount(result.stats.params)}</b> parameters <i title={`${result.stats.params.toLocaleString()} exactly`}>· {fmtBytes(result.stats.bytes_fp32)}</i></span>
          <span><b>{result.stats.macs === null ? 'n/a' : fmtCount(result.stats.macs)}</b> MACs/sample</span>
          <span><b>{result.stats.blocks}</b> blocks</span>
          <span><b>{result.stats.compounds}</b> compound</span>
          <span><b>{result.stats.depth}</b> level{result.stats.depth === 1 ? '' : 's'} deep</span>
          {Object.entries(result.stats.by_block).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => <span key={k} className="muted">{k} {Math.round((v / Math.max(1, result.stats.params)) * 100)}%</span>)}
        </div>
      )}
      <div className="bd-main">
        <div className="bd-side">
          <div className="tier-tabs" role="tablist" aria-label="Block tier">
            {(['fundamental', 'compound'] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tier === t} className={tier === t ? 'on' : ''} onClick={() => setTier(t)}>
                {t === 'fundamental' ? 'Fundamental' : 'Compound'} <span className="mono">{Object.values(blocks).filter((b) => b.tier === t).length}</span>
              </button>
            ))}
          </div>
          <p className="hint" style={{ marginTop: 8 }}>{tier === 'fundamental' ? 'Native blocks: one PyTorch layer each.' : 'Reusable subgraphs built from the blocks below them. Save your own from the canvas.'}</p>
          <input className="input" type="search" placeholder="Search blocks…" aria-label="Search blocks" value={find} onChange={(e) => setFind(e.target.value)} style={{ width: '100%', minWidth: 0, marginBottom: 10 }} />
          {[...new Set([...FAMILY_ORDER, ...Object.values(blocks).map((b) => b.family)])].map((f) => {
            const items = Object.values(blocks).filter((b) => b.tier === tier && b.family === f && `${b.name} ${b.description}`.toLowerCase().includes(find.toLowerCase()));
            if (!items.length) return null;
            return (
              <div key={f} style={{ ['--fam' as string]: `var(--fam-${f}, var(--fam-core))` }}>
                <div className="bd-fam">{f}</div>
                {items.map((b) => (
                  <div key={b.name} className="bd-pal" draggable title="Drag onto the canvas, or double-click"
                    onDragStart={(e) => e.dataTransfer.setData('application/sentinel-block', b.name)}
                    onDoubleClick={() => addBlock(b.name, nextPos(g.nodes))}>
                    {b.tier === 'compound' && '◆ '}{b.name}<small>{b.description}</small>
                    {b.uses.length > 0 && <small className="uses">built from {b.uses.join(' · ')}</small>}
                  </div>
                ))}
              </div>
            );
          })}
          {tier === 'compound' && Object.values(blocks).every((b) => b.tier !== 'compound') && <p className="hint">No compound blocks yet.</p>}
        </div>
        <div className="bd-canvas">
          <ReactFlow
            nodes={g.nodes} edges={g.edges} nodeTypes={nodeTypes} colorMode="dark" fitView fitViewOptions={{ maxZoom: 1, padding: 0.25 }} minZoom={0.15}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} isValidConnection={isValidConnection}
            onNodeDragStart={() => snapshot()} onBeforeDelete={async () => { snapshot(); return true; }}
            onDragOver={(e) => e.preventDefault()} onDrop={onDrop}
            deleteKeyCode={['Backspace', 'Delete']} proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={22} color="#222" />
            <Controls showInteractive={false} />
            {g.nodes.some((n) => n.type === 'block') && <MiniMap pannable zoomable maskColor="rgba(0,0,0,.6)" nodeColor={(n) => (n.type === 'block' ? famColor(blocks[(n.data as { block: string }).block]?.family) : '#555')} />}
            {!g.nodes.some((n) => n.type === 'block') && (
              <div className="bd-hint">
                <Icon name="builder" size={16} />
                <span><b>Start your graph.</b> Drag a block from the left (or double-click it), then wire output ports to input ports.</span>
              </div>
            )}
          </ReactFlow>
        </div>
        <div className="bd-side">
          <Inspector node={selected} result={result} blocks={blocks} siblings={g.nodes} training={g.training}
            setTraining={(t) => apply((p) => ({ ...p, training: t }), true)} patchData={patchData} rename={rename} remove={removeSelected} applyDataset={applyDataset} />
        </div>
      </div>
      {saving && result && (
        <SaveModal spec={toSpec(g)} result={result} blocks={blocks} domains={[...new Set(ready.map((c) => c.domain))].sort()}
          families={[...new Set([...FAMILY_ORDER, ...Object.values(blocks).map((b) => b.family)])]}
          onClose={() => setSaving(false)} onSaved={refreshReady} onSavedBlock={refreshBlocks} />
      )}
      {code && (
        <Modal wide title={code.title} onClose={() => setCode(null)}>
          <pre className="code">{code.text}</pre>
          <div className="modal-foot"><button className="btn" onClick={() => { navigator.clipboard?.writeText(code.text); toast('Copied'); }}>Copy</button><button className="btn primary" onClick={() => setCode(null)}>Close</button></div>
        </Modal>
      )}
    </BuilderCtx.Provider>
  );
}

export function Builder(props: Props) {
  return <ReactFlowProvider><Inner {...props} /></ReactFlowProvider>;
}
