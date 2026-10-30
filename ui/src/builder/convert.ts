import type { Edge } from '@xyflow/react';
import type { BlockInfo, GraphSpec, Shape } from '../api';
import type { BNode, Graph } from './types';

export const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function toSpec(g: Graph): GraphSpec {
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const spec: GraphSpec = { inputs: {}, nodes: {}, edges: [], outputs: {} };
  for (const n of g.nodes) {
    if (n.type === 'input') spec.inputs[n.data.name] = n.data.shape;
    else if (n.type === 'block') {
      spec.nodes[n.id] = { block: n.data.block, params: n.data.params };
      const share = n.data.share;
      if (share && byId.get(share)?.type === 'block') spec.nodes[n.id].share = share;
    }
  }
  for (const e of g.edges) {
    const s = byId.get(e.source), t = byId.get(e.target);
    if (!s || !t || s.type === 'output') continue;
    const from = s.type === 'input' ? s.data.name : `${s.id}.${e.sourceHandle}`;
    if (t.type === 'output') spec.outputs[t.data.name] = from;
    else if (t.type === 'block') spec.edges.push(`${from} -> ${t.id}.${e.targetHandle}`);
  }
  if (g.training) spec.training = g.training;
  return spec;
}

const edge = (source: string, sourceHandle: string, target: string, targetHandle: string): Edge => ({
  id: `${source}.${sourceHandle}->${target}.${targetHandle}`, source, sourceHandle, target, targetHandle,
});

/** Build editor state from a validated spec, laid out left-to-right by dependency depth. */
export function fromSpec(spec: GraphSpec, blocks: Record<string, BlockInfo>): Graph {
  const inputNames = Object.keys(spec.inputs);
  const normSrc = (s: string): [string, string] => {
    if (inputNames.includes(s)) return [`in:${s}`, 'out'];
    if (s.includes('.')) { const [n, p] = s.split('.'); return [n, p]; }
    return [s, blocks[spec.nodes[s].block.split('@')[0]].outputs[0]];
  };
  const normDst = (s: string): [string, string] => {
    if (s.includes('.')) { const [n, p] = s.split('.'); return [n, p]; }
    return [s, blocks[spec.nodes[s].block.split('@')[0]].inputs[0]];
  };
  const edges: Edge[] = [];
  for (const line of spec.edges) {
    const [a, b] = line.split('->').map((x) => x.trim());
    const [sn, sp] = normSrc(a), [tn, tp] = normDst(b);
    edges.push(edge(sn, sp, tn, tp));
  }
  const depth = new Map<string, number>();
  const d = (id: string): number => {
    if (id.startsWith('in:')) return 0;
    const known = depth.get(id);
    if (known !== undefined) return known;
    depth.set(id, 1);
    const ups = edges.filter((e) => e.target === id).map((e) => e.source);
    const v = ups.length ? 1 + Math.max(...ups.map(d)) : 1;
    depth.set(id, v);
    return v;
  };
  const nodes: BNode[] = [];
  inputNames.forEach((name, i) => nodes.push({ id: `in:${name}`, type: 'input', position: { x: 0, y: i * 140 }, data: { name, shape: spec.inputs[name] } }));
  const cols = new Map<number, number>();
  for (const [id, n] of Object.entries(spec.nodes)) {
    const k = d(id), row = cols.get(k) ?? 0; cols.set(k, row + 1);
    nodes.push({ id, type: 'block', position: { x: k * 270, y: row * 170 }, data: { block: n.block.split('@')[0], params: n.params ?? {}, share: n.share ?? null } });
  }
  const maxK = Math.max(1, ...cols.keys());
  Object.entries(spec.outputs).forEach(([name, src], i) => {
    nodes.push({ id: `out:${name}`, type: 'output', position: { x: (maxK + 1) * 270, y: i * 140 }, data: { name } });
    const [sn, sp] = normSrc(src.trim());
    edges.push(edge(sn, sp, `out:${name}`, 'in'));
  });
  return { nodes, edges, training: spec.training ?? null };
}

export const emptyGraph = (): Graph => ({
  nodes: [{ id: 'in:x', type: 'input', position: { x: 0, y: 80 }, data: { name: 'x', shape: ['B', 2] } }],
  edges: [],
  training: null,
});

/** True if `to` can already reach `from` (adding from->to would create a cycle). */
export function wouldCycle(edges: Edge[], from: string, to: string): boolean {
  if (from === to) return true;
  const seen = new Set<string>();
  const stack = [to];
  while (stack.length) {
    const n = stack.pop()!;
    if (n === from) return true;
    if (seen.has(n)) continue;
    seen.add(n);
    for (const e of edges) if (e.source === n) stack.push(e.target);
  }
  return false;
}

export const shapeText = (s: Shape | null | undefined) => (s ? `[${s.join(', ')}]` : '');
