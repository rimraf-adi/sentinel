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

/** Build editor state from a validated spec with intelligent spatial diversity and archetype recognition (e.g. U-Net, residual loops, branched models). */
export function fromSpec(spec: GraphSpec, blocks: Record<string, BlockInfo>): Graph {
  const inputNames = Object.keys(spec.inputs);
  const normSrc = (s: string): [string, string] => {
    if (inputNames.includes(s)) return [`in:${s}`, 'out'];
    if (s.includes('.')) { const [n, p] = s.split('.'); return [n, p]; }
    const bName = spec.nodes[s]?.block?.split('@')[0];
    const outPort = blocks[bName]?.outputs?.[0] ?? 'out';
    return [s, outPort];
  };
  const normDst = (s: string): [string, string] => {
    if (s.includes('.')) { const [n, p] = s.split('.'); return [n, p]; }
    const bName = spec.nodes[s]?.block?.split('@')[0];
    const inPort = blocks[bName]?.inputs?.[0] ?? 'in';
    return [s, inPort];
  };

  const edges: Edge[] = [];
  for (const line of spec.edges) {
    const [a, b] = line.split('->').map((x) => x.trim());
    const [sn, sp] = normSrc(a), [tn, tp] = normDst(b);
    edges.push(edge(sn, sp, tn, tp));
  }

  const nodes: BNode[] = [];
  const nodeKeys = Object.keys(spec.nodes);

  // Check for U-Net / Encoder-Decoder topology
  const downRegex = /^(d\d+|down\d+|enc\d+)$/i;
  const upRegex = /^(u\d+|up\d+|dec\d+)$/i;
  const midRegex = /^(mid|bottleneck|latent|z)$/i;

  const hasDown = nodeKeys.some((k) => downRegex.test(k) || spec.nodes[k]?.block?.includes('down'));
  const hasUp = nodeKeys.some((k) => upRegex.test(k) || spec.nodes[k]?.block?.includes('up'));
  const hasMid = nodeKeys.some((k) => midRegex.test(k));
  const isUnet = (hasDown && hasUp) || (hasDown && hasMid && nodeKeys.length >= 3);

  if (isUnet) {
    const downs = nodeKeys
      .filter((k) => downRegex.test(k) || spec.nodes[k]?.block?.includes('down'))
      .sort((a, b) => {
        const numA = parseInt(a.replace(/\D/g, '') || '0', 10);
        const numB = parseInt(b.replace(/\D/g, '') || '0', 10);
        return numA - numB;
      });
    const ups = nodeKeys
      .filter((k) => upRegex.test(k) || spec.nodes[k]?.block?.includes('up'))
      .sort((a, b) => {
        const numA = parseInt(a.replace(/\D/g, '') || '0', 10);
        const numB = parseInt(b.replace(/\D/g, '') || '0', 10);
        return numB - numA; // e.g. u2 then u1
      });
    const mids = nodeKeys.filter((k) => midRegex.test(k));
    const heads = nodeKeys.filter((k) => !downs.includes(k) && !ups.includes(k) && !mids.includes(k));

    const maxLevel = downs.length;
    const levels = new Map<string, number>();
    downs.forEach((k, idx) => levels.set(k, idx));
    mids.forEach((k) => levels.set(k, maxLevel));
    ups.forEach((k, idx) => {
      const matched = ups.length - 1 - idx;
      levels.set(k, Math.min(matched, maxLevel - 1));
    });
    heads.forEach((k) => levels.set(k, 0));

    let curX = 0;
    inputNames.forEach((name, i) => {
      nodes.push({ id: `in:${name}`, type: 'input', position: { x: curX, y: 80 + i * 160 }, data: { name, shape: spec.inputs[name] } });
    });
    curX += 220;

    const ordered = [...downs, ...mids, ...ups, ...heads];
    for (const id of ordered) {
      const n = spec.nodes[id];
      const lvl = levels.get(id) ?? 0;
      const y = 80 + lvl * 170;
      nodes.push({ id, type: 'block', position: { x: curX, y }, data: { block: n.block.split('@')[0], params: n.params ?? {}, share: n.share ?? null } });
      curX += 250;
    }

    Object.entries(spec.outputs).forEach(([name, src], i) => {
      nodes.push({ id: `out:${name}`, type: 'output', position: { x: curX, y: 80 + i * 160 }, data: { name } });
      const [sn, sp] = normSrc(src.trim());
      edges.push(edge(sn, sp, `out:${name}`, 'in'));
      curX += 200;
    });

    return { nodes, edges, training: spec.training ?? null };
  }

  // General layout: layered DAG with bypass detection & spatial divergence
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

  for (const k of nodeKeys) d(k);

  // Check for bypass edges (skip connections spanning > 1 rank)
  const bypassNodes = new Set<string>();
  for (const e of edges) {
    const sRank = d(e.source);
    const tRank = d(e.target);
    if (tRank - sRank > 1 && nodeKeys.includes(e.source) && nodeKeys.includes(e.target)) {
      for (const k of nodeKeys) {
        if (sRank < d(k) && d(k) < tRank) {
          bypassNodes.add(k);
        }
      }
    }
  }

  // Group nodes by rank column
  const cols = new Map<number, string[]>();
  for (const k of [...nodeKeys].sort((a, b) => d(a) - d(b))) {
    const r = d(k);
    const list = cols.get(r) ?? [];
    list.push(k);
    cols.set(r, list);
  }

  const hasEarlyBranchOutputs = Object.values(spec.outputs).some((srcExpr) => {
    const srcNode = srcExpr.trim().split('.')[0];
    const maxRank = Math.max(1, ...cols.keys());
    return nodeKeys.includes(srcNode) && d(srcNode) < maxRank;
  });

  const baseY = bypassNodes.size > 0 || hasEarlyBranchOutputs ? 160 : 80;

  inputNames.forEach((name, i) => {
    nodes.push({ id: `in:${name}`, type: 'input', position: { x: 0, y: baseY + i * 180 }, data: { name, shape: spec.inputs[name] } });
  });

  const nodePositions = new Map<string, { x: number; y: number }>();

  for (const [colRank, group] of cols.entries()) {
    group.forEach((id, rowIdx) => {
      const n = spec.nodes[id];
      let y: number;
      if (bypassNodes.has(id)) {
        y = 30; // elevated track for residual transformation branch
      } else {
        y = baseY + rowIdx * 180;
      }
      const x = colRank * 270;
      nodePositions.set(id, { x, y });
      nodes.push({ id, type: 'block', position: { x, y }, data: { block: n.block.split('@')[0], params: n.params ?? {}, share: n.share ?? null } });
    });
  }

  const maxK = Math.max(1, ...cols.keys());

  Object.entries(spec.outputs).forEach(([name, src], idx) => {
    const srcNode = src.trim().split('.')[0];
    const [sn, sp] = normSrc(src.trim());
    let outX: number;
    let outY: number;

    if (nodePositions.has(srcNode) && d(srcNode) < maxK) {
      // Early branch output (e.g. mu or logvar from VAE encoder)
      const srcPos = nodePositions.get(srcNode)!;
      outX = srcPos.x + 280;
      outY = idx % 2 === 1 ? srcPos.y - 140 : srcPos.y + 140;
    } else {
      outX = (maxK + 1) * 270;
      outY = baseY + idx * 160;
    }

    nodes.push({ id: `out:${name}`, type: 'output', position: { x: outX, y: outY }, data: { name } });
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
