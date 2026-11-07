import type { BlockInfo, Component, NodeDesc, ValidateOk } from '../api';
import { fmtBytes, fmtCount } from '../ui';
import type { BNode } from './types';

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

export interface InspectorProps {
  node: BNode | null;
  result: ValidateOk | null;
  blocks: Record<string, BlockInfo>;
  component: Component | null;
}

export function Inspector({ node, result, blocks, component }: InspectorProps) {
  if (node?.type === 'input') {
    const shape = result?.inputs[node.data.name];
    return (
      <div className="inspector-panel">
        <div className="eyebrow">GRAPH INPUT</div>
        <h3 style={{ margin: '4px 0 12px' }}>{node.data.name}</h3>
        <div className="kv">
          <span>Port</span><b>in:{node.data.name}</b>
          <span>Tensor Shape</span><b className="mono">[{node.data.shape.join(', ')}]</b>
          {shape && <span>Inferred Shape</span>}
          {shape && <b className="mono">[{shape.join(', ')}]</b>}
        </div>
        <p className="hint" style={{ marginTop: 12 }}>
          Entrypoint receiving training batch or inference features from the dataset.
        </p>
      </div>
    );
  }

  if (node?.type === 'output') {
    return (
      <div className="inspector-panel">
        <div className="eyebrow">GRAPH OUTPUT</div>
        <h3 style={{ margin: '4px 0 12px' }}>{node.data.name}</h3>
        <div className="kv">
          <span>Port</span><b>out:{node.data.name}</b>
          <span>Role</span><b>Predictions / Logits</b>
        </div>
        <p className="hint" style={{ marginTop: 12 }}>
          Terminal output node evaluated against dataset target during training and inference.
        </p>
      </div>
    );
  }

  if (node?.type === 'block') {
    const b = blocks[node.data.block];
    const r = result?.nodes[node.id];
    return (
      <div className="inspector-panel">
        <div className="row sp" style={{ alignItems: 'flex-start' }}>
          <div>
            <div className="eyebrow">{b?.family?.toUpperCase() ?? 'BLOCK'}</div>
            <h3 style={{ margin: '4px 0' }}>{node.id}</h3>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{node.data.block}</span>
          </div>
          <span className={'chip ' + (b?.tier === 'compound' ? 'on' : '')}>{b?.tier ?? 'fundamental'}</span>
        </div>

        <p className="hint" style={{ margin: '8px 0 14px' }}>{b?.description}</p>

        {r && (
          <div className="kv" style={{ marginBottom: 14 }}>
            <span>Learnable Params</span>
            <b>{r.shared ? 'shared (counted once)' : `${r.params.toLocaleString()} · ${fmtBytes(r.params * 4)}`}</b>
            <span>Compute / Sample</span>
            <b>{r.macs === null ? 'n/a' : `${fmtCount(r.macs)} MACs`}</b>
            {node.data.share && <span>Weights Reused From</span>}
            {node.data.share && <b className="mono">{node.data.share}</b>}
          </div>
        )}

        <div className="stat-title" style={{ marginTop: 12, marginBottom: 6 }}>PORTS & SHAPES</div>
        <div className="kv" style={{ marginBottom: 14 }}>
          {b?.inputs.map((p) => {
            const inKey = r?.inputs[p];
            const inShape = inKey ? (result?.inputs[inKey] ?? result?.nodes[inKey.split('.')[0]]?.outputs[inKey.split('.')[1]]) : null;
            return (
              <div key={`in-${p}`} style={{ display: 'contents' }}>
                <span>in: {p}</span>
                <b className="mono">{inShape ? `[${inShape.join(', ')}]` : '—'}</b>
              </div>
            );
          })}
          {b?.outputs.map((p) => {
            const outShape = r?.outputs[p];
            return (
              <div key={`out-${p}`} style={{ display: 'contents' }}>
                <span>out: {p}</span>
                <b className="mono">{outShape ? `[${outShape.join(', ')}]` : '—'}</b>
              </div>
            );
          })}
        </div>

        {Object.keys(node.data.params).length > 0 && (
          <>
            <div className="stat-title" style={{ marginTop: 12, marginBottom: 6 }}>LAYER HYPERPARAMETERS</div>
            <div className="kv" style={{ marginBottom: 14 }}>
              {Object.entries(node.data.params).map(([k, v]) => (
                <div key={k} style={{ display: 'contents' }}>
                  <span>{k}</span>
                  <b className="mono">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</b>
                </div>
              ))}
            </div>
          </>
        )}

        {b?.tier === 'compound' && (
          <div className="inner" style={{ marginTop: 12 }}>
            <div className="stat-title">NESTED COMPOUND ARCHITECTURE</div>
            <div className="chips" style={{ margin: '6px 0 8px' }}>{b.uses.map((u) => <span className="chip" key={u}>{u}</span>)}</div>
            {r?.inner
              ? <Inner nodes={r.inner.nodes} order={r.inner.order} depth={0} />
              : <p className="hint">Inner graph hierarchy compiled automatically.</p>}
          </div>
        )}
      </div>
    );
  }

  // No node selected: show Component & Model overview
  return (
    <div className="inspector-panel">
      <div className="eyebrow">DAG ARCHITECTURE OVERVIEW</div>
      <h3 style={{ margin: '4px 0' }}>{component?.name ?? 'Model Graph'}</h3>
      {component && <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{component.component_id}</span>}
      {component?.description && <p className="hint" style={{ margin: '8px 0 14px' }}>{component.description}</p>}

      {result?.stats && (
        <div className="kv" style={{ marginBottom: 14 }}>
          <span>Total Parameters</span>
          <b>{fmtCount(result.stats.params)} ({fmtBytes(result.stats.bytes_fp32)})</b>
          <span>Compute / Sample</span>
          <b>{result.stats.macs === null ? 'n/a' : `${fmtCount(result.stats.macs)} MACs`}</b>
          <span>Blocks</span>
          <b>{result.stats.blocks} fundamental · {result.stats.compounds} compound</b>
          <span>Nesting Depth</span>
          <b>{result.stats.depth} level{result.stats.depth === 1 ? '' : 's'}</b>
        </div>
      )}

      {result?.stats?.by_block && (
        <div style={{ marginBottom: 14 }}>
          <div className="stat-title" style={{ marginBottom: 6 }}>WEIGHTS BREAKDOWN</div>
          <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
            {Object.entries(result.stats.by_block)
              .filter(([, v]) => v > 0)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => (
                <span key={k} className="chip">
                  {k}: {Math.round((v / Math.max(1, result.stats!.params)) * 100)}%
                </span>
              ))}
          </div>
        </div>
      )}

      {result?.order && (
        <div style={{ marginBottom: 14 }}>
          <div className="stat-title" style={{ marginBottom: 6 }}>EXECUTION ORDER ({result.order.length} STEPS)</div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6, maxHeight: 180, overflowY: 'auto' }}>
            {result.order.map((id, i) => (
              <div key={id} style={{ display: 'flex', gap: 8 }}>
                <span style={{ color: 'var(--text-muted)', width: 20 }}>{i + 1}.</span>
                <b>{id}</b>
                <span style={{ color: 'var(--text-muted)' }}>({result.nodes[id]?.block})</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="hint" style={{ marginTop: 16 }}>
        💡 Click any node on the DAG canvas to inspect its tensor shapes and parameters.
      </p>
    </div>
  );
}
