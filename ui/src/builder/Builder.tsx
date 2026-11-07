import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider, applyNodeChanges, useReactFlow,
  type NodeChange,
} from '@xyflow/react';
import { api, type BlockInfo, type Component, type ValidateOk } from '../api';
import { Icon } from '../icons';
import { Modal, fmtBytes, fmtCount, useToast } from '../ui';
import { BlockNodeView, BuilderCtx, InputNodeView, OutputNodeView } from './BlockNode';
import { emptyGraph, fromSpec, toSpec } from './convert';
import { Inspector } from './Inspector';
import type { BNode, Graph } from './types';

const FAM_COLOR: Record<string, string> = {
  core: '#94a3b8',
  shape: '#2dd4bf',
  sequence: '#fbbf24',
  attention: '#c4a7e7',
  vision: '#f0a0c0',
  generative: '#fb923c',
  graph: '#67d4e8',
};
const famColor = (f?: string) => (f && FAM_COLOR[f]) || '#94a3b8';
const nodeTypes = { block: BlockNodeView, input: InputNodeView, output: OutputNodeView };

type Status = { kind: 'ok' | 'err' | 'idle'; text: string };

interface Props {
  initialComponent: string | null;
  onConsumed: () => void;
  refreshJobs: () => void;
  go: (t: string) => void;
}

function Inner({ initialComponent, onConsumed, refreshJobs, go }: Props) {
  const toast = useToast();
  const rf = useReactFlow();
  const [g, setGraph] = useState<Graph>(emptyGraph);
  const [blocks, setBlocks] = useState<Record<string, BlockInfo>>({});
  const [components, setComponents] = useState<Component[]>([]);
  const [selectedCompId, setSelectedCompId] = useState<string>('');
  const [result, setResult] = useState<ValidateOk | null>(null);
  const [status, setStatus] = useState<Status>({ kind: 'idle', text: 'Select a component to inspect' });
  const [selectedNode, setSelectedNode] = useState<BNode | null>(null);
  const [code, setCode] = useState<{ title: string; text: string } | null>(null);

  /* ---------- load catalogs ---------- */
  useEffect(() => {
    api.blocks().then((l) => setBlocks(Object.fromEntries(l.map((b) => [b.name, b])))).catch((e: Error) => toast(e.message, true));
    api.components().then((all) => {
      const graphComps = all.filter((x) => x.graph);
      setComponents(graphComps);
    }).catch(() => {});
  }, [toast]);

  const loadComponent = useCallback(async (componentId: string) => {
    if (!componentId) return;
    setSelectedCompId(componentId);
    setSelectedNode(null);
    try {
      const r = await api.validate({ component_id: componentId });
      if (!r.ok) {
        toast(r.error, true);
        setStatus({ kind: 'err', text: r.error });
        return;
      }
      setResult(r);
      setStatus({
        kind: 'ok',
        text: `valid ✔ ${r.order.length} blocks · ${fmtCount(r.stats?.params ?? 0)} params`,
      });
      const newGraph = fromSpec(r.spec, blocks);
      setGraph(newGraph);
      setTimeout(() => rf.fitView({ padding: 0.25, maxZoom: 1 }), 60);
    } catch (e) {
      toast((e as Error).message, true);
      setStatus({ kind: 'err', text: (e as Error).message });
    }
  }, [blocks, rf, toast]);

  useEffect(() => {
    if (!initialComponent) {
      // Auto-load the first available component if none selected
      if (!selectedCompId && components.length > 0 && Object.keys(blocks).length > 0) {
        loadComponent(components[0].component_id);
      }
      return;
    }
    if (Object.keys(blocks).length) {
      loadComponent(initialComponent);
      onConsumed();
    }
  }, [initialComponent, blocks, components, selectedCompId, loadComponent, onConsumed]);

  const onNodesChange = useCallback((c: NodeChange[]) => {
    setGraph((p) => ({
      ...p,
      nodes: applyNodeChanges(c, p.nodes as never) as unknown as BNode[],
    }));
  }, []);

  const viewPyTorch = async () => {
    if (!selectedCompId && !g.nodes.some((n) => n.type === 'block')) return;
    try {
      const r = await api.compose(
        selectedCompId ? { component_id: selectedCompId } : { graph: toSpec(g) }
      );
      setCode({
        title: `Generated PyTorch — ${selectedCompId || 'Graph Model'}`,
        text: r.code,
      });
    } catch (e) {
      toast((e as Error).message, true);
    }
  };

  const runComponent = async () => {
    if (!selectedCompId) return;
    try {
      await api.runGraph({ component_id: selectedCompId });
      toast(`Launched run for ${selectedCompId}`);
      refreshJobs();
      go('runs');
    } catch (e) {
      toast((e as Error).message, true);
    }
  };

  const currentComponent = useMemo(
    () => components.find((c) => c.component_id === selectedCompId) ?? null,
    [components, selectedCompId]
  );

  const ctx = useMemo(() => ({ blocks, result, errorNode: null }), [blocks, result]);

  // Keep selectedNode in sync if graph updates
  const activeNode = useMemo(() => {
    if (!selectedNode) return null;
    return g.nodes.find((n) => n.id === selectedNode.id) ?? selectedNode;
  }, [g.nodes, selectedNode]);

  return (
    <BuilderCtx.Provider value={ctx}>
      <div className="page-header" style={{ marginBottom: 12 }}>
        <div>
          <div className="eyebrow">AGENT-FIRST ARCHITECTURE</div>
          <h1 className="page-title">DAG Visualizer</h1>
          <p className="page-sub">
            Interactive topology, port shapes, and compute profiles for declarative models.
          </p>
        </div>
      </div>

      <div className="bd-bar">
        <select
          value={selectedCompId}
          onChange={(e) => loadComponent(e.target.value)}
          style={{ minWidth: 260 }}
          aria-label="Select model architecture to visualize"
        >
          <option value="" disabled>Select model architecture…</option>
          {components.map((c) => (
            <option key={c.component_id} value={c.component_id}>
              {c.name} ({c.component_id}) {c.origin ? `[${c.origin}]` : ''}
            </option>
          ))}
        </select>

        <span className="bd-sep" aria-hidden />

        <button className="btn" onClick={() => rf.fitView({ padding: 0.25, maxZoom: 1 })}>
          Fit view
        </button>
        <button className="btn" onClick={viewPyTorch} disabled={!selectedCompId && !g.nodes.length}>
          View PyTorch
        </button>
        <button className="btn primary" onClick={runComponent} disabled={!selectedCompId}>
          Launch Run
        </button>

        <div className={'bd-status ' + status.kind} title={status.text}>
          {status.text}
        </div>
      </div>

      {result?.stats && (
        <div className="bd-stats" role="status" aria-label="Model size">
          <span>
            <b>{fmtCount(result.stats.params)}</b> parameters{' '}
            <i title={`${result.stats.params.toLocaleString()} bytes`}>
              · {fmtBytes(result.stats.bytes_fp32)}
            </i>
          </span>
          <span>
            <b>{result.stats.macs === null ? 'n/a' : fmtCount(result.stats.macs)}</b> MACs/sample
          </span>
          <span>
            <b>{result.stats.blocks}</b> fundamental blocks
          </span>
          <span>
            <b>{result.stats.compounds}</b> compound blocks
          </span>
          <span>
            <b>{result.stats.depth}</b> nesting level{result.stats.depth === 1 ? '' : 's'}
          </span>
          {Object.entries(result.stats.by_block)
            .filter(([, v]) => v > 0)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([k, v]) => (
              <span key={k} className="muted">
                {k} {Math.round((v / Math.max(1, result.stats!.params)) * 100)}%
              </span>
            ))}
        </div>
      )}

      <div className="bd-main">
        <div className="bd-canvas">
          <ReactFlow
            nodes={g.nodes}
            edges={g.edges}
            nodeTypes={nodeTypes}
            colorMode="dark"
            fitView
            fitViewOptions={{ maxZoom: 1, padding: 0.25 }}
            minZoom={0.15}
            onNodesChange={onNodesChange}
            nodesDraggable={true}
            nodesConnectable={false}
            elementsSelectable={true}
            onNodeClick={(_, n) => setSelectedNode(n as BNode)}
            onPaneClick={() => setSelectedNode(null)}
            deleteKeyCode={null}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={22} color="#222" />
            <Controls showInteractive={false} />
            {g.nodes.some((n) => n.type === 'block') && (
              <MiniMap
                pannable
                zoomable
                maskColor="rgba(0,0,0,.6)"
                nodeColor={(n) =>
                  n.type === 'block'
                    ? famColor(blocks[(n.data as { block: string }).block]?.family)
                    : '#555'
                }
              />
            )}
            {!g.nodes.some((n) => n.type === 'block') && (
              <div className="bd-hint">
                <Icon name="builder" size={16} />
                <span>Select a component from the dropdown above to inspect its DAG.</span>
              </div>
            )}
          </ReactFlow>
        </div>

        <div className="bd-side">
          <Inspector
            node={activeNode}
            result={result}
            blocks={blocks}
            component={currentComponent}
          />
        </div>
      </div>

      {code && (
        <Modal wide title={code.title} onClose={() => setCode(null)}>
          <pre className="code">{code.text}</pre>
          <div className="modal-foot">
            <button
              className="btn"
              onClick={() => {
                navigator.clipboard?.writeText(code.text);
                toast('Copied to clipboard');
              }}
            >
              Copy
            </button>
            <button className="btn primary" onClick={() => setCode(null)}>
              Close
            </button>
          </div>
        </Modal>
      )}
    </BuilderCtx.Provider>
  );
}

export function Visualizer(props: Props) {
  return (
    <ReactFlowProvider>
      <Inner {...props} />
    </ReactFlowProvider>
  );
}

// Keep Builder as an alias for backwards compatibility
export const Builder = Visualizer;
