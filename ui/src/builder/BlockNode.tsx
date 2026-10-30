import { createContext, useContext } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { BlockNode as BN, Ctx, InputNode, OutputNode } from './types';
import { shapeText } from './convert';

export const BuilderCtx = createContext<Ctx>({ blocks: {}, result: null, errorNode: null });

export function BlockNodeView({ id, data, selected }: NodeProps<BN>) {
  const { blocks, result, errorNode } = useContext(BuilderCtx);
  const info = blocks[data.block];
  const r = result?.nodes[id];
  if (!info) return <div className="bnode bad"><div className="bnode-h"><b>{id}</b><span>unknown: {data.block}</span></div></div>;
  const srcShape = (key?: string) => {
    if (!result || !key) return null;
    if (result.inputs[key]) return result.inputs[key];
    const [n, p] = key.split('.');
    return result.nodes[n]?.outputs[p] ?? null;
  };
  return (
    <div className={'bnode' + (selected ? ' selected' : '') + (errorNode === id ? ' bad' : '')} style={{ ['--fam' as string]: `var(--fam-${info.family})` }}>
      <div className="bnode-h"><b>{id}</b><span>{data.block}{data.share ? ` ⇢ ${data.share}` : ''}</span></div>
      <div className="bnode-b">
        <div>
          {info.inputs.map((p) => (
            <div className="bport" key={p}>
              <Handle type="target" position={Position.Left} id={p} />
              <span>{p}</span><span className="shape">{shapeText(srcShape(r?.inputs[p]))}</span>
            </div>
          ))}
        </div>
        <div>
          {info.outputs.map((p) => (
            <div className="bport out" key={p}>
              <Handle type="source" position={Position.Right} id={p} />
              <span>{p}</span><span className="shape">{shapeText(r?.outputs[p])}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function InputNodeView({ data, selected }: NodeProps<InputNode>) {
  return (
    <div className={'bnode io' + (selected ? ' selected' : '')}>
      <div className="bnode-h"><b>input {data.name}</b></div>
      <div className="bnode-b"><div /><div className="bport out"><Handle type="source" position={Position.Right} id="out" /><span className="shape">{shapeText(data.shape)}</span></div></div>
    </div>
  );
}

export function OutputNodeView({ data, selected }: NodeProps<OutputNode>) {
  return (
    <div className={'bnode io' + (selected ? ' selected' : '')}>
      <div className="bnode-h"><b>output {data.name}</b></div>
      <div className="bnode-b"><div className="bport"><Handle type="target" position={Position.Left} id="in" /><span className="shape">logits</span></div><div /></div>
    </div>
  );
}
