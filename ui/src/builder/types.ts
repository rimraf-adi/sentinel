import type { Node, Edge } from '@xyflow/react';
import type { BlockInfo, Shape, Training, ValidateOk } from '../api';

export interface BlockData extends Record<string, unknown> {
  block: string;
  params: Record<string, unknown>;
  share: string | null;
}
export interface InputData extends Record<string, unknown> { name: string; shape: Shape }
export interface OutputData extends Record<string, unknown> { name: string }

export type BlockNode = Node<BlockData, 'block'>;
export type InputNode = Node<InputData, 'input'>;
export type OutputNode = Node<OutputData, 'output'>;
export type BNode = BlockNode | InputNode | OutputNode;

export interface Graph { nodes: BNode[]; edges: Edge[]; training: Training | null }

export interface Ctx {
  blocks: Record<string, BlockInfo>;
  result: ValidateOk | null;
  errorNode: string | null;
}
