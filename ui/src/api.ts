// Typed client for the Sentinel daemon REST API.

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'pruned' | 'cancelled';

export interface JobSpec {
  id: string;
  name: string;
  run_mode: 'command' | 'sdk' | 'framework';
  command: string | null;
  component_id: string | null;
  working_dir: string;
  created_at: string;
}
export interface JobRecord {
  spec: JobSpec;
  status: JobStatus;
  pid: number | null;
  exit_code: number | null;
  error_message: string | null;
  started_at: string | null;
  finished_at: string | null;
}
export interface GpuStats {
  index: number;
  name: string;
  vram_used_bytes: number;
  vram_total_bytes: number;
  gpu_util_percent: number;
  temperature_c: number;
}
export interface Hardware {
  timestamp: string;
  machine_id: string;
  cpu_util_percent: number;
  ram_used_bytes: number;
  ram_total_bytes: number;
  gpus: GpuStats[];
}
export interface ParamSpec {
  type: 'string' | 'int' | 'float' | 'bool' | 'choice';
  default: unknown;
  options: string[];
  range: [number, number] | null;
  help: string;
}
export interface Component {
  component_id: string;
  name: string;
  description: string;
  domain: string;
  extends: string | null;
  parameters: Record<string, ParamSpec>;
  locked: Record<string, unknown>;
  graph?: unknown;
}
export interface Dataset {
  name: string;
  path: string;
  format: string;
  description: string;
  split: string | null;
  is_external: boolean;
  exists?: boolean;
  file_count?: number;
}
export interface RepoStatus {
  is_sentinel_repo: boolean;
  workspace_name: string;
  workspace_path: string;
  full_workspace_path: string;
  git_branch: string;
  git_clean: boolean;
  lock_file_found: boolean;
  convert_commands: string[];
}
export interface MetricPoint { step: number; value: number; timestamp: number }
export interface TrackedRun {
  run_id: string;
  experiment_id: string;
  run_name: string;
  status: 'running' | 'finished' | 'failed' | 'killed' | 'unknown';
  start_time: number | null;
  end_time: number | null;
  params: Record<string, string>;
  latest: Record<string, number>;
}
export interface BlockParam {
  name: string;
  required: boolean;
  default: unknown;
  help: string;
}
export interface BlockInfo {
  name: string;
  family: string;
  description: string;
  inputs: string[];
  outputs: string[];
  params: BlockParam[];
}
export type Shape = (string | number)[];
export interface GraphSpec {
  inputs: Record<string, Shape>;
  nodes: Record<string, { block: string; params: Record<string, unknown>; share?: string | null }>;
  edges: string[];
  outputs: Record<string, string>;
  training?: Training | null;
}
export interface Training {
  loss: string;
  optimizer: string;
  lr: number;
  epochs: number;
  batch_size?: number | null;
  data: { csv: string; features: string[]; target: string; reshape?: number[] | null; test_split?: number; seed?: number };
}
export interface ValidateOk {
  ok: true;
  id: string;
  order: string[];
  inputs: Record<string, Shape>;
  nodes: Record<string, { block: string; inputs: Record<string, string>; outputs: Record<string, Shape> }>;
  outputs: Record<string, string>;
  trainable: boolean;
  spec: GraphSpec;
}
export interface ApiErr {
  ok: false;
  error: string;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  const text = await r.text();
  let body: unknown;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!r.ok) {
    const msg = typeof body === 'string' ? body : (body as { error?: string })?.error ?? r.statusText;
    throw new Error(msg || `HTTP ${r.status}`);
  }
  return body as T;
}
const post = <T>(path: string, body?: unknown) =>
  req<T>(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  jobs: () => req<JobRecord[]>('/api/jobs'),
  logs: (id: string) => req<{ logs: string[] }>(`/api/jobs/${id}/logs`),
  stop: (id: string) => post<unknown>(`/api/jobs/${id}/stop`),
  submit: (name: string, command: string) =>
    post<JobRecord>('/api/jobs', {
      id: crypto.randomUUID(),
      name,
      run_mode: 'command',
      command,
      parameters: {},
      env_vars: {},
      gpu_ids: [],
      working_dir: '.',
      created_at: new Date().toISOString(),
    }),
  hardware: () => req<Hardware>('/api/hardware'),
  components: () => req<Component[]>('/api/components'),
  runComponent: (component_id: string, overrides: Record<string, unknown>) => post<JobRecord>('/api/jobs/run-component', { component_id, overrides }),
  repo: () => req<RepoStatus>('/api/repo/status'),
  initRepo: () => post<unknown>('/api/repo/init'),
  datasets: () => req<Dataset[]>('/api/datasets'),
  addDataset: (d: { name: string; path: string; format: string; description: string }) => post<unknown>('/api/datasets', d),
  removeDataset: (name: string) => req<unknown>(`/api/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  files: () => req<{ name: string; path: string }[]>('/api/files'),
  file: (path: string) => req<{ content: string; path: string }>(`/api/files/content?path=${encodeURIComponent(path)}`),
  trackedRuns: () => req<TrackedRun[]>('/api/tracking/runs'),
  trackedRun: (id: string) => req<{ run: TrackedRun; metrics: Record<string, MetricPoint[]> }>(`/api/tracking/runs/${id}`),
  blocks: () => req<BlockInfo[]>('/api/blocks'),
  validate: (b: { graph?: GraphSpec; component_id?: string; overrides?: Record<string, unknown> }) =>
    post<ValidateOk | ApiErr>('/api/graph/validate', b).catch((e: Error) => ({ ok: false, error: e.message }) as ApiErr),
  compose: (b: { graph?: GraphSpec; component_id?: string; overrides?: Record<string, unknown>; model_only?: boolean }) =>
    post<{ ok: true; code: string }>('/api/graph/compose', b),
  runGraph: (b: { graph?: GraphSpec; component_id?: string; overrides?: Record<string, unknown> }) => post<JobRecord>('/api/graph/run', b),
};
