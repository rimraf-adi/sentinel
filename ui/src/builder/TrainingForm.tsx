import { useEffect, useState } from 'react';
import { api, type CatChoice, type CatParam, type Named, type Training, type TrainingCatalog } from '../api';
import { Field } from '../ui';

let cached: TrainingCatalog | null = null;
export function useTrainingCatalog() {
  const [c, setC] = useState<TrainingCatalog | null>(cached);
  useEffect(() => { if (!cached) api.trainingCatalog().then((x) => { cached = x; setC(x); }).catch(() => undefined); }, []);
  return c;
}

export const nameOf = (n: Named | undefined, fallback = ''): string => (n === undefined ? fallback : typeof n === 'string' ? n : n.name);
const settingsOf = (n: Named | undefined): Record<string, unknown> => {
  if (!n || typeof n === 'string') return {};
  const { name: _name, ...rest } = n;
  return rest;
};
/** A choice with no settings is written as a bare name, the way hand-written specs do. */
const build = (name: string, settings: Record<string, unknown>): Named => (Object.keys(settings).length ? { name, ...settings } : name);

const parse = (s: string): unknown => { try { return JSON.parse(s); } catch { return s; } };
const show = (v: unknown) => (v === undefined || v === null ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v));

function Text({ value, onCommit, mono, placeholder }: { value: string; onCommit: (v: string) => void; mono?: boolean; placeholder?: string }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return <input className={mono ? 'mono' : ''} placeholder={placeholder} value={v} onChange={(e) => setV(e.target.value)} onBlur={() => v !== value && onCommit(v)} onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} />;
}

function Area({ value, onCommit, placeholder }: { value: string; onCommit: (v: string) => void; placeholder?: string }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return <textarea className="mono" rows={9} spellCheck={false} placeholder={placeholder} value={v} onChange={(e) => setV(e.target.value)} onBlur={() => v !== value && onCommit(v)} />;
}

const select = (opts: string[], cur: string, on: (v: string) => void) => (
  <select value={cur} onChange={(e) => on(e.target.value)}>{opts.map((o) => <option key={o} value={o}>{o || '(none)'}</option>)}</select>
);

/** One setting of a loss/optimizer/scheduler. Empty (or equal to the default) means "leave it out". */
function ParamField({ p, value, onChange }: { p: CatParam; value: unknown; onChange: (v: unknown) => void }) {
  const cur = value === undefined ? p.default : value;
  if (typeof p.default === 'boolean')
    return <Field label={p.name} hint={p.help}><input type="checkbox" checked={!!cur} onChange={(e) => onChange(e.target.checked === p.default ? undefined : e.target.checked)} /></Field>;
  if ((p.options?.length ?? 0) > 0)
    return <Field label={p.name} hint={p.help}>{select(p.options ?? [], String(cur), (v) => onChange(v === String(p.default) ? undefined : parse(v)))}</Field>;
  return <Field label={p.name} hint={p.help}><Text value={show(value)} placeholder={show(p.default)} onCommit={(s) => onChange(s.trim() === '' ? undefined : parse(s))} /></Field>;
}

function NamedPicker({ label, choices, value, onChange, none }: { label: string; choices: CatChoice[]; value: Named | undefined; onChange: (v: Named | undefined) => void; none?: boolean }) {
  const name = nameOf(value);
  const def = choices.find((c) => c.name === name);
  const settings = settingsOf(value);
  const names = none ? ['', ...choices.map((c) => c.name)] : choices.map((c) => c.name);
  return (
    <>
      <Field label={label} hint={def?.help}>{select(names, name, (v) => onChange(v ? v : undefined))}</Field>
      {def?.params.map((pp) => (
        <ParamField key={pp.name} p={pp} value={settings[pp.name]} onChange={(v) => {
          const next = { ...settings };
          if (v === undefined) delete next[pp.name]; else next[pp.name] = v;
          onChange(build(name, next));
        }} />
      ))}
    </>
  );
}

const num = (s: string): number | undefined => { const v = parseFloat(s); return Number.isFinite(v) ? v : undefined; };
const int = (s: string): number | undefined => { const v = parseInt(s, 10); return Number.isFinite(v) ? v : undefined; };

/** Task, loss, optimizer, scheduler and training strategies, all driven by the server's catalog. */
export function TrainingForm({ t, set, nodeIds }: { t: Training; set: (patch: Partial<Training>) => void; nodeIds: string[] }) {
  const cat = useTrainingCatalog();
  const [open, setOpen] = useState(false);
  if (!cat) return <p className="hint">Loading training options…</p>;

  const lossName = nameOf(t.loss);
  const inferred = cat.tasks.find((k) => k.losses.some((l) => l.name === lossName))?.name ?? 'classification';
  const taskName = t.task ?? inferred;
  const task = cat.tasks.find((k) => k.name === taskName) ?? cat.tasks[0];
  const cust = taskName === 'custom' || lossName === 'custom';
  const unsupervised = taskName === 'reconstruction' || taskName === 'vae';
  const pickTask = (name: string) => {
    const k = cat.tasks.find((x) => x.name === name)!;
    const keep = k.losses.some((l) => l.name === lossName);
    set({ task: name, loss: keep ? t.loss : k.losses[0].name, ...(name === 'custom' && !t.custom ? { custom: { source: 'import torch\n\ndef my_loss(output, target):\n    return (output - target.view_as(output)).abs().mean()\n', loss: 'my_loss' } } : {}) });
  };
  const patchData = (d: Partial<Training['data']>) => {
    const data = { ...t.data, ...d };
    for (const k of Object.keys(data) as (keyof Training['data'])[]) if (data[k] === undefined) delete data[k];
    set({ data });
  };
  const csv = (s: string) => s.split(',').map((x) => x.trim()).filter(Boolean);
  const es = t.early_stopping;

  return (
    <>
      <Field label="Task" hint={task.help}>{select(cat.tasks.map((k) => k.name), taskName, pickTask)}</Field>
      <NamedPicker label="Loss" choices={task.losses} value={t.loss} onChange={(v) => v && set({ loss: v })} />
      <p className="hint">Needs: {task.needs}. Reports {task.metrics.join(', ')}.</p>

      {cust && (
        <>
          <Field label="Custom code" hint="Python defining your loss (and optional metric)"><Area value={t.custom?.source ?? ''} placeholder="def my_loss(output, target): ..." onCommit={(v) => set({ custom: { ...t.custom, source: v, file: undefined } })} /></Field>
          <Field label="Loss function"><Text mono value={t.custom?.loss ?? ''} onCommit={(v) => set({ custom: { ...t.custom, loss: v || undefined } })} /></Field>
          <Field label="Metric function" hint="optional, returns a number"><Text mono value={t.custom?.metric ?? ''} onCommit={(v) => set({ custom: { ...t.custom, metric: v || undefined } })} /></Field>
          <Field label="Metric mode">{select(['min', 'max'], t.custom?.mode ?? 'min', (v) => set({ custom: { ...t.custom, mode: v } }))}</Field>
          <Field label="Or a file" hint="path inside the project; replaces the code above"><Text mono value={t.custom?.file ?? ''} onCommit={(v) => set({ custom: v ? { ...t.custom, file: v, source: undefined } : { ...t.custom, file: undefined } })} /></Field>
        </>
      )}

      <Field label="CSV path"><Text mono value={t.data.csv} onCommit={(v) => patchData({ csv: v })} /></Field>
      <Field label="Feature columns" hint="comma separated"><Text value={t.data.features.join(', ')} onCommit={(v) => patchData({ features: csv(v) })} /></Field>
      {!unsupervised && (taskName === 'multilabel'
        ? <Field label="Target columns" hint="comma separated 0/1 columns"><Text value={(t.data.targets ?? []).join(', ')} onCommit={(v) => patchData({ targets: csv(v), target: undefined })} /></Field>
        : <Field label="Target column"><Text value={t.data.target ?? ''} onCommit={(v) => patchData({ target: v || undefined })} /></Field>)}
      {taskName === 'ranking' && <Field label="Query column" hint="items with the same value are ranked together"><Text value={t.data.group ?? ''} onCommit={(v) => patchData({ group: v || undefined })} /></Field>}

      <NamedPicker label="Optimizer" choices={cat.optimizers} value={t.optimizer} onChange={(v) => v && set({ optimizer: v })} />
      <Field label="Learning rate"><Text value={String(t.lr)} onCommit={(v) => set({ lr: num(v) ?? t.lr })} /></Field>
      <Field label="Weight decay" hint="empty = none"><Text value={show(t.weight_decay)} onCommit={(v) => set({ weight_decay: num(v) })} /></Field>
      <Field label="Epochs"><Text value={String(t.epochs)} onCommit={(v) => set({ epochs: int(v) ?? t.epochs })} /></Field>
      <Field label="Batch size" hint="empty = full batch"><Text value={show(t.batch_size)} onCommit={(v) => set({ batch_size: v.trim() ? int(v) : null })} /></Field>

      <button className="btn sm" style={{ margin: '6px 0' }} onClick={() => setOpen(!open)}>{open ? 'Hide' : 'Show'} schedule &amp; strategies</button>
      {open && (
        <>
          <NamedPicker label="LR schedule" none choices={cat.schedulers} value={t.scheduler} onChange={(v) => set({ scheduler: v })} />
          <Field label="Clip gradients" hint="max norm, empty = off"><Text value={show(t.clip_grad)} onCommit={(v) => set({ clip_grad: num(v) })} /></Field>
          <Field label="Accumulate batches" hint="empty = 1"><Text value={show(t.accumulate)} onCommit={(v) => set({ accumulate: int(v) })} /></Field>
          <Field label="Mixed precision">{select(cat.amp, t.amp ?? cat.amp[0], (v) => set({ amp: v === cat.amp[0] ? undefined : v }))}</Field>
          <Field label="Device">{select(cat.devices, t.device ?? cat.devices[0], (v) => set({ device: v === cat.devices[0] ? undefined : v }))}</Field>
          <Field label="Early stopping" hint="patience in epochs, empty = off"><Text value={show(es?.patience)} onCommit={(v) => { const n = int(v); set({ early_stopping: n ? { ...es, patience: n } : undefined }); }} /></Field>
          {es && <Field label="Monitor" hint="metric to watch">{select(task.metrics, es.monitor ?? task.primary, (v) => set({ early_stopping: { ...es, monitor: v } }))}</Field>}
          <Field label="Restore best epoch"><input type="checkbox" checked={!!t.save_best} onChange={(e) => set({ save_best: e.target.checked || undefined })} /></Field>
          <Field label="Freeze nodes" hint={`comma separated, from: ${nodeIds.join(', ') || '—'}`}><Text value={(t.freeze ?? []).join(', ')} onCommit={(v) => set({ freeze: csv(v).length ? csv(v) : undefined })} /></Field>
          <Field label="Start from weights" hint="path to a saved model.pt"><Text mono value={t.init_from ?? ''} onCommit={(v) => set({ init_from: v || undefined })} /></Field>
        </>
      )}
    </>
  );
}
