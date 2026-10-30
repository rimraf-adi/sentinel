import { useState } from 'react';
import { api, type CsvSchema, type Dataset } from './api';
import { usePoll } from './hooks';

export interface Picked { path: string; schema: CsvSchema }

/** Choose a mounted dataset (or a project path), read its columns, and report the schema. */
export function DatasetPicker({ onPick }: { onPick: (p: Picked) => void }) {
  const { data } = usePoll<Dataset[]>(api.datasets, 10000);
  const [custom, setCustom] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = async (path: string, fetcher: () => Promise<CsvSchema>) => {
    setBusy(true); setError(null);
    try { onPick({ path, schema: await fetcher() }); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const csvs = (data ?? []).filter((d) => d.format === 'csv');

  return (
    <div>
      <div className="field">
        <label>Dataset <small>mounted CSV files</small></label>
        <select value="" disabled={busy} onChange={(e) => { const d = csvs.find((x) => x.name === e.target.value); if (d) load(d.path, () => api.schemaOfDataset(d.name)); }}>
          <option value="">{csvs.length ? 'Choose a dataset…' : 'No CSV datasets mounted'}</option>
          {csvs.map((d) => <option key={d.name} value={d.name}>{d.name} — {d.path}</option>)}
        </select>
      </div>
      <div className="field">
        <label>…or a project path</label>
        <div className="row">
          <input className="mono" value={custom} placeholder="data/other.csv" onChange={(e) => setCustom(e.target.value)} />
          <button className="btn sm" disabled={!custom.trim() || busy} onClick={() => load(custom.trim(), () => api.schemaOfPath(custom.trim()))}>Inspect</button>
        </div>
      </div>
      {error && <p className="err-text">{error}</p>}
    </div>
  );
}

/** Pick features and the target from a schema; shows the inferred task. */
export function SchemaEditor({ schema, features, target, onChange }: {
  schema: CsvSchema; features: string[]; target: string; onChange: (features: string[], target: string) => void;
}) {
  const tcol = schema.columns.find((c) => c.name === target);
  const classification = !!tcol && (tcol.kind === 'categorical' || (tcol.kind === 'integer' && tcol.distinct <= 20 && !tcol.distinct_capped));
  const toggle = (n: string) => onChange(features.includes(n) ? features.filter((f) => f !== n) : [...features, n], target);
  return (
    <div>
      <p className="hint">{schema.rows.toLocaleString()} rows · {classification ? `classification, ${tcol?.distinct} classes` : 'regression'}{(schema.columns.some((c) => c.missing > 0)) ? ' · rows with missing values are skipped' : ''}</p>
      <div style={{ maxHeight: 190, overflow: 'auto', border: '1px solid var(--border)', borderRadius: 6 }}>
        <table style={{ fontSize: 11 }}>
          <thead><tr><th>Feature</th><th>Target</th><th>Column</th><th>Type</th><th>Values</th></tr></thead>
          <tbody>
            {schema.columns.map((c) => {
              const usable = c.kind !== 'categorical';
              return (
                <tr key={c.name}>
                  <td><input type="checkbox" disabled={!usable || c.name === target} checked={features.includes(c.name)} onChange={() => toggle(c.name)} /></td>
                  <td><input type="radio" name="sv-target" checked={c.name === target} onChange={() => onChange(features.filter((f) => f !== c.name), c.name)} /></td>
                  <td className="mono">{c.name}</td><td>{c.kind}</td>
                  <td className="mono" style={{ color: 'var(--text-muted)' }}>{c.kind === 'categorical' ? `${c.distinct}${c.distinct_capped ? '+' : ''}: ${c.examples.join(', ')}` : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export const taskOf = (schema: CsvSchema, target: string) => {
  const c = schema.columns.find((x) => x.name === target);
  const cls = !!c && (c.kind === 'categorical' || (c.kind === 'integer' && c.distinct <= 20 && !c.distinct_capped));
  return { classification: cls, loss: cls ? 'cross_entropy' : 'mse', outputs: cls ? (c?.distinct ?? 2) : 1 };
};
