import { useState } from 'react';
import { api, type Dataset } from '../api';
import { Modal, useToast } from '../ui';

export function Datasets({ datasets, refresh }: { datasets: Dataset[]; refresh: () => void }) {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [f, setF] = useState({ name: '', path: '', format: 'csv', description: '' });
  const add = async () => {
    try { await api.addDataset(f); setOpen(false); setF({ name: '', path: '', format: 'csv', description: '' }); toast('Dataset mounted'); refresh(); }
    catch (e) { toast((e as Error).message, true); }
  };
  const del = async (n: string) => { try { await api.removeDataset(n); refresh(); } catch (e) { toast((e as Error).message, true); } };
  return (
    <>
      <div className="page-header">
        <div><div className="eyebrow">DATA</div><h1 className="page-title">Datasets</h1><p className="page-sub">Registered data sources, in-repo or mounted externally.</p></div>
        <button className="btn primary" onClick={() => setOpen(true)}>+ Mount dataset</button>
      </div>
      <div className="card" style={{ padding: 0 }}>
        <table>
          <thead><tr><th>Name</th><th>Path</th><th>Format</th><th>Location</th><th /></tr></thead>
          <tbody>
            {datasets.length === 0 && <tr><td colSpan={5} className="empty">No datasets mounted.</td></tr>}
            {datasets.map((d) => (
              <tr key={d.name}>
                <td><b>{d.name}</b><div style={{ color: 'var(--text-muted)', fontSize: 11 }}>{d.description}</div></td>
                <td className="mono" style={{ fontSize: 11 }}>{d.path}</td><td>{d.format}</td>
                <td><span className="tag">{d.is_external ? 'EXTERNAL' : 'IN REPO'}</span></td>
                <td><div className="row" style={{ justifyContent: 'flex-end' }}>
                  <button className="btn sm" onClick={() => navigator.clipboard?.writeText(d.path)}>Copy path</button>
                  <button className="btn sm danger" onClick={() => del(d.name)}>Unmount</button></div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {open && (
        <Modal title="Mount dataset" onClose={() => setOpen(false)}>
          <div className="field"><label>Name</label><input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></div>
          <div className="field"><label>Path</label><input className="mono" value={f.path} onChange={(e) => setF({ ...f, path: e.target.value })} placeholder="data/demo.csv" /></div>
          <div className="field"><label>Format</label>
            <select value={f.format} onChange={(e) => setF({ ...f, format: e.target.value })}>{['csv', 'parquet', 'json', 'images', 'text'].map((x) => <option key={x}>{x}</option>)}</select></div>
          <div className="field"><label>Description</label><input value={f.description} onChange={(e) => setF({ ...f, description: e.target.value })} /></div>
          <div className="modal-foot"><button className="btn" onClick={() => setOpen(false)}>Cancel</button>
            <button className="btn primary" disabled={!f.name || !f.path} onClick={add}>Mount</button></div>
        </Modal>
      )}
    </>
  );
}
