import { useState } from 'react';
import { api } from '../api';
import { Modal, useToast } from '../ui';

export function Sweeps({ refresh, go }: { refresh: () => void; go: (t: string) => void }) {
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [trials, setTrials] = useState(8);
  const [epochs, setEpochs] = useState(5);
  const launch = async () => {
    try {
      await api.submit('asha-sweep-run', `uv run python recipes/sweep_orchestrator.py --trials ${trials} --max-epochs ${epochs}`);
      setOpen(false); toast('Sweep started'); refresh(); go('runs');
    } catch (e) { toast((e as Error).message, true); }
  };
  return (
    <>
      <div className="page-header">
        <div><div className="eyebrow">HYPERPARAMETER TUNING</div><h1 className="page-title">Sweeps</h1><p className="page-sub">Optuna-backed ASHA multi-trial exploration.</p></div>
        <button className="btn primary" onClick={() => setOpen(true)}>Configure sweep</button>
      </div>
      <div className="card empty">
        <b style={{ color: '#fff' }}>No active sweep campaigns</b>
        <p style={{ maxWidth: 480, margin: '6px auto 14px' }}>Graph components declare sweepable hyperparameters (learning rate, width, depth); run concurrent trials with early stopping.</p>
        <button className="btn sm" onClick={() => setOpen(true)}>Configure new sweep</button>
      </div>
      {open && (
        <Modal title="Configure sweep" sub="recipes/sweep_orchestrator.py" onClose={() => setOpen(false)}>
          <div className="field"><label>Trials</label><input type="number" min={1} value={trials} onChange={(e) => setTrials(+e.target.value)} /></div>
          <div className="field"><label>Max epochs per trial</label><input type="number" min={1} value={epochs} onChange={(e) => setEpochs(+e.target.value)} /></div>
          <div className="modal-foot"><button className="btn" onClick={() => setOpen(false)}>Cancel</button><button className="btn primary" onClick={launch}>Launch sweep</button></div>
        </Modal>
      )}
    </>
  );
}
