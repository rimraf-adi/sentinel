import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import type { JobStatus } from './api';

export function Modal({ title, sub, onClose, children, wide }: { title: string; sub?: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  return (
    <div className="modal-bg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={'modal' + (wide ? ' wide' : '')}>
        <div className="row sp" style={{ marginBottom: 14 }}>
          <div>
            <h3>{title}</h3>
            {sub && <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{sub}</div>}
          </div>
          <button className="btn sm" onClick={onClose}>✕</button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="field">
      <label>{label}{hint && <small>{hint}</small>}</label>
      {children}
    </div>
  );
}

const STATUS: Record<JobStatus, { label: string; dot: string }> = {
  pending: { label: 'Pending', dot: '' },
  running: { label: 'Running', dot: 'info pulse' },
  completed: { label: 'Completed', dot: 'ok' },
  failed: { label: 'Failed', dot: 'err' },
  pruned: { label: 'Pruned', dot: 'warn' },
  cancelled: { label: 'Cancelled', dot: 'warn' },
};
export function StatusPill({ status }: { status: JobStatus }) {
  const s = STATUS[status] ?? STATUS.pending;
  return <span className="pill"><span className={'dot ' + s.dot} />{s.label}</span>;
}

export const fmtGB = (b: number) => (b / 1024 ** 3).toFixed(2);
export const fmtTime = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '—');

type Toast = { msg: string; err?: boolean } | null;
const ToastCtx = createContext<(msg: string, err?: boolean) => void>(() => {});
export const useToast = () => useContext(ToastCtx);
export function ToastProvider({ children }: { children: ReactNode }) {
  const [t, setT] = useState<Toast>(null);
  const show = useCallback((msg: string, err?: boolean) => {
    setT({ msg, err });
    setTimeout(() => setT(null), 4200);
  }, []);
  return (
    <ToastCtx.Provider value={show}>
      {children}
      {t && <div className={'toast' + (t.err ? ' err' : '')}>{t.msg}</div>}
    </ToastCtx.Provider>
  );
}
