import { useEffect, useState } from 'react';
import { api } from '../api';
import { useToast } from '../ui';

export function Code() {
  const toast = useToast();
  const [files, setFiles] = useState<{ name: string; path: string }[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [text, setText] = useState('');
  useEffect(() => { api.files().then((f) => { setFiles(f); if (f[0]) setActive(f[0].path); }).catch(() => {}); }, []);
  useEffect(() => {
    if (!active) return;
    api.file(active).then((r) => setText(r.content)).catch((e: Error) => setText(`Failed to load: ${e.message}`));
  }, [active]);
  return (
    <>
      <div className="page-header"><div><div className="eyebrow">READ-ONLY</div><h1 className="page-title">Code</h1>
        <p className="page-sub">Browse project files. Edit them in your local IDE.</p></div>
        <button className="btn" onClick={() => { navigator.clipboard?.writeText(text); toast('Copied'); }}>Copy</button></div>
      <div className="code-layout">
        <div className="file-list">{files.map((f) => <div key={f.path} className={'file-item' + (f.path === active ? ' active' : '')} onClick={() => setActive(f.path)}>{f.path}</div>)}
          {files.length === 0 && <div className="empty">No files.</div>}</div>
        <textarea className="viewer" readOnly spellCheck={false} value={text} />
      </div>
    </>
  );
}
