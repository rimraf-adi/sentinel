import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type Component } from './api';
import { Icon, type IconName } from './icons';

interface Item { id: string; group: string; label: string; hint?: string; icon: IconName; run: () => void }

export function CommandPalette({ onClose, go, openInBuilder, tabs }: {
  onClose: () => void; go: (t: string) => void; openInBuilder: (id: string) => void; tabs: { id: string; label: string }[];
}) {
  const [q, setQ] = useState('');
  const [sel, setSel] = useState(0);
  const [components, setComponents] = useState<Component[]>([]);
  const list = useRef<HTMLUListElement>(null);
  useEffect(() => { api.components().then(setComponents).catch(() => {}); }, []);

  const items = useMemo<Item[]>(() => {
    const done = (f: () => void) => () => { onClose(); f(); };
    return [
      ...tabs.map((t) => ({ id: 'go-' + t.id, group: 'Go to', label: t.label, icon: t.id as IconName, run: done(() => go(t.id)) })),
      { id: 'new', group: 'Actions', label: 'New graph in the Builder', hint: 'start from a blank canvas', icon: 'plus' as IconName, run: done(() => openInBuilder('__new__')) },
      { id: 'compare', group: 'Actions', label: 'Compare experiments', hint: 'overlay metric curves', icon: 'chart' as IconName, run: done(() => go('experiments')) },
      { id: 'vitals', group: 'Actions', label: 'Watch system vitals', hint: 'CPU, GPU, network, disks', icon: 'system' as IconName, run: done(() => go('system')) },
      ...components.filter((c) => c.graph).map((c) => ({ id: 'open-' + c.component_id, group: 'Open in Builder', label: c.name, hint: c.component_id, icon: 'builder' as IconName, run: done(() => openInBuilder(c.component_id)) })),
    ];
  }, [components, tabs, go, openInBuilder, onClose]);

  const shown = useMemo(() => {
    const toks = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!toks.length) return items;
    return items
      .map((i) => { const hay = `${i.label} ${i.hint ?? ''} ${i.group}`.toLowerCase(); return { i, ok: toks.every((t) => hay.includes(t)), s: toks.reduce((a, t) => a + (i.label.toLowerCase().startsWith(t) ? 2 : i.label.toLowerCase().includes(t) ? 1 : 0), 0) }; })
      .filter((x) => x.ok).sort((a, b) => b.s - a.s).map((x) => x.i);
  }, [items, q]);

  useEffect(() => setSel(0), [q]);
  useEffect(() => { list.current?.querySelector('[aria-selected=true]')?.scrollIntoView({ block: 'nearest' }); }, [sel]);

  const key = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(shown.length - 1, s + 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
    else if (e.key === 'Enter') { e.preventDefault(); shown[sel]?.run(); }
    else if (e.key === 'Escape') { e.preventDefault(); onClose(); }
  };

  let last = '';
  return (
    <div className="cmdk-bg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="cmdk" role="dialog" aria-modal="true" aria-label="Command palette" onKeyDown={key}>
        <input autoFocus role="combobox" aria-expanded="true" aria-controls="cmdk-list" aria-activedescendant={shown[sel] ? 'cmdk-' + shown[sel].id : undefined}
          placeholder="Search pages, actions and components…" value={q} onChange={(e) => setQ(e.target.value)} />
        <ul id="cmdk-list" role="listbox" ref={list}>
          {shown.length === 0 && <li style={{ cursor: 'default' }}>Nothing matches “{q}”.</li>}
          {shown.map((it, i) => {
            const head = it.group !== last ? <li key={'g' + it.group} className="grp" role="presentation" style={{ cursor: 'default' }}>{it.group}</li> : null;
            last = it.group;
            return [head,
              <li key={it.id} id={'cmdk-' + it.id} role="option" aria-selected={i === sel} onMouseMove={() => setSel(i)} onClick={it.run}>
                <Icon name={it.icon} /><span>{it.label}</span>{it.hint && <small className="mono">{it.hint}</small>}
              </li>];
          })}
        </ul>
        <footer><span><span className="kbd">↑</span> <span className="kbd">↓</span> navigate</span><span><span className="kbd">↵</span> open</span><span><span className="kbd">esc</span> close</span></footer>
      </div>
    </div>
  );
}
