import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import { CommandPalette } from './CommandPalette';
import { usePoll, useTelemetry } from './hooks';
import { Icon, type IconName } from './icons';
import { Logo } from './Logo';
import { ToastProvider } from './ui';
import { Overview } from './pages/Overview';
import { Runs } from './pages/Runs';
import { Experiments } from './pages/Experiments';
import { Sweeps } from './pages/Sweeps';
import { Components } from './pages/Components';
import { Builder as Visualizer } from './builder/Builder';
import { Datasets } from './pages/Datasets';
import { System } from './pages/System';

const TABS = [
  { id: 'overview', label: 'Overview', group: 'Workspace' },
  { id: 'runs', label: 'Runs', group: 'Workspace' },
  { id: 'experiments', label: 'Experiments', group: 'Workspace' },
  { id: 'sweeps', label: 'Sweeps', group: 'Workspace' },
  { id: 'system', label: 'System', group: 'Workspace' },
  { id: 'components', label: 'Components', group: 'Project' },
  { id: 'visualizer', label: 'DAG Visualizer', group: 'Project' },
  { id: 'datasets', label: 'Datasets', group: 'Project' },
] as const;
type Tab = (typeof TABS)[number]['id'];

const fromHash = (): Tab => {
  const h = location.hash.slice(1);
  if (h === 'builder') return 'visualizer';
  return (TABS.find((t) => t.id === h)?.id ?? 'overview');
};
const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

export function App() {
  const [tab, setTab] = useState<Tab>(fromHash);
  const [visualizerComponent, setVisualizerComponent] = useState<string | null>(null);
  const [palette, setPalette] = useState(false);
  const repo = usePoll(api.repo, 5000);
  const jobs = usePoll(api.jobs, 2000);
  const datasets = usePoll(api.datasets, 5000);
  const { hw, online } = useTelemetry();

  useEffect(() => {
    const on = () => setTab(fromHash());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPalette((p) => !p); }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, []);
  const go = useCallback((t: string) => { location.hash = t; }, []);
  const openInVisualizer = useCallback((id: string) => { setVisualizerComponent(id); go('visualizer'); }, [go]);
  const closePalette = useCallback(() => setPalette(false), []);

  const r = repo.data;
  const running = jobs.data?.filter((j) => j.status === 'running').length ?? 0;
  const name = r?.workspace_name ?? 'sentinel';
  const groups = [...new Set(TABS.map((t) => t.group))];
  const title = TABS.find((t) => t.id === tab)!.label;

  return (
    <ToastProvider>
      <a className="skip" href="#main">Skip to content</a>
      <div className="app">
        <aside className="sidebar">
          <div className="brand"><Logo size={28} /><div className="brand-text"><b>sentinel</b><small>TRAINING CONTROL</small></div></div>
          <div className="ws-card">
            <b>{name}</b>
            <span className="mono">{r?.workspace_path ?? '…'}</span>
            <div className="row" style={{ marginTop: 6 }}>
              <span className="tag">{r?.git_branch ?? 'main'}</span>
              <span className="tag">{r ? (r.git_clean ? 'clean' : 'modified') : '…'}</span>
            </div>
          </div>
          <nav aria-label="Main">
            {groups.map((g) => (
              <div key={g} style={{ marginBottom: 14 }}>
                <div className="nav-title">{g}</div>
                {TABS.filter((t) => t.group === g).map((t) => (
                  <button key={t.id} className={'nav-item' + (t.id === tab ? ' active' : '')} aria-current={t.id === tab ? 'page' : undefined} onClick={() => go(t.id)} title={t.label}>
                    <Icon name={t.id as IconName} />
                    <span className="nav-label">{t.label}</span>
                    {t.id === 'runs' && running > 0 && <span className="badge" aria-label={`${running} running`}>{running}</span>}
                    {t.id === 'datasets' && <span className="badge">{datasets.data?.length ?? 0}</span>}
                  </button>
                ))}
              </div>
            ))}
          </nav>
        </aside>
        <div className="main">
          <header className="topbar">
            <div className="crumb">{name} / <b>{title}</b></div>
            <div className="pills">
              <button className="search-btn" onClick={() => setPalette(true)} aria-label="Open command palette" aria-keyshortcuts="Control+K Meta+K">
                <span className="row" style={{ gap: 8 }}><Icon name="search" size={14} /><span className="hint-text">Search or jump to…</span></span>
                <span className="kbd">{isMac ? '⌘' : 'Ctrl'} K</span>
              </button>
              {r && !r.lock_file_found && <span className="pill"><span className="dot warn" />sentinel.lock missing</span>}
              <span className="pill" role="status"><span className={'dot ' + (online ? 'ok' : 'err')} />{online ? 'daemon online' : 'daemon offline'}</span>
            </div>
          </header>
          <main id="main" tabIndex={-1} className={'content' + (tab === 'visualizer' ? ' fill' : '')}>
            {tab === 'overview' && <Overview repo={r ?? null} jobs={jobs.data ?? []} hw={hw} onRepoChanged={repo.refresh} go={go} />}
            {tab === 'runs' && <Runs jobs={jobs.data ?? []} refresh={jobs.refresh} go={go} />}
            {tab === 'experiments' && <Experiments go={go} />}
            {tab === 'sweeps' && <Sweeps refresh={jobs.refresh} go={go} />}
            {tab === 'system' && <System hw={hw} />}
            {tab === 'components' && <Components refresh={jobs.refresh} go={go} openInVisualizer={openInVisualizer} />}
            {tab === 'visualizer' && <Visualizer initialComponent={visualizerComponent} onConsumed={() => setVisualizerComponent(null)} refreshJobs={jobs.refresh} go={go} />}
            {tab === 'datasets' && <Datasets datasets={datasets.data ?? []} refresh={datasets.refresh} />}
          </main>
        </div>
      </div>
      {palette && <CommandPalette onClose={closePalette} go={go} openInVisualizer={openInVisualizer} tabs={TABS.map((t) => ({ id: t.id, label: t.label }))} />}
    </ToastProvider>
  );
}
