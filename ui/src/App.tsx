import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import { usePoll, useTelemetry } from './hooks';
import { ToastProvider } from './ui';
import { Overview } from './pages/Overview';
import { Runs } from './pages/Runs';
import { Experiments } from './pages/Experiments';
import { Sweeps } from './pages/Sweeps';
import { Components } from './pages/Components';
import { Builder } from './builder/Builder';
import { Datasets } from './pages/Datasets';
import { Machines } from './pages/Machines';
import { Code } from './pages/Code';

const TABS = [
  { id: 'overview', label: 'Overview', group: 'Workspace' },
  { id: 'runs', label: 'Runs', group: 'Workspace' },
  { id: 'experiments', label: 'Experiments', group: 'Workspace' },
  { id: 'sweeps', label: 'Sweeps', group: 'Workspace' },
  { id: 'components', label: 'Components', group: 'Project' },
  { id: 'builder', label: 'Builder', group: 'Project' },
  { id: 'datasets', label: 'Datasets', group: 'Project' },
  { id: 'machines', label: 'Machines', group: 'Project' },
  { id: 'code', label: 'Code (Read-Only)', group: 'Project' },
] as const;
type Tab = (typeof TABS)[number]['id'];

const fromHash = (): Tab => (TABS.find((t) => t.id === location.hash.slice(1))?.id ?? 'overview');

export function App() {
  const [tab, setTab] = useState<Tab>(fromHash);
  const [builderComponent, setBuilderComponent] = useState<string | null>(null);
  const repo = usePoll(api.repo, 5000);
  const jobs = usePoll(api.jobs, 2000);
  const datasets = usePoll(api.datasets, 5000);
  const { hw, online } = useTelemetry();

  useEffect(() => {
    const on = () => setTab(fromHash());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const go = useCallback((t: string) => { location.hash = t; }, []);
  const openInBuilder = useCallback((id: string) => { setBuilderComponent(id); go('builder'); }, [go]);

  const r = repo.data;
  const running = jobs.data?.filter((j) => j.status === 'running').length ?? 0;
  const name = r?.workspace_name ?? 'sentinel';
  const groups = [...new Set(TABS.map((t) => t.group))];
  const title = TABS.find((t) => t.id === tab)!.label;

  return (
    <ToastProvider>
      <div className="app">
        <aside className="sidebar">
          <div className="brand"><div className="brand-mark" /><div><b>sentinel</b><small>TRAINING CONTROL</small></div></div>
          <div className="ws-card">
            <b>{name}</b>
            <span className="mono">{r?.workspace_path ?? '…'}</span>
            <div className="row" style={{ marginTop: 6 }}>
              <span className="tag">{r?.git_branch ?? 'main'}</span>
              <span className="tag">{r ? (r.git_clean ? 'clean' : 'modified') : '…'}</span>
            </div>
          </div>
          {groups.map((g) => (
            <div key={g}>
              <div className="nav-title">{g}</div>
              {TABS.filter((t) => t.group === g).map((t) => (
                <button key={t.id} className={'nav-item' + (t.id === tab ? ' active' : '')} onClick={() => go(t.id)}>
                  {t.label}
                  {t.id === 'runs' && <span className="badge">{running}</span>}
                  {t.id === 'datasets' && <span className="badge">{datasets.data?.length ?? 0}</span>}
                </button>
              ))}
            </div>
          ))}
        </aside>
        <div className="main">
          <header className="topbar">
            <div className="crumb">{name} / <b>{title}</b></div>
            <div className="pills">
              {r && !r.lock_file_found && <span className="pill"><span className="dot warn" />sentinel.lock missing</span>}
              <span className="pill"><span className={'dot ' + (online ? 'ok' : 'err')} />{online ? 'daemon online' : 'daemon offline'}</span>
            </div>
          </header>
          <main className={'content' + (tab === 'builder' || tab === 'code' ? ' fill' : '')}>
            {tab === 'overview' && <Overview repo={r ?? null} jobs={jobs.data ?? []} hw={hw} onRepoChanged={repo.refresh} go={go} />}
            {tab === 'runs' && <Runs jobs={jobs.data ?? []} refresh={jobs.refresh} />}
            {tab === 'experiments' && <Experiments />}
            {tab === 'sweeps' && <Sweeps refresh={jobs.refresh} go={go} />}
            {tab === 'components' && <Components refresh={jobs.refresh} go={go} openInBuilder={openInBuilder} />}
            {tab === 'builder' && <Builder initialComponent={builderComponent} onConsumed={() => setBuilderComponent(null)} refreshJobs={jobs.refresh} go={go} />}
            {tab === 'datasets' && <Datasets datasets={datasets.data ?? []} refresh={datasets.refresh} />}
            {tab === 'machines' && <Machines hw={hw} />}
            {tab === 'code' && <Code />}
          </main>
        </div>
      </div>
    </ToastProvider>
  );
}
