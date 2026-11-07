import { useState, useEffect } from 'react';
import { api, IntegrationItem, IntegrationsResponse } from '../api';
import { Icon } from '../icons';

interface IntegrationsPageProps {
  go: (tab: string) => void;
}

export function Integrations({ go }: IntegrationsPageProps) {
  const [data, setData] = useState<IntegrationsResponse | null>(null);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const intRes = await api.integrations();
      if (intRes) setData(intRes);
    } catch (e) {
      console.error('Failed to load integrations:', e);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleToggle = async (item: IntegrationItem) => {
    setActionInProgress(item.name);
    try {
      if (item.active) {
        await api.toggleIntegration(item.name);
      } else {
        await api.addIntegration(item.name, item.description);
      }
      await loadData();
    } finally {
      setActionInProgress(null);
    }
  };

  const copyCommand = (cmd: string, id: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const integrations = data?.integrations ?? [];

  return (
    <div className="flow-page" style={{ padding: '24px 32px', maxWidth: 1240, margin: '0 auto' }}>
      {/* Header */}
      <div className="row sp" style={{ marginBottom: 24, flexWrap: 'wrap', gap: 16 }}>
        <div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--accent)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>
            MODULAR EXTENSIBILITY · CAPABILITY PACKS
          </div>
          <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span>Integrations &amp; Protocol Spaces</span>
            <span className="tag" style={{ background: 'rgba(56, 189, 248, 0.1)', color: 'var(--accent)', borderColor: 'rgba(56, 189, 248, 0.25)', fontSize: 11 }}>
              sentinel integrate
            </span>
          </h2>
          <p style={{ margin: '6px 0 0', color: 'var(--text-muted)', fontSize: 13 }}>
            Modular telemetry, xAI, gradient inspection, and optimization packs installed on-demand from sentinel-hub or custom org spaces.
          </p>
        </div>

        <div className="row" style={{ gap: 10, alignItems: 'center' }}>
          <button
            className="btn outline small mono"
            onClick={() => copyCommand('sentinel integrate list', 'list-btn')}
            style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <Icon name="terminal" size={13} />
            <span>{copiedCmd === 'list-btn' ? 'Copied!' : 'sentinel integrate list'}</span>
          </button>
        </div>
      </div>

      {/* Protocol Space Overview Banner */}
      <div
        style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '16px 20px',
          marginBottom: 24,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: 16,
        }}
      >
        <div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            Active Space Registry
          </div>
          <div className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)', wordBreak: 'break-all' }}>
            {data?.space || 'https://github.com/rimraf-adi/sentinel-hub'}
          </div>
        </div>

        <div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            Organization / Protocol
          </div>
          <div className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
            {data?.org || 'community'} · <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>sentinel.protocol.yaml</span>
          </div>
        </div>

        <div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
            Active Capabilities
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 2 }}>
            <span className="dot ok" />
            <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>
              {integrations.filter((i) => i.active).length} of {integrations.length} Packs Integrated
            </span>
          </div>
        </div>
      </div>

      {/* Integrations Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 16 }}>
        {integrations.map((item) => {
          const inProgress = actionInProgress === item.name;
          return (
            <div
              key={item.name}
              style={{
                background: 'var(--surface)',
                border: `1px solid ${item.active ? 'rgba(56, 189, 248, 0.4)' : 'var(--border)'}`,
                borderRadius: 8,
                padding: 18,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                transition: 'border-color 0.15s ease',
              }}
            >
              <div>
                <div className="row sp" style={{ marginBottom: 10, alignItems: 'center' }}>
                  <div className="row" style={{ gap: 8, alignItems: 'center' }}>
                    <span className={`dot ${item.active ? 'ok' : 'idle'}`} />
                    <span className="mono" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>
                      {item.name}
                    </span>
                    <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      v{item.version}
                    </span>
                  </div>
                  <span
                    className="tag mono"
                    style={{
                      fontSize: 10,
                      background: item.active ? 'rgba(16, 185, 129, 0.12)' : 'var(--surface-sunken)',
                      color: item.active ? 'var(--ok)' : 'var(--text-muted)',
                      borderColor: item.active ? 'rgba(16, 185, 129, 0.3)' : 'var(--border)',
                    }}
                  >
                    {item.active ? 'INTEGRATED' : 'AVAILABLE'}
                  </span>
                </div>

                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 6 }}>
                  {item.title || item.name}
                </div>

                <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5, margin: '0 0 16px' }}>
                  {item.description}
                </p>
              </div>

              <div>
                <div className="row sp" style={{ alignItems: 'center', paddingTop: 12, borderTop: '1px solid var(--border)' }}>
                  {item.active && item.ui_tab ? (
                    <button
                      className="btn outline small mono"
                      onClick={() => go(item.ui_tab!)}
                      style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}
                    >
                      <span>Open Tab</span>
                      <Icon name="arrow-right" size={12} />
                    </button>
                  ) : (
                    <button
                      className="btn outline small mono"
                      onClick={() => copyCommand(`sentinel integrate add ${item.name}`, item.name)}
                      style={{ fontSize: 11, color: 'var(--text-muted)' }}
                    >
                      {copiedCmd === item.name ? 'Copied CLI!' : `integrate add ${item.name}`}
                    </button>
                  )}

                  <button
                    className={`btn small mono ${item.active ? 'outline' : 'primary'}`}
                    disabled={inProgress}
                    onClick={() => handleToggle(item)}
                    style={{ fontSize: 11, minWidth: 90 }}
                  >
                    {inProgress ? 'Updating...' : item.active ? 'Deactivate' : 'Integrate'}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Org Protocol & Spaces Section */}
      <div
        style={{
          marginTop: 36,
          background: 'var(--surface-sunken)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: 20,
        }}
      >
        <h3 style={{ margin: '0 0 8px', fontSize: 15, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Icon name="layers" size={16} />
          <span>Custom Organization Spaces &amp; Policies</span>
        </h3>
        <p style={{ margin: '0 0 16px', color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.5 }}>
          Create a private GitHub/GitLab repository to publish internal model templates, evaluation gates, and governance policies.
          Your team can switch spaces using the CLI without any centralized cloud infrastructure:
        </p>

        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '12px 16px',
            fontFamily: 'monospace',
            fontSize: 12,
            color: 'var(--accent)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>sentinel config set space https://github.com/your-org/sentinel-space</span>
          <button
            className="btn ghost small"
            onClick={() => copyCommand('sentinel config set space https://github.com/your-org/sentinel-space', 'cfg-space')}
            style={{ fontSize: 11, color: 'var(--text-muted)' }}
          >
            {copiedCmd === 'cfg-space' ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>
    </div>
  );
}
