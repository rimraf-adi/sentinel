import type { Hardware } from '../api';
import { fmtGB } from '../ui';

export function Machines({ hw }: { hw: Hardware | null }) {
  return (
    <>
      <div className="page-header"><div><div className="eyebrow">INFRASTRUCTURE</div><h1 className="page-title">Machines</h1>
        <p className="page-sub">Live telemetry from the local worker. Remote SSH machines are configured in machines.yaml.</p></div></div>
      <div className="card">
        <div className="row sp"><b>local</b><span className="pill"><span className={'dot ' + (hw ? 'ok' : 'err')} />{hw ? 'online' : 'waiting for telemetry'}</span></div>
        {hw && (
          <div className="grid stats gap-t">
            <div><div className="stat-title">CPU</div><div className="stat-value">{hw.cpu_util_percent.toFixed(1)}%</div></div>
            <div><div className="stat-title">Memory</div><div className="stat-value">{fmtGB(hw.ram_used_bytes)} <small>/ {fmtGB(hw.ram_total_bytes)} GB</small></div></div>
            {hw.gpus.map((g) => (
              <div key={g.index}><div className="stat-title">GPU {g.index} · {g.name}</div>
                <div className="stat-value">{g.gpu_util_percent.toFixed(0)}% <small>{fmtGB(g.vram_used_bytes)}/{fmtGB(g.vram_total_bytes)} GB · {g.temperature_c.toFixed(0)}°C</small></div></div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
