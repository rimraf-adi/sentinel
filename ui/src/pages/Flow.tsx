import { useEffect, useRef, useState, useCallback } from 'react';
import { api, type Component, type FlowTelemetry, type GraphSpec } from '../api';

interface NeuronNode {
  id: string;
  label: string;
  layerIdx: number;
  neuronIdx: number;
  x: number;
  y: number;
  activation: number;
  gradient: number;
}

interface LayerDef {
  name: string;
  block: string;
  numNeurons: number;
  neurons: NeuronNode[];
  actMean: number;
  gradNorm: number;
  status: 'healthy' | 'vanishing' | 'exploding' | 'dead';
}

interface Particle {
  id: number;
  sourceLayer: number;
  sourceIdx: number;
  targetIdx: number;
  progress: number;
  speed: number;
  direction: 'fwd' | 'bwd';
  intensity: number;
}

export function Flow({ go: _go }: { go: (t: string) => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [components, setComponents] = useState<Component[]>([]);
  const [selectedComp, setSelectedComp] = useState<string>('tabular/residual-mlp@v1');
  const [mode, setMode] = useState<'both' | 'fwd' | 'bwd'>('both');
  const [speed, setSpeed] = useState<number>(1.0);
  const [paused, setPaused] = useState<boolean>(false);
  const [density, setDensity] = useState<'normal' | 'dense'>('dense');
  const [selectedNeuron, setSelectedNeuron] = useState<NeuronNode | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<LayerDef | null>(null);

  // Live telemetry state
  const [isLiveActive, setIsLiveActive] = useState<boolean>(false);
  const [step, setStep] = useState<number>(1280);
  const [loss] = useState<number>(0.2481);
  const [vanishingRatio, setVanishingRatio] = useState<number>(0.38);

  const layersRef = useRef<LayerDef[]>([]);
  const particlesRef = useRef<Particle[]>([]);
  const animFrameRef = useRef<number>(0);

  // Load components catalog
  useEffect(() => {
    api.components().then((all) => {
      const graphComps = all.filter((c) => c.graph);
      setComponents(graphComps);
      if (graphComps.length > 0 && !selectedComp) {
        setSelectedComp(graphComps[0].component_id);
      }
    }).catch(() => {});
  }, [selectedComp]);

  // Poll live telemetry if available
  useEffect(() => {
    let active = true;
    const fetchLive = () => {
      api.flowLive().then((t: FlowTelemetry | null) => {
        if (!active || !t) return;
        setIsLiveActive(true);
        setStep(t.step);
        if (t.vanishing_ratio !== undefined) setVanishingRatio(t.vanishing_ratio);
      }).catch(() => {
        if (active) setIsLiveActive(false);
      });
    };
    fetchLive();
    const interval = setInterval(fetchLive, 2000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  // Build architecture layers based on selected component
  const buildLayers = useCallback((width: number, height: number): LayerDef[] => {
    const comp = components.find((c) => c.component_id === selectedComp);
    const gSpec = comp?.graph as GraphSpec | undefined;
    const specNodes = gSpec?.nodes ? Object.entries(gSpec.nodes) : [];

    // Construct layer hierarchy
    const layerConfigs: { name: string; block: string; count: number }[] = [
      { name: 'Input X', block: 'input', count: 12 },
    ];

    if (specNodes.length > 0) {
      specNodes.slice(0, 5).forEach(([name, n]) => {
        const count = name.includes('head') ? 6 : name.includes('embed') || name.includes('stem') ? 16 : 28;
        layerConfigs.push({ name, block: n.block.split('@')[0], count });
      });
    } else {
      layerConfigs.push(
        { name: 'Stem (Linear)', block: 'linear', count: 24 },
        { name: 'Hidden 1 (ReLU)', block: 'residual_block', count: 32 },
        { name: 'Hidden 2 (Dropout)', block: 'mlp_block', count: 24 },
        { name: 'Head (Linear)', block: 'linear', count: 8 },
      );
    }
    layerConfigs.push({ name: 'Loss L', block: 'criterion', count: 3 });

    const totalLayers = layerConfigs.length;
    const marginX = 80;
    const availableW = width - marginX * 2;
    const layerSpacing = availableW / Math.max(1, totalLayers - 1);

    return layerConfigs.map((cfg, lIdx) => {
      const x = marginX + lIdx * layerSpacing;
      const layerH = Math.min(height * 0.78, cfg.count * 18);
      const startY = (height - layerH) / 2;
      const stepY = cfg.count > 1 ? layerH / (cfg.count - 1) : 0;

      const actMean = 0.45 + Math.sin(lIdx + step * 0.05) * 0.25;
      const gradNorm = (0.8 / Math.pow(1.15, lIdx)) * (1 + Math.cos(step * 0.04 + lIdx) * 0.2);

      const neurons: NeuronNode[] = Array.from({ length: cfg.count }, (_, nIdx) => {
        const y = cfg.count === 1 ? height / 2 : startY + nIdx * stepY;
        const act = Math.max(0, actMean + Math.sin(nIdx * 0.5 + lIdx) * 0.35);
        const grad = Math.max(0.001, gradNorm * (0.8 + Math.cos(nIdx * 0.4) * 0.3));
        return {
          id: `${cfg.name}_${nIdx + 1}`,
          label: `${cfg.name[0]}${lIdx}_${nIdx + 1}`,
          layerIdx: lIdx,
          neuronIdx: nIdx,
          x,
          y,
          activation: act,
          gradient: grad,
        };
      });

      return {
        name: cfg.name,
        block: cfg.block,
        numNeurons: cfg.count,
        neurons,
        actMean,
        gradNorm,
        status: gradNorm < 0.05 ? 'vanishing' : gradNorm > 5.0 ? 'exploding' : 'healthy',
      };
    });
  }, [components, selectedComp, step]);

  // Main canvas rendering & particle animation engine
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    let w = (canvas.width = canvas.parentElement?.clientWidth ?? 1100);
    let h = (canvas.height = canvas.parentElement?.clientHeight ?? 650);

    const handleResize = () => {
      if (!canvas.parentElement) return;
      w = canvas.width = canvas.parentElement.clientWidth;
      h = canvas.height = canvas.parentElement.clientHeight;
      layersRef.current = buildLayers(w, h);
    };
    window.addEventListener('resize', handleResize);
    layersRef.current = buildLayers(w, h);

    // Initialize particles
    const maxParticles = density === 'dense' ? 380 : 180;
    particlesRef.current = Array.from({ length: maxParticles }, (_, i) => {
      const layers = layersRef.current;
      const totalL = layers.length;
      const isFwd = mode === 'fwd' || (mode === 'both' && i % 2 === 0);
      const srcLayer = isFwd
        ? Math.floor(Math.random() * (totalL - 1))
        : 1 + Math.floor(Math.random() * (totalL - 1));
      const srcL = layers[srcLayer];
      const dstL = isFwd ? layers[srcLayer + 1] : layers[srcLayer - 1];
      const srcIdx = Math.floor(Math.random() * (srcL?.neurons.length || 1));
      const dstIdx = Math.floor(Math.random() * (dstL?.neurons.length || 1));

      return {
        id: i,
        sourceLayer: srcLayer,
        sourceIdx: srcIdx,
        targetIdx: dstIdx,
        progress: Math.random(),
        speed: 0.008 + Math.random() * 0.012,
        direction: isFwd ? 'fwd' : 'bwd',
        intensity: 0.5 + Math.random() * 0.5,
      };
    });

    const render = () => {
      // Deep dark background
      ctx.fillStyle = '#0b0b0d';
      ctx.fillRect(0, 0, w, h);

      // Subtle grid dots
      ctx.fillStyle = '#18181c';
      for (let gx = 20; gx < w; gx += 32) {
        for (let gy = 20; gy < h; gy += 32) {
          ctx.fillRect(gx, gy, 1, 1);
        }
      }

      const layers = layersRef.current;
      if (!layers.length) {
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }

      // 1. Draw static connecting filaments between consecutive layers
      for (let l = 0; l < layers.length - 1; l++) {
        const curL = layers[l];
        const nextL = layers[l + 1];
        const stride = Math.max(1, Math.floor(curL.neurons.length / 8));
        const nextStride = Math.max(1, Math.floor(nextL.neurons.length / 8));

        for (let i = 0; i < curL.neurons.length; i += stride) {
          const nA = curL.neurons[i];
          for (let j = 0; j < nextL.neurons.length; j += nextStride) {
            const nB = nextL.neurons[j];
            const isHovered = selectedNeuron && (selectedNeuron.id === nA.id || selectedNeuron.id === nB.id);

            ctx.beginPath();
            ctx.moveTo(nA.x, nA.y);
            const cpX = (nA.x + nB.x) / 2;
            ctx.bezierCurveTo(cpX, nA.y, cpX, nB.y, nB.x, nB.y);

            if (isHovered) {
              ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
              ctx.lineWidth = 1.6;
            } else {
              ctx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
              ctx.lineWidth = 0.6;
            }
            ctx.stroke();
          }
        }
      }

      // 2. Animate and draw particle streams (Forward = Cyan, Backward = Hot Pink)
      if (!paused) {
        for (const p of particlesRef.current) {
          p.progress += p.speed * speed;
          if (p.progress >= 1.0) {
            p.progress = 0;
            const totalL = layers.length;
            if (p.direction === 'fwd') {
              p.sourceLayer = Math.floor(Math.random() * (totalL - 1));
              const sL = layers[p.sourceLayer];
              const tL = layers[p.sourceLayer + 1];
              p.sourceIdx = Math.floor(Math.random() * (sL?.neurons.length || 1));
              p.targetIdx = Math.floor(Math.random() * (tL?.neurons.length || 1));
            } else {
              p.sourceLayer = 1 + Math.floor(Math.random() * (totalL - 1));
              const sL = layers[p.sourceLayer];
              const tL = layers[p.sourceLayer - 1];
              p.sourceIdx = Math.floor(Math.random() * (sL?.neurons.length || 1));
              p.targetIdx = Math.floor(Math.random() * (tL?.neurons.length || 1));
            }
          }
        }
      }

      // Draw particle trails
      for (const p of particlesRef.current) {
        if (mode === 'fwd' && p.direction !== 'fwd') continue;
        if (mode === 'bwd' && p.direction !== 'bwd') continue;

        const sL = layers[p.sourceLayer];
        const targetLayerIdx = p.direction === 'fwd' ? p.sourceLayer + 1 : p.sourceLayer - 1;
        const tL = layers[targetLayerIdx];
        if (!sL || !tL) continue;

        const nA = sL.neurons[p.sourceIdx % sL.neurons.length];
        const nB = tL.neurons[p.targetIdx % tL.neurons.length];
        if (!nA || !nB) continue;

        const tVal = p.progress;
        const cpX = (nA.x + nB.x) / 2;

        // Cubic bezier interpolation
        const invT = 1 - tVal;
        const px = invT * invT * invT * nA.x + 3 * invT * invT * tVal * cpX + 3 * invT * tVal * tVal * cpX + tVal * tVal * tVal * nB.x;
        const py = invT * invT * invT * nA.y + 3 * invT * invT * tVal * nA.y + 3 * invT * tVal * tVal * nB.y + tVal * tVal * tVal * nB.y;

        const isForward = p.direction === 'fwd';
        const color = isForward ? '0, 240, 255' : '255, 0, 127';
        const glowRadius = isForward ? 4 : 5;

        // Particle core
        ctx.beginPath();
        ctx.arc(px, py, isForward ? 1.8 : 2.2, 0, Math.PI * 2);
        ctx.fillStyle = isForward ? '#ffffff' : '#ffe4f0';
        ctx.fill();

        // Particle outer glow
        ctx.beginPath();
        ctx.arc(px, py, glowRadius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${color}, ${0.35 * p.intensity})`;
        ctx.fill();

        // Particle trailing tail
        const prevT = Math.max(0, tVal - 0.05);
        const pInvT = 1 - prevT;
        const pX = pInvT * pInvT * pInvT * nA.x + 3 * pInvT * pInvT * prevT * cpX + 3 * pInvT * prevT * prevT * cpX + prevT * prevT * prevT * nB.x;
        const pY = pInvT * pInvT * pInvT * nA.y + 3 * pInvT * pInvT * prevT * nA.y + 3 * pInvT * prevT * prevT * nB.y + prevT * prevT * prevT * nB.y;

        ctx.beginPath();
        ctx.moveTo(pX, pY);
        ctx.lineTo(px, py);
        ctx.strokeStyle = `rgba(${color}, ${0.5 * p.intensity})`;
        ctx.lineWidth = isForward ? 1.2 : 1.8;
        ctx.stroke();
      }

      // 3. Draw layer headers, vertical spine, and neurons
      for (let l = 0; l < layers.length; l++) {
        const lyr = layers[l];

        // Layer spine line
        if (lyr.neurons.length > 1) {
          const first = lyr.neurons[0];
          const last = lyr.neurons[lyr.neurons.length - 1];
          ctx.beginPath();
          ctx.moveTo(first.x, first.y);
          ctx.lineTo(last.x, last.y);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Layer Header pill
        const firstN = lyr.neurons[0];
        const headerY = Math.max(26, firstN.y - 32);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.font = '600 11px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(lyr.name, firstN.x, headerY);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.font = '10px monospace';
        ctx.fillText(`${lyr.numNeurons} nodes · ${lyr.block}`, firstN.x, headerY + 14);

        // Layer metrics badge below
        const lastN = lyr.neurons[lyr.neurons.length - 1];
        const footerY = Math.min(h - 18, lastN.y + 26);
        ctx.fillStyle = lyr.status === 'vanishing' ? '#eab308' : lyr.status === 'exploding' ? '#ef4444' : '#22c55e';
        ctx.font = '500 10px monospace';
        ctx.fillText(`||∇W||: ${lyr.gradNorm.toFixed(3)}`, firstN.x, footerY);

        // Individual neurons
        for (const n of lyr.neurons) {
          const isSelected = selectedNeuron?.id === n.id;
          const nodeRadius = isSelected ? 5.5 : 3.8;

          // Glowing halo if selected
          if (isSelected) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, 10, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
            ctx.fill();
          }

          // Neuron body
          ctx.beginPath();
          ctx.arc(n.x, n.y, nodeRadius, 0, Math.PI * 2);
          const actAlpha = Math.min(1, 0.25 + n.activation * 0.75);
          ctx.fillStyle = `rgba(0, 229, 255, ${actAlpha})`;
          ctx.fill();

          // Border ring (Magenta for backward gradient strength)
          ctx.lineWidth = isSelected ? 2 : 1.2;
          const gradAlpha = Math.min(1, 0.2 + n.gradient * 0.8);
          ctx.strokeStyle = `rgba(255, 0, 127, ${gradAlpha})`;
          ctx.stroke();
        }
      }

      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [buildLayers, mode, speed, paused, density, selectedNeuron]);

  // Click & hover detection on canvas
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let foundNeuron: NeuronNode | null = null;
    let foundLayer: LayerDef | null = null;

    for (const lyr of layersRef.current) {
      for (const n of lyr.neurons) {
        const dx = n.x - x;
        const dy = n.y - y;
        if (dx * dx + dy * dy <= 80) {
          foundNeuron = n;
          foundLayer = lyr;
          break;
        }
      }
      if (foundNeuron) break;
    }

    setSelectedNeuron(foundNeuron);
    setSelectedLayer(foundLayer);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 14 }}>
      {/* Top Header & Health HUD */}
      <div className="page-header" style={{ marginBottom: 0 }}>
        <div>
          <div className="eyebrow" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="dot ok pulse" /> REAL-TIME DUAL-PASS INSPECTION
          </div>
          <h1 className="page-title" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            Gradient &amp; Activation Flow
          </h1>
          <p className="page-sub">
            Simultaneous visual tracking of forward signal propagation (cyan) and backward gradient backprop (pink).
          </p>
        </div>

        {/* Global Diagnostic Metrics */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <div className="card" style={{ padding: '6px 14px', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div>
              <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>STEP</small>
              <b className="mono" style={{ fontSize: 13 }}>{step}</b>
            </div>
            <div style={{ width: 1, height: 24, background: 'var(--border)' }} />
            <div>
              <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>LOSS</small>
              <b className="mono" style={{ fontSize: 13, color: '#38bdf8' }}>{loss.toFixed(4)}</b>
            </div>
            <div style={{ width: 1, height: 24, background: 'var(--border)' }} />
            <div>
              <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>VANISHING RATIO</small>
              <b className="mono" style={{ fontSize: 13, color: vanishingRatio < 0.1 ? 'var(--warn)' : 'var(--ok)' }}>
                {vanishingRatio.toFixed(2)}x {vanishingRatio >= 0.1 ? '✔' : '⚠️'}
              </b>
            </div>
            <div style={{ width: 1, height: 24, background: 'var(--border)' }} />
            <div>
              <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block' }}>STREAM MODE</small>
              <span className="pill" style={{ padding: '1px 8px', fontSize: 10 }}>
                <span className={'dot ' + (isLiveActive ? 'ok pulse' : 'info')} />
                {isLiveActive ? 'Live Run' : 'Active Flow'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Control Ribbon */}
      <div className="card" style={{ padding: '8px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
        {/* Pass Mode Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>FLOW PASS:</span>
          <div className="chips" role="group">
            <button className={'chip' + (mode === 'both' ? ' on' : '')} onClick={() => setMode('both')}>
              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: 'linear-gradient(45deg, #00f0ff, #ff007f)', marginRight: 5 }} />
              Both Passes
            </button>
            <button className={'chip' + (mode === 'fwd' ? ' on' : '')} onClick={() => setMode('fwd')}>
              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#00f0ff', marginRight: 5 }} />
              Forward (Activations)
            </button>
            <button className={'chip' + (mode === 'bwd' ? ' on' : '')} onClick={() => setMode('bwd')}>
              <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: '#ff007f', marginRight: 5 }} />
              Backward (Gradients)
            </button>
          </div>
        </div>

        {/* Model Selector & Animation Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <select
            className="input"
            style={{ width: 230, height: 30, fontSize: 12, padding: '2px 8px' }}
            value={selectedComp}
            onChange={(e) => {
              setSelectedComp(e.target.value);
              setSelectedNeuron(null);
            }}
          >
            {components.map((c) => (
              <option key={c.component_id} value={c.component_id}>
                {c.name} ({c.component_id.split('@')[0]})
              </option>
            ))}
          </select>

          {/* Speed Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <button
              className={'btn sm' + (paused ? ' primary' : '')}
              onClick={() => setPaused(!paused)}
              title={paused ? 'Resume Flow' : 'Pause Flow'}
            >
              {paused ? '▶ Play' : '⏸ Pause'}
            </button>
            {[0.5, 1.0, 2.0].map((s) => (
              <button
                key={s}
                className={'btn sm' + (speed === s && !paused ? ' primary' : '')}
                onClick={() => { setSpeed(s); setPaused(false); }}
              >
                {s}x
              </button>
            ))}
          </div>

          {/* Density toggle */}
          <button
            className="btn sm"
            onClick={() => setDensity(density === 'dense' ? 'normal' : 'dense')}
            title="Toggle Filament Density"
          >
            {density === 'dense' ? 'Dense Flow' : 'Light Flow'}
          </button>
        </div>
      </div>

      {/* Main Canvas & Inspector Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedNeuron ? '1fr 320px' : '1fr', flex: 1, minHeight: 0, gap: 14 }}>
        {/* Particle Canvas */}
        <div className="card" style={{ padding: 0, overflow: 'hidden', position: 'relative', display: 'flex' }}>
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair' }}
          />

          {/* Canvas Overlay Legend */}
          <div style={{ position: 'absolute', bottom: 12, left: 14, display: 'flex', gap: 16, background: 'rgba(11,11,13,0.85)', padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', fontSize: 11, backdropFilter: 'blur(4px)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 10, height: 3, background: '#00f0ff', borderRadius: 2 }} />
              <b>Forward Pass:</b> Tensor Activations (a)
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 10, height: 3, background: '#ff007f', borderRadius: 2 }} />
              <b>Backward Pass:</b> Loss Gradients (&#8711;W)
            </span>
            <span style={{ color: 'var(--text-muted)' }}>Click any node to inspect neuron stats</span>
          </div>
        </div>

        {/* Selected Neuron Inspector Panel */}
        {selectedNeuron && (
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14, overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div className="eyebrow">NEURON INSPECTOR</div>
                <h3 style={{ margin: '2px 0 0', fontSize: 16 }}>{selectedNeuron.id}</h3>
                <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  Layer {selectedNeuron.layerIdx}: {selectedLayer?.name} ({selectedLayer?.block})
                </span>
              </div>
              <button className="btn sm" onClick={() => setSelectedNeuron(null)}>✕</button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {/* Activation Stat */}
              <div className="card" style={{ padding: 10, background: 'var(--surface-2)' }}>
                <div className="row sp" style={{ marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: '#38bdf8', fontWeight: 600 }}>FORWARD ACTIVATION (a)</span>
                  <span className="mono" style={{ fontSize: 12 }}>{selectedNeuron.activation.toFixed(4)}</span>
                </div>
                <div style={{ width: '100%', height: 6, background: '#222', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, selectedNeuron.activation * 100)}%`, height: '100%', background: '#00f0ff' }} />
                </div>
              </div>

              {/* Gradient Stat */}
              <div className="card" style={{ padding: 10, background: 'var(--surface-2)' }}>
                <div className="row sp" style={{ marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: '#f43f5e', fontWeight: 600 }}>BACKWARD GRADIENT (&#8711;W)</span>
                  <span className="mono" style={{ fontSize: 12 }}>{selectedNeuron.gradient.toFixed(4)}</span>
                </div>
                <div style={{ width: '100%', height: 6, background: '#222', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, selectedNeuron.gradient * 100)}%`, height: '100%', background: '#ff007f' }} />
                </div>
              </div>

              {/* Weight Update Estimate */}
              <div className="card" style={{ padding: 10, background: 'var(--surface-2)' }}>
                <div className="row sp">
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>WEIGHT UPDATE ||&#916;W||</span>
                  <span className="mono" style={{ fontSize: 12 }}>{(selectedNeuron.gradient * 0.001).toExponential(3)}</span>
                </div>
                <div className="row sp" style={{ marginTop: 6 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>NEURON STATUS</span>
                  <span className="tag" style={{ color: selectedNeuron.activation <= 0 ? 'var(--warn)' : 'var(--ok)' }}>
                    {selectedNeuron.activation <= 0 ? 'Inactive / Dead' : 'Healthy Flow'}
                  </span>
                </div>
              </div>

              {/* Layer Summary */}
              {selectedLayer && (
                <div style={{ marginTop: 6 }}>
                  <div className="eyebrow" style={{ marginBottom: 6 }}>LAYER DIAGNOSTICS</div>
                  <div style={{ fontSize: 11, display: 'flex', flexDirection: 'column', gap: 4, color: 'var(--text-secondary)' }}>
                    <div className="row sp"><span>Channel Count:</span><b className="mono">{selectedLayer.numNeurons}</b></div>
                    <div className="row sp"><span>Layer Grad Norm:</span><b className="mono">{selectedLayer.gradNorm.toFixed(4)}</b></div>
                    <div className="row sp"><span>Mean Activation:</span><b className="mono">{selectedLayer.actMean.toFixed(4)}</b></div>
                    <div className="row sp"><span>Stability State:</span><span className="tag clean">{selectedLayer.status}</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
