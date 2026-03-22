import { useState, useEffect, useRef, useCallback } from 'react';

// ─── CONSTANTS ───────────────────────────────────────────────────────────────

const C = {
  bg:       '#080b0f',
  panel:    '#0d1117',
  border:   '#1c2333',
  accent:   '#39d353',
  amber:    '#f0a500',
  spike:    '#ff4757',
  blue:     '#58a6ff',
  muted:    '#484f58',
  text:     '#cdd9e5',
  dim:      '#768390',
  grid:     '#161b22',
};

// ─── SHARED COMPONENTS ───────────────────────────────────────────────────────

function Tab({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      background: 'none',
      border: 'none',
      borderBottom: active ? `2px solid ${C.accent}` : '2px solid transparent',
      color: active ? C.accent : C.dim,
      fontFamily: 'monospace',
      fontSize: 12,
      letterSpacing: 2,
      padding: '12px 20px',
      cursor: 'pointer',
      textTransform: 'uppercase',
      transition: 'color 0.2s',
    }}>{label}</button>
  );
}

function Slider({ label, value, min, max, step, unit, color, onChange }) {
  const pct = ((value - min) / (max - min)) * 100;
  const fmt = step < 0.1 ? value.toFixed(2) : step < 1 ? value.toFixed(1) : value.toFixed(0);
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ color: C.muted, fontSize: 10, fontFamily: 'monospace', letterSpacing: 1 }}>{label}</span>
        <span style={{ color: color || C.accent, fontSize: 11, fontFamily: 'monospace', fontWeight: 700 }}>{fmt}{unit}</span>
      </div>
      <div style={{ position: 'relative', height: 3, background: C.border, borderRadius: 2 }}>
        <div style={{ position: 'absolute', left: 0, width: `${pct}%`, height: '100%', background: color || C.accent, borderRadius: 2 }} />
        <input type='range' min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          style={{ position: 'absolute', top: -8, left: 0, width: '100%', opacity: 0, cursor: 'pointer', height: 18, margin: 0 }} />
      </div>
    </div>
  );
}

// ─── DEMO 1: LIF NEURON ───────────────────────────────────────────────────────

function LIFDemo() {
  const canvasRef = useRef(null);
  const stateRef = useRef({ V: -70, running: true, spikeCount: 0 });
  const histRef = useRef([]);
  const animRef = useRef(null);
  const [params, setParams] = useState({ tau_m: 20, V_rest: -70, V_threshold: -55, V_reset: -80, R: 10, I: 2.5, noise: 0.5 });
  const [stats, setStats] = useState({ V: -70, hz: 0, spikes: 0 });
  const [running, setRunning] = useState(true);
  const spikeTimesRef = useRef([]);
  const setP = (k, v) => setParams(p => ({ ...p, [k]: v }));

  useEffect(() => {
    const dt = 0.1;
    const WINDOW = 300;
    const STEPS_PER_FRAME = 3;
    const DISPLAY_STEPS = Math.floor(WINDOW / dt);
    let V = params.V_rest;
    let lastFrame = null;

    const frame = (ts) => {
      if (!lastFrame) lastFrame = ts;
      lastFrame = ts;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      const { tau_m, V_rest, V_threshold, V_reset, R, I, noise } = params;

      if (running) {
        for (let s = 0; s < STEPS_PER_FRAME; s++) {
          const n = noise > 0 ? (Math.random() * 2 - 1) * noise : 0;
          const dV = (dt / tau_m) * (-(V - V_rest) + R * (I + n));
          V += dV;
          if (V >= V_threshold) {
            V = V_reset;
            spikeTimesRef.current.push(histRef.current.length * dt);
            stateRef.current.spikeCount++;
          }
          histRef.current.push(V);
          if (histRef.current.length > DISPLAY_STEPS * 2) histRef.current.shift();
        }
      }

      // clear
      ctx.fillStyle = C.bg;
      ctx.fillRect(0, 0, W, H);

      // grid
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      for (let v = -90; v <= -40; v += 10) {
        const y = H - ((v - (V_rest - 15)) / 50) * H;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.font = '9px monospace';
        ctx.fillText(`${v}`, 4, y - 2);
      }

      // threshold
      const thY = H - ((V_threshold - (V_rest - 15)) / 50) * H;
      ctx.strokeStyle = C.amber; ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
      ctx.beginPath(); ctx.moveTo(0, thY); ctx.lineTo(W, thY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.amber; ctx.font = '9px monospace';
      ctx.fillText('threshold', W - 65, thY - 3);

      // rest
      const restY = H - ((V_rest - (V_rest - 15)) / 50) * H;
      ctx.strokeStyle = C.blue; ctx.lineWidth = 1; ctx.setLineDash([2, 6]);
      ctx.beginPath(); ctx.moveTo(0, restY); ctx.lineTo(W, restY); ctx.stroke();
      ctx.setLineDash([]);

      // voltage trace
      const hist = histRef.current;
      const start = Math.max(0, hist.length - DISPLAY_STEPS);
      ctx.beginPath(); ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5;
      ctx.shadowColor = C.accent; ctx.shadowBlur = 3;
      let prevV = null;
      for (let i = start; i < hist.length; i++) {
        const v = hist[i];
        const x = ((i - start) / DISPLAY_STEPS) * W;
        const y = H - ((v - (V_rest - 15)) / 50) * H;
        if (i === start || Math.abs(v - prevV) > 25) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        prevV = v;
      }
      ctx.stroke(); ctx.shadowBlur = 0;

      // hz calc
      const now = hist.length * dt;
      const recent = spikeTimesRef.current.filter(t => t > now - 500);
      const hz = recent.length / 0.5;

      stateRef.current.V = V;
      setStats({ V: V.toFixed(1), hz: hz.toFixed(1), spikes: stateRef.current.spikeCount });
      animRef.current = requestAnimationFrame(frame);
    };

    animRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, params]);

  return (
    <div style={{ display: 'flex', gap: 0, height: '100%' }}>
      <div style={{ width: 200, padding: 20, borderRight: `1px solid ${C.border}`, flexShrink: 0, overflowY: 'auto' }}>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 16 }}>MEMBRANE</div>
        <Slider label='TAU_M' value={params.tau_m} min={5} max={50} step={1} unit='ms' onChange={v => setP('tau_m', v)} />
        <Slider label='V_REST' value={params.V_rest} min={-80} max={-60} step={1} unit='mV' color={C.blue} onChange={v => setP('V_rest', v)} />
        <Slider label='V_THRESHOLD' value={params.V_threshold} min={-65} max={-45} step={1} unit='mV' color={C.amber} onChange={v => setP('V_threshold', v)} />
        <Slider label='V_RESET' value={params.V_reset} min={-90} max={-70} step={1} unit='mV' color={C.muted} onChange={v => setP('V_reset', v)} />
        <Slider label='R_MEMBRANE' value={params.R} min={1} max={30} step={1} unit='MΩ' onChange={v => setP('R', v)} />
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, margin: '18px 0 14px' }}>INPUT</div>
        <Slider label='CURRENT' value={params.I} min={0} max={6} step={0.1} unit='nA' color={C.spike} onChange={v => setP('I', v)} />
        <Slider label='NOISE σ' value={params.noise} min={0} max={3} step={0.1} unit='nA' color={C.muted} onChange={v => setP('noise', v)} />
        <div style={{ marginTop: 20, padding: 12, background: C.panel, border: `1px solid ${C.border}`, borderRadius: 4 }}>
          <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 8 }}>EQUATION</div>
          <div style={{ fontSize: 10, color: C.accent, lineHeight: 1.9, fontFamily: 'monospace' }}>
            τ·dV/dt =<br/>-(V - V<sub>rest</sub>)<br/>+ R·I(t)
          </div>
        </div>
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '10px 20px', borderBottom: `1px solid ${C.border}`, display: 'flex', gap: 32, alignItems: 'center' }}>
          {[
            { label: 'MEMBRANE V', val: `${stats.V} mV`, color: C.accent },
            { label: 'FIRING RATE', val: `${stats.hz} Hz`, color: C.spike },
            { label: 'SPIKE COUNT', val: stats.spikes, color: C.amber },
          ].map(({ label, val, color }) => (
            <div key={label}>
              <div style={{ fontSize: 9, color: C.muted, letterSpacing: 1 }}>{label}</div>
              <div style={{ fontSize: 20, color, fontFamily: 'monospace', fontWeight: 700 }}>{val}</div>
            </div>
          ))}
          <div style={{ marginLeft: 'auto' }}>
            <button onClick={() => setRunning(r => !r)} style={{
              background: running ? `${C.spike}22` : `${C.accent}22`,
              border: `1px solid ${running ? C.spike : C.accent}`,
              color: running ? C.spike : C.accent,
              fontFamily: 'monospace', fontSize: 11, letterSpacing: 2,
              padding: '6px 14px', cursor: 'pointer', borderRadius: 2,
            }}>{running ? 'PAUSE' : 'RUN'}</button>
          </div>
        </div>
        <div style={{ flex: 1, padding: 16 }}>
          <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 8 }}>VOLTAGE TRACE (300ms window)</div>
          <canvas ref={canvasRef} width={900} height={260}
            style={{ width: '100%', height: 260, display: 'block', borderRadius: 4, border: `1px solid ${C.border}` }} />
        </div>
      </div>
    </div>
  );
}

// ─── DEMO 2: HOPFIELD NETWORK ─────────────────────────────────────────────────

const DIGIT_PATTERNS = {
  0: [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
  1: [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],
  2: [0,1,1,1,0, 1,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,1,1,1,1],
  3: [1,1,1,1,0, 0,0,0,0,1, 0,0,1,1,0, 0,0,0,0,1, 0,0,0,0,1, 1,1,1,1,0],
  4: [0,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1,1,1,1,1, 0,0,0,1,0, 0,0,0,1,0],
  5: [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,0,1, 0,0,0,0,1, 1,1,1,1,0],
  6: [0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
  7: [1,1,1,1,1, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 0,1,0,0,0],
  8: [0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
  9: [0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,1, 0,0,0,0,1, 0,0,0,0,1, 0,1,1,1,0],
};

function softmax(arr, beta = 3.0) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(beta * (x - max)));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function hopfieldUpdate(state, patterns, beta = 3.0) {
  const sims = patterns.map(p => p.reduce((s, v, i) => s + v * state[i], 0));
  const weights = softmax(sims, beta);
  const result = new Array(state.length).fill(0);
  patterns.forEach((p, pi) => p.forEach((v, i) => { result[i] += weights[pi] * v; }));
  return result;
}

function HopfieldDemo() {
  const allPatterns = Object.entries(DIGIT_PATTERNS).map(([d, p]) => ({
    digit: parseInt(d),
    pattern: p.map(v => v === 1 ? 1 : -1),
  }));

  const [selectedDigit, setSelectedDigit] = useState(0);
  const [noiseLevel, setNoiseLevel] = useState(0.3);
  const [beta, setBeta] = useState(3.0);
  const [phase, setPhase] = useState('original'); // original | corrupted | retrieving | retrieved
  const [displayPattern, setDisplayPattern] = useState(allPatterns[0].pattern);
  const [originalPattern, setOriginalPattern] = useState(allPatterns[0].pattern);
  const [iterCount, setIterCount] = useState(0);
  const [matchDigit, setMatchDigit] = useState(null);

  const corrupt = useCallback(() => {
    const orig = allPatterns.find(p => p.digit === selectedDigit).pattern;
    setOriginalPattern(orig);
    const corrupted = orig.map(v => Math.random() < noiseLevel ? -v : v);
    setDisplayPattern(corrupted);
    setPhase('corrupted');
    setIterCount(0);
    setMatchDigit(null);
  }, [selectedDigit, noiseLevel, allPatterns]);

  const retrieve = useCallback(() => {
    if (phase !== 'corrupted') return;
    setPhase('retrieving');
    const patterns = allPatterns.map(p => p.pattern);
    let state = [...displayPattern];
    let iter = 0;

    const step = () => {
      const newState = hopfieldUpdate(state, patterns, beta);
      const binarized = newState.map(v => v >= 0 ? 1 : -1);
      setDisplayPattern([...binarized]);
      iter++;
      setIterCount(iter);

      const changed = binarized.some((v, i) => v !== state.map(x => x >= 0 ? 1 : -1)[i]);
      if (!changed || iter >= 15) {
        // find closest match
        let bestMatch = -1, bestSim = -Infinity;
        allPatterns.forEach(({ digit, pattern }) => {
          const sim = pattern.reduce((s, v, i) => s + v * binarized[i], 0);
          if (sim > bestSim) { bestSim = sim; bestMatch = digit; }
        });
        setMatchDigit(bestMatch);
        setPhase('retrieved');
        return;
      }
      state = newState;
      setTimeout(step, 200);
    };
    setTimeout(step, 100);
  }, [phase, displayPattern, beta, allPatterns]);

  const reset = () => {
    const orig = allPatterns.find(p => p.digit === selectedDigit).pattern;
    setOriginalPattern(orig);
    setDisplayPattern(orig);
    setPhase('original');
    setMatchDigit(null);
    setIterCount(0);
  };

  const COLS = 5, ROWS = 6;
  const cellSize = 28;

  const phaseColor = { original: C.accent, corrupted: C.amber, retrieving: C.blue, retrieved: C.spike };

  return (
    <div style={{ display: 'flex', gap: 0, height: '100%' }}>
      <div style={{ width: 200, padding: 20, borderRight: `1px solid ${C.border}`, flexShrink: 0 }}>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 14 }}>SELECT DIGIT</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 4, marginBottom: 20 }}>
          {Object.keys(DIGIT_PATTERNS).map(d => (
            <button key={d} onClick={() => { setSelectedDigit(parseInt(d)); }} style={{
              background: selectedDigit === parseInt(d) ? `${C.accent}33` : C.panel,
              border: `1px solid ${selectedDigit === parseInt(d) ? C.accent : C.border}`,
              color: selectedDigit === parseInt(d) ? C.accent : C.dim,
              fontFamily: 'monospace', fontSize: 13, padding: '6px 0',
              cursor: 'pointer', borderRadius: 3,
            }}>{d}</button>
          ))}
        </div>
        <Slider label='NOISE LEVEL' value={noiseLevel} min={0.1} max={0.6} step={0.05} unit='' color={C.amber} onChange={setNoiseLevel} />
        <Slider label='BETA (sharpness)' value={beta} min={0.5} max={8} step={0.5} unit='' color={C.blue} onChange={setBeta} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 20 }}>
          <button onClick={corrupt} style={{
            background: `${C.amber}22`, border: `1px solid ${C.amber}`, color: C.amber,
            fontFamily: 'monospace', fontSize: 10, letterSpacing: 2, padding: '8px',
            cursor: 'pointer', borderRadius: 2,
          }}>CORRUPT</button>
          <button onClick={retrieve} disabled={phase !== 'corrupted'} style={{
            background: phase === 'corrupted' ? `${C.accent}22` : C.panel,
            border: `1px solid ${phase === 'corrupted' ? C.accent : C.muted}`,
            color: phase === 'corrupted' ? C.accent : C.muted,
            fontFamily: 'monospace', fontSize: 10, letterSpacing: 2, padding: '8px',
            cursor: phase === 'corrupted' ? 'pointer' : 'not-allowed', borderRadius: 2,
          }}>RETRIEVE</button>
          <button onClick={reset} style={{
            background: C.panel, border: `1px solid ${C.border}`, color: C.dim,
            fontFamily: 'monospace', fontSize: 10, letterSpacing: 2, padding: '8px',
            cursor: 'pointer', borderRadius: 2,
          }}>RESET</button>
        </div>
        <div style={{ marginTop: 20, padding: 12, background: C.panel, border: `1px solid ${C.border}`, borderRadius: 4 }}>
          <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 6 }}>STATUS</div>
          <div style={{ fontSize: 11, color: phaseColor[phase], fontFamily: 'monospace', textTransform: 'uppercase' }}>{phase}</div>
          {iterCount > 0 && <div style={{ fontSize: 10, color: C.dim, marginTop: 4 }}>iter: {iterCount}</div>}
          {matchDigit !== null && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 9, color: C.muted }}>RETRIEVED</div>
              <div style={{ fontSize: 28, color: C.spike, fontFamily: 'monospace', fontWeight: 700 }}>{matchDigit}</div>
            </div>
          )}
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 32 }}>
        <div style={{ display: 'flex', gap: 48, alignItems: 'flex-start' }}>
          {/* Original */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 12 }}>STORED MEMORY</div>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${COLS}, ${cellSize}px)`, gap: 2 }}>
              {originalPattern.map((v, i) => (
                <div key={i} style={{
                  width: cellSize, height: cellSize, borderRadius: 3,
                  background: v === 1 ? C.accent : C.grid,
                  boxShadow: v === 1 ? `0 0 6px ${C.accent}66` : 'none',
                  transition: 'background 0.1s',
                }} />
              ))}
            </div>
          </div>

          {/* Arrow */}
          <div style={{ display: 'flex', alignItems: 'center', paddingTop: 80, color: C.muted, fontSize: 20 }}>→</div>

          {/* Current state */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 9, color: phaseColor[phase], letterSpacing: 2, marginBottom: 12, textTransform: 'uppercase' }}>
              {phase === 'original' ? 'pattern' : phase === 'corrupted' ? 'corrupted input' : phase === 'retrieving' ? `retrieving... (${iterCount})` : 'retrieved'}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${COLS}, ${cellSize}px)`, gap: 2 }}>
              {displayPattern.map((v, i) => {
                const orig = originalPattern[i];
                const isFlipped = phase !== 'original' && v !== orig;
                return (
                  <div key={i} style={{
                    width: cellSize, height: cellSize, borderRadius: 3,
                    background: v === 1
                      ? (isFlipped ? C.amber : (phase === 'retrieved' ? C.spike : C.accent))
                      : C.grid,
                    boxShadow: v === 1
                      ? `0 0 6px ${isFlipped ? C.amber : (phase === 'retrieved' ? C.spike : C.accent)}66`
                      : 'none',
                    transition: 'background 0.15s',
                  }} />
                );
              })}
            </div>
          </div>
        </div>

        <div style={{ fontSize: 10, color: C.muted, fontFamily: 'monospace', maxWidth: 440, textAlign: 'center', lineHeight: 1.7 }}>
          {phase === 'original' && 'Select a digit, click CORRUPT to add noise, then RETRIEVE to watch the Hopfield network complete the memory.'}
          {phase === 'corrupted' && `${Math.round(noiseLevel * 100)}% of pixels flipped. Amber pixels are corrupted. Click RETRIEVE to run the softmax update rule.`}
          {phase === 'retrieving' && 'Running continuous Hopfield update: s = Xᵀ · softmax(β · Xs)'}
          {phase === 'retrieved' && matchDigit !== null && `Network retrieved digit ${matchDigit}. ${matchDigit === selectedDigit ? 'Correct retrieval.' : 'Spurious state — try lower noise or higher beta.'}`}
        </div>
      </div>
    </div>
  );
}

// ─── DEMO 3: NETWORK RASTER ───────────────────────────────────────────────────

function NetworkDemo() {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const simRef = useRef(null);
  const [params, setParams] = useState({ N: 20, connectivity: 0.15, w_exc: 1.2, w_inh: 2.0, I_mean: 1.5, I_spread: 0.3 });
  const [stats, setStats] = useState({ totalSpikes: 0, hz: 0 });
  const setP = (k, v) => setParams(p => ({ ...p, [k]: v }));

  useEffect(() => {
    const { N, connectivity, w_exc, w_inh, I_mean, I_spread } = params;
    const dt = 0.1, tau_m = 20, R = 10, V_rest = -70, V_threshold = -55, V_reset = -80, tau_syn = 5;
    const N_inh = Math.floor(N * 0.2);

    // initialise
    const V = new Float32Array(N).fill(V_rest);
    const I_ext = Float32Array.from({ length: N }, () => I_mean + (Math.random() * 2 - 1) * I_spread);
    const g_exc = new Float32Array(N);
    const g_inh = new Float32Array(N);
    const refractory = new Float32Array(N);

    // weight matrix
    const W = new Float32Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i !== j && Math.random() < connectivity) {
          W[i * N + j] = i >= (N - N_inh) ? -w_inh : w_exc * Math.random();
        }
      }
    }

    const spikeRecord = []; // {t, neuron}
    let step = 0;
    const MAX_HISTORY = 3000;

    simRef.current = { V, I_ext, g_exc, g_inh, refractory, W, spikeRecord, step };

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const tick = () => {
      const s = simRef.current;
      const STEPS_PER_FRAME = 5;

      for (let ss = 0; ss < STEPS_PER_FRAME; ss++) {
        const t = s.step * dt;
        for (let i = 0; i < N; i++) {
          const I_syn = s.g_exc[i] * (0 - s.V[i]) + s.g_inh[i] * (-80 - s.V[i]);
          const dV = (dt / tau_m) * (-(s.V[i] - V_rest) + R * (s.I_ext[i] + I_syn));
          s.V[i] += dV;
          s.refractory[i] -= dt;
          if (s.V[i] >= V_threshold && s.refractory[i] <= 0) {
            s.V[i] = V_reset;
            s.refractory[i] = 2.0;
            s.spikeRecord.push({ t, neuron: i });
            for (let j = 0; j < N; j++) s.g_exc[j] += Math.max(0, s.W[i * N + j]) * 0.3;
            for (let j = 0; j < N; j++) s.g_inh[j] += Math.max(0, -s.W[i * N + j]) * 0.3;
          }
        }
        for (let i = 0; i < N; i++) {
          s.g_exc[i] -= (s.g_exc[i] / tau_syn) * dt;
          s.g_inh[i] -= (s.g_inh[i] / tau_syn) * dt;
        }
        s.step++;
      }

      // trim spike record
      const tNow = s.step * dt;
      while (s.spikeRecord.length > 0 && s.spikeRecord[0].t < tNow - MAX_HISTORY * dt) s.spikeRecord.shift();

      // draw
      const W_c = canvas.width, H_c = canvas.height;
      ctx.fillStyle = C.bg;
      ctx.fillRect(0, 0, W_c, H_c);

      // grid
      for (let n = 0; n < N; n++) {
        const y = (n / N) * H_c;
        ctx.strokeStyle = n === N - N_inh ? `${C.spike}33` : C.grid;
        ctx.lineWidth = n === N - N_inh ? 1 : 0.5;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W_c, y); ctx.stroke();
      }

      // neuron labels
      ctx.font = '8px monospace';
      for (let n = 0; n < N; n += 4) {
        const y = ((n + 0.5) / N) * H_c;
        ctx.fillStyle = n >= N - N_inh ? C.spike : C.muted;
        ctx.fillText(n, 2, y + 3);
      }

      // inhibitory label
      const inhibY = ((N - N_inh) / N) * H_c;
      ctx.fillStyle = `${C.spike}88`;
      ctx.font = '8px monospace';
      ctx.fillText('← inhibitory', 18, inhibY - 3);

      // spikes
      const WINDOW = 200;
      s.spikeRecord.forEach(({ t, neuron }) => {
        const age = tNow - t;
        if (age > WINDOW) return;
        const x = W_c - (age / WINDOW) * W_c;
        const y = ((neuron + 0.5) / N) * H_c;
        const alpha = 1 - age / WINDOW;
        const isInh = neuron >= N - N_inh;
        ctx.fillStyle = isInh
          ? `rgba(255,71,87,${alpha})`
          : `rgba(57,211,83,${alpha})`;
        ctx.fillRect(x - 1, y - 2, 3, 4);
      });

      // time axis
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, H_c - 1); ctx.lineTo(W_c, H_c - 1); ctx.stroke();

      const recent = s.spikeRecord.filter(s => s.t > tNow - 1000);
      setStats({ totalSpikes: s.spikeRecord.length, hz: (recent.length / 1).toFixed(1) });

      animRef.current = requestAnimationFrame(tick);
    };

    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [params]);

  return (
    <div style={{ display: 'flex', gap: 0, height: '100%' }}>
      <div style={{ width: 200, padding: 20, borderRight: `1px solid ${C.border}`, flexShrink: 0 }}>
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 16 }}>NETWORK</div>
        <Slider label='N NEURONS' value={params.N} min={10} max={40} step={1} unit='' onChange={v => setP('N', v)} />
        <Slider label='CONNECTIVITY' value={params.connectivity} min={0.05} max={0.5} step={0.05} unit='' onChange={v => setP('connectivity', v)} />
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, margin: '18px 0 14px' }}>WEIGHTS</div>
        <Slider label='W_EXC' value={params.w_exc} min={0.1} max={3} step={0.1} unit='' color={C.accent} onChange={v => setP('w_exc', v)} />
        <Slider label='W_INH' value={params.w_inh} min={0.1} max={5} step={0.1} unit='' color={C.spike} onChange={v => setP('w_inh', v)} />
        <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, margin: '18px 0 14px' }}>INPUT</div>
        <Slider label='I_MEAN' value={params.I_mean} min={0.5} max={3} step={0.1} unit='nA' onChange={v => setP('I_mean', v)} />
        <Slider label='I_SPREAD' value={params.I_spread} min={0} max={1} step={0.05} unit='nA' color={C.muted} onChange={v => setP('I_spread', v)} />
        <div style={{ marginTop: 20, padding: 12, background: C.panel, border: `1px solid ${C.border}`, borderRadius: 4 }}>
          <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 8 }}>LEGEND</div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
            <div style={{ width: 10, height: 10, background: C.accent, borderRadius: 1 }} />
            <span style={{ fontSize: 10, color: C.dim }}>excitatory</span>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ width: 10, height: 10, background: C.spike, borderRadius: 1 }} />
            <span style={{ fontSize: 10, color: C.dim }}>inhibitory (20%)</span>
          </div>
        </div>
        <div style={{ marginTop: 12, padding: 12, background: C.panel, border: `1px solid ${C.border}`, borderRadius: 4 }}>
          <div style={{ fontSize: 9, color: C.muted, letterSpacing: 2, marginBottom: 6 }}>STATS</div>
          <div style={{ fontSize: 10, color: C.dim }}>network rate</div>
          <div style={{ fontSize: 18, color: C.accent, fontFamily: 'monospace' }}>{stats.hz} <span style={{ fontSize: 10 }}>Hz</span></div>
        </div>
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '10px 20px', borderBottom: `1px solid ${C.border}`, fontSize: 9, color: C.muted, letterSpacing: 2 }}>
          SPIKE RASTER (200ms window) — each row is one neuron, dots are spikes
        </div>
        <div style={{ flex: 1, padding: 16 }}>
          <canvas ref={canvasRef} width={900} height={400}
            style={{ width: '100%', height: '100%', display: 'block', borderRadius: 4, border: `1px solid ${C.border}` }} />
        </div>
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState(0);

  const tabs = [
    { label: 'LIF Neuron', component: <LIFDemo /> },
    { label: 'Hopfield Memory', component: <HopfieldDemo /> },
    { label: 'E-I Network', component: <NetworkDemo /> },
  ];

  return (
    <div style={{ background: C.bg, minHeight: '100vh', color: C.text, fontFamily: 'monospace', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: '16px 24px', display: 'flex', alignItems: 'baseline', gap: 16 }}>
        <div>
          <div style={{ fontSize: 14, color: C.accent, letterSpacing: 3, textTransform: 'uppercase' }}>
            Computational Neuroscience
          </div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 3 }}>
            From-scratch implementations in JavaScript · LIF · Hodgkin-Huxley · Hopfield · STDP
          </div>
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 10, color: C.muted }}>
          github.com/Geomaniac15
        </div>
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: `1px solid ${C.border}`, paddingLeft: 8, display: 'flex' }}>
        {tabs.map((t, i) => <Tab key={i} label={t.label} active={tab === i} onClick={() => setTab(i)} />)}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {tabs[tab].component}
      </div>
    </div>
  );
}
