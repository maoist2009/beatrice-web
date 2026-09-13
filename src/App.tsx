import { useCallback, useEffect, useRef, useState } from "react";
import {
  AudioLines, CheckCircle2, ChevronDown, CircleGauge, Download, FileAudio, FileMusic, Loader2,
  Menu, Mic, SlidersHorizontal, Square, UploadCloud, X, Zap,
} from "lucide-react";
import { collectFiles, exportParaphernaliaZip, parseParaphernalia, type Paraphernalia, type ParaphernaliaFiles } from "./lib/paraphernalia";
import { deleteModel, listModels, loadModel, saveModel, storageEstimate, type CachedModelMeta } from "./lib/cache";
import { CATALOG, EXTERNAL_MODELS, customHuggingFaceEntry, fetchCatalogModel, type CatalogEntry } from "./lib/catalog";
import { createVoiceEngine, hardTerminateWorker, ORT_VERSION, webgpuAvailable, type VoiceEngine, type BackendChoice, type GpuMode, type GpuPrecision, type EngineStats, type FrameResult } from "./lib/engine";
import { formatLabel } from './lib/formats';
import { BEATRICE_NOTICE } from './lib/notices';
import { binToMidi, midiToHz } from "./lib/dsp";
import { decodeAudioFile, encodeWav } from "./lib/wav";
import { lastRun } from './lib/last-run';

const HIST = 600; // frames kept for the scopes (6 s)

interface Clip { name: string; url: string; pcm: Float32Array; sampleRate: number; duration: number }

export default function App() {
  const [gpuOk, setGpuOk] = useState<boolean | null>(null);
  // Adreno 6xx WebGPU has two known failure modes: fp16/mediump garble and dispatch-bound
  // throughput (CPU often wins there). Default such devices to the measured Auto backend
  // with forced FP32. Choices persist across reloads (memory regression testing friendly).
  const isQualcomm = typeof navigator !== 'undefined' && /adreno|qualcomm/i.test(navigator.userAgent);
  // Backend defaults to WebGPU everywhere (Adreno included): the GPU path must be measured,
  // not silently bypassed. Auto remains available as the explicit benchmark tool.
  const [backend, setBackend] = useState<BackendChoice>(() => {
    const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('beatrice.backend') : null;
    return saved === 'webgpu' || saved === 'wasm' || saved === 'auto' ? saved : 'webgpu';
  });
  const [gpuPrecision, setGpuPrecision] = useState<GpuPrecision>(() => {
    const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('beatrice.gpuPrecision') : null;
    if (saved === 'auto' || saved === 'fp32') return saved;
    return /adreno|qualcomm/i.test(navigator.userAgent) ? 'fp32' : 'auto';
  });
  useEffect(() => { try { localStorage.setItem('beatrice.backend', backend); } catch { /* private mode */ } }, [backend]);
  useEffect(() => { try { localStorage.setItem('beatrice.gpuPrecision', gpuPrecision); } catch { /* private mode */ } }, [gpuPrecision]);
  // Experimental: replay recorded GPU commands to kill per-chunk dispatch overhead.
  // Session-only by design — never silently persisted as a stability-critical default.
  const [graphCapture, setGraphCapture] = useState(false);
  const [ctxFrames, setCtxFrames] = useState(64);
  const [chunkFrames, setChunkFrames] = useState(8);
  const [gpuMode, setGpuMode] = useState<GpuMode>('bound');
  const [stage, setStage] = useState('Not initialized');
  const [model, setModel] = useState<Paraphernalia | null>(null);
  const [cached, setCached] = useState<CachedModelMeta[]>([]);
  const [storage, setStorage] = useState<{ usage: number; quota: number } | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [dl, setDl] = useState<{ id: string; pct: number } | null>(null);
  const [engine, setEngine] = useState<VoiceEngine | null>(null);
  const [customUrl, setCustomUrl] = useState('');

  const [error, setError] = useState<string | null>(() => {
    const previous = lastRun();
    return previous?.pending ? `Previous task did not shut down normally. Last stage: ${previous.stage}. Try compatibility mode or WASM and save diagnostics if it repeats.` : null;
  });
  const downloadController = useRef<AbortController | null>(null);
  const engineRef = useRef<VoiceEngine | null>(null);
  const buildController = useRef<AbortController | null>(null);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<EngineStats | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [live, setLive] = useState<FrameResult | null>(null);
  const [p, setP] = useState({ speaker: 0, pitchShift: 0, formantShift: 0, intonation: 1, correction: 0, correctionType: 0 as 0 | 1, minMidi: 33.125, maxMidi: 80.875, inputGain: 0, outputGain: 0, monitor: false, averageSourcePitch: 52, vqNeighbors: 0, convert: true });
  const [drawer, setDrawer] = useState<"left" | "right" | null>(null);
  const [inClip, setInClip] = useState<Clip | null>(null);
  const [outClip, setOutClip] = useState<{ url: string; duration: number } | null>(null);
  const [conv, setConv] = useState<{ pct: number; msg: string } | null>(null);
  const [diag, setDiag] = useState<any>(null);
  const [recording, setRecording] = useState<number>(-1); // seconds or -1
  const recorder = useRef<{ mr: MediaRecorder; chunks: Blob[]; st: MediaStream; t0: number } | null>(null);
  const hist = useRef<FrameResult[]>([]);
  const pitchCanvas = useRef<HTMLCanvasElement>(null);
  const phoneCanvas = useRef<HTMLCanvasElement>(null);
  const phoneOffscreen = useRef<HTMLCanvasElement | null>(null);
  const phoneImage = useRef<ImageData | null>(null);
  const log = useCallback((s: string) => {
    setLogs(l => [...l.slice(-99), `${new Date().toLocaleTimeString()} ${s}`]);
    if (s.startsWith('ERROR')) setError(s);
  }, []);
  const locked = !!busy || !!dl || !!conv || running || recording >= 0;
  useEffect(() => { engineRef.current = engine; }, [engine]);
  useEffect(() => () => {
    downloadController.current?.abort(); buildController.current?.abort(); void engineRef.current?.dispose();
    recorder.current?.st.getTracks().forEach(t => t.stop());
  }, []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setDrawer(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  useEffect(() => {
    // Page going away (including Android same-tab reload): hard-terminate the shared
    // inference worker so its ORT WASM heap and GPU buffers are freed for good instead of
    // stacking inside the reused renderer process.
    const hard = () => hardTerminateWorker();
    window.addEventListener('pagehide', hard);
    window.addEventListener('beforeunload', hard);
    return () => { window.removeEventListener('pagehide', hard); window.removeEventListener('beforeunload', hard); };
  }, []);

  useEffect(() => { webgpuAvailable().then((ok) => { setGpuOk(ok); if (!ok) setBackend("wasm"); }); refreshCache(); }, []);
  const refreshCache = () => { listModels().then(setCached).catch(() => {}); storageEstimate().then(setStorage); };

  // ---------- model loading ----------
  const loadFiles = async (files: ParaphernaliaFiles, persist: boolean) => {
    const pp = parseParaphernalia(files, { metadataOnly: true });
    await engineRef.current?.dispose(); engineRef.current = null; setEngine(null); setStats(null);
    model?.voices.forEach(v => { if (v.portraitUrl) URL.revokeObjectURL(v.portraitUrl); });
    setModel(pp);
    hist.current = []; setLive(null); setDiag(null); setError(null);
    log(`${formatLabel(pp.format)} selected; original TOML version ${pp.version}`);
    // Match ProcessorProxy/ProcessorCore1/2 model-load behavior: source pitch stays at
    // the official 52.0-unit default, while the model's TOML average_pitch (MIDI value)
    // becomes the initial pitch shift. Do not convert TOML average_pitch to 96-bin space;
    // the official VST does not do that conversion at this boundary.
    const targetPitch = pp.voices[0]?.averagePitch ?? 52;
    setP((s) => ({
      ...s,
      speaker: 0,
      formantShift: 0,
      pitchShift: Math.max(-24, Math.min(24, targetPitch - 52)),
      averageSourcePitch: 52,
      minMidi: 33.125,
      maxMidi: 80.875,
      vqNeighbors: pp.format === 'beatrice-rc0' ? 0 : s.vqNeighbors,
    }));
    log(`loaded "${pp.name}" v${pp.version}: ${pp.speakers.nSpeakers} speakers, ${(Object.values(pp.sizes).reduce((a, b) => a + b, 0) / 1e6).toFixed(1)} MB of float16 weights`);
    if (persist) {
      const digest = await crypto.subtle.digest('SHA-256', files.waveform_generator);
      const key = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('');
      await saveModel(`${pp.version}|${pp.name}|${key}`, pp.name, pp.version, files); refreshCache(); log('cached in IndexedDB');
    }
  };
  const onPick = async (fl: FileList | null, persist = true) => {
    if (!fl || !fl.length || locked) return;
    setBusy("Reading files…");
    try { await loadFiles(await collectFiles(Array.from(fl)), persist); } catch (e) { log(`ERROR: ${(e as Error).message}`); } finally { setBusy(null); }
  };
  const loadCached = async (key: string) => {
    if (locked) return;
    setBusy("Loading from cache…");
    try { const f = await loadModel(key); if (f) await loadFiles(f, false); } catch (e) { log(`ERROR: ${(e as Error).message}`); } finally { setBusy(null); }
  };
  const loadFromHF = async (entry: CatalogEntry) => {
    if (locked) return;
    const controller = new AbortController(); downloadController.current = controller;
    setDl({ id: entry.id, pct: 0 });
    log(`downloading "${entry.name}" from HuggingFace (${(entry.bytes / 1e6).toFixed(1)} MB, CORS)…`);
    try {
      const files = await fetchCatalogModel(entry, (loaded, total) => setDl({ id: entry.id, pct: total ? Math.min(100, Math.round((loaded / total) * 100)) : 0 }), controller.signal);
      setDl(null); setBusy("Parsing paraphernalia…");
      await loadFiles(files, true);
    } catch (e) { log((e as Error).name === 'AbortError' ? 'Download cancelled' : `ERROR: ${(e as Error).message}`); } finally { setDl(null); setBusy(null); downloadController.current = null; }
  };
  const downloadFiles = async (files: ParaphernaliaFiles, name: string) => {
    if (locked) return;
    setBusy("Packing original paraphernalia…");
    try {
      const blob = await exportParaphernaliaZip(files);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${name.replace(/[^a-z0-9._-]+/gi, "_") || "beatrice_model"}.zip`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
      log(`exported ${name}: ${(blob.size / 1e6).toFixed(1)} MB paraphernalia ZIP`);
    } catch (e) { log(`ERROR: ${(e as Error).message}`); } finally { setBusy(null); }
  };
  const downloadCached = async (m: CachedModelMeta) => {
    if (locked) return;
    try { const files = await loadModel(m.key); if (files) await downloadFiles(files, m.name); }
    catch (e) { log(`ERROR: ${(e as Error).message}`); }
  };

  // ---------- engine ----------
  const buildEngine = async () => {
    if (!model || locked) return;
    const controller = new AbortController(); buildController.current = controller;
    setBusy(`Building ONNX graphs from float16 weights and compiling on ${backend}…`);
    try {
      setError(null);
      await engineRef.current?.dispose(); engineRef.current = null; setEngine(null);
      const e = await createVoiceEngine(model, { backend, ctxFrames, chunkFrames, gpuMode, gpuPrecision, graphCapture, signal: controller.signal, onLog: log, onStage: setStage });
      if (controller.signal.aborted) { await e.dispose(); return; }
      e.onStats = (s) => setStats({ ...s });
      e.onFrames = (fr) => { hist.current.push(...fr); if (hist.current.length > HIST) hist.current.splice(0, hist.current.length - HIST); setLive(fr[fr.length - 1]); };
      e.onDiag = (d) => setDiag(d);
      e.onState = state => { setRunning(state === 'live'); if (state === 'error') { setConv(null); setBusy(null); } };
      engineRef.current = e; setEngine(e); setStats({ ...e.stats });
      pushParams(e);
    } catch (e) { if (!controller.signal.aborted) log(`ERROR: ${(e as Error).message}`); }
    finally { if (buildController.current === controller) { buildController.current = null; setBusy(null); } }
  };
  const pushParams = (e = engine!) => e?.setParams({
    speaker: p.speaker, pitchShift: p.pitchShift, formantShift: p.formantShift, intonationIntensity: p.intonation, pitchCorrection: p.correction,
    pitchCorrectionType: p.correctionType, minMidi: p.minMidi, maxMidi: p.maxMidi, inputGain: 10 ** (p.inputGain / 20), outputGain: 10 ** (p.outputGain / 20), monitor: p.monitor,
    averageSourcePitch: p.averageSourcePitch, vqNeighbors: p.vqNeighbors, convert: p.convert,
  });
  useEffect(() => { pushParams(); }, [p, engine]); // eslint-disable-line react-hooks/exhaustive-deps

  const startMic = async () => {
    if (!engine || locked) return;
    setBusy('Starting microphone...');
    try {
      await engine.start(deviceId || undefined); setRunning(true);
      const ds = await navigator.mediaDevices.enumerateDevices(); setDevices(ds.filter((d) => d.kind === "audioinput"));
      log(`audio started @ ${engine.stats.sampleRate} Hz, chain latency ${engine.stats.latencyMs.toFixed(1)} ms`);
    } catch (e) { log(`ERROR: ${(e as Error).message}`); } finally { setBusy(null); }
  };
  const stopMic = () => { engine?.stop(); setRunning(false); };
  const resetWorker = async () => {
    buildController.current?.abort(); buildController.current = null;
    const current = engineRef.current;
    engineRef.current = null; setEngine(null); setRunning(false); setConv(null); setBusy(null);
    await current?.dispose(); setStats(null); setStage('Worker released. Rebuild to continue.');
  };
  const downloadDiagnostics = () => {
    const mem = (performance as unknown as { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
    const blob = new Blob([JSON.stringify({ app: 'Beatrice Web stability build 3 (shared worker)', format: model?.format, stage, backend, gpuMode, gpuPrecision, ctxFrames, chunkFrames, stats, overruns: stats?.overruns ?? 0, jsHeap: mem ? { usedMB: +(mem.usedJSHeapSize / 1048576).toFixed(1), totalMB: +(mem.totalJSHeapSize / 1048576).toFixed(1), limitMB: +(mem.jsHeapSizeLimit / 1048576).toFixed(1) } : null, logs, userAgent: navigator.userAgent, note: 'Tracked buffers are not total GPU/process memory. No audio or model bytes included.' }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = 'beatrice-diagnostics.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  };

  // ---------- offline conversion ----------
  const setInput = async (name: string, data: ArrayBuffer) => {
    if (conv) return;
    try {
      const d = await decodeAudioFile(data);
      let url = "";
      url = URL.createObjectURL(encodeWav(d.pcm, d.sampleRate));
      if (inClip) URL.revokeObjectURL(inClip.url);
      if (outClip) URL.revokeObjectURL(outClip.url);
      setInClip({ name, url, pcm: d.pcm, sampleRate: d.sampleRate, duration: d.duration });
      setOutClip(null);
      log(`decoded "${name}": ${d.duration.toFixed(2)} s @ ${d.sampleRate} Hz×${d.channels}`);
    } catch (e) { log(`ERROR decoding audio: ${(e as Error).message}`); }
  };
  const startRecording = async () => {
    if (locked) return;
    try {
      const st = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/wav") ? "audio/wav" : "";
      const mr = new MediaRecorder(st, mime ? { mimeType: mime } : undefined);
      recorder.current = { mr, chunks: [], st, t0: Date.now() };
      mr.ondataavailable = (e) => e.data.size && recorder.current!.chunks.push(e.data);
      mr.onstop = async () => {
        const blob = new Blob(recorder.current!.chunks, { type: mr.mimeType });
        recorder.current = null;
        setRecording(-1);
        setInput(`recording (${mime || mr.mimeType || "audio"}).wav`, await blob.arrayBuffer());
      };
      mr.start();
      setRecording(0);
      const tick = () => { if (recorder.current) { setRecording((Date.now() - recorder.current.t0) / 1000); requestAnimationFrame(tick); } };
      tick();
    } catch (e) { log(`ERROR: ${(e as Error).message}`); }
  };
  const stopRecording = () => { recorder.current?.mr.stop(); recorder.current?.st.getTracks().forEach((t) => t.stop()); };
  const convertFile = async () => {
    if (!engine || !inClip || locked) return;
    setConv({ pct: 0, msg: "starting…" });
    try {
      const t0 = performance.now();
      const r = await engine.convertBuffer(inClip.pcm, inClip.sampleRate, (f, msg) => setConv({ pct: Math.round(f * 100), msg }));
      const wav = encodeWav(r.audio, r.sampleRate);
      if (outClip) URL.revokeObjectURL(outClip.url);
      setOutClip({ url: URL.createObjectURL(wav), duration: r.audio.length / r.sampleRate });
      hist.current = r.frames.slice(-HIST); // show the clip's analysis in the scopes
      log(`converted "${inClip.name}" (${r.audio.length / r.sampleRate}s @ ${r.sampleRate} Hz) in ${((performance.now() - t0) / 1000).toFixed(1)} s → ${r.frames.length} frames`);
    } catch (e) { log(`ERROR: ${(e as Error).message}`); } finally { setConv(null); }
  };

  // ---------- scopes ----------
  useEffect(() => {
    let raf = 0;
    let renderedLast: FrameResult | undefined | null = null;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const c = pitchCanvas.current, h = hist.current;
      if (renderedLast === h[h.length - 1]) return;
      renderedLast = h[h.length - 1];
      if (c) {
        const g = c.getContext("2d")!; const W = c.width, H = c.height;
        g.fillStyle = "#0b1020"; g.fillRect(0, 0, W, H);
        const lo = 30, hi = 96; const y = (midi: number) => H - ((midi - lo) / (hi - lo)) * H;
        g.strokeStyle = "#1e293b"; g.lineWidth = 1;
        for (let m = 36; m <= 96; m += 12) { g.beginPath(); g.moveTo(0, y(m)); g.lineTo(W, y(m)); g.stroke(); g.fillStyle = "#475569"; g.font = "10px ui-monospace"; g.fillText(`${["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][m % 12]}${Math.floor(m / 12) - 1}`, 4, y(m) - 2); }
        const n = h.length; const x = (i: number) => (i / HIST) * W;
        for (let i = 0; i < n; i++) { const e = h[i].energy; const a = Math.min(1, Math.max(0, (e + 2.5) / 3)); g.fillStyle = `rgba(56,189,248,${a * 0.12})`; g.fillRect(x(i), 0, W / HIST + 1, H); }
        g.lineWidth = 2;
        for (let i = 1; i < n; i++) {
          const v = h[i].unvoiced > 0.5 || h[i].energy < -2.2;
          g.strokeStyle = v ? "rgba(148,163,184,0.25)" : "#38bdf8"; g.beginPath(); g.moveTo(x(i - 1), y(h[i - 1].midi)); g.lineTo(x(i), y(h[i].midi)); g.stroke();
          g.strokeStyle = v ? "rgba(251,146,60,0.2)" : "#fb923c"; g.beginPath(); g.moveTo(x(i - 1), y(binToMidi(h[i - 1].targetBin))); g.lineTo(x(i), y(binToMidi(h[i].targetBin))); g.stroke();
        }
      }
      const pc = phoneCanvas.current;
      if (pc && h.length) {
        // One ImageData blit instead of ~77k fillRect calls per redraw (that was starving the inference thread on phones).
        const g = pc.getContext("2d")!; const W = pc.width, H = pc.height;
        const rows = h[0].phone.length, cols = HIST;
        const off = phoneOffscreen.current ?? (phoneOffscreen.current = document.createElement("canvas"));
        if (off.width !== cols || off.height !== rows) { off.width = cols; off.height = rows; }
        const og = off.getContext("2d")!;
        const img = phoneImage.current && phoneImage.current.width === cols && phoneImage.current.height === rows ? phoneImage.current : (phoneImage.current = og.createImageData(cols, rows));
        const px = img.data; px.fill(0);
        for (let i = 0; i < h.length; i++) {
          const ph = h[i].phone;
          for (let c = 0; c < rows; c++) {
            const v = Math.tanh(ph[c] * 0.5), a = Math.min(255, Math.round(Math.abs(v) * 255));
            const o = ((rows - 1 - c) * cols + i) * 4;
            if (v >= 0) { px[o] = 244; px[o + 1] = 114; px[o + 2] = 182; } else { px[o] = 96; px[o + 1] = 165; px[o + 2] = 250; }
            px[o + 3] = a;
          }
        }
        og.putImageData(img, 0, 0);
        g.fillStyle = "#0b1020"; g.fillRect(0, 0, W, H);
        g.imageSmoothingEnabled = false;
        g.drawImage(off, 0, 0, W, H);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  const voice = model?.voices.find(v => v.id === p.speaker);
  const isBeta = model?.format === 'beatrice-beta2';
  const fmt = (b: number) => b >= 1e6 ? `${(b / 1e6).toFixed(2)} MB` : `${(b / 1e3).toFixed(1)} kB`;

  // ================================================================ side panels
  const modelPanel = (
    <>
      <Card title="Model" icon={<FileMusic size={15} />}>
        <div className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">Ready-to-use (HuggingFace, cross-origin, cached)</div>
        <ul className="space-y-1">
          {CATALOG.map((m) => (
            <li key={m.id} className="bg-slate-900/60 rounded-lg px-2.5 py-2 text-sm">
              <div className="flex items-center gap-2">
                <div className="flex-1 min-w-0 truncate">{m.name} <span className="text-slate-500 text-xs">· {(m.bytes / 1e6).toFixed(0)} MB</span></div>
                <button className="btn-sm bg-sky-700" disabled={locked} onClick={() => loadFromHF(m)}>
                  {dl?.id === m.id ? `${dl.pct}%` : <span className="inline-flex items-center gap-1"><Download size={12} />get</span>}
                </button>
                <a className="btn-sm" href={m.url} download title="Download the original archive without loading it"><Download size={12} /> ZIP</a>
              </div>
              {dl?.id === m.id && <div className="h-1 mt-2 bg-slate-800 rounded overflow-hidden"><div className="h-full bg-sky-400 transition-all" style={{ width: `${dl.pct}%` }} /></div>}
              <div className="text-[11px] text-slate-500 mt-1">{m.note}</div>
              <div className="flex flex-wrap gap-x-2 gap-y-1 mt-1 text-[11px]"><span className="text-sky-300">{m.version}</span><span className="text-slate-400">{m.author}</span><a href={m.termsUrl} target="_blank" rel="noreferrer" className="underline text-slate-300">Terms / credit</a><span className="text-slate-500">SHA-256 pinned</span></div>
            </li>
          ))}
        </ul>
        {dl && <button className="btn-sm mt-2" onClick={() => downloadController.current?.abort()}>Cancel download</button>}
        <form className="mt-3 space-y-2" onSubmit={event => { event.preventDefault(); try { void loadFromHF(customHuggingFaceEntry(customUrl)); } catch (e) { log(`ERROR: ${(e as Error).message}`); } }}>
          <label className="lbl" htmlFor="hf-model-url">Other public HuggingFace ZIP</label>
          <input id="hf-model-url" type="url" required className="inp w-full text-xs" disabled={locked} value={customUrl} onChange={e => setCustomUrl(e.target.value)} placeholder="https://huggingface.co/.../resolve/main/model.zip" />
          <button className="btn-sm" disabled={locked || !customUrl}>Download and load</button>
          <p className="text-[10px] text-slate-500">Downloads go directly to your browser. Beatrice beta.2 / rc.0 only; no RVC .pth/.index conversion or server inference.</p>
        </form>
        <div className="text-[11px] uppercase tracking-wider text-slate-500 mt-3 mb-1.5">Your own paraphernalia (beta.2 / rc.0)</div>
        <div className="flex flex-wrap gap-2">
          <label className="btn cursor-pointer"><UploadCloud size={14} /> Folder<input disabled={locked} type="file" className="hidden" multiple {...({ webkitdirectory: "", directory: "" } as any)} onChange={(e) => { void onPick(e.target.files); e.target.value = ''; }} /></label>
          <label className="btn cursor-pointer"><FileAudio size={14} /> .zip / files<input disabled={locked} type="file" className="hidden" multiple accept=".zip,.bin,.toml,.png,.jpg,.txt,.md" onChange={(e) => { void onPick(e.target.files); e.target.value = ''; }} /></label>
        </div>
        {cached.length > 0 && (
          <>
            <div className="text-[11px] uppercase tracking-wider text-slate-500 mt-3 mb-1.5">Cached {storage && <span className="normal-case text-slate-600">({fmt(storage.usage)}/{fmt(storage.quota)})</span>}</div>
            <ul className="space-y-1">
              {cached.map((m) => (
                <li key={m.key} className="flex items-center gap-2 text-sm bg-slate-900/60 rounded-lg px-2.5 py-1.5">
                  <span className="truncate flex-1" title={m.key}>{m.name} <span className="text-slate-500 text-xs">v{m.version} · {fmt(m.bytes)}</span></span>
                  <button disabled={locked} className="btn-sm" onClick={() => loadCached(m.key)}>load</button>
                  <button disabled={locked} className="btn-sm" onClick={() => downloadCached(m)} aria-label={`Export ${m.name}`} title="Export cached weights, TOML, formant embeddings and attribution"><Download size={12} /></button>
                  <button disabled={locked} className="btn-sm text-red-300" aria-label={`Delete ${m.name}`} onClick={() => deleteModel(m.key).then(refreshCache).catch(e => log(`ERROR: ${String(e)}`))}><X size={12} /></button>
                </li>
              ))}
            </ul>
          </>
        )}
        {busy && <div className="mt-3 text-sky-300 text-sm flex items-center gap-2"><Loader2 size={14} className="animate-spin" />{busy}</div>}
      </Card>

      {model && (
        <Card title={`${model.name} · ${model.speakers.nSpeakers} voices`} icon={<AudioLines size={15} />}>
          <div className="text-xs text-slate-500 mb-2"><b className="text-emerald-300">{formatLabel(model.format)}</b><br />TOML: {model.version}{isBeta && ' (beta.2 Trainer export format)'}</div>
          <div className="grid grid-cols-4 gap-2 max-h-64 overflow-auto pr-1">
            {model.voices.map((v) => (
              <button disabled={!!conv || !!busy || !!dl} key={v.id} onClick={() => setP((s) => ({ ...s, speaker: v.id }))} className={`rounded-lg overflow-hidden border text-left ${p.speaker === v.id ? "border-sky-400 ring-1 ring-sky-400" : "border-slate-800 hover:border-slate-600"}`}>
                {v.portraitUrl ? <img src={v.portraitUrl} alt="" loading="lazy" decoding="async" width={64} height={64} className="w-full aspect-square object-cover" /> : <div className="w-full aspect-square bg-slate-800 grid place-items-center text-slate-600"><Mic size={18} /></div>}
                <div className="px-1.5 py-1 text-[10px] leading-tight truncate">{v.name}</div>
              </button>
            ))}
          </div>
          {voice && <div className="mt-2 text-xs text-slate-400 whitespace-pre-wrap line-clamp-4" title={voice.description}>{voice.description}{voice.averagePitch != null && `\navg_pitch ${voice.averagePitch.toFixed(2)} MIDI ≈ ${midiToHz(voice.averagePitch).toFixed(0)} Hz`}</div>}
          <button disabled={locked} className="btn-sm mt-2" onClick={() => downloadFiles(model.files, model.name)}><Download size={12} /> export loaded model ZIP</button>
        </Card>
      )}

      <Card title="More voice models" icon={<Download size={15} />}>
        <p className="text-[11px] text-slate-500 mb-2">Catalog ZIPs download directly to your browser. Other authors provide models through these source pages; import a compatible paraphernalia ZIP after obtaining it.</p>
        <ul className="space-y-1.5">
          {EXTERNAL_MODELS.map((m) => (
            <li key={m.url} className="rounded-lg bg-slate-900/60 p-2 text-xs">
              <div className="flex gap-2 items-center"><span className="flex-1 font-medium">{m.name}</span><span className={m.version.endsWith("rc.0") ? "text-emerald-300" : "text-amber-300"}>{m.version}</span><a className="btn-sm" href={m.url} target="_blank" rel="noreferrer">open</a></div>
              <div className="text-slate-500 mt-1">{m.source} · {m.note}</div>
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Runtime" icon={<Zap size={15} />}>
        <fieldset disabled={locked} className="min-w-0">
        <div className="grid grid-cols-3 gap-1.5 mb-3">
            <button className="btn-sm justify-center" onClick={() => { setCtxFrames(128); setChunkFrames(4); }}>low latency<br className="hidden sm:block" /> 40 ms</button>
            <button className="btn-sm justify-center" onClick={() => { setCtxFrames(256); setChunkFrames(8); }}>balanced<br className="hidden sm:block" /> 80 ms</button>
            <button className="btn-sm justify-center" onClick={() => { setCtxFrames(256); setChunkFrames(20); }}>robust<br className="hidden sm:block" /> 200 ms</button>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <label className="flex flex-col gap-1"><span className="lbl">Backend</span>
            <select className="inp" value={backend} onChange={(e) => setBackend(e.target.value as BackendChoice)}><option value="auto">Auto — bench both, use faster</option><option value="webgpu" disabled={gpuOk === false}>WebGPU</option><option value="wasm">WASM (safe CPU)</option></select></label>
          <label className="flex flex-col gap-1"><span className="lbl">GPU memory mode</span><select disabled={backend === 'wasm'} className="inp" value={gpuMode} onChange={e => setGpuMode(e.target.value as GpuMode)}><option value="bound">Bounded GPU I/O</option><option value="compatible">Compatibility (CPU I/O)</option></select></label>
          <label className="flex flex-col gap-1"><span className="lbl">GPU precision</span>
            <select disabled={backend === 'wasm'} className="inp" value={gpuPrecision} onChange={e => setGpuPrecision(e.target.value as GpuPrecision)}><option value="auto">Auto (fp16 if driver-safe)</option><option value="fp32">Force FP32 (Adreno-safe)</option></select></label>
          <label className="flex flex-col gap-1"><span className="lbl">{isBeta ? 'Attention memory (rc.0 only)' : 'Attention memory (rc.0)'}</span>
            <select disabled={isBeta} className="inp" value={ctxFrames} onChange={(e) => setCtxFrames(Number(e.target.value))}><option value={64}>0.64 s</option><option value={128}>1.28 s</option><option value={256}>2.56 s</option><option value={512}>5.12 s</option></select></label>
          <label className="flex flex-col gap-1 col-span-2"><span className="lbl">Chunk (buffering latency)</span>
            <select className="inp" value={chunkFrames} onChange={(e) => setChunkFrames(Number(e.target.value))}><option value={4}>4 · 40 ms</option><option value={8}>8 · 80 ms</option><option value={12}>12 · 120 ms</option><option value={20}>20 · 200 ms</option><option value={40}>40 · 400 ms</option></select></label>
        </div>
        <label className="flex items-start gap-2 text-xs mt-2 leading-5">
          <input type="checkbox" className="size-4 accent-sky-400 mt-0.5" checked={graphCapture} disabled={backend === 'wasm' || locked} onChange={e => setGraphCapture(e.target.checked)} />
          <span><b className="text-slate-300">Graph capture</b> (experimental): record the GPU command sequence once, replay it every chunk — removes most per-chunk dispatch overhead, the main thing holding WebGPU back on mobile. Falls back to capture OFF automatically if the driver rejects it.</span>
        </label>
        </fieldset>
        <p className="text-[11px] text-slate-400 mt-2">Inference runs in an isolated, terminable Worker. Graph capture defaults <b className="text-slate-200">OFF</b> for stability (experimental toggle above; auto-fallback if the driver rejects it). Bounded I/O keeps recurrent state on GPU with a 32 MiB app-owned buffer cap; compatibility mode uses CPU I/O but still runs supported network operators on GPU.</p>
        <p className="text-[11px] text-amber-300/90 mt-2">Beatrice 2.0.0-rc.1/rc.2/rc.3 are VST releases, not new model formats — the trainer still exports <code>PARAPHERNALIA_VERSION = &quot;2.0.0-rc.0&quot;</code>, so the rc.3 official models (つくよみちゃん / 刻鳴時雨 / OLUNE) load with this rc.0 path.</p>
        <p className="text-[11px] text-slate-500 mt-2">These catalog voices are Japanese or English; Mandarin quality is not validated. The trainer documents Japanese content-extractor data and English synthesis pretraining. Changing a speaker model alone does not guarantee accurate Chinese pronunciation.</p>
        {isQualcomm && <p className="text-[11px] text-red-300/90 mt-2">Qualcomm Adreno GPU detected: precision defaults to <b>Force FP32</b>, because Adreno 6xx fp16/mediump paths can garble this network's output. The backend stays WebGPU so the GPU path is measured, not bypassed — build once with <b>Auto</b> to get both ms/chunk numbers in the log, and try <b>Graph capture</b> to push the GPU path further. See “Why WASM can legitimately beat WebGPU on Adreno 6xx” below.</p>}
        <p className="text-[10px] text-slate-500 mt-1">Adreno 6/7: use an up-to-date WebGPU-capable browser, FP32, a 0.64 s attention cache and 80-200 ms chunks. Hardware/driver support varies; no family-wide compatibility or latency guarantee. If unstable, rebuild in compatibility mode or WASM.</p>
        <div className="flex flex-wrap gap-2 mt-3">
          <button className="btn" disabled={!model || locked} onClick={buildEngine}><Zap size={14} /> Build & compile</button>
          {!running
            ? <button className="btn bg-emerald-600 hover:bg-emerald-500" disabled={!engine || locked} onClick={startMic}><Mic size={14} /> Live mic</button>
            : <button className="btn bg-red-700 hover:bg-red-600" onClick={stopMic}><Square size={14} /> Stop</button>}
        </div>
        {devices.length > 0 && <select className="inp mt-2 w-full text-xs" value={deviceId} onChange={(e) => setDeviceId(e.target.value)}><option value="">default input</option>{devices.map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label || d.deviceId}</option>)}</select>}
        {stats && (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 mt-3 text-xs">
            <dt className="text-slate-500">end-to-end latency</dt><dd>not measured</dd>
            <dt className="text-slate-500">networks / chunk</dt><dd>{stats.inferMs.toFixed(1)} ms</dd>
            <dt className="text-slate-500">synthesis / chunk</dt><dd>{stats.synthMs.toFixed(1)} ms</dd>
            <dt className="text-slate-500">RTF</dt><dd className={stats.rtf > 0.9 ? "text-red-300" : "text-emerald-300"}>{stats.measured ? stats.rtf.toFixed(3) : 'waiting for completed audio chunk'}</dd>
            <dt className="text-slate-500">frames / underruns</dt><dd>{stats.frames} / {stats.underruns}</dd>
            <dt className="text-slate-500">overrun safe-pauses</dt><dd className={stats.overruns ? 'text-amber-300' : ''}>{stats.overruns ?? 0}{stats.overruns ? ' (mic paused, engine kept — Live mic resumes)' : ''}</dd>
            <dt className="text-slate-500">graph bytes</dt><dd>{fmt(stats.graphBytes)}</dd>
            <dt className="text-slate-500">graph capture</dt><dd>{stats.capture ? 'ON — replaying recorded commands' : 'OFF — per-op dispatch'}</dd>
            <dt className="text-slate-500">actual chunk</dt><dd>{(stats.chunkFrames ?? chunkFrames) * 10} ms</dd>
            <dt className="text-slate-500">tracked GPU buffers</dt><dd>{stats.resources?.gpuBuffers ?? 0} / {fmt(stats.resources?.gpuBytes ?? 0)}</dd>
            <dt className="text-slate-500">sessions / in-flight</dt><dd>{stats.resources?.sessions ?? 0} / {stats.resources?.inFlight ?? 0}</dd>
            {stats.adapter && <><dt className="text-slate-500">GPU adapter</dt><dd className="break-words">{[stats.adapter.vendor, stats.adapter.architecture, stats.adapter.description].filter(Boolean).join(' / ') || 'not exposed by browser'}</dd></>}
          </dl>
        )}
        <p role="status" className="text-xs text-slate-400 mt-2 break-words">{stage}</p>
        <HeapMeter />
        <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
          <summary className="text-xs text-slate-300 cursor-pointer select-none">Why WASM can legitimately beat WebGPU on Adreno 6xx — and how to verify it on this device</summary>
          <ul className="text-[11px] text-slate-500 mt-2 space-y-1.5 list-disc pl-4">
            <li><b className="text-slate-300">Dispatch overhead dominates.</b> Every 40–200 ms chunk executes three networks as a long sequence of small WebGPU dispatches. Each dispatch costs JS encoding + Dawn validation + Android driver submission ≈ 0.2–1 ms on mobile (≈ 0.02–0.05 ms on desktop). ~100 dispatches/chunk ⇒ 20–100 ms of pure overhead before a single multiply runs.</li>
            <li><b className="text-slate-300">fp16 is locked out by correctness.</b> GPU throughput on these nets mainly comes from fp16. Adreno 6xx fp16 corrupts this network (the garble) ⇒ forced fp32 ⇒ the GPU's main advantage is gone while its overhead stays.</li>
            <li><b className="text-slate-300">Per-run readbacks.</b> Vocoder output is read back to CPU after each network run (JS float64 phase synthesis); every mapAsync is a sync point serializing the pipeline.</li>
            <li><b className="text-slate-300">Graph capture is the one real fix.</b> Replaying recorded commands collapses per-run dispatch cost — that's the experimental toggle above, OFF by default for stability (auto fallback if the driver rejects it).</li>
          </ul>
          <p className="text-[11px] text-slate-500 mt-2">Verify: build once with <b className="text-slate-300">Auto</b> — the log prints measured WebGPU vs WASM ms/chunk. Then rebuild with <b className="text-slate-300">Graph capture</b> and compare “networks / chunk”. If capture-OFF fp32 WebGPU still loses, the residual gap is driver-side dispatch latency — not addressable from this app's layer, which is exactly what the measurement proves.</p>
        </details>
        <p className="text-[10px] text-slate-500 mt-1">Tracked GPU I/O excludes ORT kernels/weights/driver allocation; it is not total VRAM usage.</p>
        <div className="flex gap-2 mt-3"><button className="btn-sm" onClick={() => void resetWorker()}>Stop / release Worker</button><button className="btn-sm" onClick={downloadDiagnostics}>Save diagnostics</button></div>
      </Card>
    </>
  );

  const paramsPanel = (
    <>
      <Card title="Voice parameters" icon={<SlidersHorizontal size={15} />}>
        <fieldset className="min-w-0" disabled={!!conv || !!busy || !!dl}>
        <Slider label="Pitch shift" unit={(p.pitchShift >= 0 ? "+" : "") + p.pitchShift + " st"} min={-24} max={24} step={1} value={p.pitchShift} onChange={(v) => setP((s) => ({ ...s, pitchShift: v }))} />
        <Slider label="Formant shift" unit={`${(p.formantShift >= 0 ? "+" : "") + p.formantShift} st → #${Math.round(p.formantShift * 2 + 4)}`} min={-2} max={2} step={0.5} value={p.formantShift} onChange={(v) => setP((s) => ({ ...s, formantShift: v }))} />
        <Slider label="Intonation intensity" unit={p.intonation.toFixed(2)} min={-1} max={3} step={0.05} value={p.intonation} onChange={(v) => setP((s) => ({ ...s, intonation: v }))} />
        <Slider label="Pitch correction" unit={p.correction.toFixed(2)} min={0} max={1} step={0.05} value={p.correction} onChange={(v) => setP((s) => ({ ...s, correction: v }))} />
        <div className="flex gap-2 text-xs mb-3">
          <button className={`btn-sm flex-1 ${p.correctionType === 0 ? "bg-sky-700" : ""}`} onClick={() => setP((s) => ({ ...s, correctionType: 0 }))}>type 0 · x|x|⁻ᵖ</button>
          <button className={`btn-sm flex-1 ${p.correctionType === 1 ? "bg-sky-700" : ""}`} onClick={() => setP((s) => ({ ...s, correctionType: 1 }))}>type 1 · |x|^(1/(1−p))</button>
        </div>
        <Slider label="Average source pitch (official unit)" unit={p.averageSourcePitch.toFixed(3)} min={0} max={128} step={0.125} value={p.averageSourcePitch} onChange={(v) => setP((s) => ({ ...s, averageSourcePitch: v }))} />
        <Slider label="Min source pitch" unit={`MIDI ${p.minMidi.toFixed(3)}`} min={0} max={128} step={0.125} value={p.minMidi} onChange={(v) => setP((s) => ({ ...s, minMidi: v }))} />
        <Slider label="Max source pitch" unit={`MIDI ${p.maxMidi.toFixed(3)}`} min={0} max={128} step={0.125} value={p.maxMidi} onChange={(v) => setP((s) => ({ ...s, maxMidi: v }))} />
        <Slider label="Input gain" unit={`${p.inputGain} dB`} min={-60} max={20} step={1} value={p.inputGain} onChange={(v) => setP((s) => ({ ...s, inputGain: v }))} />
        <Slider label="Output gain" unit={`${p.outputGain} dB`} min={-60} max={20} step={1} value={p.outputGain} onChange={(v) => setP((s) => ({ ...s, outputGain: v }))} />
        {!isBeta && <Slider label="VQ neighbours (kNN-VC top-k)" unit={p.vqNeighbors === 0 ? "off" : String(p.vqNeighbors)} min={0} max={8} step={1} value={p.vqNeighbors} onChange={(v) => setP((s) => ({ ...s, vqNeighbors: v }))} />}
        {isBeta && <p className="text-xs text-slate-500 mb-3">beta.2 does not contain VQ or channel RMS normalization. These are not emulated.</p>}
        <label className="flex items-center gap-2 text-xs mt-1 leading-5"><input type="checkbox" className="size-4 accent-sky-400" checked={p.convert} onChange={(e) => setP((s) => ({ ...s, convert: e.target.checked }))} /> Conversion ON (realtime: synthesise the 24 kHz output)</label>
        <label className="flex items-center gap-2 text-xs mt-1.5 leading-5"><input type="checkbox" className="size-4 accent-sky-400" checked={p.monitor} onChange={(e) => setP((s) => ({ ...s, monitor: e.target.checked }))} /> Dry monitor when OFF (passes input through the official 16k/24k/48k chain)</label>
        </fieldset>
      </Card>

      {engine && (
        <Card title="Verification" icon={<CheckCircle2 size={15} />}>
          <details>
            <summary className="text-xs text-slate-400 cursor-pointer select-none flex items-center gap-1"><ChevronDown size={12} /> version-specific weight layout checks (not a native VST parity certificate)</summary>
            <table className="w-full text-[11px] mt-2">
              <tbody>
                {engine.reports.map((r) => (
                  <tr key={r.name} className="border-t border-slate-800 align-top">
                    <td className="py-1 pr-2 font-mono">{r.name}</td>
                    <td className="pr-2 whitespace-nowrap">{fmt(r.bytes)}</td>
                    <td className="pr-2"><StatusPill s={r.status} /></td>
                    <td className="text-slate-500">{r.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </Card>
      )}
    </>
  );

  // ================================================================ render
  return (
    <div className="h-dvh flex flex-col bg-[#070a14] text-slate-200 font-sans overflow-hidden">
      <header className="h-16 shrink-0 border-b border-slate-800 px-3 sm:px-5 py-2.5 flex items-center gap-2 sm:gap-3">
        <button className="xl:hidden btn-sm" onClick={() => setDrawer(drawer === "left" ? null : "left")} aria-label="model panel"><Menu size={16} /></button>
        <div className="min-w-0">
          <h1 className="text-base sm:text-lg font-semibold tracking-tight truncate">Beatrice 2 <span className="text-sky-400">Web</span></h1>
          <p className="text-[10px] sm:text-xs text-slate-500 truncate">Local voice conversion · beta.2 / rc.0 · ORT Web {ORT_VERSION}</p>
        </div>
        <div className="ml-auto flex items-center gap-2 text-[10px] sm:text-xs">
          <Badge ok={gpuOk === true}>{gpuOk === null ? "…" : gpuOk ? "WebGPU" : "WASM"}</Badge>
          {stats && <Badge ok={stats.backend === "webgpu"}>{stats.backend}{stats.measured ? ` · RTF ${stats.rtf.toFixed(2)}` : ' · ready'}</Badge>}
        </div>
        <button className="xl:hidden btn-sm" onClick={() => setDrawer(drawer === "right" ? null : "right")} aria-label="voice parameters"><SlidersHorizontal size={16} /></button>
      </header>

      <div className="flex-1 flex min-h-0 relative">
        {/* left sidebar */}
        <Side visible={drawer !== "left"} side="left" onClose={() => setDrawer(null)}>
          {modelPanel}
        </Side>
        {/* right sidebar */}
        <Side visible={drawer !== "right"} side="right" onClose={() => setDrawer(null)}>
          {paramsPanel}
        </Side>

        {/* center */}
        <main className="xl:order-2 flex-1 min-w-0 overflow-y-auto p-3 sm:p-5 space-y-4">
          {error && <div role="alert" className="rounded-lg border border-red-900 bg-red-950/30 px-3 py-2 text-sm text-red-200 flex gap-2"><span className="flex-1 break-words">{error}</span><button aria-label="Dismiss error" onClick={() => setError(null)}><X size={16} /></button></div>}
          {/* -------- offline conversion -------- */}
          <Card title="File & recording conversion" icon={<FileAudio size={15} />} accent>
            <div className="grid md:grid-cols-2 gap-3">
              <DropZone onFile={setInput}>
                <div className="flex flex-col gap-2">
                  {inClip ? (
                    <>
                      <div className="flex items-center gap-2">
                        <FileAudio size={16} className="text-sky-300 shrink-0" />
                        <div className="min-w-0 text-xs">
                          <div className="truncate font-medium">{inClip.name}</div>
                          <div className="text-slate-500">{inClip.duration.toFixed(2)} s @ {inClip.sampleRate} Hz</div>
                        </div>
                      </div>
                      <Waveform pcm={inClip.pcm} />
                      <audio controls src={inClip.url} className="w-full h-8" />
                    </>
                  ) : (
                    <div className="py-4 text-center text-xs text-slate-500">
                      drop a <b className="text-slate-300">.wav / .mp3 / .ogg</b> here<br />or
                    </div>
                  )}
                  <div className="flex flex-wrap gap-2">
                    <label className="btn-sm cursor-pointer"><UploadCloud size={12} /> choose file<input disabled={locked} type="file" className="hidden" accept=".wav,.mp3,.ogg,.m4a,.flac,.opus,.webm,audio/*" onChange={async (e) => { const el = e.currentTarget, f = el.files?.[0]; if (f) await setInput(f.name, await f.arrayBuffer()); el.value = ""; }} /></label>
                    {recording < 0
                      ? <button disabled={locked} className="btn-sm text-red-300" onClick={startRecording}><Mic size={12} /> record mic</button>
                      : <button className="btn-sm bg-red-700" onClick={stopRecording}><Square size={12} /> {recording.toFixed(1)} s — stop</button>}
                  </div>
                </div>
              </DropZone>
              <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-2.5 flex flex-col gap-2">
                {conv ? (
                  <div className="my-auto">
                    <div className="flex justify-between text-xs text-slate-400"><span className="truncate">{conv.msg}</span><span>{conv.pct}%</span></div>
                    <div className="h-1.5 mt-2 bg-slate-800 rounded overflow-hidden"><div className="h-full bg-sky-400 transition-all" style={{ width: `${conv.pct}%` }} /></div>
                  </div>
                ) : outClip ? (
                  <>
                    <div className="flex items-center gap-2 text-xs">
                      <CheckCircle2 size={16} className="text-emerald-300 shrink-0" />
                      <div className="min-w-0"><div className="font-medium">converted</div><div className="text-slate-500">{outClip.duration.toFixed(2)} s</div></div>
                    </div>
                    <audio controls src={outClip.url} className="w-full h-8" autoPlay />
                    <a className="btn-sm self-start" href={outClip.url} download={`${(inClip?.name ?? "audio").replace(/\.[^.]+$/, "")}.beatrice.wav`}><Download size={12} /> download WAV</a>
                  </>
                ) : (
                  <div className="py-4 text-center text-xs text-slate-500 my-auto">converted audio appears here</div>
                )}
              </div>
            </div>
            <button className="btn mt-3 w-full sm:w-auto bg-sky-700" disabled={!engine || !inClip || locked} onClick={convertFile}>
              <CircleGauge size={15} /> Convert{!engine ? " — build the engine first" : running ? " — stop the mic first" : ""}
            </button>
            {conv && <button className="btn-sm ml-2 mt-3" onClick={() => engineRef.current?.stop()}>Cancel conversion</button>}
          </Card>

          {/* -------- live scopes -------- */}
          <Card title="Pitch — source (blue) vs. target after official pitch math (orange)" icon={<AudioLines size={15} />}>
            <canvas ref={pitchCanvas} width={1200} height={240} className="w-full rounded bg-[#0b1020] touch-none" />
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 mt-2 text-xs">
              <Stat k="source" v={live ? `${live.hz.toFixed(1)} Hz` : "—"} />
              <Stat k="target" v={live ? `${live.targetHz.toFixed(1)} Hz` : "—"} />
              <Stat k="unvoiced p" v={live ? live.unvoiced.toExponential(1) : "—"} />
              <Stat k="½ / 2×" v={live ? `${live.half.toFixed(2)} / ${live.dbl.toFixed(2)}` : "—"} />
              <Stat k="energy" v={live ? live.energy.toFixed(2) : "—"} />
              <Stat k="frame" v={live ? `${live.frame}` : "—"} />
            </div>
            {diag && (
              <div className="mt-2 text-[10px] font-mono text-slate-500 grid grid-cols-2 sm:grid-cols-4 gap-1">
                <span>ir (log-amp) [{diag.ir.mn.toFixed(1)},{diag.ir.mx.toFixed(1)}]</span>
                <span>aperiodicity [{diag.ap.mn.toFixed(2)},{diag.ap.mx.toFixed(2)}]</span>
                <span>post-filter [{(diag.pf.rms as number).toFixed(3)} rms]</span>
                <span>f0 {diag.f0.toFixed(1)} Hz</span>
              </div>
            )}
          </Card>
          <Card title={`Phone units (${isBeta ? 256 : 128} channels at 100 Hz)`} icon={<AudioLines size={15} />}>
            <canvas ref={phoneCanvas} width={1200} height={180} className="w-full rounded bg-[#0b1020] touch-none" />
          </Card>

          <Card title={isBeta ? 'Beatrice beta.2: stateful pipeline' : 'Beatrice rc.0: stateful pipeline'} icon={<Zap size={15} />}>
            <ol className="text-xs text-slate-400 space-y-1 list-decimal pl-4">
              <li><b className="text-slate-200">Audio</b>: browser decoding and resampling for files; AudioWorklet for microphone mode. 16 kHz input, 24 kHz synthesis. Browser audio I/O is not native VST I/O.</li>
              <li><b className="text-slate-200">PitchEstimator</b>: hop 160 / window 560; {isBeta ? '6 ConvNeXt blocks, 384 bins, band width 48. The beta missing post-sum activation is deliberately preserved.' : '9 ConvNeXt blocks, 448 bins, band width 4.'}</li>
              <li><b className="text-slate-200">PhoneExtractor</b>: {isBeta ? '6 strided convolutions, 3 recurrent GRU layers, 8 ConvNeXt blocks. Explicit persistent states; no VQ/RMS normalization.' : '6 strided convolutions, 20 attention blocks, speaker codebook VQ and RMS normalization in a bounded context.'}</li>
              <li><b className="text-slate-200">Conditioning</b>: {isBeta ? 'Reflect-aligned energy/pitch features, separate speaker and formant embeddings. The synthesis F0 uses unshifted-in-time pitch.' : 'Speaker/codebook/key-value embeddings plus pitch and formant controls.'}</li>
              <li><b className="text-slate-200">Vocoder</b>: {isBeta ? 'Stateful prenet and weight-standardized IR/noise/filter generators; intermediate width 768, no speaker cross-attention.' : 'Cross-attention prenet and IR/noise/filter generators.'}</li>
              <li><b className="text-slate-200">Synthesis</b> (JS, float64 phase): pitch-synchronous 512-tap overlap-add, rectangular analysis/Hann synthesis noise, 768-point post filter with the 120-sample offset.</li>
            </ol>
            <p className="text-xs text-slate-500 mt-2">{isBeta ? 'beta.2 was numerically compared with the pinned official PyTorch fused-FP16 reference using both downloadable voices. WASM inference is tested; mobile GPU/native VST parity is not certified.' : 'The existing rc.0 path remains available. A matching file size is not evidence of complete numerical or timing parity.'} RVC v2 is a different model family and is not implemented here.</p>
          </Card>
          <Card title="Log" icon={<Menu size={15} />}>
            <pre className="text-[11px] leading-4 text-slate-400 max-h-44 overflow-auto whitespace-pre-wrap">{logs.join("\n") || "—"}</pre>
          </Card>
          <footer className="text-[11px] text-slate-500 pb-4">App source: <a className="underline" href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank" rel="noreferrer">GPLv3</a>. Voice models retain their own terms and credits.<details className="mt-2"><summary className="cursor-pointer">Beatrice attribution and license</summary><pre className="whitespace-pre-wrap mt-2">{BEATRICE_NOTICE}</pre></details></footer>
        </main>
      </div>
    </div>
  );
}

// ================================================================ small components
function Card({ title, icon, children, accent }: { title: string; icon?: React.ReactNode; children: React.ReactNode; accent?: boolean }) {
  return (
    <section className={`rounded-xl border bg-slate-950/60 p-3.5 ${accent ? "border-sky-900/60" : "border-slate-800"}`}>
      <h2 className="text-sm font-semibold text-slate-300 mb-2.5 flex items-center gap-2">{icon}{title}</h2>
      {children}
    </section>
  );
}

function Side({ children, side, visible, onClose }: { children: React.ReactNode; side: "left" | "right"; visible: boolean; onClose: () => void }) {
  const cls = `z-40 bg-slate-950/95 border-slate-800 overflow-y-auto overscroll-contain p-3 space-y-4 shrink-0 fixed top-16 bottom-0 w-[88vw] max-w-[360px] xl:static xl:block xl:w-[300px] xl:h-full xl:max-w-none xl:bg-transparent ${visible ? 'hidden' : 'block drawer-enter'} ${side === 'left' ? 'left-0 border-r xl:order-1' : 'right-0 border-l xl:order-3'}`;
  return (
    <>
      {!visible && <button className="xl:hidden fixed inset-0 bg-black/50 z-30" onClick={onClose} aria-label="Close sidebar" />}
      <aside className={cls} aria-label={side === 'left' ? 'Models and runtime' : 'Voice settings'}>
        <button className="btn-sm xl:hidden" onClick={onClose}><X size={16} /> Close</button>
        {children}
      </aside>
    </>
  );
}

function DropZone({ children, onFile }: { children: React.ReactNode; onFile: (name: string, data: ArrayBuffer) => void }) {
  const [over, setOver] = useState(false);
  return (
    <div
      className={`rounded-lg border border-dashed p-2.5 transition-colors ${over ? "border-sky-400 bg-sky-950/20" : "border-slate-800 bg-slate-950/50"}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={async (e) => {
        e.preventDefault(); setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f.name, await f.arrayBuffer());
      }}
    >{children}</div>
  );
}

function Waveform({ pcm }: { pcm: Float32Array }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const g = c.getContext("2d")!; const W = c.width, H = c.height;
    g.fillStyle = "#0b1020"; g.fillRect(0, 0, W, H);
    g.fillStyle = "#38bdf8";
    const n = pcm.length, per = Math.max(1, Math.floor(n / W));
    for (let x = 0; x < W; x++) {
      let mn = 1, mx = -1;
      for (let i = x * per; i < Math.min(n, (x + 1) * per); i++) { const v = pcm[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      const y0 = H / 2 - mx * H * 0.45, y1 = H / 2 - mn * H * 0.45;
      g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
    }
  }, [pcm]);
  return <canvas ref={ref} width={600} height={44} className="w-full rounded bg-[#0b1020]" />;
}

function Badge({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return <span className={`px-2 py-1 rounded-full border ${ok ? "border-emerald-700 text-emerald-300" : "border-amber-700 text-amber-300"}`}>{children}</span>;
}
function Stat({ k, v }: { k: string; v: string }) {
  return <div className="bg-slate-900/60 rounded px-2 py-1"><div className="text-slate-500 text-[10px]">{k}</div><div className="font-mono text-slate-200">{v}</div></div>;
}
function StatusPill({ s }: { s: string }) {
  const cls = s === "verified" ? "bg-emerald-900 text-emerald-200" : "bg-amber-900 text-amber-200";
  return <span className={`px-2 py-0.5 rounded ${cls}`}>{s}</span>;
}
/** Chrome-only live JS heap meter (green/yellow/red). Reload a few times and watch whether
 *  used/max still climbs — if it stays flat, the per-reload WASM-arena stacking is gone. */
function HeapMeter() {
  const supported = typeof performance !== 'undefined' && !!(performance as unknown as { memory?: unknown }).memory;
  const [s, setS] = useState<{ used: number; total: number; limit: number; max: number } | null>(null);
  useEffect(() => {
    if (!supported) return;
    let max = 0;
    const read = () => {
      const m = (performance as unknown as { memory: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
      max = Math.max(max, m.usedJSHeapSize);
      setS({ used: m.usedJSHeapSize, total: m.totalJSHeapSize, limit: m.jsHeapSizeLimit, max });
    };
    read();
    const t = setInterval(read, 1000);
    return () => clearInterval(t);
  }, [supported]);
  if (!supported) return <p className="text-[10px] text-slate-600 mt-2">JS heap meter: not exposed by this browser (Chrome-only performance.memory). Open in Chrome to watch reload memory growth.</p>;
  const pct = s ? s.used / s.limit : 0;
  const bar = pct < 0.5 ? 'bg-emerald-400' : pct < 0.75 ? 'bg-amber-400' : 'bg-red-500';
  const label = pct < 0.5 ? 'text-emerald-300' : pct < 0.75 ? 'text-amber-300' : 'text-red-300';
  const mb = (n: number) => (n / 1048576).toFixed(0);
  return (
    <div className="mt-3">
      <div className="flex justify-between text-[10px] text-slate-500 mb-1">
        <span>JS heap (Chrome, reload watch)</span>
        {s && <span className={`font-mono ${label}`}>{mb(s.used)} MB used · max {mb(s.max)} MB · limit {mb(s.limit)} MB</span>}
      </div>
      <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
        <div className={`h-full transition-all ${bar}`} style={{ width: `${Math.min(100, pct * 100)}%` }} />
      </div>
    </div>
  );
}

function Slider({ label, unit, min, max, step, value, onChange }: { label: string; unit: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void }) {
  return (
    <label className="block mb-2.5 text-xs">
      <div className="flex justify-between text-slate-400 mb-0.5"><span>{label}</span><span className="font-mono text-slate-300">{unit}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-sky-400 h-6" />
    </label>
  );
}
