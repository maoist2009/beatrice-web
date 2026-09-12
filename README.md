# Beatrice Web

A static, browser-only Beatrice voice converter. The application source is GPL-3.0-only;
upstream MIT notices and model-specific terms remain applicable. See `LICENSE` and
the in-app notices. No audio is uploaded and there is no inference server.

## Supported Models

- Beatrice 2.0.0-rc.0: incremental convolutions and bounded attention KV state, 128 phone channels, VQ and speaker cross-attention.
- Beatrice 2.0.0-beta.2: new incremental implementation, 256 phone channels, three GRU layers,
  384 pitch bins and the original 48-bin pitch sampling band. No VQ, channel RMS norm or cross-attention.
- The beta.2 Trainer exports `model.version = "2.0.0-beta.1"`. This is a paraphernalia format version,
  not a reason to rename the file or pretend it is rc.0. Both beta.1/beta.2 labels route to the verified
  beta.2 layout. Older beta.0 and arbitrary architecture modifications are rejected.

**RVC v2 is a different architecture and is not implemented.** The beta.2 packages discussed here
belong to Beatrice, not RVC. `.pth`/`.index` files cannot be converted by renaming them.

The catalog includes pinned HuggingFace ZIP URLs and SHA-256 hashes for:

- `hecko/beatrice-old-tts` (8 voices, rc.0).
- `wok000/vcclient_model/beatrice_v2_rc0`: 100 JVS corpus voices. This is NOT the separate
  Official Model 1 containing Tsukuyomi-chan, Tokina Shigure and OLUNE; the previous catalog label was wrong.
- `yasyune/Shigure_Tokina_Beatrice_2.0.0-beta.2`: Tokina Shigure / 刻鳴時雨, CV 丸ころ, manager 瓶詰め.
- `yasyune/Kurage_Kikoto_Beatrice_2.0.0-beta.2`: Kikoto Kurage / 黄琴海月.

The browser downloads the original files directly, including model cards and credit links. Other public
HF paraphernalia ZIP URLs can be entered manually. Downloads are limited to 160 MiB and expanded packages
to 192 MiB to limit memory spikes. Models must still match a supported architecture.

Cached models can be exported without changing weight bytes. Beta exports include the required separate
`formant_shift_embeddings.bin`; license/credit text is retained. Cache listings read a small metadata
store instead of cloning every model into memory.

## Using It

1. Get a catalog model, or import a supported folder/ZIP.
2. Select a backend and chunk size, then choose **Build & compile**.
3. Upload an MP3/WAV, or record a short clip; choose **Convert** and download the result as WAV.
4. For live mode, use headphones, a secure origin (HTTPS or localhost), and grant microphone permission.

Both paths keep convolution states; beta.2 also keeps GRU states, rc.0 keeps a bounded attention KV cache.
The context selector only changes rc.0 attention memory. Chunk size is **not** end-to-end latency.
Use an 80-200 ms chunk and 0.64 s attention memory as a conservative starting point. No Adreno 6/7/830
hardware stability or latency measurements have been performed in this environment.

The entire inference engine runs in a dedicated, terminable Web Worker. WebGPU uses the primitive GRU
decomposition; WASM uses the native GRU operator. Graph capture is OFF in all modes. Bounded GPU I/O
keeps recurrent state on GPU without capture; compatibility mode runs the network with CPU-backed I/O.
Initialization failures fall back to compatibility WebGPU and then WASM. A watchdog and the explicit
**Stop / release Worker** action allow recovery from stuck kernels without a JS-main-thread deadlock.

## Stability Fixes

The former idle `pump().finally(() => pump())` loop re-created microtasks even when no audio was ready.
A subprocess reproduced this without a model or GPU: a 20 ms timer never fired within 1500 ms.
`CooperativePump` now checks readiness before allocating a task and runs one chunk per macrotask.
An overrun pauses inference instead of resetting/rebasing inconsistent counters in a retry loop.

Sessions retain I/O metadata only, not serialized ONNX weight buffers. ModelProto serialization is flattened
once. UI model previews do not eagerly decode large speaker embeddings; only the worker does that.
Shared portraits use one object URL (the JVS pack uses the same portrait for 100 voices) and images load
lazily instead of creating 100 independently decoded image resources.
GPU allocations are tracked immediately and rolled back on partial failure. Runs cannot overlap, disposal
waits for in-flight work, and user-owned GPU buffers are destroyed after session release. Graph capture
is not used. A 32 MiB cap applies to app-owned GPU I/O, **not** total ORT/driver VRAM.

Input messages are credit-limited; the playback ring is capped at 100 audio frames. Output workspaces
are reused, files are capped at 60 seconds, and diagnostic traces are bounded. **Save diagnostics** records
the current stage, timings and tracked allocations without including the user's audio/model bytes.
A bounded PCM snapshot is taken before each asynchronous live inference step so that a slow GPU cannot
read overwritten data after the live input ring wraps. No progress for 30 seconds triggers worker
termination (120 seconds during initialization). A tiny session-storage breadcrumb records the last stage
if a renderer dies before it can return an error.

The 13% device crash is not conclusively diagnosed as OOM without device crash/memory logs. The verified
scheduling bug and resource-lifetime defects were fixed; do not read passing desktop tests as a guarantee
against browser/driver OOM on all phones.

## Frequently asked

**rc.1 / rc.2 / rc.3?** Those are VST releases, not model formats. The trainer still writes
`PARAPHERNALIA_VERSION = "2.0.0-rc.0"` (repo HEAD `f34836d`), so the rc.3 official models
(つくよみちゃん / 刻鳴時雨 / OLUNE) are rc.0 files and load with the rc.0 path here. rc.2's
formant fix was host-side; this app already indexes formant embeddings in half-semitone steps.

**Why is Chinese poor?** Mandarin performance is not validated. Upstream documents Japanese data for
content recognition and English synthesis pretraining, and the listed target voices are Japanese/English.
Training a new speaker does not automatically fix language coverage in the frozen extractor.

**Is the GRU stuck on CPU?** No. The WebGPU path uses MatMul/Sigmoid/Tanh/Mul/Add instead of a native GRU
node. WASM uses native GRU. Operator support alone does not imply a speedup; dispatch and driver costs
must be measured on the actual device.

## Deployment Size

Deploy **only `dist/`**. The production build is one HTML file. Models, ONNX graphs and ORT WASM are not
base64-inlined into it: models are downloaded/cached on demand and ONNX graphs are constructed in the
browser. The runtime's matching JS/WASM version is loaded from jsDelivr.

`tools/audit-dist.mjs` checks a 512 KiB uncompressed HTML budget and rejects unexpected assets.
`tools/`, `docs/`, `.research/`, Python environments, test audio, Playwright binaries and model fixtures
are not imported by the frontend and must never be moved to `public/`. Tailwind also excludes the
research/test directories from class scanning. The normal production build does not run/download tests.

## Reproducible Validation

Run the optional development scripts separately, from the repository root:

1. `node tools/research-beta.mjs`: invokes **curl** with pinned source/model revisions; checks archive hashes.
2. `node tools/setup-reference.mjs`: installs CPU-only PyTorch/NumPy/ONNX/ORT under ignored `.research/`.
3. `node tools/run-beta-tests.mjs`: executes the actual TS graph builders and runtime against the pinned
   upstream PyTorch classes loaded with real FP16 weights. Also tests FFT, pitch DSP, cache and ZIP export.
4. `node tools/setup-browser.mjs`: optional headless Chromium setup, kept under `.research/`.
5. After the production build, `node tools/browser-smoke.mjs` tests the built HTML, real HF download,
   WAV conversion, cache reload, export, and desktop/mobile layout. Static HTML is intercepted in the
   test browser; no application server code is added.
6. `node tools/audit-dist.mjs` checks deployment contents and size.
7. `node tools/run-stability-unit.mjs` tests the production scheduler and session lifetime with fault injection.
8. `node tools/browser-stability.mjs` tests repeated one-second files, live UI responsiveness and worker release
   against the built app. `TEST_GPU=1` selects headless Chromium software WebGPU; it is not Adreno hardware.

`docs/beta2-validation.json` records provenance and measured results. Graph tests use chunks of 1, 4 and
20 frames and real speech with both downloadable beta models. Synthesis comparisons replay the same
initial phase and noise stream, because the official model is stochastic. This validates the fused
FP16 export path, not the proprietary native library or all possible hardware.

The tests are development tools, not browser polyfills. Their dependencies are not bundled into `dist`.
After preserving the validation report, `node tools/clean-research.mjs` removes the ignored development
downloads, model fixtures, Python libraries and test browser. Subsequent validation can recreate them.