# Official Beatrice Call Parity Report

Checked sources by git clone:

- Official VST: `vendor/prj-beatrice-vst`, commit `e66048ee1e533782184bc7732d477c3e662b4c3b`.
- Official trainer/exporter: `vendor/fierce-cats-beatrice-trainer`, commit `f34836de014b86956096878aecb8d3b17feaaa0b`.

## Conclusion

The web implementation is **not provably byte-for-byte identical to the official VST**.
The official VST calls a closed/prebuilt `beatrice.lib` (`Beatrice20rc0_*` / `Beatrice20b1_*`),
while the web app reconstructs the networks as ONNX graphs and executes them through ORT Web.
The repo's own validation says this explicitly: its reports validate the fused trainer/FP16
export path and mark native `beatrice.lib` parity as unverified.

## Matches Confirmed

- Model routing: `2.0.0-beta.1` -> beta.2 layout and `2.0.0-rc.0` -> rc.0 layout.
- Header constants: input hop 160 @ 16 kHz, output hop 240 @ 24 kHz, 96 pitch bins/octave,
  rc.0 128 phone channels / 448 pitch bins, beta.2 256 / 384.
- rc.0 call order: phone extraction, pitch estimation, official pitch arithmetic, then waveform
  generation; speaker additive/formant/KV data are supplied to the corresponding graph stages.
- beta.2 call order and feature alignment follow the trainer: phone lookahead, pitch/energy
  reflection alignment, 384-bin band width 48, 256-channel conditioning, and 24 kHz synthesis.
- Live audio's worklet resampler ports the official `AnyFreqInOut` topology, including the 480
  block, 2-in/3-out 80-sample bridge, 32-tap sinc and 0.99 cutoff factors.
- FP16 binary parsing and the exported weight layouts match the trainer dump order; the existing
  streaming tests compare incremental graphs to full-window ONNX references.

## Mismatches Fixed In This Build

- Official defaults are now used: average source pitch `52.0`, source pitch range `33.125..80.875`,
  and rc.0 VQ neighbors `0` (not trainer `vq_topk=4`).
- On model load, the web UI now keeps official source pitch `52.0` and initializes pitch shift as
  `clamp(TOML average_pitch - 52.0, -24, 24)`, matching the official `parameter_schema.cc` logic.
- UI parameter ranges now match the official schema: intonation `-1..3`, gains `-60..20 dB`,
  source pitch `0..128` with 1/8-unit steps.

## Remaining Non-Identity Areas

- The official native implementation owns its internal streaming contexts. The web rc.0 phone
  attention cache is an explicit selectable bound (`ctxFrames`); no native context length is exposed
  by the VST source, so native state parity cannot be asserted.
- Offline file conversion uses browser `OfflineAudioContext` resampling, not the official streaming
  sinc resampler. Live mode uses the closer worklet port; file mode therefore has different boundary
  samples even when the 16 kHz model calls are equivalent.
- Browser audio gain smoothing and VST `Gain::Context` smoothing are not yet identical sample for
  sample; this affects transitions, not the neural graph's weights or tensor order.
- Voice morphing, VST lock behavior, latency reporting, silence flags, and DAW state serialization
  are not implemented as native VST features in the browser UI.
- WebGPU/WASM changes numeric execution and scheduling; ORT graph/provider behavior cannot be called
  native-library identical without a golden native output comparison.

## What Would Prove Full Identity

For the same model archive, exact PCM input, exact parameter state, fixed synth RNG/phase, and the
same chunk boundaries, run the official VST/native `beatrice.lib` and web WASM path while capturing
phone, pitch, conditioning, vocoder tensors and final audio. Compare each tensor with declared
tolerances. Existing web tests do not perform this native-library comparison.