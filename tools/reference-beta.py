"""Development-only parity oracle. Loads real FP16 exports into the pinned upstream PyTorch classes.

No training checkpoint, random substitute weights, server, or modified architecture is used. Parameters
are loaded by intercepting the official dump traversal. Fused WS weights bypass re-standardization;
norm/gamma/scales omitted by the exporter are neutralized using upstream merge_weights().
"""
import ast
import gc
import io
import json
import math
import os
import types
import warnings
import wave
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import onnx
import onnxruntime as ort

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
warnings.filterwarnings('ignore', category=FutureWarning)
root = Path('.research')
tree = ast.parse((root / 'beta2.py').read_text())
keep = {
    'dump_params', 'dump_layer', 'CausalConv1d', 'WSConv1d', 'WSLinear', 'ConvNeXtBlock', 'ConvNeXtStack',
    'FeatureExtractor', 'FeatureProjection', 'PhoneExtractor', 'PitchEstimator', 'Vocoder', 'ConverterNetwork',
    'extract_pitch_features', 'overlap_add', 'generate_noise', 'GradientEqualizerFunction',
}
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
nodes += [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in keep]
module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
scope = dict(torch=torch, nn=nn, F=F, np=np, math=math, warnings=warnings, os=os, partial=partial,
             weight_norm=weight_norm, remove_weight_norm=remove_weight_norm,
             torchaudio=types.SimpleNamespace(transforms=types.SimpleNamespace(MelSpectrogram=lambda **kw: nn.Identity())))
exec(compile(module, 'official-beta2-inference.py', 'exec'), scope)

def load_dump(model, path):
    values = np.fromfile(path, dtype='<f2').astype(np.float32)
    offset = 0
    original = scope['dump_params']
    def load(param, f):
        nonlocal offset
        if param is None:
            return
        count = param.numel()
        if offset + count > len(values):
            raise AssertionError('Official dump traversal exceeds the model file')
        with torch.no_grad():
            param.copy_(torch.from_numpy(values[offset:offset + count].reshape(tuple(param.shape))))
        offset += count
    scope['dump_params'] = load
    try:
        model.dump(io.BytesIO())
    finally:
        scope['dump_params'] = original
    assert offset == len(values), (path, offset, len(values))
    for layer in model.modules():
        if isinstance(layer, (scope['WSConv1d'], scope['WSLinear'])):
            layer.standardized_weight = types.MethodType(lambda self: self.weight, layer)

def rng(seed=12345):
    def draw(*shape, **kwargs):
        nonlocal seed
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        a = np.empty(math.prod(shape), dtype=np.float32)
        for i in range(len(a)):
            seed = (1664525 * seed + 1013904223) & 0xffffffff
            a[i] = seed / 4294967296
        return torch.from_numpy(a.reshape(shape)).to(kwargs.get('device', 'cpu'))
    return draw

with wave.open(str(root / 'jfk.wav'), 'rb') as wav:
    assert wav.getsampwidth() == 2 and wav.getframerate() == 16000
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
    if wav.getnchannels() > 1:
        pcm = pcm.reshape(-1, wav.getnchannels()).mean(1)
length = 233  # non-multiple of all tested chunk sizes; exercises startup, GRU carry, ring wrap and final crop
pcm = pcm[6000:6000 + length * 160].copy()
assert len(pcm) == length * 160
x = torch.from_numpy(pcm)[None, None, :]

def stats(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    d = a.astype(np.float64) - b
    return dict(max_abs=float(np.max(np.abs(d))), rms_relative=float(np.sqrt(np.mean(d * d)) / max(1e-12, np.sqrt(np.mean(b.astype(np.float64) ** 2)))))

results = []
for name in ['shigure', 'kurage']:
    folder = root / name
    phone = scope['PhoneExtractor']().eval()
    phone.remove_weight_norm(); phone.merge_weights(); load_dump(phone, folder / 'phone_extractor.bin')
    pitch = scope['PitchEstimator']().eval()
    pitch.merge_weights(); load_dump(pitch, folder / 'pitch_estimator.bin')
    net = scope['ConverterNetwork'](phone, pitch, 1, 256).eval()
    net.merge_weights(); load_dump(net, folder / 'waveform_generator.bin')
    with torch.no_grad():
        net.embed_speaker.weight.copy_(torch.from_numpy(np.fromfile(folder / 'speaker_embeddings.bin', dtype='<f2').astype(np.float32).reshape(1, 256)))
        net.embed_formant_shift.weight.copy_(torch.from_numpy(np.fromfile(folder / 'formant_shift_embeddings.bin', dtype='<f2').astype(np.float32).reshape(9, 256)))
    golden = {}
    def capture(key):
        def hook(_m, _i, output): golden[key] = output.detach().clone()
        return hook
    def voc_in(_m, inputs):
        golden['embedding'] = inputs[0].detach().clone()
        golden['f0'] = inputs[1].detach().clone()
    handles = [net.vocoder.register_forward_pre_hook(voc_in)]
    for key, layer in [('ir', net.vocoder.ir_generator_post), ('aperiodicity', net.vocoder.aperiodicity_generator_post), ('post_filter', net.vocoder.post_filter_generator_post)]:
        handles.append(layer.register_forward_hook(capture(key)))
    with torch.inference_mode():
        golden['phone'] = phone(x, return_stats=False)
        golden['pitch'], golden['energy'] = pitch(x)
        golden['instfreq'], golden['corr_diff'], _ = scope['extract_pitch_features'](x.squeeze(1))
        golden['quantized'], golden['features'] = pitch.sample_pitch(golden['pitch'], return_features=True)
        orig_rand = torch.rand
        torch.rand = rng()
        try:
            golden['audio'] = net(x, torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.0]))
        finally:
            torch.rand = orig_rand
    for h in handles: h.remove()
    golden['input'] = x
    golden['ir_window'] = net.vocoder.ir_window.detach()
    outdir = root / f'golden-{name}'
    outdir.mkdir(exist_ok=True)
    manifest = {}
    for key, value in golden.items():
        a = value.detach().float().numpy()
        assert np.isfinite(a).all(), key
        a.tofile(outdir / f'{key}.f32')
        manifest[key] = list(a.shape)
    (outdir / 'shapes.json').write_text(json.dumps(manifest))
    print('[upstream reference]', name, 'samples', golden['audio'].numel(), 'RMS', golden['audio'].square().mean().sqrt().item(), flush=True)

    so = ort.SessionOptions(); so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for chunk in [1, 4, 20]:
        for stage in ['phone', 'phone-primitives', 'pitch', 'vocoder']:
            path = root / 'graphs' / f'{name}-{stage}-{chunk}'
            onnx.checker.check_model(str(path) + '.onnx')
            state_specs = json.loads(Path(str(path) + '.json').read_text())
            session = ort.InferenceSession(str(path) + '.onnx', so, providers=['CPUExecutionProvider'])
            outputs = [o.name for o in session.get_outputs()]
            state = {s['input']: np.zeros(s['dims'], dtype=np.float32) for s in state_specs}
            collected = {key: [] for key in (['phone'] if stage.startswith('phone') else ['pitch'] if stage == 'pitch' else ['ir', 'aperiodicity', 'post_filter'])}
            def cut(a, start, count):
                result = np.zeros((*a.shape[:-1], count), dtype=np.float32)
                lo, hi = max(0, start), min(a.shape[-1], start + count)
                if hi > lo: result[..., lo - start:hi - start] = a[..., lo:hi]
                return result
            for at in range(0, length, chunk):
                feeds = dict(state)
                if stage.startswith('phone'):
                    feeds['wav'] = cut(x.numpy(), at * 160 - 40, chunk * 160 + 80)
                elif stage == 'pitch':
                    feeds['instfreq'] = cut(golden['instfreq'].numpy(), at, chunk + 1)
                    feeds['corr_diff'] = cut(golden['corr_diff'].numpy(), at, chunk + 1)
                    feeds['valid'] = (np.arange(at, at + chunk + 1) < length).astype(np.float32)[None, None, :]
                else:
                    feeds['x'] = cut(golden['embedding'].numpy(), at, chunk + 2)
                result = dict(zip(outputs, session.run(None, feeds)))
                for s in state_specs: state[s['input']] = result[s['output']]
                for key in collected:
                    output = 'units' if key == 'phone' else 'logits' if key == 'pitch' else key
                    collected[key].append(result[output])
            for key, parts in collected.items():
                a = np.concatenate(parts, -1)[..., :length]
                b = golden[key].numpy()
                stage_label = f'{stage}/{key}' if stage != key else stage
                report = stats(a, b)
                peak = float(np.max(np.abs(b)))
                assert report['max_abs'] < max(8e-5, peak * 4e-4) and report['rms_relative'] < 2e-4, (name, chunk, key, report)
                results.append(dict(model=name, chunk=chunk, stage=stage_label, **report))
                print('[torch/onnx]', name, chunk, stage_label, json.dumps(report), flush=True)
            del session
            gc.collect()
    del net, phone, pitch, golden
    gc.collect()
(root / 'torch-onnx-results.json').write_text(json.dumps(results, indent=2))
print('REFERENCE PARITY PASS: official fused FP16 PyTorch -> actual TypeScript-generated stateful ONNX graphs', flush=True)