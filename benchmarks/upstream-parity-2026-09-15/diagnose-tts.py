import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

root = Path('/home/kadir/kadir_projects/github/voicehub')
cache = root / '.cache/upstream-parity'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


worker = module('parity_worker', root / 'scripts/upstream_parity_worker.py')
runner = module('parity_runner', root / 'scripts/benchmark_upstream_parity.py')
side, name, request_path = sys.argv[1:]
request = json.loads(Path(request_path).read_text())
dest = cache / 'tts-diagnostics' / name / side
dest.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
with runner.execution_lock(cache / 'execution.lock'):
    infer = worker.prepare(request, side)
    cells = dict(zip(infer.__code__.co_freevars, [c.cell_contents for c in infer.__closure__]))
    captures = {}
    if name == 'echo':
        target = importlib.import_module(
            'inference' if side == 'upstream' else 'voicehub.models.echo.sampling')
        original = target.ae_decode

        def decode(codec, pca, z):
            captures['latents'] = z.detach().float().cpu().numpy()
            out = original(codec, pca, z)
            captures['full_audio'] = out.detach().float().cpu().numpy()
            return out

        target.ae_decode = decode
    else:
        wrapper = cells['model']
        language_model = wrapper if side == 'upstream' else wrapper.model
        codec = cells['codec'] if side == 'upstream' else wrapper.codec
        generate = language_model.generate

        def tracked_generate(*args, **kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            captures['input_ids'] = ids.detach().cpu().numpy()
            out = generate(*args, **kwargs)
            seq = out if isinstance(out, torch.Tensor) else out.sequences
            captures['generated_ids'] = seq.detach().cpu().numpy()
            return out

        language_model.generate = tracked_generate
        method = 'decode' if side == 'upstream' else 'decode_code'
        decode = getattr(codec, method)

        def tracked_decode(codes, *args, **kwargs):
            captures['codes'] = codes.detach().cpu().numpy()
            return decode(codes, *args, **kwargs)

        setattr(codec, method, tracked_decode)
    random.seed(request['seed'])
    np.random.seed(request['seed'])
    torch.manual_seed(request['seed'])
    with torch.inference_mode():
        out = infer(request['samples'][0])
    for key, value in captures.items():
        np.save(dest / (key + '.npy'), value)
    np.save(dest / 'audio.npy', out['audio_array'])
    out.pop('audio_array')
    out['request_sha256'] = hashlib.sha256(Path(request_path).read_bytes()).hexdigest()
    out['worker_sha256'] = hashlib.sha256(
        (root / 'scripts/upstream_parity_worker.py').read_bytes()).hexdigest()
    (dest / 'report.json').write_text(json.dumps(out, indent=2) + '\n')
    if side == 'upstream':
        assert not any(k == 'voicehub' or k.startswith('voicehub.') for k in sys.modules)
print(name, side, 'diagnostic saved', flush=True)
