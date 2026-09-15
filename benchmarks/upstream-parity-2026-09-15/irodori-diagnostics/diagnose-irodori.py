import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/kadir/kadir_projects/github/voicehub/scripts')
from upstream_parity_worker import prepare

side, request_path, output_path = sys.argv[1:]
request = json.loads(Path(request_path).read_text())
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
infer = prepare(request, side)
closure = inspect.getclosurevars(infer).nonlocals
runtime = closure['runtime'] if side == 'upstream' else closure['model'].model
captured = {}


def save(key, value):
    if torch.is_tensor(value):
        captured[key] = {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'values': value.detach().float().cpu().flatten()[:100000].clone()
        }
    elif isinstance(value, (tuple, list)):
        for i, part in enumerate(value):
            save(key + '.' + str(i), part)


def hook(name):

    def capture(module, args, kwargs, out):
        if name + ':seen' not in captured:
            captured[name + ':seen'] = True
            save(name + ':input', args)
            for key, value in kwargs.items():
                save(name + ':kw:' + key, value)
            save(name + ':output', out)

    return capture


targets = ('text_encoder', 'text_norm', 'blocks.0', 'duration_predictor')
for name, module in runtime.model.named_modules():
    if name and (name.startswith(targets) or '.' not in name):
        module.register_forward_hook(hook(name), with_kwargs=True)
original_decode = runtime.codec.decode_latent


def decode(*args, **kwargs):
    save('codec:latent', args)
    out = original_decode(*args, **kwargs)
    save('codec:output', out)
    return out


runtime.codec.decode_latent = decode
random.seed(request['seed'])
np.random.seed(request['seed'])
torch.manual_seed(request['seed'])
with torch.inference_mode():
    result = infer(request['samples'][0])
save('audio:final', torch.as_tensor(result['audio_array']))
torch.save(captured, output_path)
print(side, len(captured), output_path)
