---
description: Install experimental audio and quantization backends from source.
---

# Optional source backends

These projects are opt-in. VoiceHub does not report them as applied public passes
until a model has reversible validation and real-checkpoint evidence.

## HQQ and GemLite

[HQQ](../optimizations/hqq.md) quantizes eligible `nn.Linear` weights.
[GemLite](../optimizations/gemlite.md) provides compatible low-bit matrix
kernels. Install pinned source revisions:

```bash
python -m pip install \
  "hqq @ git+https://github.com/dropbox/hqq.git@d88a488ec8aa2d58362ef2038a52bca862db2e74" \
  "gemlite @ git+https://github.com/dropbox/gemlite.git@3dc52c3115fee49a09d00fd9e470ef6396885949"
```

Use them only on supported linear layers. They do not replace convolution
kernels used by many speech decoders.

## audio.cpp

[audio.cpp](../optimizations/audio-cpp.md) is a separate C++/GGML
runtime, not a Python optimization pass. Build its CLI from a pinned checkout:

```bash
git clone https://github.com/0xShug0/audio.cpp.git
cd audio.cpp
git checkout 748c5e28f6a7228b8f38ad7142ca97d29584544b
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target audiocpp_cli -j 8
```

Keep conversion, checkpoint compatibility, and runtime output checks outside
VoiceHub until an adapter implements the full optimization lifecycle.

## vLLM and SGLang

[vLLM](../optimizations/vllm.md) and
[SGLang](../optimizations/sglang.md) run as separate HTTP servers. Their
CUDA and PyTorch stacks stay outside the VoiceHub environment.

```python
from voicehub.llm_serving import list_llm_backend_support

for support in list_llm_backend_support():
    print(support.model_type, support.backend.value)
```

Use only a listed model/backend pair. VoiceHub does not silently fall back to
native inference.

## Validation

Before publishing a backend result, record the model, checkpoint revision,
device, dtype, input, warm-up, latency, memory, and audio-quality comparison.
See the [optimization workflow](tts-optimization.md).
