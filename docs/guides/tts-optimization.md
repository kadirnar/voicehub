---
description: Apply reversible, capability-checked TTS optimizations and measure their speed, memory, and quality.
---

# TTS optimization

Start with a correct eager result. Test one optimization at a time on the same
text, seed, device, dtype, and model revision.

Unsupported automatic policies use the model's native path and record the
decision. Unsupported explicit requirements raise an error.

## Inspect support

```python
from voicehub.optimization import get_tts_optimization_support, list_tts_optimization_support

support = get_tts_optimization_support("qwen3tts")
print(support.to_dict())

for item in list_tts_optimization_support():
    print(item.model_type, item.attention_implementations, item.kernel_backends)
```

This table covers automatic TTS selection. Explicit extensions validate the runtime directly; see [Add an optimization](../project/adding-an-optimization.md).

## Use a quality-preserving default

```python
from voicehub import AutoModelForTextToSpeech
from voicehub.optimization import TTSOptimizationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_type="qwen3tts",
    device="cuda",
    lazy_load=True,
)

config = TTSOptimizationConfig(
    attn_implementation="auto",
    kernel_backend="auto",
    compile="auto",
    diffusion_cache="disabled",
    diffusion_sampling="disabled",
)
result = model.optimize(config)
print(result.manifest())
```

Keep the manifest of requested, applied, and skipped changes with your results.

Restore the original runtime before testing another profile:

```python
model.restore_tts_optimization(mode="inference")
```

Calling restore without an active policy raises an error.

## Understand the controls

| Control | Values | Safe starting choice |
| --- | --- | --- |
| Attention | `auto`, `native`, `sdpa`, `flash_attention_4` | `auto` |
| Kernels | `auto`, `native`, `torch`, `triton`, `cuda_extension` | `auto` |
| Compile | `auto`, `required`, `disabled` | `auto` |
| Diffusion cache | `auto`, `required`, `disabled` | `disabled` |
| Diffusion sampling | `auto`, `required`, `disabled` | `disabled` |

`auto` stays native/eager unless retained real-checkpoint evidence explicitly
validates automatic use. `required` fails if preparation or execution fails.

NeuTTS, VITS, and VUI reject inference compilation because fixed-seed
real-checkpoint tests changed tokens or audio. Auto stays eager, explicit
compile raises, and training compilation remains available.

Diffusion caching, fewer sampling steps, and altered guidance schedules are
approximate techniques. They may change the audio, so leave them disabled
when quality regression is unacceptable. Explicit Triton or CUDA-extension
kernels can also have small numerical differences. Validate them before
production rather than assuming equal quality.

## Compare profiles fairly

Use a sample long enough to expose steady-state behavior:

```python
TEXT = """
VoiceHub keeps optimization experiments reproducible by using the same model,
prompt, seed, device, and precision for every run. This longer passage should
produce more than ten seconds of speech, which makes warm latency, real-time
factor, and peak memory easier to compare without letting startup overhead
dominate the result.
""".strip()
```

Measure the eager baseline, each technique alone, and the combined automatic
profile:

| Profile | Configuration |
| --- | --- |
| Eager baseline | Do not call `optimize()` |
| SDPA only | `attn_implementation="sdpa"`, native kernels, compile disabled |
| Compile only | Native attention and kernels, `compile="required"` |
| Automatic | Attention, kernels, and compile set to `auto` |

Skip a profile when `get_tts_optimization_support()` says it is unsupported.
Use separate model instances, or restore the active policy between profiles.

This compact helper reports latency, audio duration, real-time factor (RTF),
and allocated CUDA memory:

```python
import time
import torch


def benchmark_once(model, text):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    torch.manual_seed(7)
    started = time.perf_counter()
    output = model.generate(text, seed=7)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    sample_count = (
        output.audio.shape[-1]
        if hasattr(output.audio, "shape")
        else len(output.audio)
    )
    duration = sample_count / output.sample_rate
    peak_gib = (
        torch.cuda.max_memory_allocated() / 2**30
        if torch.cuda.is_available()
        else None
    )
    if duration < 10:
        raise RuntimeError(f"Expected at least 10 seconds, got {duration:.2f}")
    return {
        "latency_s": latency,
        "audio_s": duration,
        "rtf": latency / duration,
        "peak_allocated_gib": peak_gib,
    }
```

Run one untimed warm-up, then at least three measured repetitions. Compilation
usually makes the first call slower, so report cold and warm latency
separately. `max_memory_allocated()` is tensor memory, not total process or
reserved GPU memory.

Calculate changes against eager:

```python
speedup = eager["latency_s"] / candidate["latency_s"]
memory_drop_percent = (
    100
    * (eager["peak_allocated_gib"] - candidate["peak_allocated_gib"])
    / eager["peak_allocated_gib"]
)
```

Lower RTF is better; `RTF < 1` is faster than real time. Do not publish a
speedup without the model revision, hardware, software versions, precision,
audio duration, warm-up count, repetitions, and peak-memory definition.

## Check quality

Performance alone is not acceptance. Keep the eager audio and compare:

- intelligibility with the same ASR and word-error-rate calculation;
- speaker similarity when voice conditioning is used;
- loudness, clipping, duration, and silence;
- human listening with randomized, blinded samples.

Reject a profile if it produces invalid audio or a meaningful quality
regression. Small waveform differences do not by themselves prove a quality
change.

See the current [RTX 4090 speech benchmark report](rtx-4090-speech-benchmarks.md)
and the earlier [RTX 5090 TTS report](rtx-5090-tts-benchmarks.md) for measured
VoiceHub results. Re-run the procedure on your own hardware before choosing a
profile.

## Configure optimization during loading

```python
from voicehub import AutoModelForTextToSpeech
from voicehub.optimization import TTSOptimizationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    "F5TTS_v1_Base",
    model_type="f5tts",
    device="cuda",
    optimization_config=TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
    ),
)
```

With lazy loading, the policy is applied when the runtime loads. If runtime
validation fails, call `clear_optimization_config()` before loading natively.

## Inspect a plan without weights

```python
from voicehub.optimization import TTSOptimizationConfig
from voicehub.optimization import OptimizationContext

plan = TTSOptimizationConfig().resolve(
    "qwen3tts",
    mode="inference",
    context=OptimizationContext(
        mode="inference",
        device="cuda",
        dtype="bfloat16",
    ),
)
print(plan.manifest())
```

Resolution is side-effect free. Actual runtime compatibility is checked again
when the plan is applied.

## Training optimization

Runtime optimization does not make an inference-only artifact trainable.
Validate training support first, then use the model's source-backed profile.
See the [training guide](training.md), [training support matrix](../models/training-support.md),
[VITS guide](vits-optimization.md), [codec/LLM guide](codec-optimization.md),
and [diffusion guide](diffusion-optimization.md).

## Common failures

- Unsupported explicit backend: inspect support and use `auto` or `native`.
- First compiled call is slow: separate cold compilation from warm timings.
- CUDA out of memory: reduce batch or sequence size; do not hide it as a
  fallback.
- Invalid input or device errors: fix the input or environment. Automatic
  optimization does not swallow model errors.
- A second `optimize()` call fails: restore the active policy first.
