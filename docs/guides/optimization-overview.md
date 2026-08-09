---
description: Discover VoiceHub optimization passes and apply, report, and restore them through one speech-model lifecycle.
---

# Optimization overview

VoiceHub exposes one optimization lifecycle across TTS, ASR, and VAD. A pass
can target speed, memory, or both, but compatibility and output behavior depend
on the loaded model, execution mode, hardware, dtype, and input shape.

Use the same explicit lifecycle for every registered model:

```python
def run_optimization(model, pass_name):
    print(model.available_optimization_passes())
    result = model.apply_optimization_plan(pass_name, mode="inference")
    print(model.optimization_manifest(mode="inference"))
    model.restore_optimization_plan(mode="inference")
    return result
```

The list reports global discovery, not compatibility with one runtime;
validation happens before mutation, a failure rolls back earlier reversible
passes, and the manifest records what was actually applied.

| Technique | Public pass | Evidence boundary |
| --- | --- | --- |
| [Compilation](../optimizations/compile.md) | `compile` | The concrete graph, mode, device, dtype, and fixed-seed evidence decide support. |
| [Attention backends](../optimizations/flash-attention-4.md) | `flash-attention-4` | Requires a compatible attention surface and optional CUDA backend. |
| [General kernels](../optimizations/custom-kernels.md) | `custom-kernels` | Uses registered kernels only after runtime validation. |
| [Codec kernels](../optimizations/codec-kernels.md) | `codec-kernels` | Applies only to discovered codec operations and records each selected backend. |
| [Diffusion caching](../optimizations/diffusion-cache.md) | `diffusion-cache` | Approximate reuse may change generated audio. |
| [Diffusion sampling](../optimizations/diffusion-sampling.md) | `diffusion-sampling` | Step, guidance, or solver changes may change generated audio. |

## Compilation

Compilation can reduce Python overhead and fuse operations after an initial
warm-up. VoiceHub checks the model-owned compile targets and preserves the
original runtime for deterministic restoration. Models with failed
real-checkpoint equivalence stay eager when automatic selection is requested
and reject an explicit requirement.

## Attention backends

An attention backend is useful only when the loaded architecture exposes the
required attention protocol. VoiceHub does not select one from a provider
name. The pass validates the concrete call, optional dependency, CUDA device,
dtype, and execution mode before changing the runtime.

## Kernels

General and codec-specific kernel passes resolve registered implementations by
capability. They do not build an extension implicitly, and they retain
canonical state-dict keys. Explicit unsupported backends fail instead of
silently using a different implementation.

## Diffusion caching

Diffusion caching reuses intermediate work across related sampling steps. It
is architecture- and schedule-sensitive and can trade quality for latency or
memory. Keep it disabled when the model has no retained checkpoint evidence or
when exact output behavior is required.

## Diffusion sampling

Sampling optimization can reduce steps, alter guidance, or select a different
solver. These are semantic changes rather than guaranteed-equivalent kernels.
Record the complete configuration and compare audio quality against the eager
baseline before retaining a policy.

## Boundaries

VoiceHub currently has no registry-wide public quantization pass. Quantized
checkpoint formats and provider-local loaders therefore remain model-specific
and are not advertised as universal optimization support.

[HQQ](../optimizations/hqq.md), [GemLite](../optimizations/gemlite.md), and
[audio.cpp](../optimizations/audio-cpp.md) have dedicated source-install pages.
They remain opt-in until a model meets the public pass lifecycle and evidence
contract.

Parallelism is a training or serving topology, not a reversible model pass.
Continuous batching belongs to a serving scheduler, not the model-mutation
lifecycle. Both stay outside this registry until they have a model-independent
public contract and complete coverage.

## Next steps

- Follow the [TTS optimization workflow](tts-optimization.md) for fair
  performance and quality comparisons.
- Review [optional source backends](optional-backends.md) for HQQ, GemLite, and
  audio.cpp boundaries.
- Use the [codec guide](codec-optimization.md) for operation-level kernel
  selection.
- Use the [diffusion guide](diffusion-optimization.md) for approximate cache
  and sampling policies.
- Inspect the shared [optimization API](../reference/api.md#optimization) and
  [contribution contract](../project/adding-an-optimization.md).
