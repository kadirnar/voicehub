---
description: Usage, support boundaries, paper, and source links for Diffusion cache.
---

# Diffusion cache

Reuse architecture-owned diffusion block residuals within one isolated generation request.

## Use

```python
result = model.apply_optimization_plan(
    "diffusion-cache",
    mode="inference",
)
print(result.manifest())
```

## Support

| Property | Value |
| --- | --- |
| Availability | Registered public pass: `diffusion-cache` |
| Fidelity | Approximate; generated audio may change |
| Runtime | CPU, CUDA, or MPS; inference only |
| Registry name | `diffusion-cache` |
| Pass ID | `voicehub.diffusion-block-cache` |
| Pass version | `1` |
| Restore | `model.restore_optimization_plan(mode="inference")` |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858); [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://arxiv.org/abs/2411.19108)
- **Upstream GitHub:** [DeepCache](https://github.com/horseee/DeepCache); [TeaCache](https://github.com/ali-vilab/TeaCache)
- **VoiceHub source:** [VoiceHub implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/optimization/diffusion_cache.py)

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/diffusion-optimization.md) and
[optimization API](../reference/api.md#optimization).
