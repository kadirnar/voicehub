---
description: Usage, support boundaries, paper, and source links for Diffusion sampling.
---

# Diffusion sampling

Reduce model evaluations through explicit schedule, guidance, cache, or solver policies.

## Use

```python
result = model.apply_optimization_plan(
    "diffusion-sampling",
    mode="inference",
)
print(result.manifest())
```

## Support

| Property | Value |
| --- | --- |
| Availability | Registered public pass: `diffusion-sampling` |
| Fidelity | Approximate; generated audio may change |
| Runtime | CPU, CUDA, or MPS; inference only |
| Registry name | `diffusion-sampling` |
| Pass ID | `voicehub.diffusion-sampling` |
| Pass version | `1` |
| Restore | `model.restore_optimization_plan(mode="inference")` |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling](https://arxiv.org/abs/2206.00927); [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **Upstream GitHub:** [DPM-Solver](https://github.com/LuChengTHU/dpm-solver)
- **VoiceHub source:** [VoiceHub implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/optimization/diffusion_sampling.py)

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/diffusion-optimization.md) and
[optimization API](../reference/api.md#optimization).
