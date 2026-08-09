---
description: Usage, support boundaries, paper, and source links for Custom kernels.
---

# Custom kernels

Select registered Triton or CUDA operators on modules that expose the general kernel protocol.

## Use

```python
result = model.apply_optimization_plan(
    "custom-kernels",
    mode="inference",
)
print(result.manifest())
```

## Support

| Property | Value |
| --- | --- |
| Availability | Registered public pass: `custom-kernels` |
| Fidelity | Operator-equivalent intent; validate dtype-specific tolerances |
| Runtime | CPU, CUDA, or MPS; accelerated backends require CUDA |
| Registry name | `custom-kernels` |
| Pass ID | `custom-kernels` |
| Pass version | `1` |
| Restore | `model.restore_optimization_plan(mode="inference")` |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/10.1145/3315508.3329973)
- **Upstream GitHub:** [Triton](https://github.com/triton-lang/triton)
- **VoiceHub source:** [VoiceHub implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/optimization/accelerators.py)

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/optimization-overview.md) and
[optimization API](../reference/api.md#optimization).
