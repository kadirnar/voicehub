---
description: Usage, support boundaries, paper, and source links for FlashAttention-4.
---

# FlashAttention-4

Select FlashAttention-4 only on native attention modules that expose its validated policy surface.

## Use

```python
result = model.apply_optimization_plan(
    "flash-attention-4",
    mode="inference",
)
print(result.manifest())
```

## Support

| Property | Value |
| --- | --- |
| Availability | Registered public pass: `flash-attention-4` |
| Fidelity | Exact attention intent; validate backend tolerances on the target GPU |
| Runtime | CUDA with float16 or bfloat16 when the backend is required |
| Registry name | `flash-attention-4` |
| Pass ID | `flash-attention-4` |
| Pass version | `1` |
| Restore | `model.restore_optimization_plan(mode="inference")` |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- **Upstream GitHub:** [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- **VoiceHub source:** [VoiceHub implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/optimization/accelerators.py)

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/optimization-overview.md) and
[optimization API](../reference/api.md#optimization).
