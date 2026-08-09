---
description: Usage, support boundaries, paper, and source links for Torch compile.
---

# Torch compile

Compile model-owned execution methods while preserving checkpoint keys and reversible eager fallbacks.

## Use

```python
print(model.available_optimization_passes())
result = model.apply_optimization_plan("compile", mode="inference")
print(result.manifest())
model.restore_optimization_plan(mode="inference")
```

## Support

| Property | Value |
| --- | --- |
| Availability | Registered public pass: `compile` |
| Fidelity | Exact intent; verify numerical and audio equivalence for the concrete graph |
| Runtime | CPU or CUDA; float32, float16, or bfloat16 |
| Registry name | `compile` |
| Pass ID | `torch.compile` |
| Pass version | `1` |
| Restore | `model.restore_optimization_plan(mode="inference")` |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://pytorch.org/assets/pytorch2-2.pdf)
- **Upstream GitHub:** [PyTorch](https://github.com/pytorch/pytorch)
- **VoiceHub source:** [VoiceHub implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/optimization/torch_compile.py)

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/optimization-overview.md) and
[optimization API](../reference/api.md#optimization).
