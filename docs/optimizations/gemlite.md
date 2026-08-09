---
description: Usage, support boundaries, paper, and source links for GemLite.
---

# GemLite

Run compatible low-bit matrix kernels; this is not yet a VoiceHub public pass.

## Use

```bash
python -m pip install \
  "gemlite @ git+https://github.com/dropbox/gemlite.git@3dc52c3115fee49a09d00fd9e470ef6396885949"
```

## Support

| Property | Value |
| --- | --- |
| Availability | Optional library; no registered VoiceHub pass |
| Fidelity | Kernel- and quantization-dependent |
| Runtime | Supported CUDA GPUs |
| Registry name | Not registered; do not report this backend as an applied VoiceHub pass |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** [GemLite: Towards Building Custom Low-Bit Fused CUDA Kernels](https://dropbox.github.io/gemlite_blogpost/)
- **Upstream GitHub:** [GemLite](https://github.com/dropbox/gemlite)
- **VoiceHub source:** No VoiceHub pass implementation; use the external source project directly.

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/optional-backends.md) and
[optimization API](../reference/api.md#optimization).
