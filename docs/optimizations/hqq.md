---
description: Usage, support boundaries, paper, and source links for HQQ.
---

# HQQ

Quantize eligible linear weights without calibration data; this is not yet a VoiceHub public pass.

## Use

```bash
python -m pip install \
  "hqq @ git+https://github.com/dropbox/hqq.git@d88a488ec8aa2d58362ef2038a52bca862db2e74"
```

## Support

| Property | Value |
| --- | --- |
| Availability | Optional library; no registered VoiceHub pass |
| Fidelity | Quantized; measure task quality and generated audio |
| Runtime | Backend-dependent |
| Registry name | Not registered; do not report this backend as an applied VoiceHub pass |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** [Half-Quadratic Quantization of Large Machine Learning Models](https://dropbox.github.io/hqq_blog/)
- **Upstream GitHub:** [HQQ](https://github.com/dropbox/hqq)
- **VoiceHub source:** No VoiceHub pass implementation; use the external source project directly.

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/optional-backends.md) and
[optimization API](../reference/api.md#optimization).
