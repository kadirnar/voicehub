---
description: Usage, support boundaries, paper, and source links for audio.cpp.
---

# audio.cpp

Build a separate C++/GGML audio runtime; this is not a Python optimization pass.

## Use

```bash
git clone https://github.com/0xShug0/audio.cpp.git
cd audio.cpp
git checkout 748c5e28f6a7228b8f38ad7142ca97d29584544b
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target audiocpp_cli -j 8
```

## Support

| Property | Value |
| --- | --- |
| Availability | External runtime; no registered VoiceHub pass |
| Fidelity | Model conversion and runtime dependent |
| Runtime | CPU and backend-dependent accelerators |
| Registry name | Not registered; do not report this backend as an applied VoiceHub pass |

Unsupported explicit configurations must fail before mutation. A pass that
does not match a model reports `not-applicable`; it is not an acceleration.

## Paper and GitHub

- **Paper:** No dedicated upstream research paper is published for this integration.
- **Upstream GitHub:** [audio.cpp](https://github.com/0xShug0/audio.cpp)
- **VoiceHub source:** No VoiceHub pass implementation; use the external source project directly.

## Verify

Compare the eager and optimized paths with the same checkpoint, input, seed,
warm-up, device, and dtype. Record latency, memory, output quality, the exact
source revision, and the optimization manifest.

See the [related workflow](../guides/optional-backends.md) and
[optimization API](../reference/api.md#optimization).
