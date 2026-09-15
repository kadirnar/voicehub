---
description: Get started with VoiceHub, explore speech models, and compare inference against original repositories.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-eyebrow">Getting started</p>

# Welcome to VoiceHub Docs!

<p class="vh-doc-tagline">A shared Python interface for speech synthesis, transcription, and voice activity detection.</p>

<div class="vh-welcome-banner" role="img" aria-label="VoiceHub documentation: text to speech, speech recognition, and voice activity detection">
  <div>
    <img src="assets/voicehub-mark.svg" alt="" width="48" height="48">
    <strong>VoiceHub</strong>
    <span>Speech models. One interface.</span>
  </div>
  <div class="vh-welcome-tasks" aria-hidden="true"><span>Text → Audio</span><span>Audio → Text</span><span>Audio → Speech regions</span></div>
</div>

## Why VoiceHub?

Work with **68 integrations**: **34 TTS backends**, **23 ASR providers**,
and **11 VAD providers**, each with a documented configuration, model, and processor contract.

- **One model lifecycle** — discover, configure, load, and run speech models through consistent APIs.
- **Architecture-aware behavior** — preserve each model's conditioning, codecs, decoding, and outputs.
- **Reproducible comparisons** — inspect checkpoint provenance, inference settings, timings, and measured differences.
- **Explicit support** — check each integration's requirements and tested limits before choosing a model.

[Explore the library architecture →](concepts/architecture.md)

## Get started

<div class="grid cards" markdown>

-   **Explore models**

    Browse TTS, ASR, and VAD integrations, checkpoints, and their capabilities.

    [Explore models →](models/providers/index.md)

-   **Inference guides**

    Install VoiceHub and run your first speech request with the shared API.

    [Start generating →](getting-started/quickstart.md)

-   **Fine-tuning**

    Check training support and prepare data for your selected architecture.

    [Fine-tuning guide →](guides/training.md)

-   **Examples**

    Follow runnable notebooks for inference, data preparation, and training.

    [Open examples →](guides/notebook.md)

</div>

!!! note "Check the evidence"
    A registered model is not proof of upstream quality or performance parity.
    The [upstream comparison guide](guides/upstream-parity.md) distinguishes
    actual paired measurements, failures, and outstanding evaluations.
    Waveform agreement, transcript error rates, and inference timing are
    separate checks.

Checkpoint weights are downloaded lazily. Training tools are available through
`voicehub[training]`. VoiceHub's license covers the library; integrated source,
checkpoints, codecs, and datasets may have separate terms.

</div>
