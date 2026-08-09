---
description: Load pretrained speech models, run pipelines, and inspect training support with VoiceHub.
---

# Quickstart

VoiceHub keeps configuration, processing, pretrained loading, inference, and
training behind a small public surface shared by text-to-speech (TTS),
automatic speech recognition (ASR), and voice activity detection (VAD).

This quickstart shows you how to:

- load a pretrained speech model;
- run inference with `pipeline()`; and
- inspect training support before constructing a `Trainer`.

## Set up

Create an isolated environment and install VoiceHub. Choose one package
manager and keep using the same environment for the rest of the guide.

=== "uv"

    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
    ```

=== "pip"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
    ```

Windows PowerShell users activate the environment with
`.venv\Scripts\Activate.ps1`. Install the correct PyTorch build before
VoiceHub when the selected checkpoint requires CUDA or another accelerator.
The [Installation](installation.md) guide covers source, editable, cache, and
offline workflows.

## Agent skills

The source repository includes focused instructions that help coding agents
follow VoiceHub's model, documentation, and release contracts. They are
available in a Git checkout; they are not runtime features of the wheel and do
not authorize commits, publication, or other external changes.

=== "Codex"

    ```text
    Read .ai/AGENTS.md, .ai/GOAL.md, and .ai/LOOP.md.
    Use .ai/skills/add-or-validate-speech-model/SKILL.md to audit one model integration.
    Preserve uv.lock and report every inaccessible checkpoint as unverified.
    ```

=== "Other agents"

    ```text
    Follow the canonical instructions under .ai/.
    Select the matching SKILL.md before changing model, documentation, or release contracts.
    Run the focused regression first and never count skipped hardware checks as passes.
    ```

Start with the repository's
[model-integration skill](https://github.com/kadirnar/voicehub/blob/main/.ai/skills/add-or-validate-speech-model/SKILL.md)
for model work. The
[documentation-parity skill](https://github.com/kadirnar/voicehub/blob/main/.ai/skills/match-transformers-docs/SKILL.md)
and
[release-evidence skill](https://github.com/kadirnar/voicehub/blob/main/.ai/skills/prepare-release-evidence/SKILL.md)
cover their corresponding quality gates.

## Pretrained models

Three shared contracts keep pretrained integrations predictable.

| Contract | Purpose |
| --- | --- |
| `VoiceHubConfig` | Stores serializable model attributes and the immutable checkpoint identity |
| `PreTrainedSpeechModel` | Defines the common loading, saving, device, and task-model lifecycle |
| Processor | Converts text or audio into the model-specific inputs declared by `AutoProcessor` |

Prefer the Auto classes because they resolve the configuration, processor, and
task wrapper from registry metadata. Pass `model_type` when the checkpoint
does not contain a VoiceHub `config.json`.

```python
from voicehub import AutoConfig, AutoModelForTextToSpeech, AutoProcessor

checkpoint = "parler-tts/parler-tts-mini-v1"
config = AutoConfig.from_pretrained(checkpoint, model_type="parlertts")
processor = AutoProcessor.from_pretrained(checkpoint, config=config)
model = AutoModelForTextToSpeech.from_pretrained(
    checkpoint,
    config=config,
    device="cuda",
    lazy_load=True,
)
print(config.model_type, type(processor).__name__, model.is_loaded)
```

`lazy_load=True` resolves the public contract without allocating the
checkpoint. Call `model.load()` when a service should fail during startup
rather than on its first inference request.

!!! tip

    Skip to [Trainer](#trainer) when you already have a model, dataset,
    processor, and collator for a supported differentiable training path.

## Pipeline

`pipeline()` is the shortest task-aware inference API. It selects the correct
Auto model, preserves the model's normalized output type, and accepts either a
checkpoint source or an already constructed model.

=== "Text to speech"

    Create a TTS pipeline with an explicit checkpoint and registry key.

    ```python
    from voicehub import pipeline

    synthesizer = pipeline(
        task="text-to-speech",
        model="parler-tts/parler-tts-mini-v1",
        model_type="parlertts",
        device="cuda",
    )
    ```

    Generate speech and inspect the normalized `TTSOutput`.

    ```python
    speech = synthesizer(
        "VoiceHub uses one predictable speech-model lifecycle.",
        description="A clear speaker talks at a steady pace.",
    )
    print(speech.sample_rate, speech.file_path)
    ```

=== "Automatic speech recognition"

    Create an ASR pipeline. The selected checkpoint determines its language
    and hardware boundaries.

    ```python
    from voicehub import pipeline

    transcriber = pipeline(
        task="automatic-speech-recognition",
        model="Qwen/Qwen3-ASR-0.6B",
        model_type="asr_qwen3",
        device="cuda",
    )
    ```

    Pass a local audio path and read the normalized `ASROutput`.

    ```python
    transcript = transcriber("speech.wav", language="English")
    print(transcript.text)
    ```

=== "Voice activity detection"

    Create a VAD pipeline. Omitting `model` uses the registry's declared task
    default when one exists.

    ```python
    from voicehub import pipeline

    detector = pipeline(
        task="voice-activity-detection",
        model_type="vad_silero",
    )
    ```

    Detect speech regions and inspect the normalized `VADOutput`.

    ```python
    detection = detector("speech.wav", threshold=0.55)
    for segment in detection.segments:
        print(segment.start, segment.end, segment.score)
    ```

!!! tip

    The [Pipeline guide](../guides/inference.md) covers batching boundaries,
    task parameters, chunking, streaming, large inputs, save/reload, and
    failure behavior.

## Trainer

`Trainer` provides shared evaluation, checkpoint, resume, and reporting
orchestration only when an integration declares a real differentiable
objective. Inspect that contract before allocating a training runtime.

```python
from voicehub import Trainer, TrainingArguments, get_training_spec

training_spec = get_training_spec("parlertts")
if not training_spec.supports_training:
    raise RuntimeError(f"Training is not supported: {training_spec.support.value}")

arguments = TrainingArguments(
    output_dir="runs/parlertts-smoke",
    max_steps=1,
    per_device_train_batch_size=1,
    report_to="none",
)
print(Trainer.__name__, training_spec.support.value, arguments.max_steps)
```

The [training guide](../guides/training.md) adds the checkpoint-specific model,
dataset, processor, collator, one-step validation, save, and exact-resume
boundaries required before calling `Trainer.train()`.

## Next steps

- [Base classes](../concepts/architecture.md): trace configuration,
  processing, registry, model, output, and serialization contracts.
- [Pipeline](../guides/inference.md): build reproducible TTS, ASR, and VAD
  requests.
- [Trainer](../guides/trainer.md): understand the shared training
  orchestration boundary.
- [Quantization and optimization](../guides/optimization-overview.md): inspect
  registry-wide support before applying a public pass.
- [Resources](../guides/notebook.md): run short workflows and generated model
  notebooks.
