---
description: Run TTS, ASR, and VAD through one task-aware VoiceHub inference API.
---

# Inference

The `pipeline()` function is the shortest path from a speech task and checkpoint
to a normalized VoiceHub output. It selects the task-specific auto factory,
keeps model construction lazy, and forwards inference to `generate()`,
`transcribe()`, or `detect()`.

```python
from voicehub import pipeline

synthesizer = pipeline(
    task="text-to-speech",
    model="parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = synthesizer("VoiceHub provides one task-aware speech pipeline.")
print(output.sample_rate, output.audio.shape)
```

Install VoiceHub first. GPU users should install the correct PyTorch build for
their system before installing the package. See [Installation](../getting-started/installation.md).

## Tasks

Choose a canonical task name or a documented alias such as `tts`, `asr`,
`stt`, or `vad`. Discovery and construction do not import every model runtime.

### Text to speech

A text-to-speech pipeline returns `TTSOutput`. The example below verifies the
actual duration because word count cannot guarantee speaking time.

```python
from voicehub import TTSGenerationConfig, pipeline

text = (
    "VoiceHub keeps speech experiments simple and reproducible. This longer "
    "sample checks pacing, pronunciation, pauses, volume, and consistent tone "
    "across several complete sentences. We measure the returned waveform "
    "instead of guessing its duration, then preserve the prompt, seed, model "
    "revision, and output file for a fair comparison."
)
synthesizer = pipeline(
    task="text-to-speech",
    model="parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = synthesizer(
    text,
    description="A clear speaker talks at a steady, natural pace.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file="artifacts/parler.wav",
    ),
)
sample_count = output.audio.shape[-1] if hasattr(output.audio, "shape") else len(output.audio)
duration = sample_count / output.sample_rate
if duration < 10:
    raise RuntimeError(f"Expected at least 10 seconds, got {duration:.2f}")
```

Conditioning fields remain model-specific. Check the
[TTS capability matrix](../models/tts-capabilities.md) before switching models.

### Automatic speech recognition

An automatic speech-recognition pipeline accepts the same audio envelope as
the underlying model and returns `ASROutput`.

```python
from voicehub import pipeline

transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-small",
    model_type="asr_transformers",
    device="cuda",
)
output = transcriber("samples/interview.wav", sampling_rate=16_000)
print(output.text)
```

See [Speech recognition](speech-recognition.md) for decoding controls,
timestamps, language behavior, and checkpoint-specific limitations.

### Voice activity detection

A voice-activity-detection pipeline returns `VADOutput` with normalized speech
segments.

```python
from voicehub import pipeline

detector = pipeline(
    task="voice-activity-detection",
    model="silero_vad",
    model_type="vad_silero",
    device="cpu",
)
output = detector("samples/meeting.wav", sampling_rate=16_000)
for segment in output.segments:
    print(segment.start, segment.end)
```

See [Voice activity detection](voice-activity-detection.md) for threshold,
frame, and boundary behavior.

## Parameters

`pipeline()` separates task selection from checkpoint loading options:

| Parameter | Purpose |
| --- | --- |
| `task` | Canonical speech task or documented alias |
| `model` | Repository ID, local artifact, or an already configured model object |
| `model_type` | Canonical registry key when the artifact does not identify its integration |
| `config` / `config_kwargs` | Complete configuration or configuration overrides |
| `device` | Device passed to the task-specific model factory |
| `inference_strategy` | Registered inference strategy applied by the model lifecycle |
| `model_kwargs` | Additional model-loader options such as `lazy_load` |

Loader options cannot be combined with an existing model object. Configure
that object first, then wrap it with `pipeline(task, model=model)`.

### Device

Omit `device` to use the model factory default. Use `device="cpu"` for a
portable baseline or a supported accelerator such as `device="cuda"`. The
pipeline reports `speech_pipeline.device` but does not move an existing model
or silently fall back when the requested hardware is unavailable.

### Batch inference

VoiceHub does not provide a universal vectorized batch contract because a
Python list can represent either a waveform or several inputs. Use explicit
sequential orchestration unless the selected model or serving backend
documents batching:

```python
paths = ["samples/one.wav", "samples/two.wav"]
outputs = [transcriber(path) for path in paths]
```

Measure throughput and memory before increasing provider-specific batch size.

### Task-specific parameters

Call-time keyword arguments are forwarded to the selected model and validated
there. Examples include TTS conditioning, ASR decoding settings, and VAD
thresholds. Unknown fields fail instead of being silently ignored.

## Chunking and streaming

Chunk boundaries affect transcripts, timestamps, and speech segments. The
pipeline does not invent a universal chunking policy. Use only the streaming
or chunk controls documented by the selected integration, and retain overlap
and boundary settings with evaluation evidence.

## Large inputs

Decode audio lazily, downmix and resample once, and keep the original sample
rate in request metadata. For long recordings, prefer a checkpoint with a
verified chunking path. Confirm that recombined ASR timestamps or VAD segments
remain monotonic and do not duplicate overlap regions.

## Large models

Construction is lazy by default. Call `speech_pipeline.load()` during service
startup when allocation or checkpoint failures must happen before the first
request. Record checkpoint revision, VoiceHub and backend versions, device,
precision, input, and cold/warm latency before making a performance claim.

## Save and reload

Portable VoiceHub models retain their normalized configuration and task:

```python
artifact = synthesizer.save_pretrained("artifacts/parler-voicehub")
reloaded = pipeline(
    task="text-to-speech",
    model=artifact,
    model_type="parlertts",
    device="cuda",
)
```

Provider-specific or legacy artifacts require an explicitly verified
conversion path. Do not enable trust flags for an unverified file.

## Troubleshooting

- Unknown task: use `text-to-speech`, `automatic-speech-recognition`, or
  `voice-activity-detection`.
- Unknown `model_type`: select a canonical key from
  `list_model_specs(task=...)`.
- Wrong task: use the factory declared by the registry for that model.
- Unsupported argument: check the selected model's input and conditioning
  contract.
- Out of memory: choose a smaller checkpoint or provider-supported batch size;
  do not claim a precision or quantization path without quality evidence.

Continue with the [API reference](../reference/api.md),
[optimization guide](tts-optimization.md), or [inference notebook](notebook.md).
