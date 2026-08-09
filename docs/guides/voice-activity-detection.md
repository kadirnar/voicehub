---
description: Detect and normalize speech regions with VoiceHub VAD providers.
---

# Voice activity detection

VoiceHub exposes VAD integrations through
`AutoModelForVoiceActivityDetection` and normalizes results as `VADOutput`.
Use the [ASR/VAD matrix](../models/asr-vad-support.md#vad-providers) for exact
checkpoint, streaming, score, and fine-tuning support.

## Install

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

No VAD-specific inference extra is required.

## Discover detectors

```python
from voicehub import AutoModelForVoiceActivityDetection

for spec in AutoModelForVoiceActivityDetection.available_models():
    print(
        spec.model_type,
        spec.architecture,
        spec.default_model_path,
        spec.capabilities,
    )
```

## Detect speech

Omitting the checkpoint source uses the registered Silero default:

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    model_type="vad_silero",
    device="cpu",
)
result = model.detect(
    "meeting.wav",
    threshold=0.55,
    min_speech_duration_ms=250,
    min_silence_duration_ms=150,
    speech_pad_ms=30,
)

for segment in result.segments:
    print(f"{segment.start:.3f}s -> {segment.end:.3f}s")
```

`model(audio, ...)` and `model.detect(audio, ...)` are equivalent. Files,
tensors, arrays, mappings, and `AudioInput` use the same audio envelope as
[speech recognition](speech-recognition.md#audio-inputs).

## Choose a detector

| Need | Model type |
| --- | --- |
| Small recurrent neural VAD | `vad_silero` |
| Fixed-point, low-overhead frames | `vad_webrtc` |
| Explainable energy baseline | `vad_auditok` |
| Native Silero/TEN with Sherpa-compatible endpoints | `vad_sherpa_onnx` |
| Trainable Wav2Vec2 frame classifier | `vad_transformers` |
| PyanNet segmentation or powerset output | `vad_pyannote` / `vad_pyannote_segmentation` |
| Speech plus SNR/C50 estimates | `vad_pyannote_brouhaha` |
| Native SpeechBrain-compatible CRDNN | `vad_speechbrain` |
| Native multilingual MarbleNet Frame-VAD | `vad_nemo` |
| Native FSMN frame scores | `vad_funasr` |

Names describe execution families. They do not mean VoiceHub imports the
similarly named upstream framework.

## Configure segmentation

```python
from voicehub import VADInferenceConfig

segmentation = VADInferenceConfig(
    threshold=0.55,
    onset=0.60,
    offset=0.45,
    min_speech_duration_ms=250,
    min_silence_duration_ms=120,
    speech_pad_ms=30,
    max_speech_duration_s=30.0,
    return_frames=False,
)
result = model.detect("meeting.wav", inference_config=segmentation)
```

| Field | Meaning |
| --- | --- |
| `threshold` | Speech threshold in `[0, 1]` |
| `onset` / `offset` | Optional hysteresis thresholds |
| `min_speech_duration_ms` | Reject shorter regions |
| `min_silence_duration_ms` | Bridge shorter silence gaps |
| `speech_pad_ms` | Extend accepted regions |
| `max_speech_duration_s` | Optional maximum region length |
| `window_size_samples` | Provider-supported window override |
| `return_frames` | Return frame scores when computed |

These fields are a common vocabulary, not universal capabilities. WebRTC has
binary decisions rather than neural probabilities. Auditok does not return
calibrated probability scores. Unsupported settings raise.

## Output

`VADOutput` contains ordered, non-overlapping `SpeechSegment` values:

```python
print(result.duration, result.sample_rate, result.speech_duration)
print(result.contains(3.5))
for segment in result.segments:
    print(segment.start, segment.end, segment.score, segment.label)
```

`score` can be `None` when the detector does not compute a calibrated score.
Provider-specific frame values and acoustic estimates remain in metadata.

## Streaming

Use one isolated session per request:

```python
with model.stream(sampling_rate=16_000, return_frames=True) as session:
    for chunk in microphone_chunks:
        frame_scores = session.push(chunk)
        print(frame_scores)
    final = session.flush()

for segment in final.segments:
    print(segment.start, segment.end)
```

Native streaming providers own recurrent and endpoint state inside the
session. Do not share a session across callers. For providers without an
incremental override, the common session buffers audio until `flush()`.

## Feed regions to ASR

Keep VAD and ASR as explicit stages. VAD timestamps remain on the source
recording:

```python
from voicehub import AutoModelForSpeechRecognition

asr = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    device="cuda",
)

for segment in result.segments:
    print("Transcribe source interval:", segment.start, segment.end)
```

Slice or stream the original waveform with those intervals, retain the
timebase offset, and then transcribe. Do not concatenate separated regions
when word timestamps must map back to the original recording.

## Fine-tuning

Only detectors with a differentiable graph are trainable. WebRTC and Auditok
are deterministic algorithms and intentionally fail training validation.

```python
model = AutoModelForVoiceActivityDetection.from_pretrained(
    model_type="vad_silero",
    device="cuda",
    lazy_load=True,
)
training_spec = model.validate_training_support()
print(training_spec.support.value, training_spec.family_name)
```

Records normally contain audio plus time intervals or frame labels. The exact
label layout, objective, sample rate, frozen components, and accepted artifact
format are listed in the [ASR/VAD matrix](../models/asr-vad-support.md#fine-tuning-boundaries)
and [speech data guide](speech-data.md).

## Safety and troubleshooting

- Validate finite mono audio and the expected sample rate.
- Tune thresholds on representative held-out recordings, not the training
  split.
- Report false accepts and false rejects alongside latency.
- Never trust-convert an unverified ONNX, JIT, or pickle artifact.
- Use separate sessions for concurrent streams.
- Keep consent and provenance when VAD regions are used to build training
  data.

See the [API reference](../reference/api.md) and
[inference notebook](notebook.md) for runnable workflows.
