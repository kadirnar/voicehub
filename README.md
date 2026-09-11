<h2 align="center">Unified speech inference, training, and optimization</h2>

<div align="center">
  <img width="100%" alt="Abstract sound waves representing VoiceHub's unified speech toolkit" src="https://raw.githubusercontent.com/kadirnar/voicehub/main/assets/readme-hero.png">
</div>

VoiceHub provides one API for text-to-speech (TTS), speech recognition (ASR),
voice activity detection (VAD), and audio codecs. It supports Python 3.10–3.12.

Version 0.4 uses one shared model lifecycle and keeps each native graph beside
its model adapter. See the [architecture](docs/concepts/architecture.md) and
[breaking-change migration guide](docs/project/migration-0.4.md).

## Install

Clone the source repository and install the library:

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
python -m pip install .
```

Install the correct [PyTorch build](https://pytorch.org/get-started/locally/)
for your hardware first.

## Models

Every registered model has a dedicated page in the
[model list](https://kadirnar.github.io/voicehub/models/providers/).

```python
from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = model.generate(
    "VoiceHub uses one predictable speech model API.",
    generation_config=TTSGenerationConfig(output_file="speech.wav", seed=42),
)
print(output.file_path)
```

See the [TTS capabilities](https://kadirnar.github.io/voicehub/models/tts-capabilities/)
and [ASR/VAD support](https://kadirnar.github.io/voicehub/models/asr-vad-support/)
tables for task-specific inputs.

## Four tasks

| Task      | Model method                      | Factory                              |
| --------- | --------------------------------- | ------------------------------------ |
| TTS       | `generate(text)`                  | `AutoModelForTextToSpeech`           |
| STT / ASR | `transcribe(audio)`               | `AutoModelForSpeechRecognition`      |
| VAD       | `detect(audio)`                   | `AutoModelForVoiceActivityDetection` |
| Codec     | `encode(audio)` / `decode(codes)` | `AutoModelForAudioCodec`             |

`AutoModel` loads any registered task. All factories share `from_pretrained`,
`from_config`, lazy loading, and `save_pretrained`.

```python
from voicehub import AutoModel, list_model_specs

print([spec.model_type for spec in list_model_specs(task="codec")])
codec = AutoModel.from_pretrained("descript/dac_44khz", model_type="dac", device="cpu")
encoded = codec.encode("speech.wav")
reconstructed = codec.decode(encoded)
print(reconstructed.audio.shape, reconstructed.sample_rate)
```

## Train

```python
from voicehub.training import get_training_spec

spec = get_training_spec("dia")
print(spec.support.value, spec.family_name)
```

Use the [training guide](https://kadirnar.github.io/voicehub/guides/training/)
and [training support](https://kadirnar.github.io/voicehub/models/training-support/)
matrix.

## Optimize

```python
from voicehub.optimization import TTSOptimizationConfig

result = model.optimize(
    TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
    )
)
print(result.manifest())
```

Use the [optimization catalog](https://kadirnar.github.io/voicehub/optimizations/)
and [TTS benchmark evidence](https://kadirnar.github.io/voicehub/guides/tts-model-benchmarks/)
on the target hardware.

## Documentation

Read the [installation guide](https://kadirnar.github.io/voicehub/getting-started/installation/),
[quickstart](https://kadirnar.github.io/voicehub/getting-started/quickstart/),
and [API reference](https://kadirnar.github.io/voicehub/reference/api/).

VoiceHub is Apache-2.0. Check each checkpoint's separate license before use.
