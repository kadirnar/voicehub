<h2 align="center">Unified speech inference, training, and optimization</h2>

<div align="center">
  <img width="100%" alt="Abstract sound waves representing VoiceHub's unified speech toolkit" src="https://raw.githubusercontent.com/kadirnar/voicehub/main/assets/readme-hero.png">
</div>

VoiceHub provides one API for text-to-speech (TTS), speech recognition (ASR),
and voice activity detection (VAD). It supports Python 3.10–3.12.

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

## Train

```python
from voicehub import get_training_spec

spec = get_training_spec("dia")
print(spec.support.value, spec.family_name)
```

Use the [training guide](https://kadirnar.github.io/voicehub/guides/training/)
and [training support](https://kadirnar.github.io/voicehub/models/training-support/)
matrix.

## Optimize

```python
from voicehub import TTSOptimizationConfig

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
