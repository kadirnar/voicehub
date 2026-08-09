<h2 align="center">Unified speech inference, training, and optimization</h2>

<div align="center">
  <img width="100%" alt="Abstract sound waves representing VoiceHub's unified speech toolkit" src="https://raw.githubusercontent.com/kadirnar/voicehub/main/assets/readme-hero.png">
</div>

VoiceHub provides one API for text-to-speech (TTS), speech recognition (ASR),
and voice activity detection (VAD). It supports Python 3.10–3.12.

## Install from source

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
python -m pip install .
```

Install the training tools only when needed:

```bash
python -m pip install ".[training]"
```

Install the correct PyTorch build for your hardware first. See the
[installation guide](https://kadirnar.github.io/voicehub/getting-started/installation/).

## TTS

```python
from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

tts_model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = tts_model.generate(
    "VoiceHub uses one predictable speech model API.",
    generation_config=TTSGenerationConfig(output_file="speech.wav", seed=42),
)
print(output.file_path, output.sample_rate)
```

Model-specific inputs are listed in the
[TTS matrix](https://kadirnar.github.io/voicehub/models/tts-capabilities/).

## ASR

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    device="cuda",
)
print(model.transcribe("speech.wav", language="English").text)
```

## VAD

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(model_type="vad_silero")
output = model.detect("speech.wav", threshold=0.55)
for segment in output.segments:
    print(segment.start, segment.end)
```

See the [ASR and VAD matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/)
for checkpoints and supported inputs.

## Optimize

```python
from voicehub import TTSOptimizationConfig

result = tts_model.optimize(
    TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
    )
)
print(result.manifest())
```

Benchmark on the target hardware. Start with the
[optimization guide](https://kadirnar.github.io/voicehub/guides/tts-optimization/),
[model benchmarks](https://kadirnar.github.io/voicehub/guides/tts-model-benchmarks/),
and [RTX 4090 results](https://kadirnar.github.io/voicehub/guides/rtx-4090-speech-benchmarks/).

## Train

```python
from voicehub import get_training_spec

spec = get_training_spec("dia")
print(spec.support.value, spec.family_name)
```

Use the [training guide](https://kadirnar.github.io/voicehub/guides/training/),
[training matrix](https://kadirnar.github.io/voicehub/models/training-support/),
and [data guide](https://kadirnar.github.io/voicehub/guides/data-preparation/).

## Models and notebooks

Each registered model has a dedicated page in the left sidebar:
[model guides](https://kadirnar.github.io/voicehub/models/providers/).
See the [notebook guide](https://kadirnar.github.io/voicehub/guides/notebook/)
for hardware notes and opt-in execution flags.

| Notebook                    | GitHub                                                                                  | Colab                                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| TTS, ASR, and VAD inference | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/inference.ipynb)        | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/inference.ipynb)        |
| Data preparation            | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) |
| Fine-tuning                 | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/training.ipynb)         | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/training.ipynb)         |
| Dia workflow                | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)     | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)     |

## Development

```bash
python -m pip install -e ".[test,training,docs]"
python -m pytest
python scripts/check_distribution.py
```

Useful docs: [Pipeline](https://kadirnar.github.io/voicehub/guides/inference/),
[architecture](https://kadirnar.github.io/voicehub/concepts/architecture/),
[add a model](https://kadirnar.github.io/voicehub/project/adding-a-model/), and
[API reference](https://kadirnar.github.io/voicehub/reference/api/).

## License

VoiceHub is Apache-2.0. Check each checkpoint's separate license before use.
