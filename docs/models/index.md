---
description: Browse every VoiceHub TTS, ASR, and VAD model.
---

# Models

VoiceHub documents every registered TTS, ASR, and VAD model on its own page.
Each page includes a short example, supported inputs, training and optimization
status, and paper and GitHub links when available.

## Model list

Open the [complete model list](providers/index.md), or inspect the registry in
Python:

```python
from voicehub import list_model_specs

for model in list_model_specs():
    print(model.task.value, model.model_type)
```

Use the task matrices for a compact comparison:

- [TTS capabilities](tts-capabilities.md)
- [ASR and VAD support](asr-vad-support.md)

## Train

Check the [training support matrix](training-support.md) before choosing a
checkpoint or preparing a dataset.

```python
from voicehub import get_training_spec

print(get_training_spec("dia").support.value)
```

## Optimize

Start with the [optimization catalog](../optimizations/index.md). Optimization
passes validate support before changing a model.

```python
def show_optimizations(model):
    print(model.available_optimization_passes())
```
