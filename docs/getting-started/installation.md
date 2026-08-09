---
description: Install VoiceHub from source and configure model caching.
---

# Installation

VoiceHub supports Python 3.10 through 3.12 and PyTorch 2.8. Install VoiceHub
from source while package-index installation is unavailable. Check the
[release-readiness report](../project/release-readiness.md) before a release.

## Virtual environment

[uv](https://docs.astral.sh/uv/) manages environments and packages. Follow its
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
then create an environment:

```bash
uv venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the same environment with:

```powershell
.venv\Scripts\Activate.ps1
```

With `pip`, create the environment with `python -m venv .venv` and replace
`uv pip install` with `python -m pip install`.

## Python

Install the current source:

```bash
uv pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Add training dependencies only when needed:

```bash
uv pip install "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

Choose a hardware-specific command from the
[PyTorch installer](https://pytorch.org/get-started/locally/) first. A CPU-only
environment can use:

```bash
uv pip install "torch>=2.8,<2.9" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Verify discovery without downloading a checkpoint or importing PyTorch:

```python
import sys

import voicehub

models = voicehub.list_model_specs(task=None)
print("VoiceHub:", voicehub.__version__)
print("Registered models:", len(models))
print("PyTorch imported during discovery:", "torch" in sys.modules)
```

Check the accelerator separately:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Source install

Clone the repository when you want a local source tree. Pin a commit instead
of `main` for a reproducible environment.

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
uv pip install .
```

Confirm the installed version and dependency-light registry:

```bash
python -c "import voicehub; print(voicehub.__version__, len(voicehub.list_model_specs()))"
```

### Editable install

An editable install exposes local edits immediately:

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
uv pip install -e ".[test,training,docs]"
```

Update the checkout explicitly:

```bash
cd voicehub
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/) can own the Python
environment while uv installs VoiceHub from source:

```bash
conda create -n voicehub python=3.12 -y
conda activate voicehub
python -m pip install uv
uv pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

## Set up

After installation, configure where Hub-backed model files are cached and
whether network access is allowed. Legal terms, access tokens, and hardware
requirements remain specific to each model page.

### Cache directory

VoiceHub's shared Hub transport uses the same cache roots as the Hugging Face
ecosystem. An explicit `cache_dir` argument has the highest priority, followed
by these locations:

1. `HF_HUB_CACHE`
2. `HUGGINGFACE_HUB_CACHE`
3. `HF_HOME/hub`
4. `XDG_CACHE_HOME/huggingface/hub`
5. `~/.cache/huggingface/hub`

Pass a directory directly when one service should not depend on process-wide
environment variables:

```python
from voicehub import AutoConfig

config = AutoConfig.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    cache_dir="/srv/voicehub-cache",
)
print(config.model_type)
```

Pin an immutable checkpoint revision in production. A moving branch may be
refreshed when the model is loaded again.

### Offline mode

Load and run the required model once with network access so its configuration,
weights, processor assets, and model-specific files are present. Then set
either `HF_HUB_OFFLINE=1` or `VOICEHUB_OFFLINE=1` to prevent VoiceHub's shared
Hub transport from making HTTP requests:

```bash
VOICEHUB_OFFLINE=1 python app.py
```

Use `local_files_only=True` for an explicit call-level boundary. This example
checks only the cached configuration; follow the selected model page for its
complete checkpoint and processor inventory.

```python
from voicehub import AutoConfig

config = AutoConfig.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    local_files_only=True,
)
print(config.name_or_path)
```

An offline cache miss raises `FileNotFoundError` with the requested repository,
revision, file, and cache location. Do not report an offline model path as
verified until its complete checkpoint-specific inference succeeds.

Continue with the [quickstart](quickstart.md), then select a checkpoint from
the [model catalog](../models/index.md).
