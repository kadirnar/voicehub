---
description: Install VoiceHub from source on Linux, macOS, or Windows.
---

# Installation

VoiceHub supports Python 3.10–3.12. Install it from source.

## Create an environment

Choose your platform.

=== "Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "Windows"

    ```powershell
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```

## Install

Choose the correct [PyTorch build](https://pytorch.org/get-started/locally/)
for your hardware, then install VoiceHub from GitHub:

```bash
python -m pip install "voicehub @ git+https://github.com/kadirnar/voicehub.git@main"
```

Add training dependencies only when needed:

```bash
python -m pip install "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main"
```

### Editable checkout

Use an editable checkout when changing the library:

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
python -m pip install -e ".[test,training,docs]"
```

## Verify

Check the installation without downloading a checkpoint:

```bash
python -c "import voicehub; print(voicehub.__version__, len(voicehub.list_model_specs()))"
```

Check the accelerator separately:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Cache and offline mode

Pass `cache_dir` to isolate model files. Set `local_files_only=True` after
the required checkpoint and processor assets have been downloaded.

```python
from voicehub import AutoConfig

config = AutoConfig.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    cache_dir=".cache/voicehub",
    local_files_only=True,
)
print(config.model_type)
```

Choose your platform when starting an offline process.

=== "Linux"

    ```bash
    VOICEHUB_OFFLINE=1 python app.py
    ```

=== "macOS"

    ```bash
    VOICEHUB_OFFLINE=1 python app.py
    ```

=== "Windows"

    ```powershell
    $env:VOICEHUB_OFFLINE = "1"
    python app.py
    ```
