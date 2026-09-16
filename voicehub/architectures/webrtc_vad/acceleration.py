"""Lazy, optional compilation of the pinned WebRTC CPU implementation.

The Python detector remains available on platforms without a compiler.
No model load, compilation, or external runtime import happens during
registry discovery.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from voicehub.hub_transport import _FileLock


class _Accelerator:

    def __init__(self, library: Path) -> None:
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.voicehub_webrtc_frames
        self.function.argtypes = [
            ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8)
        ]
        self.function.restype = ctypes.c_int

    def __call__(self, pcm, sample_rate: int, frame_size: int, mode: int) -> list[int]:
        import torch

        from voicehub.architectures.webrtc_vad.detector import NativeWebRTCVAD

        if (not isinstance(pcm, torch.Tensor) or pcm.device.type != "cpu" or pcm.dtype != torch.int16 or
                pcm.ndim != 1 or not pcm.is_contiguous()):
            raise ValueError("The WebRTC accelerator requires contiguous, one-dimensional CPU PCM16.")
        if not NativeWebRTCVAD.valid_rate_and_frame_length(sample_rate, frame_size) or mode not in range(4):
            raise ValueError("Invalid WebRTC frame configuration.")
        count = (pcm.numel() + frame_size - 1) // frame_size
        flags = (ctypes.c_uint8 * count)()
        # Keep pcm alive throughout the synchronous native call.
        pointer = ctypes.cast(pcm.data_ptr(), ctypes.POINTER(ctypes.c_int16))
        result = self.function(pointer, pcm.numel(), sample_rate, frame_size, mode, flags)
        if result != 0:
            raise RuntimeError("The compiled WebRTC detector rejected its input.")
        return list(flags)


def _build(cache: Path, cc: str, cxx: str) -> Path:
    package = Path(__file__).resolve().parents[2]
    source = Path(__file__).with_name("source")
    batch = Path(__file__).with_name("batch.c")
    files = sorted(p for p in source.rglob("*") if p.suffix in {".c", ".h", ".cc"})
    digest = hashlib.sha256(f"{sys.platform}:{platform.machine()}:{cc}:{cxx}:v1".encode())
    for path in [Path(__file__).resolve(), batch, *files]:
        digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
    directory = cache / digest.hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    library = directory / "webrtc.so"
    with _FileLock(directory / "build.lock", timeout=120):
        if library.exists():
            return library
        with tempfile.TemporaryDirectory(dir=directory, prefix="build-") as temporary:
            build = Path(temporary)
            common = ["-O3", "-fwrapv", "-fPIC", "-pthread", "-DWEBRTC_POSIX", "-I", str(source)]
            commands = [
                [cc, *common, "-c",
                 str(batch), *(str(p) for p in files if p.suffix == ".c")],
                [cxx, *common, "-std=c++11", "-c",
                 str(source / "webrtc/rtc_base/checks.cc")],
            ]
            with (directory / "build.log").open("w") as log:
                for command in commands:
                    subprocess.run(command, cwd=build, stdout=log, stderr=log, check=True, timeout=90)
                command = [
                    cxx, "-shared", "-pthread", *(str(p) for p in sorted(build.glob("*.o"))), "-lm", "-o",
                    str(build / "webrtc.so")
                ]
                subprocess.run(command, cwd=build, stdout=log, stderr=log, check=True, timeout=90)
            (build / "webrtc.so").replace(library)
    return library


@lru_cache(maxsize=1)
def get_accelerator() -> tuple[_Accelerator | None, str]:
    """Return an optional accelerator and an explicit diagnostic status."""
    if sys.platform not in {"linux", "darwin"}:
        return None, "python: native compilation is unavailable on this platform"
    cc, cxx = shutil.which("cc"), shutil.which("c++")
    if cc is None or cxx is None:
        return None, "python: C and C++ compilers are unavailable"
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cache = root / "voicehub/webrtc-native"
    try:
        return _Accelerator(_build(cache, cc, cxx)), "compiled-c"
    except (OSError, subprocess.SubprocessError, TimeoutError) as error:
        return None, f"python: native build/load failed ({type(error).__name__}); see {cache}"
