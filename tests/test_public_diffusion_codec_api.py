import json
import subprocess
import sys


def test_optimization_facade_exports_complete_diffusion_and_codec_apis():
    import voicehub.optimization as optimization
    from voicehub.optimization import codecs, diffusion, diffusion_cache, diffusion_sampling, diffusion_solvers

    modules = (
        codecs,
        diffusion,
        diffusion_cache,
        diffusion_sampling,
        diffusion_solvers,
    )
    expected = set().union(*(set(module.__all__) for module in modules))
    assert expected <= set(optimization.__all__)
    assert set(optimization.__all__) <= set(dir(optimization))
    assert len(optimization.__all__) == len(set(optimization.__all__))
    for module in modules:
        for name in module.__all__:
            assert getattr(optimization, name) is getattr(module, name)


def test_codec_package_exports_complete_structural_api():
    import voicehub.components.audio.codecs as codecs
    from voicehub.components.audio.codecs import base, catalog

    assert set(codecs.__all__) == set(base.__all__) | set(catalog.__all__)
    assert set(codecs.__all__) <= set(dir(codecs))
    assert len(codecs.__all__) == len(set(codecs.__all__))
    for module in (base, catalog):
        for name in module.__all__:
            assert getattr(codecs, name) is getattr(module, name)


def test_root_api_keeps_inference_separate_from_specialist_namespaces():
    import voicehub

    assert {"AutoModelForAudioCodec", "CodecOutput", "AudioOutput"} <= set(voicehub.__all__)
    assert {"CodecOptimizationConfig", "DiffusionSamplingConfig", "Trainer"}.isdisjoint(voicehub.__all__)
    assert set(voicehub.__all__) <= set(dir(voicehub))


def test_public_facades_do_not_eagerly_load_compilers_or_model_graphs():
    code = """
import json
import sys

import voicehub
import voicehub.components.audio.codecs as codec_api
import voicehub.optimization as optimization

before = {
    "codec_base": "voicehub.components.audio.codecs.base" in sys.modules,
    "codec_optimization": "voicehub.optimization.codecs" in sys.modules,
    "diffusion": "voicehub.optimization.diffusion" in sys.modules,
}
discoverable = all(
    name in dir(module) and name in module.__all__
    for module, names in (
        (
            voicehub,
            ("AutoModelForAudioCodec", "CodecOutput", "AudioOutput"),
        ),
        (
            optimization,
            (
                "CodecOptimizationConfig",
                "DiffusionArchitectureKind",
            ),
        ),
        (codec_api, ("AudioCodecComponentView", "CodecCodeBatch")),
    )
    for name in names
)

_ = optimization.CodecOptimizationConfig
_ = optimization.DiffusionArchitectureKind
_ = codec_api.DenseCodecCodes
_ = codec_api.CODEC_CATALOG

blocked = (
    "torch.utils.cpp_extension",
    "voicehub.kernels.triton_activations",
    "voicehub.models.f5tts.native.modeling",
    "voicehub.models.irodoritts.native.modeling",
    "voicehub.models.vibevoice.native.diffusion",
    "voicehub.models.echo.autoencoder",
    "voicehub.models.echo.model",
)
after = {
    "blocked": sorted(name for name in blocked if name in sys.modules),
    "triton": sorted(
        name
        for name in sys.modules
        if name == "triton" or name.startswith("triton.")
    ),
}
print(json.dumps({
    "after": after,
    "before": before,
    "discoverable": discoverable,
}))
"""
    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "after": {
            "blocked": [],
            "triton": [],
        },
        "before": {
            "codec_base": False,
            "codec_optimization": False,
            "diffusion": False,
        },
        "discoverable": True,
    }
