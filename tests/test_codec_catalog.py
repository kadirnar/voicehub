from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from voicehub.components.audio.codecs.catalog import (
    CODEC_ALIASES,
    CODEC_CATALOG,
    LLM_TTS_CODEC_FEATURE,
    REGISTERED_LLM_TTS_MODEL_TYPES,
    CodecIntegration,
    CodecOptimizationSurface,
    CodecPrimitive,
    CodecRepresentation,
    CodecStage,
    CodecStageAvailability,
    get_codec_entries_for_model,
    get_codec_entry,
    get_codec_primitive_manifest,
    list_codec_entries,
    list_codec_primitive_manifests,
    list_registered_llm_tts_codec_model_types,
    normalize_codec_id,
    validate_codec_catalog_registry_coverage,
)
from voicehub.registry import MODEL_REGISTRY
from voicehub.runtime import get_architecture_spec
from voicehub.tasks import SpeechTask

EXPECTED_LLM_TTS_MODEL_TYPES = {
    "bark",
    "chatterbox",
    "conversationtts",
    "cosyvoice",
    "csm",
    "dia",
    "fishtts",
    "gptsovits",
    "higgstts",
    "llasa",
    "mosstts",
    "neutts",
    "omnivoice",
    "orpheustts",
    "outetts",
    "parlertts",
    "qwen3tts",
    "vibevoice",
    "voxcpm",
    "vui",
    "xtts",
    "zonos",
    "zonos2",
}

EXPECTED_CODEC_IDS = (
    "snac-24khz",
    "dac-native",
    "encodec-bark",
    "mimi-native",
    "xcodec2-llasa",
    "neucodec",
    "moss-audio-tokenizer",
    "qwen3-tts-tokenizer-12hz",
    "vibevoice-acoustic-tokenizer",
    "voxcpm-audiovae-v2",
    "higgs-audio-tokenizer-v2",
    "fish-modified-dac",
    "fluac-vui",
    "chatterbox-s3gen",
    "gpt-sovits-s2",
    "xtts2-dvae",
    "cosyvoice-speech-tokenizer",
)


def test_exact_codec_inventory_covers_every_active_llm_tts_model_once():
    entries = list_codec_entries()
    owner_bindings = tuple(owner for entry in entries for owner in entry.owners)
    owner_types = tuple(owner.model_type for owner in owner_bindings)

    assert tuple(CODEC_CATALOG) == EXPECTED_CODEC_IDS
    assert set(REGISTERED_LLM_TTS_MODEL_TYPES) == EXPECTED_LLM_TTS_MODEL_TYPES
    assert (set(list_registered_llm_tts_codec_model_types()) == EXPECTED_LLM_TTS_MODEL_TYPES)
    assert set(owner_types) == EXPECTED_LLM_TTS_MODEL_TYPES
    assert len(owner_types) == len(set(owner_types))

    for owner in owner_bindings:
        spec = MODEL_REGISTRY[owner.model_type]
        assert spec.task is SpeechTask.TEXT_TO_SPEECH
        assert spec.is_voicehub_native
        assert spec.architecture == owner.architecture_id
        architecture = get_architecture_spec(owner.architecture_id)
        assert architecture.capabilities.has_feature(LLM_TTS_CODEC_FEATURE)
    validate_codec_catalog_registry_coverage()


def test_qwen_inventory_is_full_native_encoder_quantizer_decoder():
    qwen = get_codec_entry("qwen3_tts")

    assert qwen.codec_id == "qwen3-tts-tokenizer-12hz"
    assert qwen.representation is CodecRepresentation.DENSE_DISCRETE
    assert qwen.stages.is_full_native_codec
    assert qwen.stages.encoder is CodecStageAvailability.NATIVE
    assert qwen.stages.quantizer is CodecStageAvailability.NATIVE
    assert qwen.stages.decoder is CodecStageAvailability.NATIVE
    assert qwen.implementation_paths == (
        "voicehub.models.qwen3tts.native.encoder:Qwen3TTSSpeechEncoder",
        "voicehub.models.qwen3tts.native.codec:Qwen3TTSSpeechDecoder",
    )
    assert CodecOptimizationSurface.SNAKE_BETA in qwen.optimization.surfaces
    assert not qwen.gaps


def test_cuda_graph_inventory_excludes_host_synchronized_boundaries():
    encodec = get_codec_entry("encodec")
    qwen = get_codec_entry("qwen3tts")
    moss = get_codec_entry("mosstts")
    higgs = get_codec_entry("higgs_audio_v2")

    for decoder_only in (encodec, qwen):
        assert (CodecOptimizationSurface.CUDA_GRAPH in decoder_only.optimization.surfaces)
        assert decoder_only.optimization.cuda_graph_targets == (CodecStage.DECODER, )
        assert any(
            "decoder-only" in constraint for constraint in decoder_only.optimization.cuda_graph_constraints)

    for host_synchronized in (moss, higgs):
        assert (CodecOptimizationSurface.CUDA_GRAPH not in host_synchronized.optimization.surfaces)
        assert host_synchronized.optimization.cuda_graph_targets == ()
        assert host_synchronized.optimization.cuda_graph_constraints == ()
        assert CodecOptimizationSurface.TORCH_COMPILE in (host_synchronized.optimization.surfaces)
        assert any("CUDA Graph capture is not advertised" in gap for gap in host_synchronized.gaps)


def test_inventory_distinguishes_geometry_and_native_boundaries():
    snac = get_codec_entry("orpheustts")
    vibe = get_codec_entry("vibevoice")
    voxcpm = get_codec_entry("voxcpm")
    chatterbox = get_codec_entry("chatterbox")
    gptsovits = get_codec_entry("gptsovits")
    cosyvoice = get_codec_entry("cosyvoice")

    assert snac.representation is CodecRepresentation.HIERARCHICAL_DISCRETE
    for continuous in (vibe, voxcpm):
        assert continuous.representation is CodecRepresentation.CONTINUOUS_VAE
        assert continuous.stochastic_vae
        assert continuous.separable_autoencoder
        assert continuous.stages.quantizer is CodecStageAvailability.NOT_APPLICABLE
        assert continuous.stages.has_native_encoder_decoder

    assert "decoder-only" in vibe.owners[0].variant
    assert chatterbox.integration is CodecIntegration.SPLIT_TTS_PIPELINE
    assert chatterbox.stages.decoder is CodecStageAvailability.NATIVE_SPLIT_PIPELINE
    assert gptsovits.integration is CodecIntegration.INTEGRATED_TTS_GRAPH
    assert gptsovits.stages.encoder is CodecStageAvailability.NATIVE_INTEGRATED
    assert cosyvoice.integration is CodecIntegration.SPLIT_TTS_PIPELINE
    assert cosyvoice.stages.encoder is CodecStageAvailability.NATIVE
    assert cosyvoice.stages.quantizer is CodecStageAvailability.NATIVE
    assert cosyvoice.stages.decoder is CodecStageAvailability.NATIVE_SPLIT_PIPELINE
    assert all(entry.gaps for entry in (vibe, chatterbox, gptsovits, cosyvoice))


def test_aliases_filters_and_primitive_manifest_apis_are_stable():
    qwen = get_codec_entry("qwen3tts-codec")
    dac = get_codec_entry("descript_audio_codec")

    assert normalize_codec_id(" Qwen3_TTS ") == qwen.codec_id
    assert get_codec_entries_for_model("qwen3tts") == (qwen, )
    assert get_codec_entries_for_model("unknown-model") == ()
    assert set(dac.owner_model_types) == {
        "dia",
        "outetts",
        "parlertts",
        "zonos",
        "zonos2",
    }
    continuous = list_codec_entries(representation="continuous-vae")
    assert continuous == (
        get_codec_entry("vibevoice"),
        get_codec_entry("voxcpm"),
    )

    manifest = get_codec_primitive_manifest("llasa")
    assert CodecPrimitive.FINITE_SCALAR_QUANTIZER in manifest.quantizer
    assert CodecPrimitive.SNAKE_BETA in manifest.all
    assert list_codec_primitive_manifests() == tuple(
        (codec_id, entry.primitives) for codec_id, entry in CODEC_CATALOG.items())


def test_catalog_metadata_and_public_views_are_immutable():
    entry = get_codec_entry("dac")

    with pytest.raises(FrozenInstanceError):
        entry.family = "changed"
    with pytest.raises(TypeError):
        CODEC_CATALOG["changed"] = entry
    with pytest.raises(TypeError):
        CODEC_ALIASES["changed"] = entry.codec_id


def test_catalog_and_lazy_package_export_do_not_import_model_graphs():
    code = """
import json
import sys

import voicehub.components.audio.codecs as codecs

before = "voicehub.components.audio.codecs.catalog" in sys.modules
discoverable = all(
    name in codecs.__all__ and name in dir(codecs)
    for name in (
        "CODEC_CATALOG",
        "CodecCatalogEntry",
        "get_codec_entry",
        "list_codec_entries",
    )
)
catalog = codecs.CODEC_CATALOG
implementation_modules = sorted({
    path.partition(":")[0]
    for entry in catalog.values()
    for path in entry.implementation_paths
})
loaded_graphs = [
    module for module in implementation_modules
    if module in sys.modules
]
print(json.dumps({
    "before": before,
    "discoverable": discoverable,
    "entry_count": len(catalog),
    "loaded_graphs": loaded_graphs,
}))
"""
    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "before": False,
        "discoverable": True,
        "entry_count": len(EXPECTED_CODEC_IDS),
        "loaded_graphs": [],
    }
