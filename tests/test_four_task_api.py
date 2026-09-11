"""Exercise the shared entry points with real small codec graphs on CPU."""

import json
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from voicehub import AutoConfig, AutoModel, AutoModelForAudioCodec, CodecOutput, SpeechTask, pipeline
from voicehub.model import PreTrainedSpeechModel
from voicehub.models.audio import PreTrainedASRModel, PreTrainedVADModel
from voicehub.models.codec import PreTrainedCodecModel
from voicehub.models.tts import PreTrainedTTSModel


@pytest.fixture(params=["dac", "encodec", "encodec_stereo"])
def codec(request):
    if request.param == "dac":
        values = dict(
            encoder_hidden_size=4,
            downsampling_ratios=[2, 2],
            decoder_hidden_size=16,
            n_codebooks=2,
            codebook_size=8,
            codebook_dim=2,
            sampling_rate=100)
        model_type = "dac"
    else:
        stereo = request.param == "encodec_stereo"
        values = dict(
            target_bandwidths=[0.1],
            sample_rate=100,
            channels=2 if stereo else 1,
            normalize=True,
            segment=0.5,
            overlap=0.2,
            name="tiny_encodec",
            dimension=8,
            n_filters=4,
            n_residual_layers=1,
            ratios=[2, 2],
            kernel_size=3,
            last_kernel_size=3,
            residual_kernel_size=3,
            dilation_base=2,
            compress=2,
            lstm=0,
            bins=8,
            n_q=2,
            kmeans_init=False,
            kmeans_iters=2,
            threshold_ema_dead_code=0)
        model_type = "encodec"
    config = AutoConfig.for_model(model_type, native_config=values)
    return AutoModel.from_config(config, device="cpu")


def test_all_tasks_share_the_pretrained_lifecycle():
    for model_class in (PreTrainedTTSModel, PreTrainedASRModel, PreTrainedVADModel, PreTrainedCodecModel):
        assert issubclass(model_class, PreTrainedSpeechModel)
        assert model_class._restore_voicehub_model_state is PreTrainedSpeechModel._restore_voicehub_model_state
    assert SpeechTask.coerce("codec") is SpeechTask.AUDIO_CODEC
    assert SpeechTask.coerce("stt") is SpeechTask.AUTOMATIC_SPEECH_RECOGNITION


def test_discovery_does_not_import_models_or_optional_features():
    code = '''
import json, sys
import voicehub.models.catalog
from voicehub import AutoModel, AutoConfig
specs = AutoModel.available_models()
for spec in specs:
    AutoConfig.for_model(spec.model_type)
print(json.dumps({"tasks": sorted({s.task.value for s in specs}),
                  "loaded": [m for m in ("torch", "transformers", "voicehub.training.trainer", "voicehub.optimization") if m in sys.modules]}))
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    assert info["loaded"] == []
    assert len(info["tasks"]) == 4


def test_codec_encode_decode_and_pipeline_preserve_samples(codec):
    assert not codec.is_loaded
    channels = codec.config.native_config.get("channels", 1)
    audio = torch.randn(2, channels, 79) * 0.1
    encoded = codec.encode(audio, sampling_rate=100)
    assert isinstance(encoded, CodecOutput)
    assert encoded.length == 79
    reconstructed = codec.decode(encoded)
    assert reconstructed.audio.shape == audio.shape
    assert not reconstructed.audio.requires_grad
    assert torch.isfinite(reconstructed.audio).all()
    processor = pipeline("codec", model=codec)
    torch.testing.assert_close(
        processor.decode(processor(audio, sampling_rate=100)).audio, reconstructed.audio)


def test_codec_round_trip_reloads_exact_weights(codec, tmp_path):
    channels = codec.config.native_config.get("channels", 1)
    audio = torch.randn(1, channels, 77) * 0.1
    expected = codec(audio, sampling_rate=100).audio
    codec.save_pretrained(tmp_path)
    restored = AutoModel.from_pretrained(tmp_path, device="cpu", lazy_load=False)
    assert isinstance(restored, PreTrainedCodecModel)
    assert restored.sample_rate == 100
    torch.testing.assert_close(restored(audio, sampling_rate=100).audio, expected, rtol=0, atol=0)
    assert restored.config.native_config == codec.config.native_config
    exported = codec.export_native_pretrained(tmp_path / "exported")
    from_file = AutoModelForAudioCodec.from_pretrained(
        exported / "model.safetensors", model_type=codec.config.model_type, device="cpu")
    torch.testing.assert_close(from_file(audio, sampling_rate=100).audio, expected, rtol=0, atol=0)


def test_codec_uses_the_loaded_checkpoint_rate_without_an_intermediate_resample(codec):
    channels = codec.config.native_config.get("channels", 1)
    waveform = torch.randn(1, channels, 79) * 0.1
    # A checkpoint may resolve a rate different from a generic lazy configuration.
    codec.config.sample_rate = 50
    with patch.object(codec, "_encode", return_value="captured") as encode:
        encoded = codec.encode(waveform, sampling_rate=100)
    torch.testing.assert_close(encode.call_args.args[0], waveform, rtol=0, atol=0)
    assert encoded.sample_rate == 100


def test_codec_rejects_incompatible_input_and_codes(codec):
    with pytest.raises(ValueError, match="sampling_rate"):
        codec.encode(torch.zeros(79))
    assert not codec.is_loaded
    with pytest.raises(TypeError, match="CodecOutput"):
        codec.decode(torch.zeros(1, 1, 2))
    channels = codec.config.native_config.get("channels", 1)
    encoded = codec.encode(torch.zeros(1, channels, 79), sampling_rate=100)
    with pytest.raises(ValueError, match="different codec"):
        codec.decode(replace(encoded, model_type="different"))
    with pytest.raises(ValueError, match="sample rates"):
        codec.decode(replace(encoded, sample_rate=200))


def test_codec_factory_rejects_other_tasks_before_loading():
    with pytest.raises(ValueError, match="text-to-speech"):
        AutoModelForAudioCodec.from_pretrained(model_type="vits")
    with pytest.raises(ValueError, match="audio-codec"):
        pipeline("vad", model=AutoModelForAudioCodec.from_pretrained(model_type="dac"))
