"""Regression tests for MOSS-TTS behavior audited against the official
OpenMOSS processors and ``MossTTSDelayModel.generate``."""

import dataclasses
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_native_mosstts import _TinySemanticModel, _TinyTokenizer, _tiny_tts_config  # noqa: E402

from voicehub.architectures.mosstts.codec import MossCodecDecodeOutput, MossCodecEncodeOutput  # noqa: E402
from voicehub.architectures.mosstts.modeling import _find_last_equal, build_mosstts_model  # noqa: E402
from voicehub.architectures.mosstts.processing import AUDIO_PLACEHOLDER, MossTTSProcessor  # noqa: E402
from voicehub.architectures.mosstts.runtime import (  # noqa: E402
    SOURCE_DECODE_CHUNK_SECONDS,
    MossTTSRuntime,
    default_mosstts_codec_config,
    loudness_normalize,
    uses_source_text_normalizer,
)
from voicehub.architectures.mosstts.text_normalization import normalize_tts_text  # noqa: E402
from voicehub.processing.waveform import resample_waveform_hann  # noqa: E402

TORCHAUDIO_AVAILABLE = importlib.util.find_spec("torchaudio") is not None


class _RecordingCodec:
    """Codec stub that records the boundary tensors the runtime sends."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        self.encoded = []
        self.decode_kwargs = []

    def encode(self, waveforms, lengths=None, *, num_quantizers=None):
        self.encoded.append(waveforms.detach().clone())
        frames = 3
        return MossCodecEncodeOutput(
            audio_codes=torch.zeros(1, frames, num_quantizers, dtype=torch.long),
            audio_code_lengths=torch.tensor([frames]),
        )

    def decode(self, audio_codes, lengths=None, *, chunk_duration=None):
        self.decode_kwargs.append({"chunk_duration": chunk_duration})
        samples = audio_codes.shape[1] * self.config.downsample_rate
        return MossCodecDecodeOutput(
            waveform=torch.zeros(1, 1, samples),
            waveform_lengths=torch.tensor([samples]),
            sample_rate=self.config.sample_rate,
        )


def _tiny_runtime():
    config = _tiny_tts_config("delay")
    tokenizer = _TinyTokenizer()
    model = _TinySemanticModel(config).to(torch.bfloat16)
    codec_config = dataclasses.replace(
        default_mosstts_codec_config(config),
        codebook_size=config.audio_vocab_size,
    )
    return MossTTSRuntime(
        model=model,
        tokenizer=tokenizer,
        processor=MossTTSProcessor(config, tokenizer),
        codec=_RecordingCodec(codec_config),
    )


class DelayGenerationTests(unittest.TestCase):

    def test_reference_free_prompt_generates(self):
        # The official generate() treats a missing <|audio_start|> as -1
        # (find_last_equal_C); direct TTS prompts contain none.
        config = _tiny_tts_config("delay")
        torch.manual_seed(0)
        model = build_mosstts_model(config).eval()
        prompt = MossTTSProcessor(config, _TinyTokenizer()).build_generation_prompt("hello")
        self.assertFalse(bool(prompt.input_ids[..., 0].eq(config.audio_start_token_id).any()))

        output = model.generate(
            prompt.input_ids,
            attention_mask=prompt.attention_mask,
            max_new_tokens=4,
            text_temperature=0.0,
            audio_temperature=0.0,
        )

        self.assertEqual(len(output), 1)
        start_length, sequence = output[0]
        prompt_length = prompt.input_ids.shape[1]
        last_im_start = int(torch.where(prompt.input_ids[0, :, 0].eq(config.im_start_token_id))[0][-1])
        self.assertEqual(start_length, prompt_length - (last_im_start + 3))
        self.assertEqual(sequence.shape[-1], config.channels)
        self.assertEqual(sequence.shape[0], start_length + 4)

    def test_find_last_equal_optional_sentinel(self):
        ids = torch.tensor([[1, 3, 3, 2], [1, 2, 2, 2]])
        self.assertEqual(_find_last_equal(ids, 3, required=False).tolist(), [2, -1])
        with self.assertRaisesRegex(ValueError, "must contain"):
            _find_last_equal(ids, 3)


class DelayPromptTests(unittest.TestCase):

    def test_reference_slots_are_speaker_labelled(self):
        config = _tiny_tts_config("delay")
        processor = MossTTSProcessor(config, _TinyTokenizer())
        content = processor._render_user(
            "hello",
            reference_count=2,
            instruction=None,
            duration_tokens=None,
            quality=None,
            sound_event=None,
            ambient_sound=None,
            language=None,
        )
        self.assertIn(
            f"- Reference(s):\n[S1]:\n{AUDIO_PLACEHOLDER}\n[S2]:\n{AUDIO_PLACEHOLDER}\n- Instruction:",
            content,
        )

    def test_reference_prompt_token_layout(self):
        config = _tiny_tts_config("delay")
        tokenizer = _TinyTokenizer()
        processor = MossTTSProcessor(config, tokenizer)
        reference = torch.zeros(2, config.n_vq, dtype=torch.long)
        rows = processor.build_generation_prompt(
            "hi",
            reference_codes=(reference, ),
        ).input_ids[0]
        text = rows[:, 0].tolist()
        start = text.index(config.audio_start_token_id)
        prefix = tokenizer.encode_ids("<|im_start|>user\n<user_inst>\n- Reference(s):\n[S1]:\n")
        self.assertEqual(text[:start], prefix)
        # Delay-pattern user block: (frames + n_vq - 1) user slots.
        self.assertEqual(
            text[start + 1:start + 1 + 2 + config.n_vq - 1],
            [config.audio_user_slot_token_id] * (2 + config.n_vq - 1),
        )

    def test_v15_text_normalizer_is_applied_only_when_enabled(self):
        config = _tiny_tts_config("delay")
        tokenizer = _TinyTokenizer()
        text = "Line one\nLine   two"
        plain = MossTTSProcessor(config, tokenizer).build_generation_prompt(text)
        normalized = MossTTSProcessor(
            config,
            tokenizer,
            normalize_text=True,
        ).build_generation_prompt(text)
        expected = MossTTSProcessor(config, tokenizer).build_generation_prompt("Line one。Line two")
        self.assertTrue(torch.equal(normalized.input_ids, expected.input_ids))
        self.assertFalse(torch.equal(plain.input_ids, normalized.input_ids))

    def test_normalizer_matches_upstream_cases(self):
        cases = (
            ("This   is   a   test.", "This is a test."),
            ("这 是　一 段  含有多种空白的文本。", "这是一段含有多种空白的文本。"),
            ("这是Anthropic的npm包", "这是 Anthropic 的 npm 包"),
            ("请求接入 -> 身份与策略判定 -> 域服务处理", "请求接入，身份与策略判定，域服务处理"),
            ("真的假的？？？！！！", "真的假的？！"),
            ("- 修复 .map 泄露\n- 发布 v2.3.1", "修复 .map 泄露。发布 v2.3.1"),
            ("[S1]你好。[S2]收到。", "[S1]你好。[S2]收到。"),
            ("The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog."),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_tts_text(raw), expected)
                self.assertEqual(normalize_tts_text(expected), expected)

    def test_only_v15_release_enables_normalizer(self):
        artifacts = SimpleNamespace(source="OpenMOSS-Team/MOSS-TTS-v1.5", root=Path("/nonexistent"))
        self.assertTrue(uses_source_text_normalizer(artifacts))
        artifacts = SimpleNamespace(source="OpenMOSS-Team/MOSS-TTS", root=Path("/nonexistent"))
        self.assertFalse(uses_source_text_normalizer(artifacts))


class ReferenceAudioTests(unittest.TestCase):

    def test_loudness_normalize_matches_source_formula(self):
        quiet = torch.full((1, 100), 0.01)
        loud = torch.full((1, 100), 0.5)
        mid = torch.full((1, 100), 0.08)
        # -40 dBFS -> clamped +3 dB; ~-6 dBFS -> clamped -3 dB.
        self.assertTrue(torch.allclose(loudness_normalize(quiet), quiet * 10**(3 / 20)))
        self.assertTrue(torch.allclose(loudness_normalize(loud), loud * 10**(-3 / 20)))
        gain = -20.0 - 10.0 * math.log10(0.08**2 + 1e-9)
        self.assertTrue(torch.allclose(loudness_normalize(mid), mid * 10**(gain / 20)))

    def test_reference_is_encoded_in_float32_with_source_gain(self):
        runtime = _tiny_runtime()
        waveform = torch.linspace(-0.3, 0.3, 4_800).unsqueeze(0) * 0.123456789
        runtime.encode_reference(waveform)
        sent = runtime.codec.encoded[-1]
        self.assertEqual(sent.dtype, torch.float32)
        self.assertTrue(torch.equal(sent[0], loudness_normalize(waveform)))

    def test_reference_loading_resamples_like_torchaudio_default(self):
        runtime = _tiny_runtime()
        waveform = torch.randn(1_600, generator=torch.Generator().manual_seed(0)) * 0.1
        loaded = runtime.load_reference_audio({"array": waveform, "sampling_rate": 16_000})
        self.assertTrue(torch.equal(loaded[0], resample_waveform_hann(waveform, 16_000, 24_000)))

    def test_decode_uses_source_streaming_chunks(self):
        runtime = _tiny_runtime()
        runtime.decode_codes(torch.zeros(3, runtime.config.n_vq, dtype=torch.long))
        self.assertEqual(runtime.codec.decode_kwargs[-1], {"chunk_duration": SOURCE_DECODE_CHUNK_SECONDS})
        self.assertEqual(SOURCE_DECODE_CHUNK_SECONDS, 8.0)


@unittest.skipUnless(TORCHAUDIO_AVAILABLE, "torchaudio is used only as an audit reference")
class HannResamplerTests(unittest.TestCase):

    def test_matches_torchaudio_default_resampler_exactly(self):
        import torchaudio

        generator = torch.Generator().manual_seed(0)
        for source, target, length in ((16_000, 24_000, 7_704), (44_100, 24_000, 4_321), (48_000, 16_000, 999),
                                       (22_050, 48_000, 1_000)):
            with self.subTest(source=source, target=target):
                waveform = torch.randn(2, length, generator=generator) * 0.3
                expected = torchaudio.functional.resample(waveform, source, target)
                self.assertTrue(torch.equal(resample_waveform_hann(waveform, source, target), expected))


if __name__ == "__main__":
    unittest.main()
