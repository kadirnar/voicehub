"""Regression tests for MOSS-TTS behavior audited against the official OpenMOSS
processors and ``MossTTSDelayModel.generate``."""

import dataclasses
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_native_mosstts import _tiny_tts_config, _TinySemanticModel, _TinyTokenizer  # noqa: E402

from voicehub.architectures.mosstts.codec import MossCodecDecodeOutput, MossCodecEncodeOutput  # noqa: E402
from voicehub.architectures.mosstts.modeling import _find_last_equal, build_mosstts_model  # noqa: E402
from voicehub.architectures.mosstts.processing import AUDIO_PLACEHOLDER, MossTTSProcessor  # noqa: E402
from voicehub.architectures.mosstts.runtime import (  # noqa: E402
    SOURCE_DECODE_CHUNK_SECONDS,
    MossTTSRuntime,
    default_mosstts_codec_config,
    load_mosstts_runtime,
    loudness_normalize,
    uses_source_text_normalizer,
)
from voicehub.architectures.mosstts.sampling import sample_delay_token  # noqa: E402
from voicehub.architectures.mosstts.text_normalization import normalize_tts_text  # noqa: E402
from voicehub.architectures.mosstts.tokenization import MossTextTokenizer  # noqa: E402
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

    def test_unpadded_generation_uses_maskless_attention(self):
        # The official SDPA path drops an all-ones mask; passing it would
        # select VoiceHub's float32 masked-attention fallback and change
        # bfloat16 numerics (greedy tokens diverged from the source).
        config = _tiny_tts_config("delay")
        model = build_mosstts_model(config).eval()
        prompt = MossTTSProcessor(config, _TinyTokenizer()).build_generation_prompt("hello")
        seen = []
        forward = model.forward

        def recording_forward(*args, **kwargs):
            seen.append(kwargs.get("attention_mask"))
            return forward(*args, **kwargs)

        model.forward = recording_forward
        model.generate(
            prompt.input_ids,
            attention_mask=prompt.attention_mask,
            max_new_tokens=3,
            text_temperature=0.0,
            audio_temperature=0.0,
        )
        self.assertEqual(seen, [None, None, None])

        seen.clear()
        padded = prompt.attention_mask.clone()
        padded[:, 0] = False
        model.generate(
            prompt.input_ids,
            attention_mask=padded,
            max_new_tokens=2,
            text_temperature=0.0,
            audio_temperature=0.0,
        )
        self.assertTrue(all(mask is not None for mask in seen))

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

    def test_recorded_normalizer_flag_takes_precedence(self):
        v15 = SimpleNamespace(source="OpenMOSS-Team/MOSS-TTS-v1.5", root=Path("/nonexistent"))
        local = SimpleNamespace(source="/exported", root=Path("/nonexistent"))
        self.assertTrue(uses_source_text_normalizer(local, {"voicehub_text_normalizer": True}))
        self.assertFalse(uses_source_text_normalizer(v15, {"voicehub_text_normalizer": False}))
        self.assertTrue(uses_source_text_normalizer(v15, {}))
        with self.assertRaises(TypeError):
            uses_source_text_normalizer(local, {"voicehub_text_normalizer": "yes"})

    def test_save_pretrained_round_trip_keeps_text_normalizer(self):
        config = _tiny_tts_config("delay")
        codec_config = dataclasses.replace(
            default_mosstts_codec_config(config),
            codebook_size=config.audio_vocab_size,
        )
        control_tokens = {
            "<|endoftext|>": config.pad_token_id,
            "<|im_start|>": config.im_start_token_id,
            "<|im_end|>": config.im_end_token_id,
            "<|audio_start|>": config.audio_start_token_id,
            "<|audio_end|>": config.audio_end_token_id,
            "<|audio_user_slot|>": config.audio_user_slot_token_id,
            "<|audio_assistant_gen_slot|>": config.audio_assistant_slot_token_id,
            "<|audio_assistant_delay_slot|>": config.audio_assistant_delay_slot_token_id,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "vocab.json").write_text(json.dumps({"a": 10, "b": 11, "ab": 12}))
            (source / "merges.txt").write_text("#version: 0.2\na b\n")
            (source / "tokenizer_config.json").write_text(
                json.dumps({
                    "added_tokens_decoder": {
                        str(token_id): {
                            "content": spelling,
                            "special": True
                        }
                        for spelling, token_id in control_tokens.items()
                    }
                }))
            tokenizer = MossTextTokenizer.from_files(
                source / "vocab.json",
                source / "merges.txt",
                source / "tokenizer_config.json",
                model_config=config,
            )
            for normalize_text in (True, False):
                with self.subTest(normalize_text=normalize_text):
                    runtime = MossTTSRuntime(
                        model=build_mosstts_model(config, initialize=True, dtype=torch.float32),
                        tokenizer=tokenizer,
                        processor=MossTTSProcessor(config, tokenizer, normalize_text=normalize_text),
                        codec=_RecordingCodec(codec_config),
                    )
                    # The export carries no normalizer script and its local
                    # path is not a known release, so only config.json can
                    # preserve the processor behavior.
                    exported = runtime.save_pretrained(Path(directory) / f"export-{normalize_text}")
                    self.assertFalse((exported / "tts_robust_normalizer_single_script.py").exists())
                    reloaded = load_mosstts_runtime(
                        exported,
                        compute_dtype="float32",
                        codec=_RecordingCodec(codec_config),
                    )
                    self.assertIs(reloaded.processor.normalize_text, normalize_text)
                    self.assertNotIn("voicehub_text_normalizer", reloaded.model.config.extra_config)


class LocalPromptTests(unittest.TestCase):
    """The original Local release is frame-aligned (no delay pattern)."""

    def test_generation_prompt_opens_audio_and_keeps_reference_frames(self):
        config = _tiny_tts_config("local")
        processor = MossTTSProcessor(config, _TinyTokenizer())
        reference = torch.arange(3 * config.n_vq).remainder(8).view(3, config.n_vq)

        rows = processor.build_generation_prompt("hi", reference_codes=(reference, )).input_ids[0]

        text = rows[:, 0]
        self.assertEqual(int(text[-1]), config.audio_start_token_id)
        self.assertTrue(rows[-1, 1:].eq(config.audio_pad_token_id).all())
        start = int(torch.where(text.eq(config.audio_start_token_id))[0][0])
        end = int(torch.where(text.eq(config.audio_end_token_id))[0][0])
        self.assertEqual(end - start - 1, reference.shape[0])
        self.assertTrue(text[start + 1:end].eq(config.audio_user_slot_token_id).all())
        self.assertTrue(torch.equal(rows[start + 1:end, 1:], reference))

        direct = processor.build_generation_prompt("hi").input_ids[0]
        self.assertEqual(int(direct[-1, 0]), config.audio_start_token_id)

    def test_training_target_and_decode_are_frame_aligned(self):
        config = _tiny_tts_config("local")
        processor = MossTTSProcessor(config, _TinyTokenizer())
        speech = torch.arange(4 * config.n_vq).remainder(8).view(4, config.n_vq)

        record = processor.build_training_record(text="hi", speech_tokens=speech)
        full = torch.cat([record.input_ids[0], record.labels[0, -1:].clamp_min(0)])
        text = full[:, 0]
        self.assertEqual(int(text.eq(config.audio_start_token_id).sum()), 1)
        start = int(torch.where(text.eq(config.audio_start_token_id))[0][0])
        self.assertTrue(text[start + 1:start + 5].eq(config.audio_assistant_slot_token_id).all())
        self.assertTrue(torch.equal(full[start + 1:start + 5, 1:], speech))
        self.assertEqual(int(text[start + 5]), config.audio_end_token_id)

        generated = torch.cat([
            processor._text_rows([config.audio_start_token_id]),
            processor._audio_rows(speech, text_token_id=config.audio_assistant_slot_token_id),
            processor._text_rows([config.audio_end_token_id]),
        ])
        decoded = processor.decode_generated([(0, generated)])
        self.assertTrue(torch.equal(decoded[0].audio_codes, speech))


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


def _vendored_source_sampling():
    path = (
        Path(__file__).resolve().parents[1] /
        "voicehub/models/mosstts/source/moss_tts_delay/inference_utils.py")
    spec = importlib.util.spec_from_file_location("_moss_source_inference_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DelaySamplingTests(unittest.TestCase):

    def test_seeded_draws_match_source_sample_token(self):
        source = _vendored_source_sampling()
        generator = torch.Generator().manual_seed(3)
        for rows, top_k, top_p, temperature in ((1, 50, 1.0, 1.5), (31, 25, 0.8, 1.7), (0, 25, 0.8, 1.7)):
            with self.subTest(rows=rows, top_k=top_k, top_p=top_p):
                logits = (torch.randn(rows, 1025, generator=generator) * 4).to(torch.bfloat16)
                scaled = logits / temperature
                torch.manual_seed(11)
                expected = source.sample_token(scaled.clone(), top_p=top_p, top_k=top_k, do_sample=True)
                torch.manual_seed(11)
                actual = sample_delay_token(scaled.clone(), do_sample=True, top_k=top_k, top_p=top_p)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(
                    torch.equal(
                        sample_delay_token(scaled, do_sample=False, top_k=top_k, top_p=top_p),
                        source.sample_token(scaled.clone(), top_p=top_p, top_k=top_k, do_sample=False),
                    ))


class RotaryFrequencyTests(unittest.TestCase):

    def test_meta_construction_stays_lazy(self):
        from voicehub.neural.rotary import RotaryEmbedding

        self.assertEqual(RotaryEmbedding(128, base=1e6, device="meta").inverse_frequency.device.type, "meta")
        with torch.device("meta"):
            ambient = RotaryEmbedding(128, base=1e6)
        self.assertEqual(ambient.inverse_frequency.device.type, "meta")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required to compare device arithmetic")
    def test_cuda_frequencies_match_cpu_reference(self):
        # CUDA pow rounded inverse_frequency[18] differently for Qwen3's
        # base 1e6; at position 459 that flipped a bfloat16 sine near 3*pi
        # and made long greedy MOSS generations diverge from upstream.
        from voicehub.neural.rotary import RotaryEmbedding

        cpu = RotaryEmbedding(128, base=1e6).inverse_frequency
        cuda = RotaryEmbedding(128, base=1e6, device="cuda").inverse_frequency
        self.assertTrue(torch.equal(cuda.cpu(), cpu))


class HubSnapshotLayoutTests(unittest.TestCase):

    def test_local_snapshot_with_blob_symlinks_resolves(self):
        import json
        import tempfile

        from voicehub.architectures.mosstts.artifacts import resolve_mosstts_artifacts
        from voicehub.architectures.mosstts.checkpoint import inspect_mosstts_checkpoint
        from voicehub.checkpointing.safetensors import save_safetensors

        with tempfile.TemporaryDirectory() as directory:
            # Resolved: macOS /var is a symlink and Windows may use 8.3 names.
            repository = Path(directory).resolve() / "models--org--model"
            blobs = repository / "blobs"
            snapshot = repository / "snapshots" / ("a" * 40)
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            payloads = {
                "config.json":
                b"{}",
                "vocab.json":
                b"{}",
                "merges.txt":
                b"#version: 0.2\n",
                "tokenizer_config.json":
                b"{}",
                "model.safetensors.index.json":
                json.dumps({
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    }
                }).encode(),
            }
            for index, (name, payload) in enumerate(payloads.items()):
                (blobs / f"blob{index}").write_bytes(payload)
                (snapshot / name).symlink_to(Path("../..") / "blobs" / f"blob{index}")
            for index, name in enumerate(("a", "b"), start=1):
                blob = blobs / f"shard{index}"
                save_safetensors({name: torch.zeros(2)}, blob.with_suffix(".safetensors"))
                blob.with_suffix(".safetensors").rename(blob)
                (snapshot / f"model-0000{index}-of-00002.safetensors").symlink_to(
                    Path("../..") / "blobs" / blob.name)

            artifacts = resolve_mosstts_artifacts(snapshot)
            self.assertEqual(artifacts.checkpoint.name, "model.safetensors.index.json")
            self.assertEqual(artifacts.checkpoint.parent, snapshot.absolute())
            report = inspect_mosstts_checkpoint(artifacts.checkpoint)
            self.assertEqual(report.tensor_count, 2)


@unittest.skipUnless(TORCHAUDIO_AVAILABLE, "torchaudio is used only as an audit reference")
class HannResamplerTests(unittest.TestCase):

    def test_matches_torchaudio_default_resampler_exactly(self):
        import torchaudio

        generator = torch.Generator().manual_seed(0)
        for source, target, length in ((16_000, 24_000, 7_704), (44_100, 24_000, 4_321),
                                       (48_000, 16_000, 999), (22_050, 48_000, 1_000)):
            with self.subTest(source=source, target=target):
                waveform = torch.randn(2, length, generator=generator) * 0.3
                expected = torchaudio.functional.resample(waveform, source, target)
                self.assertTrue(torch.equal(resample_waveform_hann(waveform, source, target), expected))


if __name__ == "__main__":
    unittest.main()
