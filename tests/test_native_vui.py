from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.models.vui.artifacts import VUI_CODEC_FILENAME, VUI_MODEL_FILENAME, VUI_REVISION, resolve_vui_artifacts
from voicehub.models.vui.config import Config, VuiConfig
from voicehub.models.vui.fluac import FSQ, Fluac, FluacConfig
from voicehub.models.vui.model import MHA, Vui
from voicehub.models.vui.notebook import play, plot_mel_spec
from voicehub.models.vui.patterns import DelayedPatternProvider
from voicehub.models.vui.rope import apply_rotary_emb, precompute_freqs_cis
from voicehub.models.vui.tok import CustomByT5Tokenizer
from voicehub.models.vui.tts import number_to_words, replace_numbers_with_words
from voicehub.models.vui.vad import (
    Binarize,
    EnergyVADPipeline,
    SlidingWindow,
    SlidingWindowFeature,
    detect_voice_activity,
)


class NativeVuiTests(unittest.TestCase):

    def tearDown(self):
        import voicehub.models.vui.vad as native_vad

        native_vad.pipeline = None

    def test_public_vui_modules_import_only_stdlib_torch_and_voicehub(self):
        package = Path(__file__).parents[1] / "voicehub" / "models" / "vui"
        allowed_roots = {
            "__future__",
            "abc",
            "array",
            "base64",
            "collections",
            "contextlib",
            "dataclasses",
            "functools",
            "hashlib",
            "html",
            "io",
            "logging",
            "math",
            "os",
            "pathlib",
            "re",
            "sys",
            "tempfile",
            "torch",
            "typing",
            "urllib",
            "voicehub",
            "wave",
        }
        violations = []
        for path in sorted(package.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports = [node.module]
                else:
                    continue
                for module in imports:
                    if module.split(".", 1)[0] not in allowed_roots:
                        violations.append((path.name, node.lineno, module))
        self.assertEqual(violations, [])

    def test_config_is_validated_and_serializes_without_pydantic(self):
        config = Config.from_dict({
            "name": "tiny",
            "model": {
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 2,
                "unknown_future_metadata": "ignored",
            },
            "upstream_metadata": "ignored",
        })
        self.assertIsInstance(config.model, VuiConfig)
        self.assertEqual(config.model.d_model, 32)
        self.assertEqual(
            Config.from_dict(config.model_dump()).model_dump(),
            config.model_dump(),
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            VuiConfig(d_model=30, n_heads=4)

        codec = FluacConfig.from_dict({
            "sample_rate": 22_050,
            "encoder_rates": [2, 4, 8, 8],
            "future_metadata": True,
        })
        self.assertEqual(codec.hop_length, 512)
        self.assertEqual(codec.effective_codebook_size, 1_000)
        self.assertEqual(
            FluacConfig.from_dict(codec.model_dump()).model_dump(),
            codec.model_dump(),
        )

    def test_artifact_resolver_pins_model_and_codec_to_one_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / VUI_MODEL_FILENAME
            codec_path = root / VUI_CODEC_FILENAME
            model_path.touch()
            codec_path.touch()
            commit = "1" * 40
            with (
                    patch(
                        "voicehub.models.vui.artifacts.resolve_pretrained_file",
                        side_effect=(model_path, codec_path),
                    ) as resolve,
                    patch(
                        "voicehub.models.vui.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=commit,
                    ),
            ):
                artifacts = resolve_vui_artifacts(
                    VUI_MODEL_FILENAME,
                    verify_official_integrity=False,
                )

        self.assertTrue(artifacts.official)
        self.assertEqual(artifacts.revision, commit)
        self.assertEqual(resolve.call_args_list[0].kwargs["revision"], VUI_REVISION)
        self.assertEqual(resolve.call_args_list[1].kwargs["revision"], commit)

    def test_local_artifact_resolver_requires_the_matching_codec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / VUI_MODEL_FILENAME).touch()
            with self.assertRaisesRegex(FileNotFoundError, "Fluac codec"):
                resolve_vui_artifacts(root)
            (root / VUI_CODEC_FILENAME).touch()
            artifacts = resolve_vui_artifacts(root)

        self.assertFalse(artifacts.official)
        self.assertIsNone(artifacts.revision)

    def test_registry_formats_match_the_complete_native_checkpoint_lifecycle(self):
        from voicehub.architectures.vui.registration import create_vui_architecture_spec
        from voicehub.registry import get_model_spec

        capabilities = create_vui_architecture_spec().capabilities
        self.assertEqual(
            capabilities.checkpoint_formats,
            ("pytorch", "safetensors"),
        )
        self.assertEqual(capabilities.export_formats, ("safetensors", ))
        self.assertIn(
            "torch-compile-inference-unsafe",
            capabilities.features,
        )
        self.assertIn("safetensors", get_model_spec("vui").capabilities)
        self.assertIn(
            "standalone-safetensors-export",
            get_model_spec("vui").capabilities,
        )

    def test_native_safetensors_export_round_trips_model_codec_and_configs(self):
        from voicehub.checkpointing import SafeTensorReader, save_safetensors
        from voicehub.models.vui.checkpoint import VUI_NATIVE_CODEC_FILENAME, VUI_NATIVE_MODEL_FILENAME
        from voicehub.models.vui.inference import VuiForTextToSpeech

        model_config = Config(
            name="tiny-roundtrip",
            model=VuiConfig(
                max_text_tokens=8,
                max_audio_tokens=8,
                n_quantizers=1,
                codebook_size=4,
                special_token_id=4,
                audio_eos_id=5,
                audio_pad_id=6,
                d_model=8,
                n_layers=1,
                n_heads=2,
            ),
        )
        codec_config = FluacConfig(
            sample_rate=8_000,
            encoder_dim=2,
            encoder_rates=[2],
            n_quantizers=1,
            fsq_levels=[2, 2],
            decoder_dim=8,
            decoder_rates=[2],
        )
        original = Vui(
            model_config,
            codec=Fluac(codec_config),
        ).eval()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original.save_pretrained(root)

            with SafeTensorReader(root / VUI_NATIVE_MODEL_FILENAME) as reader:
                self.assertEqual(
                    reader.metadata,
                    {
                        "component": "model",
                        "format": "voicehub-vui",
                        "format_version": "1",
                    },
                )
                self.assertFalse(any(name.startswith("codec.") for name in reader.keys()))
            with SafeTensorReader(root / VUI_NATIVE_CODEC_FILENAME) as reader:
                self.assertEqual(reader.metadata["component"], "codec")

            restored = Vui.from_pretrained(root)
            self.assertEqual(
                restored.config.model_dump(),
                original.config.model_dump(),
            )
            self.assertEqual(
                restored.codec.config.model_dump(),
                original.codec.config.model_dump(),
            )
            for name, expected in original.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[name],
                    expected,
                )

            wrapper = VuiForTextToSpeech.from_pretrained(
                root,
                device="cpu",
                lazy_load=False,
            )
            self.assertEqual(wrapper.sample_rate, 8_000)
            self.assertEqual(
                wrapper.model.config.model_dump(),
                original.config.model_dump(),
            )

            model_state = {
                name: value
                for name, value in original.state_dict().items() if not name.startswith("codec.")
            }
            save_safetensors(
                model_state,
                root / VUI_NATIVE_MODEL_FILENAME,
            )
            with self.assertRaisesRegex(ValueError, "metadata is incompatible"):
                Vui.from_pretrained(root)

    def test_native_byt5_boundary_matches_pinned_reference_vectors(self):
        tokenizer = CustomByT5Tokenizer()
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertEqual(len(tokenizer), 384)
        self.assertEqual(tokenizer.encode("A").tolist(), [68])
        self.assertEqual(tokenizer.encode("é").tolist(), [198, 172])
        self.assertEqual(
            tokenizer("A").input_ids.tolist(),
            [68, tokenizer.eos_token_id],
        )
        batch = tokenizer(
            ["A", "Hi"],
            padding="longest",
            return_tensors="pt",
        )
        self.assertEqual(batch.input_ids.tolist(), [[68, 1, 0], [75, 108, 1]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(tokenizer.decode([75, 108, 1]), "Hi")

    def test_native_rope_matches_direct_interleaved_equation(self):
        frequencies = precompute_freqs_cis(4, 3)
        values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        actual = apply_rotary_emb(frequencies[:1], values)
        pairs = values.reshape(1, 1, 1, 2, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
        expected = (values * frequencies[:1].cos() + rotated.flatten(start_dim=-2) * frequencies[:1].sin())
        torch.testing.assert_close(actual, expected)

    def test_mha_supports_equal_heads_and_future_grouped_query_layouts(self):
        for heads, kv_heads in ((4, 4), (4, 2)):
            with self.subTest(heads=heads, kv_heads=kv_heads):
                layer = MHA(
                    16,
                    heads,
                    kv_heads,
                    block_idx=0,
                    causal=True,
                    use_rotary_emb=False,
                )
                values = torch.randn(2, 5, 16, requires_grad=True)
                output = layer(values)
                self.assertEqual(output.shape, values.shape)
                output.square().mean().backward()
                self.assertIsNotNone(values.grad)
                self.assertTrue(torch.isfinite(values.grad).all())

    def test_fsq_tensor_transforms_preserve_shapes_and_gradients(self):
        quantizer = FSQ(
            levels=[8, 5, 5, 5],
            dim=4,
            channel_first=True,
        )
        values = torch.randn(2, 4, 7, requires_grad=True)
        output, indices = quantizer(values)
        self.assertEqual(output.shape, values.shape)
        self.assertEqual(indices.shape, (2, 7))
        reconstructed = quantizer.indices_to_codes(indices)
        torch.testing.assert_close(output, reconstructed)
        output.mean().backward()
        self.assertIsNotNone(values.grad)

    def test_delayed_pattern_uses_native_tensor_indexes(self):
        codes = torch.arange(2 * 3 * 5).reshape(2, 3, 5)
        pattern = DelayedPatternProvider(n_q=3).get_pattern(5)
        sequence, indexes, mask = pattern.build_pattern_sequence(
            codes,
            special_token=999,
        )
        restored, _, restored_mask = pattern.revert_pattern_sequence(
            sequence,
            special_token=-1,
        )
        self.assertEqual(indexes.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)
        torch.testing.assert_close(
            restored[restored_mask.expand_as(restored)], codes[restored_mask.expand_as(codes)])

    def test_native_vad_and_hysteresis_return_directly_visible_segments(self):
        waveform = torch.cat((
            torch.zeros(1_600),
            torch.full((3_200, ), 0.2),
            torch.zeros(1_600),
        ))
        regions = detect_voice_activity(waveform, pipe=EnergyVADPipeline())
        self.assertEqual(len(regions), 1)
        self.assertLessEqual(regions[0][0], 0.11)
        self.assertGreaterEqual(regions[0][1], 0.29)

        scores = SlidingWindowFeature(
            [[0.1], [0.9], [0.8], [0.2]],
            SlidingWindow(start=0.0, duration=0.1, step=0.1),
        )
        annotation = Binarize(onset=0.5, offset=0.4)(scores)
        segment = annotation.get_timeline()[0]
        self.assertAlmostEqual(segment.start, 0.15)
        self.assertAlmostEqual(segment.end, 0.35)

    def test_binarize_compares_thresholds_in_float32_like_pyannote(self):
        # pyannote.audio 3.3.2 reference: [(0.25, 0.45)]. A float32 score equal
        # to float32(0.8) is not above the onset.
        scores = SlidingWindowFeature(
            torch.tensor([[0.1], [0.8], [0.9], [0.85], [0.49], [0.95]]),
            SlidingWindow(start=0.0, duration=0.1, step=0.1),
        )
        timeline = Binarize(onset=0.8, offset=0.5)(scores).get_timeline().support()
        self.assertEqual([(segment.start, segment.end) for segment in timeline], [(0.25, 0.45)])

    def test_pyannote_vad_aggregation_matches_pyannote_inference(self):
        import hashlib

        from voicehub.models.vui.vad import PyannoteVADPipeline, aggregate_chunk_scores

        frames = PyannoteVADPipeline.receptive_field()
        self.assertEqual(
            (frames.start, frames.duration, frames.step),
            (0.0, 991 / 16_000, 270 / 16_000),
        )
        index = torch.arange(4 * 293, dtype=torch.float64).reshape(4, 293, 1)
        scores = ((torch.sin(index * 0.37) + 1) / 2).float()
        aggregated = aggregate_chunk_scores(scores, chunk_duration=5.0, chunk_step=0.5, frames=frames)
        # Inference.aggregate(..., hamming=True, missing=0.0) from
        # pyannote.audio 3.3.2 on the same float32 scores.
        self.assertEqual(tuple(aggregated.shape), (386, 1))
        self.assertEqual(
            hashlib.sha256(aggregated.numpy().tobytes()).hexdigest(),
            "738bc6cc204dc8daef7cbc7f39f8bd1bf92cb076368abd615e0612b78717f644",
        )
        self.assertEqual(float(aggregated[1, 0]), 0.6808077096939087)

    def test_pyannote_vad_pipeline_chunks_like_pyannote_inference(self):
        from voicehub.models.vui.vad import PyannoteVADPipeline

        class FakeSegmentation(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.batches = []

            def forward(self, chunks):
                self.batches.append(tuple(chunks.shape))
                return torch.full((chunks.shape[0], 293, 3), 0.9)

        model = FakeSegmentation()
        pipeline = PyannoteVADPipeline(model=model)
        # Frame counts produced by upstream's pyannote pipeline on Vui audio.
        for samples, frames in ((84_707, 314), (154_553, 573)):
            with self.subTest(samples=samples):
                self.assertEqual(pipeline.frame_scores(torch.zeros(samples)).data.shape[0], frames)
        model.batches.clear()
        pipeline.frame_scores(torch.zeros(400_000))
        self.assertEqual(model.batches, [(32, 80_000), (9, 80_000)])
        model.batches.clear()
        regions = detect_voice_activity(torch.zeros(84_707), pipe=pipeline)
        self.assertEqual(regions, [(0.03096875, 0.03096875 + 313 * 270 / 16_000)])

    def test_vad_backend_is_validated(self):
        from voicehub.models.vui.inference import VuiConfig
        from voicehub.models.vui.vad import EnergyVADPipeline, PyannoteVADPipeline, create_vad_pipeline

        self.assertEqual(VuiConfig().vad_backend, "pyannote")
        self.assertIsInstance(create_vad_pipeline("pyannote"), PyannoteVADPipeline)
        self.assertIsInstance(create_vad_pipeline("energy"), EnergyVADPipeline)
        with self.assertRaises(ValueError):
            VuiConfig(vad_backend="silero")

    def test_number_normalization_matches_pinned_inflect_examples(self):
        expected = {
            "0": "zero",
            "21": "twenty-one",
            "101": "one hundred and one",
            "1001": "one thousand and one",
            "12345": "twelve thousand, three hundred and forty-five",
        }
        for value, words in expected.items():
            with self.subTest(value=value):
                self.assertEqual(number_to_words(value), words)
        self.assertEqual(
            replace_numbers_with_words("Room 21"),
            "Room twenty-one ",
        )

    def test_notebook_views_need_no_display_or_array_library(self):
        audio = play(torch.linspace(-1.0, 1.0, 200), sr=16_000)
        self.assertIn("data:audio/wav;base64,", audio._repr_html_())
        view = plot_mel_spec(torch.arange(12).reshape(3, 4), title="Vui")
        svg = view._repr_svg_()
        self.assertIn("<svg", svg)
        self.assertIn("Vui", svg)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unused"
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
