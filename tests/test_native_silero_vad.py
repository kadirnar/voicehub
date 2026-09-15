from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from voicehub.architectures.registry import ArchitectureRegistry
    from voicehub.architectures.silero_vad import (
        OFFICIAL_SILERO_VAD_16K_HEADER_FINGERPRINT,
        OFFICIAL_SILERO_VAD_REVISION,
        OfficialSileroVADSafeTensorsCheckpointAdapter,
        OfficialSileroVADTorchScriptCheckpointAdapter,
        SileroVADBinaryCrossEntropyLoss,
        SileroVADConfig,
        SileroVADModel,
        SileroVADSegmentationConfig,
        SileroVADState,
        SileroVADStream,
        SpeechSegment,
        create_silero_vad_architecture_spec,
        native_silero_vad_tensor_shapes,
        official_safetensors_tensor_mapping,
        official_torchscript_tensor_mapping,
        register_silero_vad_architecture,
        segment_speech_probabilities,
        silero_vad_binary_cross_entropy,
        tensor_inventory_fingerprint,
    )


@unittest.skipUnless(torch is not None, "Native Silero VAD uses PyTorch")
class SileroVADConfigurationTests(unittest.TestCase):

    def test_sherpa_scorer_consumes_overlapping_context_without_zero_prefix(self):
        from voicehub.models.vad_sherpa_onnx.streaming import NativeSileroScorer

        model = SileroVADModel(SileroVADConfig(sampling_rate=16_000)).eval()
        scorer = NativeSileroScorer(model)
        self.assertEqual(scorer.window_size, 576)
        self.assertEqual(scorer.window_shift, 512)
        waveform = torch.randn(3 * 512 + 64, generator=torch.Generator().manual_seed(37))
        state = model.initial_state(1)
        expected = []
        with torch.inference_mode():
            recurrent = (state.hidden, state.cell)
            for start in range(0, 3 * 512, 512):
                values = waveform[start:start + 576].unsqueeze(0)
                probabilities, _, recurrent = model.forward_with_context(values, recurrent)
                expected.append(float(probabilities.item()))
        for _ in range(2):
            actual = [scorer.compute(waveform[start:start + 576]) for start in range(0, 3 * 512, 512)]
            self.assertEqual(actual, expected)
            scorer.reset()

    def test_official_sample_rate_layouts_are_derived_and_round_trip(self):
        sixteen = SileroVADConfig.from_dict({
            "sampling_rate": 16_000,
            "future_field": {
                "value": 1
            },
        })
        eight = SileroVADConfig(sampling_rate=8_000)

        self.assertEqual(
            (
                sixteen.frame_size,
                sixteen.context_size,
                sixteen.filter_length,
                sixteen.hop_length,
                sixteen.spectrum_bins,
            ),
            (512, 64, 256, 128, 129),
        )
        self.assertEqual(
            (
                eight.frame_size,
                eight.context_size,
                eight.filter_length,
                eight.hop_length,
                eight.spectrum_bins,
            ),
            (256, 32, 128, 64, 65),
        )
        self.assertEqual(
            SileroVADConfig.from_dict(sixteen.to_dict()),
            sixteen,
        )
        self.assertEqual(sixteen.extra_config["future_field"]["value"], 1)

    def test_invalid_model_dimensions_cannot_be_described(self):
        with self.assertRaisesRegex(ValueError, "8000, 16000"):
            SileroVADConfig(sampling_rate=44_100)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            SileroVADConfig(decoder_dropout=1.0)


@unittest.skipUnless(torch is not None, "Native Silero VAD uses PyTorch")
class SileroVADModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(31)

    def test_frame_graph_returns_explicit_recurrent_and_context_state(self):
        model = SileroVADModel().eval()
        frame = torch.randn(2, 512)

        output = model(frame)

        self.assertEqual(tuple(output.probabilities.shape), (2, 1))
        self.assertEqual(tuple(output.logits.shape), (2, 1))
        self.assertEqual(tuple(output.state.recurrent.shape), (2, 2, 128))
        self.assertEqual(tuple(output.state.context.shape), (2, 64))
        torch.testing.assert_close(output.state.context, frame[:, -64:])
        self.assertTrue(((output.probabilities >= 0.0) & (output.probabilities <= 1.0)).all())

    def test_arbitrary_audio_is_padded_once_and_remains_differentiable(self):
        model = SileroVADModel()
        audio = torch.randn(2, 1_000, requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        output = model.frame_probabilities(audio)
        loss = SileroVADBinaryCrossEntropyLoss(from_logits=True)(
            output.logits,
            targets,
        )
        loss.backward()

        self.assertEqual(tuple(output.probabilities.shape), (2, 2))
        self.assertEqual(output.valid_samples, 1_000)
        self.assertIsNotNone(audio.grad)
        self.assertIsNotNone(model.conv1.weight.grad)
        self.assertIsNotNone(model.lstm_cell.weight_ih.grad)
        self.assertIsNotNone(model.final_conv.weight.grad)
        self.assertFalse(model.stft_conv.weight.requires_grad)
        self.assertIsNone(model.stft_conv.weight.grad)
        torch.testing.assert_close(
            output.state.context[:, -24:],
            torch.zeros(2, 24),
        )

    def test_state_shape_and_frame_size_are_strict(self):
        model = SileroVADModel()
        with self.assertRaisesRegex(ValueError, "exactly 512"):
            model(torch.zeros(1, 511))
        bad_state = SileroVADState(
            hidden=torch.zeros(1, 128),
            cell=torch.zeros(1, 128),
            context=torch.zeros(1, 32),
        )
        with self.assertRaisesRegex(ValueError, "Context"):
            model(torch.zeros(1, 512), state=bad_state)

    def test_complete_audio_matches_public_frames_with_initial_state_and_padding(self):
        for rate in (8_000, 16_000):
            with self.subTest(sampling_rate=rate):
                model = SileroVADModel(SileroVADConfig(sampling_rate=rate)).eval()
                width = model.config.frame_size
                initial = model(torch.randn(2, width)).state
                audio = torch.randn(2, 3 * width + 17)
                actual = model.frame_probabilities(audio, state=initial)
                padded = torch.nn.functional.pad(audio, (0, width - 17))
                state, expected = initial, []
                for frame in padded.split(width, dim=1):
                    output = model(frame, state=state)
                    expected.append(output.probabilities)
                    state = output.state
                torch.testing.assert_close(actual.probabilities, torch.cat(expected, dim=1), rtol=0, atol=0)
                for name in ("hidden", "cell", "context"):
                    torch.testing.assert_close(
                        getattr(actual.state, name), getattr(state, name), rtol=0, atol=0)
                self.assertEqual(actual.valid_samples, audio.shape[1])

    def test_complete_audio_still_validates_before_scoring(self):
        model = SileroVADModel().eval()
        with self.assertRaisesRegex(ValueError, "NaN"):
            model.frame_probabilities(torch.full((1, 1024), float("nan")))
        with self.assertRaisesRegex(TypeError, "float32"):
            model.double().frame_probabilities(torch.zeros(1, 1024))

    def test_stream_sessions_are_isolated_and_resettable(self):
        model = SileroVADModel().eval()
        first = torch.randn(1, 512)
        second = torch.randn(1, 512)
        third = torch.randn(1, 512)
        stream_a = SileroVADStream(model)
        stream_b = SileroVADStream(model)
        control = SileroVADStream(model)

        first_a = stream_a.process(first)
        first_b = stream_b.process(first)
        torch.testing.assert_close(first_a, first_b)
        second_a = stream_a.process(second)
        second_b = stream_b.process(second)
        torch.testing.assert_close(second_a, second_b)

        control.process(first)
        control.process(second)
        snapshot = stream_a.state
        self.assertIsNotNone(snapshot)
        snapshot.hidden.zero_()
        torch.testing.assert_close(
            stream_a.process(third),
            control.process(third),
        )

        stream_a.reset()
        self.assertFalse(stream_a.initialized)
        torch.testing.assert_close(stream_a.process(first), first_a)

    def test_eight_kilohertz_graph_uses_the_released_shapes(self):
        model = SileroVADModel(SileroVADConfig(sampling_rate=8_000)).eval()

        output = model(torch.randn(3, 256))

        self.assertEqual(tuple(model.stft_conv.weight.shape), (130, 1, 128))
        self.assertEqual(tuple(model.conv1.weight.shape), (128, 65, 3))
        self.assertEqual(tuple(output.probabilities.shape), (3, 1))
        self.assertEqual(tuple(output.state.context.shape), (3, 32))


@unittest.skipUnless(torch is not None, "Native Silero VAD uses PyTorch")
class SileroVADCheckpointTests(unittest.TestCase):

    @staticmethod
    def _zero_source(mapping, shapes):
        return {source: torch.zeros(shapes[target]) for source, target in mapping}

    def test_real_safetensors_header_inventory_has_full_native_coverage(self):
        shapes = native_silero_vad_tensor_shapes()
        mapping = official_safetensors_tensor_mapping()

        self.assertEqual(len(shapes), 15)
        self.assertEqual(set(shapes), {target for _, target in mapping})
        self.assertEqual(set(shapes), {source for source, _ in mapping})
        self.assertEqual(
            tensor_inventory_fingerprint(shapes),
            OFFICIAL_SILERO_VAD_16K_HEADER_FINGERPRINT,
        )
        self.assertEqual(shapes["stft_conv.weight"], (258, 1, 256))
        self.assertEqual(shapes["lstm_cell.weight_ih"], (512, 128))

    def test_safetensors_adapter_is_strict_and_16khz_only(self):
        config = SileroVADConfig()
        model = SileroVADModel(config)
        shapes = native_silero_vad_tensor_shapes(config)
        source = self._zero_source(
            official_safetensors_tensor_mapping(config),
            shapes,
        )

        report = OfficialSileroVADSafeTensorsCheckpointAdapter().load(
            model,
            source,
            config.to_dict(),
        )

        self.assertTrue(report.is_compatible)
        self.assertEqual(len(report.loaded), 15)
        with self.assertRaisesRegex(ValueError, "does not release an 8 kHz"):
            official_safetensors_tensor_mapping(SileroVADConfig(sampling_rate=8_000))

    def test_merged_torchscript_state_dict_maps_both_branches_exactly(self):
        for sampling_rate in (8_000, 16_000):
            with self.subTest(sampling_rate=sampling_rate):
                config = SileroVADConfig(sampling_rate=sampling_rate)
                other = SileroVADConfig(sampling_rate=24_000 - sampling_rate)
                selected_shapes = native_silero_vad_tensor_shapes(config)
                other_shapes = native_silero_vad_tensor_shapes(other)
                source = self._zero_source(
                    official_torchscript_tensor_mapping(config),
                    selected_shapes,
                )
                source.update(self._zero_source(
                    official_torchscript_tensor_mapping(other),
                    other_shapes,
                ))

                report = (
                    OfficialSileroVADTorchScriptCheckpointAdapter().load(
                        SileroVADModel(config),
                        source,
                        config.to_dict(),
                    ))

                self.assertTrue(report.is_compatible)
                self.assertEqual(len(report.loaded), 15)
                self.assertEqual(len(report.ignored_sources), 15)


@unittest.skipUnless(torch is not None, "Native Silero VAD uses PyTorch")
class SileroVADObjectiveTests(unittest.TestCase):

    def test_weighted_binary_loss_matches_the_released_recipe(self):
        probabilities = torch.tensor([[0.8, 0.3], [0.2, 0.9]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = torch.tensor([[1.0, 0.5], [0.25, 1.0]])

        loss = silero_vad_binary_cross_entropy(
            probabilities,
            targets,
            weights=weights,
        )
        expected = (
            torch.nn.functional.binary_cross_entropy(
                probabilities,
                targets,
                reduction="none",
            ) * weights).mean()

        torch.testing.assert_close(loss, expected)

    def test_objective_rejects_invalid_probability_targets(self):
        with self.assertRaisesRegex(ValueError, "targets"):
            silero_vad_binary_cross_entropy(
                torch.tensor([0.5]),
                torch.tensor([1.5]),
            )


@unittest.skipUnless(torch is not None, "Native Silero VAD uses PyTorch")
class SileroVADSegmentationTests(unittest.TestCase):

    def test_hysteresis_minimum_durations_and_padding_are_applied(self):
        config = SileroVADSegmentationConfig(
            min_speech_duration_ms=64,
            min_silence_duration_ms=64,
            speech_pad_ms=32,
        )
        probabilities = [
            0.1,
            0.1,
            0.8,
            0.8,
            0.8,
            0.8,
            0.4,
            0.2,
            0.2,
            0.2,
            0.8,
        ]

        segments = segment_speech_probabilities(
            probabilities,
            audio_length_samples=len(probabilities) * 512,
            config=config,
        )

        self.assertEqual(segments, (SpeechSegment(512, 4096), ))

    def test_maximum_duration_splits_continuous_speech(self):
        probabilities = [0.9] * 12
        config = SileroVADSegmentationConfig(
            min_speech_duration_ms=0,
            min_silence_duration_ms=64,
            speech_pad_ms=0,
            max_speech_duration_s=0.2,
        )

        segments = segment_speech_probabilities(
            probabilities,
            audio_length_samples=len(probabilities) * 512,
            config=config,
        )

        self.assertEqual(
            segments,
            (
                SpeechSegment(0, 3072),
                SpeechSegment(3072, 6144),
            ),
        )
        self.assertTrue(all(segment.duration <= int(0.2 * 16_000) for segment in segments))

    def test_maximum_duration_prefers_a_valid_internal_silence(self):
        probabilities = [
            0.9,
            0.9,
            0.1,
            0.1,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
            0.9,
        ]
        config = SileroVADSegmentationConfig(
            min_speech_duration_ms=0,
            min_silence_duration_ms=200,
            min_silence_at_max_speech_ms=32,
            speech_pad_ms=0,
            max_speech_duration_s=0.3,
            use_max_possible_silence=True,
        )

        segments = segment_speech_probabilities(
            probabilities,
            audio_length_samples=len(probabilities) * 512,
            config=config,
        )

        self.assertEqual(segments[0], SpeechSegment(0, 1024))
        self.assertEqual(segments[1].start, 2048)

    def test_probability_count_must_match_audio_length(self):
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            segment_speech_probabilities(
                [0.1],
                audio_length_samples=513,
            )


@unittest.skipUnless(torch is not None, "Native Silero VAD uses PyTorch")
class SileroVADRegistrationTests(unittest.TestCase):

    def test_spec_pins_source_license_capabilities_and_components(self):
        spec = create_silero_vad_architecture_spec()

        self.assertEqual(spec.architecture_id, "silero-vad")
        self.assertEqual(
            spec.upstream_revision,
            OFFICIAL_SILERO_VAD_REVISION,
        )
        self.assertEqual(spec.license_id, "MIT")
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.streaming)
        self.assertTrue(spec.capabilities.supports_task("voice-activity-detection"))
        self.assertEqual(
            spec.model_builder.path,
            "voicehub.architectures.silero_vad.modeling:SileroVADModel",
        )
        self.assertEqual(
            spec.get_component_reference("torchscript-checkpoint-adapter").attribute,
            "OfficialSileroVADTorchScriptCheckpointAdapter",
        )

    def test_registration_supports_isolated_registry_and_aliases(self):
        registry = ArchitectureRegistry()

        spec = register_silero_vad_architecture(registry=registry)

        self.assertIs(registry.get("silero-vad"), spec)
        self.assertIs(registry.get("native-silero-vad"), spec)
        self.assertIs(registry.get("silero"), spec)

    def test_registration_import_does_not_load_graph_or_checkpoint_modules(self):
        script = (
            "import sys; "
            "import voicehub.architectures.silero_vad.registration as r; "
            "r.create_silero_vad_architecture_spec(); "
            "print(int('voicehub.architectures.silero_vad.modeling' in "
            "sys.modules), int('voicehub.architectures.silero_vad.checkpoint' "
            "in sys.modules))")
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(result.stdout.strip(), "0 0")


if __name__ == "__main__":
    unittest.main()
