import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from voicehub.architectures import get_architecture_spec
from voicehub.architectures.fsmn_vad.checkpoint import (
    FSMNVADSafeTensorsCheckpointAdapter,
    convert_funasr_fsmn_checkpoint,
)
from voicehub.architectures.fsmn_vad.configuration import FSMNVADConfig
from voicehub.architectures.fsmn_vad.frontend import FSMNVADFrontend
from voicehub.architectures.fsmn_vad.inference import FSMNVADDecoder
from voicehub.architectures.fsmn_vad.metadata import (
    FUNASR_HF_REVISION,
    FUNASR_MODEL_SHA256,
    FUNASR_OFFICIAL_TENSOR_FINGERPRINT,
)
from voicehub.architectures.fsmn_vad.modeling import FSMNVADModel
from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import write_json_file
from voicehub.modeling_outputs import SpeechSegment
from voicehub.models.vad_funasr import FSMNVADTrainingDataset, FunASRVADConfig, FunASRVADForVoiceActivityDetection
from voicehub.models.vad_funasr.modeling_vad_funasr import _postprocess_segments
from voicehub.processing.waveform import resample_waveform_hann
from voicehub.registry import get_model_spec
from voicehub.training import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cmvn_text(dimension):
    shift = " ".join("0" for _ in range(dimension))
    scale = " ".join("1" for _ in range(dimension))
    return (
        "<Nnet>\n"
        "<AddShift> 400 400\n"
        f"<LearnRateCoef> 0 [ {shift} ]\n"
        "<Rescale> 400 400\n"
        f"<LearnRateCoef> 0 [ {scale} ]\n"
        "</Nnet>\n")


def _native_artifact(root):
    config = FSMNVADConfig()
    model = FSMNVADModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.encoder.out_linear2.linear.bias[1] = 10.0
    root.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        model.state_dict(),
        root / "model.safetensors",
        metadata={"format": "voicehub-fsmn-vad-v1"},
    )
    write_json_file(root / "config.json", config.to_dict())
    return model


class NativeFSMNVADArchitectureTests(unittest.TestCase):

    def test_provider_import_does_not_import_external_runtimes(self):
        code = """
import json
import sys
from voicehub.models.vad_funasr import FunASRVADForVoiceActivityDetection
print(json.dumps({
    name: name in sys.modules
    for name in ("funasr", "torchaudio", "modelscope")
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout.strip()),
            {
                "funasr": False,
                "torchaudio": False,
                "modelscope": False,
            },
        )

    def test_configuration_matches_published_graph(self):
        config = FSMNVADConfig()
        self.assertEqual(config.input_dim, 400)
        self.assertEqual(config.output_dim, 248)
        self.assertEqual(config.frame_length_samples, 400)
        self.assertEqual(config.frame_shift_samples, 160)
        self.assertEqual(config.silence_pdf_ids, (0, ))
        self.assertEqual(
            FSMNVADConfig.from_dict(config.to_dict()),
            config,
        )

    def test_frontend_is_differentiable_and_applies_lfr_cmvn(self):
        config = FSMNVADConfig()
        frontend = FSMNVADFrontend(
            config,
            cmvn_shift=torch.full((400, ), 2.0),
            cmvn_scale=torch.full((400, ), 0.5),
        )
        waveform = torch.randn(1, 3_200, requires_grad=True)
        fbank = frontend.fbank(waveform)
        features = frontend(waveform)
        self.assertEqual(features.shape, (1, fbank.shape[1], 400))
        expected_first = torch.cat((
            fbank[0, 0],
            fbank[0, 0],
            fbank[0, 0],
            fbank[0, 1],
            fbank[0, 2],
        ), )
        torch.testing.assert_close(
            features[0, 0],
            (expected_first + 2.0) * 0.5,
        )
        features.square().mean().backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_fsmn_streaming_cache_matches_whole_graph(self):
        torch.manual_seed(7)
        model = FSMNVADModel(FSMNVADConfig()).eval()
        features = torch.randn(1, 43, 400)
        with torch.inference_mode():
            whole = model(features=features).probabilities
            cache = {}
            streamed = torch.cat(
                (
                    model(features=features[:, :11], cache=cache).probabilities,
                    model(features=features[:, 11:29], cache=cache).probabilities,
                    model(features=features[:, 29:], cache=cache).probabilities,
                ),
                dim=1,
            )
        torch.testing.assert_close(streamed, whole)
        self.assertEqual(
            set(cache),
            {f"cache_layer_{index}"
             for index in range(4)},
        )

    def test_native_objectives_support_binary_and_pdf_targets(self):
        model = FSMNVADModel(FSMNVADConfig())
        waveforms = torch.randn(2, 3_200)
        frame_count = model.frame_count(3_200)
        binary = model(
            waveforms,
            labels=torch.ones(2, frame_count),
        )
        self.assertEqual(binary.objective, "grouped-binary-nll")
        self.assertTrue(torch.isfinite(binary.loss))
        binary.loss.backward()
        model.zero_grad(set_to_none=True)
        pdf = model(
            waveforms,
            labels=torch.full((2, frame_count), 17, dtype=torch.long),
        )
        self.assertEqual(pdf.objective, "pdf-cross-entropy")
        self.assertTrue(torch.isfinite(pdf.loss))
        pdf.loss.backward()
        explicit_pdf = model(
            waveforms,
            labels=torch.zeros(2, frame_count, dtype=torch.long),
            target_kind="pdf",
        )
        self.assertEqual(explicit_pdf.objective, "pdf-cross-entropy")

    def test_endpoint_decoder_uses_upstream_window_and_extension_geometry(self):
        config = FSMNVADConfig()
        decoder = FSMNVADDecoder(
            config,
            speech_noise_threshold=0.0,
            max_end_silence_ms=150,
        )
        speech = torch.cat((
            torch.zeros(10),
            torch.ones(30),
            torch.zeros(30),
        ), )
        silence = 1.0 - speech
        boundaries = decoder.process(
            speech,
            silence_probabilities=silence,
            decibels=torch.zeros_like(speech),
            final=True,
        )
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].start_ms, 0)
        self.assertGreater(boundaries[0].end_ms, 400)
        self.assertLessEqual(boundaries[0].end_ms, 700)

    def test_endpoint_decoder_never_rewinds_behind_a_completed_segment(self):
        # Expected boundaries come from FunASR 16cd165 FsmnVADStreaming
        # DetectLastFrames on the same frame scores: its data buffer only
        # advances, so a new start never precedes the previous end.
        config = FSMNVADConfig()
        maximum_cut = [0.99] * 100
        quick_restart = [0.01] * 10 + [0.99] * 30 + [0.01] * 20 + [0.99] * 30 + [0.01] * 30
        cases = (
            (maximum_cut, 0.6, 800, 500, [(0, 510), (510, 1000)]),
            (quick_restart, 0.5, 100, None, [(0, 460), (460, 960)]),
        )
        for values, threshold, silence_ms, maximum_ms, expected in cases:
            speech = torch.tensor(values)
            decoder = FSMNVADDecoder(
                config,
                speech_noise_threshold=threshold,
                max_end_silence_ms=silence_ms,
                max_single_segment_ms=maximum_ms,
            )
            boundaries = decoder.process(
                speech,
                silence_probabilities=1.0 - speech,
                decibels=torch.zeros_like(speech),
                final=True,
            )
            self.assertEqual([(item.start_ms, item.end_ms) for item in boundaries], expected)

    def test_pickle_conversion_is_trust_gated_and_strict(self):
        config = FSMNVADConfig()
        model = FSMNVADModel(config)
        state = {name: tensor for name, tensor in model.state_dict().items() if name.startswith("encoder.")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            cmvn = root / "am.mvn"
            torch.save(state, checkpoint)
            cmvn.write_text(_cmvn_text(config.input_dim), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                convert_funasr_fsmn_checkpoint(
                    checkpoint,
                    cmvn,
                    root / "native",
                )
            output = convert_funasr_fsmn_checkpoint(
                checkpoint,
                cmvn,
                root / "native",
                trust_pickle_checkpoint=True,
                expected_checkpoint_sha256=_sha256(checkpoint),
                expected_cmvn_sha256=_sha256(cmvn),
            )
            restored = FSMNVADModel(config)
            with SafeTensorReader(output / "model.safetensors") as reader:
                report = FSMNVADSafeTensorsCheckpointAdapter().load_streaming(
                    restored,
                    reader,
                    config.to_dict(),
                    strict=True,
                )
            self.assertTrue(report.is_compatible)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(
                    tensor,
                    restored.state_dict()[name],
                )

    def test_pinned_artifact_metadata_is_complete(self):
        self.assertEqual(len(FUNASR_HF_REVISION), 40)
        self.assertEqual(len(FUNASR_MODEL_SHA256), 64)
        self.assertEqual(len(FUNASR_OFFICIAL_TENSOR_FINGERPRINT), 64)


class NativeFSMNVADProviderTests(unittest.TestCase):

    def test_decoder_boundaries_are_not_merged_a_second_time(self):
        # Real upstream endpoints with an 800 ms silence setting. Lookback
        # and lookahead make the final visible gap only 280 ms.
        segments = tuple(
            SpeechSegment(start=a, end=b) for a, b in ((0.19, 10.42), (11.24, 18.78), (19.06, 29.38)))
        output = _postprocess_segments(
            segments,
            duration=29.4,
            min_speech_duration_ms=0,
            speech_pad_ms=0,
        )
        self.assertEqual(output, segments)
        padded = _postprocess_segments(
            segments,
            duration=29.4,
            min_speech_duration_ms=0,
            speech_pad_ms=200,
        )
        self.assertEqual(len(padded), 2)

    def test_touching_decoder_boundaries_stay_separate_and_are_not_resplit(self):
        # FunASR fsmn-vad (max_single_segment_time=5000) cuts libri-004 into
        # touching regions. They are separate upstream segments; the 5.01 s
        # region is the decoder's own maximum-duration cut, not re-split.
        segments = tuple(
            SpeechSegment(start=a, end=b)
            for a, b in ((0.19, 5.2), (5.2, 10.21), (11.24, 16.25), (16.25, 18.78)))
        output = _postprocess_segments(
            segments,
            duration=29.4,
            min_speech_duration_ms=0,
            speech_pad_ms=0,
        )
        self.assertEqual(output, segments)
        padded = _postprocess_segments(
            segments,
            duration=29.4,
            min_speech_duration_ms=0,
            speech_pad_ms=30,
        )
        self.assertEqual(len(padded), 2)
        for segment, (start, end) in zip(padded, ((0.16, 10.24), (11.21, 18.81))):
            self.assertAlmostEqual(segment.start, start, places=9)
            self.assertAlmostEqual(segment.end, end, places=9)

    def test_other_sampling_rates_are_resampled_like_torchaudio_transform(self):
        # FunASR resamples with torchaudio.transforms.Resample(audio_fs, 16000).
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _native_artifact(root / "artifact")
            wrapper = FunASRVADForVoiceActivityDetection(
                FunASRVADConfig(name_or_path=root / "artifact"),
                device="cpu",
            ).load()
            audio = 0.1 * torch.randn(8_000, generator=torch.Generator().manual_seed(0))
            captured = []
            original = wrapper._native_inference

            def capture(waveform, **kwargs):
                captured.append(waveform)
                return original(waveform, **kwargs)

            with mock.patch.object(wrapper, "_native_inference", side_effect=capture):
                output = wrapper.detect(
                    audio,
                    sampling_rate=8_000,
                    min_speech_duration_ms=0,
                    speech_pad_ms=0,
                )
            expected = resample_waveform_hann(audio, 8_000, 16_000, match="transform")
            self.assertEqual(len(captured), 1)
            self.assertTrue(torch.equal(captured[0], expected))
            self.assertEqual(output.sample_rate, 16_000)
            self.assertAlmostEqual(output.duration, 1.0)

    def test_load_detect_train_export_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = _native_artifact(root / "artifact")
            wrapper = FunASRVADForVoiceActivityDetection(
                FunASRVADConfig(name_or_path=root / "artifact"),
                device="cpu",
            ).load()
            self.assertEqual(wrapper.training_support, "native")
            self.assertTrue(wrapper.supports_generic_finetuning)
            output = wrapper.detect(
                torch.zeros(16_000),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )
            self.assertTrue(output.segments)
            self.assertEqual(output.metadata["backend"], "voicehub-native")
            self.assertTrue(output.metadata["frame_scores_available"])
            self.assertEqual(len(output.probabilities), 98)

            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(3_200),
                    "sampling_rate": 16_000,
                    "segments": [{
                        "start": 0.0,
                        "end": 0.1,
                    }],
                },
                phase="voice_activity_detection",
            )
            trained = wrapper.model(**prepared)
            self.assertTrue(torch.isfinite(trained.loss))
            trained.loss.backward()
            adapter = wrapper.get_training_adapter()
            dataset = adapter.create_dataset([{
                "audio": torch.zeros(3_200),
                "segments": [(0.0, 0.1)],
            }])
            self.assertIsInstance(dataset, FSMNVADTrainingDataset)
            context = adapter.create_training_context({
                "audio": torch.zeros(3_200),
                "sampling_rate": 16_000,
                "segments": [(0.0, 0.1)],
            })
            training_output = adapter.execute_training_phase(context)
            self.assertTrue(training_output.loss.requires_grad)
            self.assertEqual(
                adapter.recipe_resume_configuration()["architecture"],
                "fsmn-vad",
            )
            pdf_prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(3_200),
                    "sampling_rate": 16_000,
                    "pdf_labels": torch.zeros(
                        wrapper.model.frame_count(3_200),
                        dtype=torch.long,
                    ),
                },
                phase="voice_activity_detection",
            )
            self.assertEqual(pdf_prepared["target_kind"], "pdf")
            self.assertEqual(
                wrapper.model(**pdf_prepared).objective,
                "pdf-cross-entropy",
            )

            export = wrapper.export_native_pretrained(root / "export")
            fresh = FunASRVADForVoiceActivityDetection(
                FunASRVADConfig(name_or_path=export),
                device="cpu",
            ).load()
            for name, tensor in expected.state_dict().items():
                torch.testing.assert_close(
                    tensor,
                    fresh.model.state_dict()[name],
                )

    def test_streams_are_isolated_and_match_offline_frame_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _native_artifact(root)
            wrapper = FunASRVADForVoiceActivityDetection(
                FunASRVADConfig(name_or_path=root),
                device="cpu",
            ).load()
            waveform = torch.linspace(-0.2, 0.2, 8_000)
            offline = wrapper.detect(
                waveform,
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )
            first = wrapper.stream(
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )
            second = wrapper.stream(
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )
            first.push(waveform[:2_000])
            second.push(waveform[:4_000])
            first.push(waveform[2_000:5_000])
            second.push(waveform[4_000:])
            first.push(waveform[5_000:])
            first_output = first.flush()
            second_output = second.flush()
            torch.testing.assert_close(
                torch.tensor(first_output.probabilities),
                torch.tensor(offline.probabilities),
            )
            torch.testing.assert_close(
                torch.tensor(second_output.probabilities),
                torch.tensor(offline.probabilities),
            )
            self.assertIsNot(first._encoder_cache, second._encoder_cache)

    def test_variable_length_training_batches_mask_padded_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _native_artifact(root)
            wrapper = FunASRVADForVoiceActivityDetection(
                FunASRVADConfig(name_or_path=root),
                device="cpu",
            ).load()
            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": [
                        torch.zeros(3_200),
                        torch.zeros(4_800),
                    ],
                    "sampling_rate": [16_000, 16_000],
                    "segments": [
                        [(0.0, 0.1)],
                        [(0.05, 0.2)],
                    ],
                },
                phase="voice_activity_detection",
            )
            self.assertEqual(tuple(prepared["waveforms"].shape), (2, 4_800))
            self.assertEqual(
                prepared["label_mask"].sum(dim=1).tolist(),
                [
                    wrapper.model.frame_count(3_200),
                    wrapper.model.frame_count(4_800),
                ],
            )
            output = wrapper.model(**prepared)
            self.assertTrue(torch.isfinite(output.loss))
            self.assertEqual(output.objective, "grouped-binary-nll")

    def test_official_pickle_requires_explicit_acknowledgement(self):
        wrapper = FunASRVADForVoiceActivityDetection(
            FunASRVADConfig(),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
            wrapper._load_pretrained_model()

    def test_registry_architecture_and_training_are_native(self):
        model_spec = get_model_spec("vad_funasr")
        self.assertEqual(model_spec.architecture, "fsmn-vad")
        self.assertIn("voicehub-native", model_spec.capabilities)
        architecture = get_architecture_spec("fsmn-vad")
        self.assertTrue(architecture.capabilities.training)
        self.assertTrue(architecture.capabilities.streaming)
        training = get_training_spec("vad_funasr")
        self.assertEqual(training.support.value, "native")
        self.assertEqual(training.default_phase, "voice_activity_detection")

    def test_config_and_controls_are_validated(self):
        config = FunASRVADConfig(
            name_or_path="funasr/fsmn-vad",
            revision=FUNASR_HF_REVISION,
            local_files_only=True,
        )
        self.assertEqual(config.sample_rate, 16_000)
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            FunASRVADConfig(sample_rate=8_000)
        with self.assertRaisesRegex(ValueError, "credentials"):
            FunASRVADConfig(inference_config={
                "nested": {
                    "token": "secret",
                },
            })
        with self.assertRaisesRegex(ValueError, "positive"):
            FunASRVADConfig(training_max_duration_s=float("inf"))
        wrapper = FunASRVADForVoiceActivityDetection(config)
        with self.assertRaisesRegex(ValueError, "independent"):
            wrapper._detect(
                torch.zeros(16_000),
                sampling_rate=16_000,
                onset=0.6,
                offset=0.4,
            )


if __name__ == "__main__":
    unittest.main()
