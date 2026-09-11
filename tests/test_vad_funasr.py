import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_funasr import FSMNVADTrainingDataset, FunASRVADConfig, FunASRVADForVoiceActivityDetection
from voicehub.models.vad_funasr.native.checkpoint import (
    FSMNVADSafeTensorsCheckpointAdapter,
    convert_funasr_fsmn_checkpoint,
)
from voicehub.models.vad_funasr.native.configuration import FSMNVADConfig
from voicehub.models.vad_funasr.native.frontend import FSMNVADFrontend
from voicehub.models.vad_funasr.native.inference import FSMNVADDecoder
from voicehub.models.vad_funasr.native.metadata import (
    FUNASR_HF_REVISION,
    FUNASR_MODEL_SHA256,
    FUNASR_OFFICIAL_TENSOR_FINGERPRINT,
)
from voicehub.models.vad_funasr.native.modeling import FSMNVADModel
from voicehub.registry import get_model_spec
from voicehub.runtime import get_architecture_spec
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
