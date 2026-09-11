import unittest
from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from voicehub.models.csm.modeling import CSMForTextToSpeech
from voicehub.models.fishtts.modeling import FishTTSForTextToSpeech
from voicehub.models.higgstts.inference import HiggsTTSForTextToSpeech
from voicehub.models.llasa.modeling import LlasaForTextToSpeech
from voicehub.models.mosstts.modeling import MossTTSConfig, MossTTSForTextToSpeech
from voicehub.models.mosstts.native.codec import MossCodecDecodeOutput
from voicehub.models.mosstts.native.runtime import MossTTSRuntime
from voicehub.models.neutts.modeling import NeuTTSForTextToSpeech
from voicehub.models.orpheustts.modeling import OrpheusTTSForTextToSpeech
from voicehub.models.outetts.inference import OuteTTSConfig, OuteTTSForTextToSpeech
from voicehub.models.qwen3tts.modeling import Qwen3TTSConfig, Qwen3TTSForTextToSpeech


class _TokenRow:

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _TokenBatch:

    def __init__(self, values):
        self.row = _TokenRow(values)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.row


class _InferenceMode:

    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.torch.inference_depth += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.torch.inference_depth -= 1


class _CodeMatrix:
    ndim = 2

    def numel(self):
        return 4

    def transpose(self, first, second):
        if (first, second) != (0, 1):
            raise AssertionError((first, second))
        return self


class _AudioTensor:

    def __init__(self, values):
        self.values = values
        if values and isinstance(values[0], list):
            self.shape = (len(values), len(values[0]))
        else:
            self.shape = (len(values), )
        self.ndim = len(self.shape)

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def squeeze(self, dimension):
        if dimension != 0 or self.shape[0] != 1:
            raise AssertionError((dimension, self.shape))
        return _AudioTensor(list(self.values[0]))

    def mean(self, *, dim):
        if dim != 0 or self.ndim != 2:
            raise AssertionError((dim, self.shape))
        return _AudioTensor([
            sum(channel[index] for channel in self.values) / len(self.values)
            for index in range(self.shape[1])
        ])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self


class _FakeTorch:
    long = object()

    def __init__(self):
        self.inference_depth = 0

    @property
    def inference_active(self):
        return self.inference_depth > 0

    def inference_mode(self):
        return _InferenceMode(self)

    def as_tensor(self, value, **kwargs):
        del kwargs
        if isinstance(value, _AudioTensor):
            return value
        return _CodeMatrix()

    def cat(self, values, *, dim=0):
        return ("concatenated", tuple(values), dim)


class InferencePreflightTests(unittest.TestCase):

    def test_required_conditioning_fails_before_runtime_load(self):
        requests = [
            (
                OrpheusTTSForTextToSpeech(device="cpu"),
                {},
                "non-empty `voice`",
            ),
            (
                LlasaForTextToSpeech(device="cpu"),
                {
                    "speaker_audio_path": "reference.wav"
                },
                "speaker_audio_path.*reference_text",
            ),
            (
                NeuTTSForTextToSpeech(device="cpu"),
                {},
                "speaker_audio_path.*reference_text",
            ),
            (
                FishTTSForTextToSpeech(device="cpu"),
                {
                    "speaker_audio_path": "reference.wav"
                },
                "non-empty `reference_text`",
            ),
            (
                CSMForTextToSpeech(device="cpu"),
                {
                    "reference_text": "orphaned transcript"
                },
                "speaker_audio_path.*reference_text",
            ),
        ]

        for model, options, message in requests:
            with self.subTest(model=model.config.model_type):
                with self.assertRaisesRegex(ValueError, message):
                    model.generate("hello", **options)
                self.assertFalse(model.is_loaded)

    def test_oute_rejects_ambiguous_speaker_sources_before_load(self):
        model = OuteTTSForTextToSpeech(device="cpu")

        with self.assertRaisesRegex(ValueError, "only one"):
            model.generate(
                "hello",
                speaker_audio_path="reference.wav",
                speaker_profile_path="speaker.json",
            )

        self.assertFalse(model.is_loaded)

    def test_invalid_moss_variant_fails_before_importing_runtime(self):
        model = MossTTSForTextToSpeech(
            MossTTSConfig(variant="unknown"),
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported MOSS-TTS variant"):
            model.generate("hello")

        self.assertFalse(model.is_loaded)

    def test_higgs_token_limit_fails_before_runtime_load(self):
        model = HiggsTTSForTextToSpeech(device="cpu")

        with self.assertRaisesRegex(
                ValueError,
                "greater than zero|positive integer",
        ):
            model.generate("hello", max_new_tokens=0)

        self.assertFalse(model.is_loaded)

    def test_higgs_rejects_invalid_finite_backend_options_before_load(self):
        invalid_options = (
            ({
                "stop_strings": 3
            }, "stop_strings"),
            ({
                "ras_win_max_num_repeat": "2"
            }, "ras_win_max_num_repeat"),
        )
        for options, message in invalid_options:
            with self.subTest(options=options):
                model = HiggsTTSForTextToSpeech(device="cpu")
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    model.generate("hello", **options)
                self.assertFalse(model.is_loaded)

    def test_csm_rejects_unrepresentable_duration_and_top_k(self):
        requests = (
            ({
                "max_audio_length_ms": 79
            }, "at least 80"),
            ({
                "top_k": 2_052
            }, "audio vocabulary size"),
        )
        for options, message in requests:
            with self.subTest(options=options):
                model = CSMForTextToSpeech(device="cpu")
                with self.assertRaisesRegex(ValueError, message):
                    model.generate("hello", **options)
                self.assertFalse(model.is_loaded)

    def test_fish_rejects_multiple_samples_and_unknown_options(self):
        model = FishTTSForTextToSpeech(device="cpu")
        with self.assertRaisesRegex(ValueError, "num_samples=1"):
            model.generate("hello", num_samples=2)
        self.assertFalse(model.is_loaded)

        with self.assertRaisesRegex(ValueError, "Unsupported generation option"):
            model.generate("hello", topp=0.9)
        self.assertFalse(model.is_loaded)

    def test_qwen_rejects_unknown_options_before_runtime_load(self):
        model = Qwen3TTSForTextToSpeech(device="cpu")

        with self.assertRaisesRegex(
                ValueError,
                "Unsupported Qwen3-TTS generation option.*top_pp",
        ):
            model.generate(
                "hello",
                mode="voice_design",
                top_pp=0.9,
            )

        self.assertFalse(model.is_loaded)

    def test_oute_validates_version_and_backend_contracts_before_load(self):
        legacy = OuteTTSForTextToSpeech(
            OuteTTSConfig(interface_version="V2"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "has no bundled default speaker"):
            legacy.generate("hello")
        self.assertFalse(legacy.is_loaded)

        batch = OuteTTSForTextToSpeech(
            OuteTTSConfig(backend="VLLM"),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "only supports batch"):
            batch.generate("hello", generation_type="chunked")
        self.assertFalse(batch.is_loaded)

        synchronous = OuteTTSForTextToSpeech(device="cpu")
        with self.assertRaisesRegex(ValueError, "asynchronous backend"):
            synchronous.generate("hello", generation_type="batch")
        self.assertFalse(synchronous.is_loaded)

    def test_oute_validates_context_limit_and_sampler_options(self):
        model = OuteTTSForTextToSpeech(
            OuteTTSConfig(max_seq_length=4_096),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "exceeds.*max_seq_length"):
            model.generate("hello", max_length=4_097)
        self.assertFalse(model.is_loaded)

        with self.assertRaisesRegex(ValueError, "sampler option.*top_pp"):
            model.generate("hello", sampler={"top_pp": 0.9})
        self.assertFalse(model.is_loaded)

        model._validate_sampler({
            "temperature": 0,
            "top_k": 0,
        })


class OrpheusTokenParsingTests(unittest.TestCase):

    def test_extracts_only_complete_valid_snac_frames(self):
        offset = OrpheusTTSForTextToSpeech._AUDIO_TOKEN_OFFSET
        size = OrpheusTTSForTextToSpeech._SNAC_CODEBOOK_SIZE
        frame = [offset + channel * size + channel for channel in range(7)]
        tokens = [
            10,
            OrpheusTTSForTextToSpeech._START_SPEECH_TOKEN_ID,
            *frame,
            offset,
            OrpheusTTSForTextToSpeech._END_SPEECH_TOKEN_ID,
        ]

        codes = OrpheusTTSForTextToSpeech._extract_audio_codes(_TokenBatch(tokens))

        self.assertEqual(
            codes,
            [channel * size + channel for channel in range(7)],
        )

    def test_rejects_invalid_codebook_channel(self):
        offset = OrpheusTTSForTextToSpeech._AUDIO_TOKEN_OFFSET
        tokens = [
            OrpheusTTSForTextToSpeech._START_SPEECH_TOKEN_ID,
            *([offset] * 7),
            OrpheusTTSForTextToSpeech._END_SPEECH_TOKEN_ID,
        ]

        with self.assertRaisesRegex(RuntimeError, "channel 1"):
            OrpheusTTSForTextToSpeech._extract_audio_codes(_TokenBatch(tokens))


class WrapperHelperTests(unittest.TestCase):

    def test_llasa_malformed_speech_token_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "malformed speech token"):
            LlasaForTextToSpeech._extract_speech_ids(["<|s_not-an-integer|>"])

    def test_oute_enum_errors_list_supported_values(self):

        class Backend(Enum):
            HF = "hf"
            VLLM = "vllm"

        with self.assertRaisesRegex(
                ValueError,
                "Choose one of: hf, vllm",
        ):
            OuteTTSForTextToSpeech._enum_member(
                Backend,
                "missing",
                option_name="backend",
            )

    def test_moss_variant_aliases_are_canonical(self):
        model = MossTTSForTextToSpeech(
            MossTTSConfig(variant="local-v1.5"),
            device="cpu",
        )

        self.assertEqual(model._resolve_variant(), "local_v1_5")

    def test_moss_default_codec_tracks_variant(self):
        expected = {
            "delay": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "local": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "local_v1_5": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
            "realtime": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        }
        model = MossTTSForTextToSpeech(device="cpu")

        for variant, codec_name in expected.items():
            with self.subTest(variant=variant):
                self.assertEqual(
                    model._resolve_codec_name_or_path(variant),
                    codec_name,
                )

        configured = MossTTSForTextToSpeech(
            MossTTSConfig(codec_name_or_path="custom/codec"),
            device="cpu",
        )
        self.assertEqual(
            configured._resolve_codec_name_or_path("delay"),
            "custom/codec",
        )

    def test_moss_loader_uses_native_runtime_and_codec_source(self):
        semantic_model = torch.nn.Linear(1, 1)
        runtime = SimpleNamespace(
            model=semantic_model,
            config=SimpleNamespace(variant="delay"),
            sample_rate=24_000,
        )
        model = MossTTSForTextToSpeech(
            MossTTSConfig(codec_name_or_path="custom/native-codec"),
            device="cpu",
        )

        with patch(
                "voicehub.models.mosstts.native.runtime.load_mosstts_runtime",
                return_value=runtime,
        ) as load_runtime:
            model._load_pretrained_model()

        self.assertIs(model.model, semantic_model)
        self.assertIs(model._mosstts_runtime, runtime)
        self.assertEqual(model.sample_rate, 24_000)
        self.assertEqual(
            load_runtime.call_args.kwargs["codec_source"],
            "custom/native-codec",
        )

    def test_moss_inference_transition_preserves_runtime_identities(self):
        semantic_model = torch.nn.Linear(1, 1)
        codec = object()
        prepare = Mock()
        runtime = SimpleNamespace(
            model=semantic_model,
            codec=codec,
            prepare_for_inference=prepare,
        )
        model = MossTTSForTextToSpeech(device="cpu")
        model.model = semantic_model
        model._mosstts_runtime = runtime

        model._prepare_for_inference()

        self.assertIs(model.model, semantic_model)
        self.assertIs(model._mosstts_runtime, runtime)
        self.assertIs(model._mosstts_runtime.codec, codec)
        prepare.assert_called_once_with()

    def test_moss_decodes_inside_inference_mode(self):

        class Processor:

            @staticmethod
            def _codes(value, *, name):
                del name
                return value

        class Codec:

            @staticmethod
            def decode(codes, lengths):
                del codes, lengths
                if not torch.is_inference_mode_enabled():
                    raise AssertionError("codec decode must use inference mode")
                return MossCodecDecodeOutput(
                    waveform=torch.tensor([[[0.1, -0.1]]]),
                    waveform_lengths=torch.tensor([2]),
                    sample_rate=24_000,
                )

        runtime = SimpleNamespace(
            processor=Processor(),
            codec=Codec(),
            device=torch.device("cpu"),
            sample_rate=24_000,
        )
        output = MossTTSRuntime.decode_codes(
            runtime,
            torch.zeros(2, 4, dtype=torch.long),
        )

        self.assertEqual(output.sample_rate, 24_000)
        self.assertTrue(torch.equal(output.waveform_lengths, torch.tensor([2])))

    def test_moss_native_decode_concatenates_every_audio_segment(self):
        model = MossTTSForTextToSpeech(device="cpu")
        runtime = SimpleNamespace(
            artifacts=None,
            config=SimpleNamespace(variant="delay"),
            decode_codes=Mock(
                side_effect=[
                    MossCodecDecodeOutput(
                        waveform=torch.tensor([[[0.1, 0.2]]]),
                        waveform_lengths=torch.tensor([2]),
                        sample_rate=24_000,
                    ),
                    MossCodecDecodeOutput(
                        waveform=torch.tensor([[[0.3]]]),
                        waveform_lengths=torch.tensor([1]),
                        sample_rate=24_000,
                    ),
                ]),
        )
        model._mosstts_runtime = runtime
        model._generate_code_segments = Mock(
            return_value=(
                SimpleNamespace(audio_codes=torch.zeros(1, 4, dtype=torch.long)),
                SimpleNamespace(audio_codes=torch.ones(1, 4, dtype=torch.long)),
            ))

        with patch(
                "voicehub.models.mosstts.modeling.seeded_inference",
                return_value=nullcontext(11),
        ):
            output = model._generate("hello")

        self.assertTrue(torch.allclose(
            output.audio,
            torch.tensor([0.1, 0.2, 0.3]),
        ))
        self.assertEqual(output.sample_rate, 24_000)
        self.assertEqual(runtime.decode_codes.call_count, 2)

    def test_moss_stereo_is_downmixed_to_mono(self):
        output = MossCodecDecodeOutput(
            waveform=torch.tensor([[
                [1.0, 3.0, 5.0],
                [3.0, 5.0, 7.0],
            ]]),
            waveform_lengths=torch.tensor([3]),
            sample_rate=48_000,
        )

        waveform, source_channels = MossTTSForTextToSpeech._normalize_waveform(output, )

        self.assertEqual(source_channels, 2)
        self.assertEqual(waveform.ndim, 1)
        self.assertTrue(torch.equal(waveform, torch.tensor([2.0, 4.0, 6.0])))

    def test_qwen_auto_mode_uses_loaded_checkpoint_role(self):
        backend = SimpleNamespace(
            model=SimpleNamespace(tts_model_type="voice_design"),
            generate_voice_design=Mock(return_value=([[0.1, -0.1]], 24_000)),
            generate_custom_voice=Mock(),
            generate_voice_clone=Mock(),
        )
        model = Qwen3TTSForTextToSpeech(
            Qwen3TTSConfig(name_or_path="test/base-named-directory"),
            device="cpu",
        )
        model.model = backend

        with patch(
                "voicehub.models.qwen3tts.modeling.seeded_inference",
                return_value=nullcontext(19),
        ):
            output = model.generate("hello", mode="auto", instruct="calm")

        backend.generate_voice_design.assert_called_once()
        backend.generate_custom_voice.assert_not_called()
        backend.generate_voice_clone.assert_not_called()
        self.assertEqual(output.metadata["mode"], "voice_design")

    def test_higgs_routes_seed_through_request_local_native_generator(self):
        response = SimpleNamespace(
            waveform=torch.tensor([[[0.1, -0.1]]]),
            generated_steps=2,
            sample_rate=24_000,
        )
        model = HiggsTTSForTextToSpeech(device="cpu")
        model._runtime = SimpleNamespace(generate=Mock(return_value=response))

        output = model._generate("hello", seed=43)

        self.assertEqual(
            model._runtime.generate.call_args.kwargs["seed"],
            43,
        )
        self.assertEqual(output.metadata["seed"], 43)
        self.assertEqual(output.metadata["requested_seed"], 43)
        self.assertEqual(output.metadata["backend"], "voicehub-native")

    def test_qwen_voice_design_allows_an_empty_instruction(self):
        backend = SimpleNamespace(
            model=SimpleNamespace(tts_model_type="voice_design"),
            generate_voice_design=Mock(return_value=([[0.1, -0.1]], 24_000)),
            generate_custom_voice=Mock(),
            generate_voice_clone=Mock(),
        )
        model = Qwen3TTSForTextToSpeech(device="cpu")
        model.model = backend

        with patch(
                "voicehub.models.qwen3tts.modeling.seeded_inference",
                return_value=nullcontext(7),
        ):
            output = model.generate(
                "hello",
                mode="voice_design",
                instruct="",
                seed=7,
            )

        backend.generate_voice_design.assert_called_once()
        self.assertEqual(
            backend.generate_voice_design.call_args.kwargs["instruct"],
            "",
        )
        self.assertEqual(output.metadata["seed"], 7)

    def test_higgs_defaults_to_forced_audio_generation(self):
        model = HiggsTTSForTextToSpeech(device="cpu")

        self.assertTrue(model.generation_config.force_audio_gen)
        with self.assertRaisesRegex(ValueError, "requires audio generation"):
            model.generate("hello", force_audio_gen=False)
        self.assertFalse(model.is_loaded)

    def test_csm_processor_batch_to_must_preserve_mapping_contract(self):
        batch = SimpleNamespace(to=Mock(return_value=object()))

        with self.assertRaisesRegex(TypeError, "must return a mapping"):
            CSMForTextToSpeech._move_processor_output_to_device(
                batch,
                "cpu",
            )


if __name__ == "__main__":
    unittest.main()
