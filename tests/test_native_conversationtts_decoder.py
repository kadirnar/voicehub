import ast
import unittest
from pathlib import Path

import torch

from voicehub.architectures.conversationtts.decoder import ConversationRMSNorm, build_llama32_decoder
from voicehub.models.conversationtts.configuration_conversationtts import ConversationTTSConfig
from voicehub.models.conversationtts.source.conversationtts.tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer

PROJECT_ROOT = Path(__file__).parents[1]
CONVERSATION_SOURCE = (PROJECT_ROOT / "voicehub/models/conversationtts/source/conversationtts")


class NativeConversationDecoderTests(unittest.TestCase):

    @staticmethod
    def _decoder():
        torch.manual_seed(7)
        return build_llama32_decoder(
            vocabulary_size=19,
            number_of_layers=2,
            number_of_heads=4,
            number_of_kv_heads=2,
            embedding_dimension=16,
            maximum_sequence_length=12,
            intermediate_dimension=32,
            normalization_epsilon=1e-5,
        ).eval()

    def test_parameter_namespace_matches_released_torchtune_graph(self):
        names = set(self._decoder().state_dict())
        expected = {
            "tok_embeddings.weight",
            "layers.0.attn.q_proj.weight",
            "layers.0.attn.k_proj.weight",
            "layers.0.attn.v_proj.weight",
            "layers.0.attn.output_proj.weight",
            "layers.0.mlp.w1.weight",
            "layers.0.mlp.w2.weight",
            "layers.0.mlp.w3.weight",
            "layers.0.sa_norm.scale",
            "layers.0.mlp_norm.scale",
            "norm.scale",
            "output.weight",
        }
        self.assertTrue(expected <= names)
        self.assertFalse(any("cache" in name or "theta" in name for name in names))

    def test_cached_generation_matches_full_causal_forward(self):
        decoder = self._decoder()
        embeddings = torch.randn(1, 5, 16)
        causal = torch.ones(1, 5, 5, dtype=torch.bool).tril()
        decoder.tok_embeddings = torch.nn.Identity()
        decoder.output = torch.nn.Identity()
        expected = decoder(
            embeddings,
            mask=causal,
            input_pos=torch.arange(5).unsqueeze(0),
        )

        decoder.setup_caches(1, embeddings.dtype)
        pieces = []
        cache_mask = torch.ones(1, 5, 12, dtype=torch.bool).tril()
        for index in range(5):
            pieces.append(
                decoder(
                    embeddings[:, index:index + 1],
                    mask=cache_mask[:, index:index + 1],
                    input_pos=torch.tensor([[index]]),
                ))
        actual = torch.cat(pieces, dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

        decoder.reset_caches()
        self.assertEqual(decoder.layers[0].attn.kv_cache.size, 0)

    def test_rms_norm_accumulates_in_float32_and_preserves_dtype(self):
        layer = ConversationRMSNorm(4, epsilon=1e-5).to(dtype=torch.bfloat16)
        inputs = torch.tensor(
            [[1.0, -2.0, 3.0, -4.0]],
            dtype=torch.bfloat16,
        )
        output = layer(inputs)
        self.assertEqual(output.dtype, inputs.dtype)
        expected = torch.nn.functional.rms_norm(
            inputs.float(),
            (4, ),
            layer.scale.float(),
            1e-5,
        ).to(dtype=inputs.dtype)
        torch.testing.assert_close(output, expected)

    def test_conversation_model_has_no_provider_runtime_import(self):
        runtime_files = (
            CONVERSATION_SOURCE / "models/model_new.py",
            CONVERSATION_SOURCE / "inference/generator.py",
            CONVERSATION_SOURCE / "tools/tokenizer/Text2ID/text_tokenizer.py",
            *(CONVERSATION_SOURCE / "tools/tokenizer/MimiCodec").rglob("*.py"),
        )
        forbidden = {
            "einops",
            "huggingface_hub",
            "numpy",
            "omegaconf",
            "safetensors",
            "sentencepiece",
            "tokenizers",
            "torchaudio",
            "torchtune",
            "transformers",
        }
        violations = []
        for path in runtime_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
            violations.extend((path.relative_to(PROJECT_ROOT), name) for name in imports
                              if name.split(".", 1)[0] in forbidden)
        self.assertEqual(violations, [])


class NativeConversationTokenizerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer_root = CONVERSATION_SOURCE / "llama3_2"
        cls.tokenizer = TextTokenizer(cls.tokenizer_root)

    def test_multilingual_tokens_match_the_pinned_llama32_asset(self):
        expected = {
            "Hello world!": [128000, 9906, 1917, 0, 128001],
            "[1]Hello world!": [
                128000,
                58,
                16,
                60,
                9906,
                1917,
                0,
                128001,
            ],
            "云再黑风再吼": [
                128000,
                103458,
                88356,
                57752,
                103125,
                88356,
                7305,
                120,
                128001,
            ],
            "123 1234 12 1": [
                128000,
                4513,
                220,
                4513,
                19,
                220,
                717,
                220,
                16,
                128001,
            ],
        }
        for text, token_ids in expected.items():
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.tokenize(text), token_ids)

    def test_empty_text_still_has_explicit_boundaries(self):
        self.assertEqual(
            self.tokenizer.tokenize(""),
            [self.tokenizer.bos_id, self.tokenizer.eos_id],
        )

    def test_mimi_checkpoint_is_pinned_to_the_audited_file_revision(self):
        config = ConversationTTSConfig()
        self.assertEqual(
            config.audio_tokenizer_revision,
            "a0870f178898fba98afe5ef08f0bdd2773d80f62",
        )


class ConversationPromptAudioTests(unittest.TestCase):

    def test_prompt_audio_is_resampled_like_torchaudio_functional_resample(self):
        # Upstream load_prompt_audio uses torchaudio.functional.resample; the
        # generic VoiceHub resampler differed by up to 0.13 on a 16 kHz
        # LibriSpeech prompt, changing the prompt's Mimi codes.
        import tempfile

        from voicehub.models.conversationtts.source.conversationtts.inference import generator, generator_pod, generator_pod_cn
        from voicehub.processing.waveform import resample_waveform, resample_waveform_hann, save_pcm_wave

        source = torch.sin(torch.arange(1_600, dtype=torch.float32) * 0.37) * 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.wav"
            save_pcm_wave(path, source, 16_000)
            decoded = generator.load_audio(path).waveform
            expected = resample_waveform_hann(decoded, 16_000, 24_000)
            for module in (generator, generator_pod, generator_pod_cn):
                actual = module.load_prompt_audio(str(path))
                self.assertEqual(actual.shape, (2_400, ))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(expected, resample_waveform(decoded, 16_000, 24_000)))

    def test_mimi_tokenize_resamples_each_row_without_a_channel_downmix(self):
        # Upstream MimiTokenizer.tokenize applies torchaudio.transforms.Resample
        # to a [batch, time] tensor and encodes every row as its own example.
        from voicehub.models.conversationtts.source.conversationtts.tools.tokenizer.MimiCodec import mimi_tokenizer
        from voicehub.processing.waveform import resample_waveform_hann

        class RecordingCodec:

            def __init__(self):
                self.inputs = []

            def encode(self, wav):
                self.inputs.append(wav)
                return torch.zeros(wav.shape[0], 32, 2, dtype=torch.long)

        tokenizer = mimi_tokenizer.MimiTokenizer.__new__(mimi_tokenizer.MimiTokenizer)
        torch.nn.Module.__init__(tokenizer)
        tokenizer.sr = 24_000
        tokenizer.device = torch.device("cpu")
        tokenizer.model = RecordingCodec()
        rows = torch.stack((
            torch.sin(torch.arange(1_600, dtype=torch.float32) * 0.37),
            torch.cos(torch.arange(1_600, dtype=torch.float32) * 0.11),
        )) * 0.5

        codes = tokenizer.tokenize(rows, sample_rate=16_000)

        (encoded, ) = tokenizer.model.inputs
        self.assertEqual(tuple(encoded.shape), (2, 1, 2_400))
        expected = resample_waveform_hann(rows, 16_000, 24_000, match="transform")
        torch.testing.assert_close(encoded[:, 0], expected, rtol=0, atol=0)
        self.assertEqual(tuple(codes.shape), (2, 32, 2))


if __name__ == "__main__":
    unittest.main()
