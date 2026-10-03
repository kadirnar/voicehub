import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

from voicehub.architectures.causal_lm.configuration import Qwen3Config
from voicehub.architectures.qwen3_asr.artifacts import Qwen3ASRArtifacts
from voicehub.architectures.qwen3_asr.checkpoint import native_qwen3_asr_tensor_shapes
from voicehub.architectures.qwen3_asr.configuration import Qwen3ASRArchitectureConfig, Qwen3ASRAudioConfig
from voicehub.architectures.qwen3_asr.modeling import Qwen3ASRForConditionalGeneration
from voicehub.architectures.qwen3_asr.processing import Qwen3ASRProcessor
from voicehub.architectures.qwen3_asr.runtime import Qwen3ASRRuntime, load_qwen3_asr_runtime
from voicehub.architectures.qwen3_asr.tokenization import (
    ASR_TEXT,
    EXPECTED_TOKEN_IDS,
    Qwen3ASRTokenizer,
    qwen2_pretokenize,
)
from voicehub.models.asr_qwen3 import Qwen3ASRConfig, Qwen3ASRForSpeechRecognition
from voicehub.processing.waveform import save_pcm_wave
from voicehub.tokenization.assets import encode_gpt2_token


def _tiny_config() -> Qwen3ASRArchitectureConfig:
    text = Qwen3Config(
        model_type="qwen3",
        vocab_size=151_936,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=EXPECTED_TOKEN_IDS["<|endoftext|>"],
        tie_word_embeddings=True,
    )
    audio = Qwen3ASRAudioConfig(
        num_mel_bins=128,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        max_source_positions=1_500,
        n_window=50,
        output_dim=8,
        n_window_infer=100,
        conv_chunksize=8,
        downsample_hidden_size=4,
    )
    return Qwen3ASRArchitectureConfig(
        audio_config=audio,
        text_config=text,
        audio_token_id=EXPECTED_TOKEN_IDS["<|audio_pad|>"],
        audio_start_token_id=EXPECTED_TOKEN_IDS["<|audio_start|>"],
        audio_end_token_id=EXPECTED_TOKEN_IDS["<|audio_end|>"],
        support_languages=("English", "Turkish"),
    )


def _write_tokenizer_assets(directory: Path) -> Qwen3ASRTokenizer:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    vocabulary[encode_gpt2_token(b"ab")] = 256
    (directory / "vocab.json").write_text(
        json.dumps(vocabulary),
        encoding="utf-8",
    )
    (directory / "merges.txt").write_text(
        "#version: 0.2\na b\n",
        encoding="utf-8",
    )
    records = {
        str(token_id): {
            "content": spelling,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": spelling != ASR_TEXT,
        }
        for spelling, token_id in EXPECTED_TOKEN_IDS.items()
    }
    (directory / "tokenizer_config.json").write_text(
        json.dumps({
            "add_prefix_space": False,
            "added_tokens_decoder": records,
            "errors": "replace",
        }),
        encoding="utf-8",
    )
    return Qwen3ASRTokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
        directory / "tokenizer_config.json",
    )


def _runtime(directory: Path) -> Qwen3ASRRuntime:
    directory.mkdir(parents=True, exist_ok=True)
    config = _tiny_config()
    tokenizer = _write_tokenizer_assets(directory)
    processor = Qwen3ASRProcessor(config, tokenizer)
    model = Qwen3ASRForConditionalGeneration(config)
    artifacts = Qwen3ASRArtifacts(
        source=str(directory),
        revision=None,
        config=directory / "config.json",
        checkpoint=directory / "model.safetensors",
        vocab=directory / "vocab.json",
        merges=directory / "merges.txt",
        tokenizer_config=directory / "tokenizer_config.json",
    )
    return Qwen3ASRRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config={
            "do_sample": False,
            "eos_token_id": [151_643, 151_645],
            "pad_token_id": 151_643,
        },
    )


class NativeQwen3ASRTests(unittest.TestCase):

    def test_official_namespace_is_exact_for_both_public_shapes(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "voicehub" / "architectures" / "qwen3_asr" / "SOURCE.json")
        metadata = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["main_library"]["revision"],
            "7c6daf77a2421100f5fb066495372c00129d39ff",
        )
        shapes = native_qwen3_asr_tensor_shapes(_tiny_config())
        self.assertIn("thinker.audio_tower.conv2d1.weight", shapes)
        self.assertIn("thinker.model.layers.0.self_attn.q_norm.weight", shapes)
        self.assertEqual(
            shapes["thinker.model.embed_tokens.weight"],
            (151_936, 8),
        )
        self.assertEqual(
            shapes["thinker.lm_head.weight"],
            (151_936, 8),
        )

    def test_audio_attention_stays_inside_inference_windows(self):
        # Qwen3-ASR's audio encoder attends within `n_window_infer` frame
        # windows, as in the official FlashAttention-2 and vLLM paths and
        # the Transformers port. The original Transformers backend's SDPA
        # and eager paths never build that mask and attend globally.
        torch.manual_seed(0)
        tower = Qwen3ASRForConditionalGeneration(_tiny_config()).thinker.audio_tower.eval()
        # n_window_infer=100 frames -> windows of 13 encoder tokens.
        features = torch.randn(128, 300)
        changed = features.clone()
        changed[:, 200:] += 1.0
        lengths = torch.tensor([300])
        with torch.no_grad():
            original = tower(features, feature_lengths=lengths)
            edited = tower(changed, feature_lengths=lengths)
        self.assertEqual(original.shape[0], 39)
        torch.testing.assert_close(original[:26], edited[:26], rtol=0, atol=0)
        self.assertFalse(torch.allclose(original[26:], edited[26:]))

    def test_unpadded_generation_uses_the_maskless_decoder_path(self):
        from voicehub.generation.config import GenerationConfig

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            model = runtime.model.eval()
            prepared = runtime.processor.prepare_inference_batch(
                (torch.randn(3_200) * 0.1, ),
                sampling_rates=(16_000, ),
            )
            inputs = {
                "input_features": prepared["input_features"],
                "feature_attention_mask": prepared["feature_attention_mask"],
            }
            masks = []
            decoder = model.thinker.model
            original_forward = decoder.forward

            def spy(*args, **kwargs):
                masks.append(kwargs.get("attention_mask"))
                return original_forward(*args, **kwargs)

            config = GenerationConfig(
                max_new_tokens=6,
                do_sample=False,
                eos_token_id=(151_643, 151_645),
                pad_token_id=151_643,
                use_cache=True,
            )
            with torch.no_grad(), unittest.mock.patch.object(decoder, "forward", spy):
                generated = model.generate(
                    prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    generation_config=config,
                    **inputs,
                )
            self.assertTrue(masks)
            self.assertTrue(all(mask is None for mask in masks))

            # Reference: uncached greedy decoding through the explicit
            # masked attention path.
            sequence = prepared["input_ids"]
            with torch.no_grad():
                for _ in range(int(generated.generated_lengths[0])):
                    logits = model(
                        sequence,
                        attention_mask=torch.ones_like(sequence, dtype=torch.bool),
                        use_cache=False,
                        **inputs,
                    ).logits
                    sequence = torch.cat((sequence, logits[:, -1].argmax(-1, keepdim=True)), dim=-1)
            torch.testing.assert_close(generated.sequences[:, :sequence.shape[1]], sequence)

            # Left-padded prompts keep the explicit mask.
            padded_ids = torch.cat((torch.full((1, 2), 151_643), prepared["input_ids"]), dim=-1)
            padded_mask = torch.cat(
                (torch.zeros(1, 2, dtype=torch.bool), prepared["attention_mask"]),
                dim=-1,
            )
            masks.clear()
            with torch.no_grad(), unittest.mock.patch.object(decoder, "forward", spy):
                model.generate(
                    padded_ids,
                    attention_mask=padded_mask,
                    generation_config=config,
                    **inputs,
                )
            self.assertTrue(all(mask is not None for mask in masks))

    def test_inference_context_is_inserted_verbatim(self):
        # qwen_asr passes the context as the system message unchanged, and
        # VoiceHub training prompts already did; inference used to strip it.
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            context = " Names: Quilter\n"
            prepared = runtime.processor.prepare_inference_batch(
                (torch.zeros(1_600), ),
                sampling_rates=(16_000, ),
                contexts=(context, ),
            )
            prompt = runtime.processor.tokenizer.decode(
                prepared["input_ids"][0],
                skip_special_tokens=False,
            )
            self.assertTrue(prompt.startswith(f"<|im_start|>system\n{context}<|im_end|>\n"))

    def test_processor_matches_qwen_whitespace_and_frame_boundaries(self):
        self.assertEqual(
            qwen2_pretokenize(" \tword \t2"),
            (" ", "\tword", " ", "\t", "2"),
        )
        self.assertEqual(
            qwen2_pretokenize("\t\n\t word"),
            ("\t\n", "\t", " word"),
        )
        self.assertEqual(
            qwen2_pretokenize("we're we’re"),
            ("we", "'re", " we", "’re"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            normalized = runtime.processor.materialize_audio(
                torch.tensor([2.0, -1.0]),
                sampling_rate=16_000,
            )
            torch.testing.assert_close(
                normalized.waveform,
                torch.tensor([1.0, -0.5]),
            )
            prepared = runtime.processor.prepare_inference_batch(
                (torch.zeros(8_001), ),
                sampling_rates=(16_000, ),
            )
            self.assertEqual(
                prepared["input_features"].shape[-1],
                8_001 // 160,
            )
            self.assertEqual(
                int(prepared["feature_attention_mask"].sum().item()),
                8_001 // 160,
            )

    def test_raw_training_batch_masks_prefix_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            prepared = runtime.processor.prepare_training_batch(
                (torch.zeros(1_600), ),
                ("hello", ),
                sampling_rates=(16_000, ),
                languages=("English", ),
            )
            labels = prepared["labels"]
            active = labels[labels.ne(-100)]
            self.assertEqual(
                runtime.processor.tokenizer.decode(
                    active,
                    skip_special_tokens=False,
                ),
                "language English<asr_text>hello<|im_end|>",
            )
            placeholders = (prepared["input_ids"] == runtime.config.audio_token_id)
            self.assertTrue(torch.all(labels[placeholders] == -100))
            runtime.model.gradient_checkpointing_enable()
            output = runtime.model(**prepared)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(runtime.model.thinker.audio_tower.conv2d1.weight.grad)
            self.assertIsNotNone(runtime.model.thinker.model.layers[0].self_attn.q_proj.weight.grad)

    def test_wrapper_trims_file_audio_before_training_resample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root / "runtime")
            wrapper = Qwen3ASRForSpeechRecognition(
                Qwen3ASRConfig(name_or_path=root / "runtime"),
                device="cpu",
            )
            wrapper.runtime = runtime
            wrapper.artifacts = runtime.artifacts
            wrapper.native_config = runtime.config
            wrapper.qwen3_processor = runtime.processor
            wrapper.model = runtime.model
            path = save_pcm_wave(
                root / "padded.wav",
                torch.cat((torch.zeros(400), torch.ones(400))),
                8_000,
            )
            common = {
                "sampling_rate": 8_000,
                "text": "hello",
                "language": "English",
            }

            file_batch = wrapper.prepare_training_inputs(
                {
                    **common,
                    "audio": str(path),
                    "audio_lengths": 400,
                },
                phase="speech_recognition",
            )
            tensor_batch = wrapper.prepare_training_inputs(
                {
                    **common,
                    "audio": torch.zeros(400),
                },
                phase="speech_recognition",
            )

            torch.testing.assert_close(
                file_batch["input_features"],
                tensor_batch["input_features"],
            )
            torch.testing.assert_close(
                file_batch["feature_attention_mask"],
                tensor_batch["feature_attention_mask"],
            )

    def test_native_lora_export_reloads_as_dense_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _runtime(root / "source")
            wrapper = Qwen3ASRForSpeechRecognition(
                Qwen3ASRConfig(name_or_path=str(root / "source")),
                device="cpu",
            )
            wrapper.runtime = runtime
            wrapper.artifacts = runtime.artifacts
            wrapper.native_config = runtime.config
            wrapper.qwen3_processor = runtime.processor
            wrapper.training_processor = runtime.processor
            wrapper.transformers_processor = runtime.processor
            wrapper.model = runtime.model
            injection = wrapper.enable_lora(
                rank=2,
                target_modules=("*.q_proj", "*.v_proj"),
            )
            adapter = wrapper.get_training_adapter().setup()
            self.assertEqual(
                type(adapter).__name__,
                "NativeQwen3ASRTrainingAdapter",
            )
            self.assertEqual(
                adapter.artifact_manifest()["lora"]["rank"],
                2,
            )
            self.assertTrue(injection.module_names)
            self.assertTrue(all(parameter.requires_grad for parameter in injection.parameters()))
            self.assertTrue(
                all(
                    not parameter.requires_grad for name, parameter in wrapper.model.named_parameters()
                    if ".lora_" not in name))
            with torch.no_grad():
                next(injection.parameters()).fill_(0.01)
                list(injection.parameters())[1].fill_(0.02)
            first_name = injection.module_names[0]
            first_module = injection.modules[first_name]
            expected_weight = (first_module.base.weight.detach() + first_module.adapter_delta().detach())
            destination = wrapper.export_native_pretrained(root / "export")
            reloaded = load_qwen3_asr_runtime(
                destination,
                device="cpu",
                compute_dtype="float32",
                for_training=True,
            )
            self.assertFalse(any(".base." in name for name in reloaded.model.state_dict()))
            self.assertEqual(
                set(reloaded.model.state_dict()),
                set(native_qwen3_asr_tensor_shapes(runtime.config)),
            )
            torch.testing.assert_close(
                reloaded.model.state_dict()[f"{first_name}.weight"],
                expected_weight,
            )


if __name__ == "__main__":
    unittest.main()
