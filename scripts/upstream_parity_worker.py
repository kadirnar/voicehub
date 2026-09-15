#!/usr/bin/env python3
"""Isolated real-checkpoint worker for benchmark_upstream_parity.py.

Run with SIDE REQUEST OUTPUT. Upstream imports resolve from the working
repository; the VoiceHub side must use its separately installed
environment.
"""

from __future__ import annotations

import faulthandler
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path


def prepare(request, side):
    import numpy as np
    import torch

    model_type = request["model_type"]
    device = request["device"]
    config = request.get("config", {})
    generation = request.get("generation", {})
    if side == "voicehub":
        from voicehub.registry import get_model_spec

        if request.get("text_frontend") == "misaki_en":
            if model_type != "kokoro":
                raise ValueError("The Misaki comparison frontend is specific to Kokoro.")
            from misaki import en, espeak

            g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")

            def frontend(text, *, language_code):
                if language_code != "a":
                    raise ValueError("This comparison recipe uses American English.")
                return g2p(text)[0]

            # Exercise the existing public injection API. This is explicitly
            # a caller-provided upstream frontend, not the built-in fallback.
            config = {**config, "text_frontend": frontend}
        spec = get_model_spec(model_type)
        cls = getattr(importlib.import_module(spec.module), spec.class_name)
        model = cls(model_path=request.get("checkpoint", spec.default_model_path), device=device, **config)
        model.load()

        def infer(sample):
            if request["task"] == "tts":
                kwargs = dict(generation)
                if sample.get("phonemes") and request.get("matched_phonemes"):
                    kwargs["phonemes"] = sample["phonemes"]
                out = model.generate(sample["text"], seed=request["seed"], **kwargs)
                return {
                    "audio_array": out.audio.detach().float().cpu().numpy().reshape(-1),
                    "sample_rate": out.sample_rate,
                    "metadata": out.metadata
                }
            audio = sample["audio"] if request.get("input_mode") == "file" else torch.from_numpy(
                sample["waveform"])
            if request["task"] == "asr":
                out = model.transcribe(audio, sampling_rate=sample["sampling_rate"], **generation)
                return {"text": out.text, "reference": sample["reference"]}
            out = model.detect(audio, sampling_rate=sample["sampling_rate"], **generation)
            result = {"segments": [[s.start, s.end] for s in out.segments], "metadata": out.metadata}
            if out.probabilities is not None:
                result["probabilities"] = np.asarray(out.probabilities).reshape(-1).tolist()
            return result

        return infer
    # No import of VoiceHub (including a shared frontend/segmenter) is allowed
    # in the reference process. The real upstream package owns inference.
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path.cwd() / "src"))
    if model_type == "asr_faster_whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(
            request["upstream_checkpoint"],
            device=device,
            compute_type="float32",
            cpu_threads=request.get("threads", 1),
            num_workers=1,
        )

        def infer(sample):
            segments, _ = model.transcribe(
                sample["waveform"],
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            return {
                "text": "".join(segment.text for segment in segments).strip(),
                "reference": sample["reference"]
            }

        return infer
    if model_type == "llasa":
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(request["checkpoint"])
        model = AutoModelForCausalLM.from_pretrained(request["checkpoint"], dtype=torch.float32)
        model.to(device).eval()
        codec = AutoModel.from_pretrained(config["codec_name_or_path"], dtype=torch.float32)
        codec.to(device).eval()

        def infer(sample):
            formatted = f"<|TEXT_UNDERSTANDING_START|>{sample['text']}<|TEXT_UNDERSTANDING_END|>"
            messages = [
                {
                    "role": "user",
                    "content": "Convert the text to speech:" + formatted
                },
                {
                    "role": "assistant",
                    "content": "<|SPEECH_GENERATION_START|>"
                },
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                continue_final_message=True,
                return_dict=False,
                **({
                    "date_string": config["date_string"]
                } if config.get("date_string") is not None else {}),
            ).to(device)
            end = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
            options = {
                "max_length": config.get("max_total_tokens", 2048),
                "eos_token_id": end,
                "do_sample": True,
                "temperature": generation.get("temperature", 0.8),
                "top_p": generation.get("top_p", 1.0),
                "top_k": generation.get("top_k", 50),
            }
            if "max_new_tokens" in generation:
                options.pop("max_length")
                options["max_new_tokens"] = generation["max_new_tokens"]
            ids = model.generate(inputs, **options)[0, inputs.shape[1]:]
            # Preserve the model-card recipe's final-token slicing even at
            # the cap, and report whether this actually removed an EOS.
            ended = ids[-1].item() == end
            # Transformers 5 treats a one-dimensional batch_decode input as
            # one sequence. Request token strings explicitly, preserving the
            # original recipe's per-token extraction semantics.
            tokens = tokenizer.convert_ids_to_tokens(ids[:-1].detach().cpu().tolist())
            codes = [
                int(token[4:-2]) for token in tokens if token.startswith("<|s_") and token.endswith("|>")
            ]
            audio = codec.decode(torch.tensor(codes, device=device).view(1, 1, -1)).audio_values
            return {
                "audio_array": audio[0, 0].float().cpu().numpy(),
                "sample_rate": 16000,
                "metadata": {
                    "audio_tokens": len(codes),
                    "speech_end_reached": ended,
                    "top_k": options.get("top_k", model.generation_config.top_k),
                    "codec_recipe": "Authors' HKUSTAudio/xcodec2-hf Transformers release",
                },
            }

        return infer
    if model_type == "parlertts":
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(request["checkpoint"])
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            request["checkpoint"],
            torch_dtype=torch.float32,
            attn_implementation=config.get("attention_implementation", "sdpa"),
        ).to(device).eval()

        def infer(sample):
            options = dict(generation)
            description = options.pop("description")
            conditioning = tokenizer(description, return_tensors="pt").to(device)
            prompt = tokenizer(sample["text"], return_tensors="pt").to(device)
            audio = model.generate(
                input_ids=conditioning.input_ids,
                attention_mask=conditioning.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
                **options,
            )
            return {
                "audio_array": audio.detach().float().cpu().numpy().reshape(-1),
                "sample_rate": model.config.sampling_rate,
                "metadata": {
                    "description": description
                },
            }

        return infer
    if model_type == "irodoritts":
        from unittest.mock import patch

        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
        from irodori_tts.tokenizer import PretrainedTextTokenizer

        original_tokenizer_loader = PretrainedTextTokenizer.from_pretrained

        def load_pinned_tokenizer(*, repo_id, add_bos, local_files_only):
            del repo_id, local_files_only
            return original_tokenizer_loader(
                repo_id=config["tokenizer_name_or_path"], add_bos=add_bos, local_files_only=True)

        key = RuntimeKey(
            checkpoint=str(Path(request["checkpoint"]) / "model.safetensors"),
            model_device=device,
            codec_repo=str(Path(config["codec_name_or_path"]) / "weights.pth"),
            model_precision=config.get("model_precision", "fp32"),
            codec_device=device,
            codec_precision=config.get("codec_precision", "fp32"),
            compile_model=False,
        )
        with patch.object(PretrainedTextTokenizer, "from_pretrained", side_effect=load_pinned_tokenizer):
            runtime = InferenceRuntime.from_key(key)

        def infer(sample):
            options = dict(generation)
            options["no_ref"] = options.pop("no_reference", True)
            out = runtime.synthesize(SamplingRequest(text=sample["text"], seed=request["seed"], **options))
            return {
                "audio_array": out.audio.detach().float().cpu().numpy().reshape(-1),
                "sample_rate": out.sample_rate,
                "metadata": {
                    "stage_timings": out.stage_timings,
                    "messages": out.messages,
                    "watermark_available": runtime.watermarker.ready,
                },
            }

        return infer
    if model_type == "echo":
        from functools import partial

        import inference as original

        artifacts = {
            "jordand/echo-tts-base": Path(request["checkpoint"]),
            "jordand/fish-s1-dac-min": Path(config["codec_name_or_path"]),
        }

        def pinned_artifact(repo_id, filename, **kwargs):
            del kwargs
            path = artifacts[repo_id] / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return str(path)

        original.hf_hub_download = pinned_artifact
        model = original.load_model_from_hf(device=device, delete_blockwise_modules=True)
        codec = original.load_fish_ae_from_hf(device=device)
        pca = original.load_pca_state_from_hf(device=device)
        sample_fn = partial(
            original.sample_euler_cfg_independent_guidances,
            num_steps=generation.get("num_steps", 40),
            cfg_scale_text=3.0,
            cfg_scale_speaker=8.0,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=generation.get("sequence_length", 640),
        )

        def infer(sample):
            audio, normalized = original.sample_pipeline(
                model,
                codec,
                pca,
                sample_fn,
                text_prompt=sample["text"],
                speaker_audio=None,
                rng_seed=request["seed"])
            return {
                "audio_array": audio[0].float().cpu().numpy().reshape(-1),
                "sample_rate": 44100,
                "metadata": {
                    "normalized_text": normalized,
                    "artifact_resolver": "Pinned local files"
                },
            }

        return infer
    if model_type == "asr_moonshine":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(request["checkpoint"])
        model = AutoModelForSpeechSeq2Seq.from_pretrained(request["checkpoint"], dtype=torch.float32)
        model.to(device).eval()

        def infer(sample):
            inputs = processor(
                sample["waveform"], sampling_rate=sample["sampling_rate"], return_tensors="pt").to(device)
            ids = model.generate(**inputs, **generation)
            return {
                "text": processor.batch_decode(ids, skip_special_tokens=True)[0],
                "reference": sample["reference"]
            }

        return infer
    if model_type in {"asr_wav2vec2", "asr_hubert", "asr_wavlm", "asr_medasr"}:
        from transformers import AutoModelForCTC, AutoProcessor

        processor = AutoProcessor.from_pretrained(request["checkpoint"])
        model = AutoModelForCTC.from_pretrained(request["checkpoint"], dtype=torch.float32)
        model.to(device).eval()

        def infer(sample):
            inputs = processor(
                sample["waveform"], sampling_rate=sample["sampling_rate"], return_tensors="pt").to(device)
            ids = model(**inputs).logits.argmax(dim=-1)
            return {"text": processor.batch_decode(ids)[0], "reference": sample["reference"]}

        return infer
    if model_type == "vad_nemo":
        from unittest.mock import patch

        from nemo.collections.asr.models import EncDecFrameClassificationModel
        from nemo.collections.asr.parts.utils.vad_utils import binarization, filtering

        # This released inference archive omits the training-only loss weight.
        # Permit that one known omission, while still rejecting every missing
        # model tensor or unexpected checkpoint tensor during restoration.
        load_state_dict = torch.nn.Module.load_state_dict

        def restore_inference_state(module, state_dict, strict=True, **kwargs):
            result = load_state_dict(module, state_dict, strict=False, **kwargs)
            if set(result.missing_keys) - {"loss.weight"} or result.unexpected_keys:
                raise RuntimeError(f"Unexpected MarbleNet checkpoint keys: {result}")
            return result

        with patch.object(torch.nn.Module, "load_state_dict", restore_inference_state):
            model = EncDecFrameClassificationModel.restore_from(request["checkpoint"], map_location=device)
        model.to(device).eval()
        model.preprocessor.featurizer.dither = 0.0
        if generation.get("speech_pad_ms", 0) != 0:
            raise ValueError("The MarbleNet comparison recipe requires zero speech padding.")
        postprocessing = {
            "onset": float(generation.get("threshold", 0.5)),
            "offset": float(generation.get("threshold", 0.5)),
            "frame_length_in_sec": 0.02,
            "min_duration_on": generation.get("min_speech_duration_ms", 0) / 1000.0,
            "min_duration_off": generation.get("min_silence_duration_ms", 0) / 1000.0,
        }

        def infer(sample):
            waveform = torch.from_numpy(sample["waveform"]).unsqueeze(0).to(device)
            lengths = torch.tensor([waveform.shape[-1]], device=device)
            logits = model(input_signal=waveform, input_signal_length=lengths)
            probabilities = logits.softmax(dim=-1)[0, :, 1].float().cpu()
            segments = filtering(binarization(probabilities, postprocessing), postprocessing)
            return {
                "segments": segments.tolist(),
                "probabilities": probabilities.tolist(),
                "metadata": {
                    "restore_compatibility": "Only missing training-only loss.weight is permitted"
                }
            }

        return infer
    if model_type == "dia":
        # The pinned Dia README includes this official Transformers recipe.
        from transformers import AutoProcessor, DiaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(request["checkpoint"])
        model = DiaForConditionalGeneration.from_pretrained(request["checkpoint"], dtype=torch.float32)
        model.to(device).eval()

        def infer(sample):
            inputs = processor(text=[sample["text"]], padding=True, return_tensors="pt").to(device)
            output = model.generate(**inputs, **generation)
            audio = processor.batch_decode(output)[0]
            return {"audio_array": np.asarray(audio).reshape(-1), "sample_rate": 44100}

        return infer
    if model_type == "kokoro":
        from kokoro import KModel, KPipeline

        weights = Path(request["checkpoint"])
        model = KModel(config=str(weights / "config.json"), model=str(weights / "kokoro-v1_0.pth"))
        model.to(device).eval()
        pipeline = KPipeline(lang_code="a", model=model, device=device)
        voice = torch.load(weights / "voices" / "af_heart.pt", weights_only=True, map_location=device)
        pipeline.voices["af_heart"] = voice

        def infer(sample):
            if request.get("matched_phonemes"):
                ps = sample["phonemes"]
                audio = model(ps, voice[len(ps) - 1], speed=1)
                return {"audio_array": audio.numpy(), "sample_rate": 24000, "phonemes": ps}
            results = list(pipeline(sample["text"], voice="af_heart", speed=1))
            return {
                "audio_array": torch.cat([r.audio for r in results]).numpy(),
                "sample_rate": 24000,
                "phonemes": "".join(r.phonemes for r in results)
            }

        return infer
    if model_type in {"asr_whisper", "asr_openai_whisper"}:
        import whisper

        if request.get("hf_conversion_source"):
            # Use the original model and the exact same official HF tensors as
            # VoiceHub. Read the author's namespace mapping as data; do not
            # import Transformers or VoiceHub into the reference process.
            import ast

            from safetensors.torch import load_file

            tree = ast.parse(Path(request["hf_conversion_source"]).read_text())
            mapping = next(
                ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "WHISPER_MAPPING" for t in node.targets))
            weights = Path(request["checkpoint"])
            c = json.loads((weights / "config.json").read_text())
            dimensions = whisper.ModelDimensions(
                n_mels=c["num_mel_bins"],
                n_audio_ctx=c["max_source_positions"],
                n_audio_state=c["d_model"],
                n_audio_head=c["encoder_attention_heads"],
                n_audio_layer=c["encoder_layers"],
                n_vocab=c["vocab_size"],
                n_text_ctx=c["max_target_positions"],
                n_text_state=c["d_model"],
                n_text_head=c["decoder_attention_heads"],
                n_text_layer=c["decoder_layers"])
            model = whisper.Whisper(dimensions)
            tensors = load_file(str(weights / "model.safetensors"))
            state, used = {}, set()
            for name in model.state_dict():
                renamed = name
                for old, new in mapping.items():
                    renamed = renamed.replace(old, new)
                key = "model." + renamed
                state[name] = tensors[key]
                used.add(key)
            extra = set(tensors) - used
            if extra == {"proj_out.weight"} and torch.equal(tensors["proj_out.weight"],
                                                            state["decoder.token_embedding.weight"]):
                extra.clear()
            if extra:
                raise ValueError(f"Unmapped official Whisper checkpoint keys: {sorted(extra)}")
            model.load_state_dict(state, strict=True)
            model.to(device).eval()
        else:
            model = whisper.load_model(
                request.get("upstream_checkpoint", "tiny.en"),
                device=device,
                download_root=request.get("upstream_weights"))

        def infer(sample):
            out = model.transcribe(
                sample["waveform"],
                language="en",
                fp16=False,
                temperature=0,
                condition_on_previous_text=False,
                without_timestamps=True,
                compression_ratio_threshold=None,
                logprob_threshold=None,
                no_speech_threshold=None)
            return {"text": out["text"], "reference": sample["reference"]}

        return infer
    if model_type == "vad_silero":
        from silero_vad import get_speech_timestamps

        model = torch.jit.load(request["checkpoint"], map_location=device).eval()

        def infer(sample):
            audio = torch.from_numpy(sample["waveform"]).to(device)
            sr = sample["sampling_rate"]
            segments = get_speech_timestamps(
                audio,
                model,
                sampling_rate=sr,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
                return_seconds=False)
            return {"segments": [[s["start"] / sr, s["end"] / sr] for s in segments]}

        return infer
    if model_type == "vad_webrtc":
        import webrtcvad

        def infer(sample):
            sr = sample["sampling_rate"]
            pcm = np.rint(np.clip(sample["waveform"].astype(float), -1, 1) * 32767).astype("<i2")
            frame_size = sr * 30 // 1000
            model = webrtcvad.Vad(3)
            flags = []
            for start in range(0, len(pcm), frame_size):
                frame = pcm[start:start + frame_size]
                frame = np.pad(frame, (0, frame_size - len(frame)))
                flags.append(model.is_speech(frame.tobytes(), sr))
            segments, start = [], None
            for i, flag in enumerate([*flags, False]):
                if flag and start is None:
                    start = i * frame_size
                elif not flag and start is not None:
                    segments.append([start / sr, min(i * frame_size, len(pcm)) / sr])
                    start = None
            return {"segments": segments}

        return infer
    if model_type == "vad_auditok":
        import auditok

        def infer(sample):
            pcm = np.rint(np.clip(sample["waveform"].astype(float), -1, 1) * 32767).astype("<i2")
            regions = auditok.split(
                pcm.tobytes(),
                sampling_rate=sample["sampling_rate"],
                sample_width=2,
                channels=1,
                min_dur=0.25,
                max_dur=100,
                max_silence=0.1,
                energy_threshold=50,
                analysis_window=0.05)
            return {"segments": [[r.start, r.end] for r in regions]}

        return infer
    if model_type == "asr_transformers":
        from transformers import pipeline

        model = pipeline(
            "automatic-speech-recognition",
            model=request["checkpoint"],
            device=device,
            torch_dtype=torch.float32)

        def infer(sample):
            inputs = {"raw": sample["waveform"], "sampling_rate": sample["sampling_rate"]}
            out = model(
                inputs,
                generate_kwargs={
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": 444,
                    **generation
                })
            return {"text": out["text"], "reference": sample["reference"]}

        return infer
    if model_type == "asr_qwen3":
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            request["checkpoint"], dtype=torch.float32, device_map=device, max_inference_batch_size=1)

        def infer(sample):
            out = model.transcribe(audio=(sample["waveform"], sample["sampling_rate"]), language="English")
            return {"text": out[0].text, "reference": sample["reference"]}

        return infer
    if model_type in {"asr_funasr", "vad_funasr"}:
        from funasr import AutoModel

        if request["task"] == "vad" and (generation.get("min_speech_duration_ms", 250) != 0 or
                                         generation.get("speech_pad_ms", 30) != 0):
            raise ValueError("The original FSMN recipe requires explicit zero minimum duration and padding.")
        model = AutoModel(model=request["checkpoint"], device=device, disable_update=True)

        def infer(sample):
            options = ({
                "max_end_silence_time": generation.get("min_silence_duration_ms", 100),
                "speech_noise_thres": generation.get("threshold", 0.5),
            } if request["task"] == "vad" else {})
            out = model.generate(input=sample["waveform"], fs=sample["sampling_rate"], **options)[0]
            if request["task"] == "asr":
                return {"text": out["text"], "reference": sample["reference"]}
            return {"segments": [[start / 1000, end / 1000] for start, end in out["value"]]}

        return infer
    if model_type == "asr_speechbrain":
        from speechbrain.inference.ASR import EncoderDecoderASR

        model = EncoderDecoderASR.from_hparams(source=request["checkpoint"], run_opts={"device": device})

        def infer(sample):
            wav = torch.from_numpy(sample["waveform"]).unsqueeze(0).to(device)
            text, _ = model.transcribe_batch(wav, torch.ones(1, device=device))
            return {"text": text[0], "reference": sample["reference"]}

        return infer
    if model_type == "vad_speechbrain":
        from speechbrain.inference.VAD import VAD

        if request.get("input_mode") != "file":
            raise ValueError("SpeechBrain's file-based API requires matching file input on both sides.")
        model = VAD.from_hparams(
            source=request["checkpoint"],
            savedir=str(Path.cwd() / ".cache/parity-vad"),
            run_opts={"device": device})

        def infer(sample):
            out = model.get_speech_segments(sample["audio"], close_th=0.1, len_th=0.25)
            return {"segments": out.cpu().tolist()}

        return infer
    if model_type == "asr_wenet":
        import sentencepiece
        import yaml
        from wenet.dataset.processor import compute_fbank
        from wenet.transformer.asr_model import init_asr_model

        weights = Path(request["upstream_weights"])
        settings = yaml.safe_load((weights / "train.yaml").read_text())
        settings["cmvn_file"] = str(weights / "global_cmvn")
        model = init_asr_model(settings)
        model.load_state_dict(torch.load(weights / "final.pt", map_location="cpu", weights_only=True))
        model.to(device).float().eval()
        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(weights / "train_xl_unigram5000.model"))
        units = {
            int(line.split()[1]): line.split()[0]
            for line in (weights / "units.txt").read_text().splitlines()
        }
        feature_options = {**settings["dataset_conf"]["fbank_conf"], "dither": 0.0}

        def infer(sample):
            source = {
                "key": sample["id"],
                "label": [],
                "sample_rate": sample["sampling_rate"],
                "wav": torch.from_numpy(sample["waveform"]).unsqueeze(0)
            }
            features = next(compute_fbank([source], **feature_options))["feat"].unsqueeze(0).to(device)
            lengths = torch.tensor([features.shape[1]], device=device)
            tokens, _ = model.attention_rescoring(
                features, lengths, beam_size=5, ctc_weight=0.3, reverse_weight=0.3)
            pieces = [units[token] for token in tokens if token != len(units) - 1]
            return {"text": tokenizer.decode_pieces(pieces), "reference": sample["reference"]}

        return infer
    if model_type in {"asr_nemo", "asr_parakeet_tdt", "asr_nemotron"}:
        from nemo.collections.asr.models import ASRModel

        checkpoint = request["checkpoint"]
        model = (
            ASRModel.restore_from(checkpoint) if Path(checkpoint).is_file() else ASRModel.from_pretrained(
                model_name=checkpoint))
        model.to(device).float().eval()

        def infer(sample):
            out = model.transcribe([sample["waveform"]], batch_size=1)[0]
            return {"text": out if isinstance(out, str) else out.text, "reference": sample["reference"]}

        return infer
    raise ValueError(f"An independent upstream inference recipe is required for {model_type}.")


def main():
    side, request_path, output_path = sys.argv[1:]
    output = Path(output_path)
    request = json.loads(Path(request_path).read_text())
    if request.get("diagnostics_after_seconds"):
        faulthandler.dump_traceback_later(request["diagnostics_after_seconds"])
    report = {"status": "error", "task": request["task"], "samples": []}
    try:
        import numpy as np
        import soundfile as sf
        import torch

        torch.set_num_threads(request.get("threads", 1))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report["runtime"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": request["device"],
            "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        report["runtime"]["packages"] = {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        }
        report["runtime"]["platform"] = platform.platform()
        report["runtime"]["worker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["runtime"]["repository_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                           text=True).strip()
        report["runtime"]["repository_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    text=True).strip())
        report["runtime"]["repository_diff_sha256"] = hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--", "."])).hexdigest()
        if side == "voicehub":
            source_hash = hashlib.sha256()
            root = Path.cwd()
            for path in sorted(p for p in (root / "voicehub").rglob("*")
                               if p.suffix in {".py", ".c", ".cc", ".h"}):
                source_hash.update(str(path.relative_to(root)).encode() + b"\0")
                source_hash.update(path.read_bytes())
            report["runtime"]["voicehub_source_tree_sha256"] = source_hash.hexdigest()
        files = []
        artifact_paths = {
            str(value)
            for value in (
                request.get("checkpoint"),
                request.get("upstream_checkpoint"),
                request.get("config", {}).get("codec_name_or_path"),
                request.get("config", {}).get("tokenizer_name_or_path"),
            ) if value
        }
        for value in sorted(artifact_paths):
            checkpoint = Path(value)
            if checkpoint.is_file():
                files.append(checkpoint)
            elif checkpoint.is_dir():
                for path in checkpoint.rglob("*"):
                    if (path.is_file() and
                            path.suffix in {".json", ".pth", ".pt", ".jit", ".safetensors", ".ckpt", ".yaml",
                                            ".yml", ".mvn", ".model", ".bin", ".onnx", ".txt"} and
                            ".cache" not in path.relative_to(checkpoint).parts):
                        files.append(path)
        report["checkpoint_sha256"] = {}
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            report["checkpoint_sha256"][str(path)] = digest.hexdigest()

        def sync():
            if request["device"].startswith("cuda"):
                torch.cuda.synchronize()

        def seed():
            random.seed(request["seed"])
            np.random.seed(request["seed"])
            torch.manual_seed(request["seed"])

        started = time.perf_counter()
        if request["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        infer = prepare(request, side)
        sync()
        report["load_seconds"] = time.perf_counter() - started
        report["load_peak_cuda_memory_bytes"] = (
            torch.cuda.max_memory_allocated() if request["device"].startswith("cuda") else None)
        for original in request["samples"]:
            sample = dict(original)
            if "audio" in sample:
                audio_hash = hashlib.sha256(Path(sample["audio"]).read_bytes()).hexdigest()
                if sample.get("sha256") and sample["sha256"] != audio_hash:
                    raise ValueError(f"Audio fixture hash mismatch: {sample['id']}")
                sample["waveform"], sample["sampling_rate"] = sf.read(sample["audio"], dtype="float32")
                if sample["waveform"].ndim != 1:
                    raise ValueError(
                        "Parity fixtures must be mono, with identical preprocessing on both sides.")
            timings = []
            peak_memory = []
            for index in range(request.get("warmup", 2) + request.get("repeats", 10)):
                seed()
                sync()
                if request["device"].startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode():
                    result = infer(sample)
                sync()
                elapsed = time.perf_counter() - started
                if index >= request.get("warmup", 2):
                    timings.append(elapsed)
                    if request["device"].startswith("cuda"):
                        peak_memory.append(torch.cuda.max_memory_allocated())
            if "audio_array" in result:
                audio_path = output.parent / (output.stem + "-" + sample["id"] + ".npy")
                data = result.pop("audio_array")
                np.save(audio_path, data, allow_pickle=False)
                sf.write(audio_path.with_suffix(".wav"), data, result["sample_rate"], subtype="FLOAT")
                result["audio"] = str(audio_path.resolve())
            report["samples"].append({
                "id":
                sample["id"],
                "input_sha256":
                (audio_hash if "audio" in sample else hashlib.sha256(sample["text"].encode()).hexdigest()),
                "warm_seconds":
                timings,
                "peak_cuda_memory_bytes":
                max(peak_memory) if peak_memory else None,
                **result
            })
        if side == "upstream" and any(k == "voicehub" or k.startswith("voicehub.") for k in sys.modules):
            raise RuntimeError("Upstream process imported VoiceHub; independent reference invalid.")
        report["status"] = "ok"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        traceback.print_exc()
    if os.name == "posix":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    output.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.cancel_dump_traceback_later()
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
