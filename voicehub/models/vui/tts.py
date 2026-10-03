import re
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from voicehub.models.vui.model import Vui
from voicehub.models.vui.sampling import multinomial, sample_top_k, sample_top_p, sample_top_p_top_k
from voicehub.models.vui.vad import detect_voice_activity as vad
from voicehub.processing import resample_waveform_hann


def ensure_spaces_around_tags(text: str):
    """Ensure whitespace exists before ``[`` and after ``]`` markers in the
    text."""
    # Add space before '[' if not preceded by space, '<', or '['
    text = re.sub(
        r"(?<![<\[\s])(\[)",
        lambda m: (f"\n{m.group(1)}" if m.start() > 0 and text[m.start() - 1] == "\n" else f" {m.group(1)}"),
        text,
    )
    # Add space after ']' if not preceded by digit+']' and not followed by space, '>', or ']'
    text = re.sub(
        r"(?<!\d\])(\])(?![>\]\s])",
        lambda m: (f"{m.group(1)}\n" if m.end() < len(text) and text[m.end()] == "\n" else f"{m.group(1)} "),
        text,
    )
    text = text.strip()
    return text


REPLACE = [
    ("—", ","),
    ("'", "'"),
    (":", ","),
    (";", ","),
]

wm = None

_SMALL_NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_NUMBER_SCALES = (
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
)


def _inference_precision(model: Vui):
    """Return the autocast context and matching KV-cache dtype.

    The upstream runtime is tuned for bfloat16 CUDA inference. CPU and
    MPS execute in the model's native dtype; allocating a hard-coded
    bfloat16 cache there creates mixed-dtype attention queries and keys.
    """
    if model.device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16), torch.bfloat16
    return nullcontext(), model.dtype


def asr(chunk, model=None, prefix=None):
    """Run VoiceHub's native Whisper ASR on one 16 kHz audio chunk."""
    global wm
    if model is not None:
        wm = model
    elif wm is None:
        from voicehub.models.asr_whisper_native import WhisperForSpeechRecognition

        wm = WhisperForSpeechRecognition(
            "openai/whisper-large-v3-turbo",
            device="auto",
        )
    result = wm.transcribe(
        chunk,
        sampling_rate=16_000,
        language="en",
        initial_prompt=prefix,
        return_timestamps=False,
    )
    return result.text


def _under_one_thousand(number: int) -> str:
    words: list[str] = []
    hundreds, remainder = divmod(number, 100)
    if hundreds:
        words.extend((_SMALL_NUMBERS[hundreds], "hundred"))
        if remainder:
            words.append("and")
    if remainder < 20:
        if remainder:
            words.append(_SMALL_NUMBERS[remainder])
    else:
        tens, ones = divmod(remainder, 10)
        words.append(_TENS[tens] if not ones else f"{_TENS[tens]}-{_SMALL_NUMBERS[ones]}")
    return " ".join(words)


def number_to_words(value: str | int) -> str:
    """Convert a non-negative integer to source-compatible English words."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid integer value: {value!r}.") from None
    if number < 0:
        return f"minus {number_to_words(-number)}"
    if number == 0:
        return _SMALL_NUMBERS[0]

    groups: list[int] = []
    while number:
        number, group = divmod(number, 1_000)
        groups.append(group)
    if len(groups) > len(_NUMBER_SCALES):
        return " ".join(_SMALL_NUMBERS[int(digit)] for digit in str(value))

    rendered: list[str] = []
    populated = [(index, group) for index, group in enumerate(groups) if group]
    for offset, (index, group) in enumerate(reversed(populated)):
        part = _under_one_thousand(group)
        scale = _NUMBER_SCALES[index]
        if scale:
            part = f"{part} {scale}"
        is_final = offset == len(populated) - 1
        if is_final and index == 0 and group < 100 and rendered:
            rendered.append(f"and {part}")
        else:
            rendered.append(part)
    return ", ".join(rendered[:-1]) + (
        (", " if len(rendered) > 1 and not rendered[-1].startswith("and ") else " ") +
        rendered[-1] if len(rendered) > 1 else rendered[0])


def replace_numbers_with_words(text):
    """Replace digit sequences with deterministic English words."""

    def replacement(match):
        return number_to_words(match.group()) + " "

    return re.sub(r"\d+", replacement, text)


valid_non_speech = ["breath", "sigh", "laugh", "tut", "hesitate"]
valid_non_speech = [f"[{v}]" for v in valid_non_speech]


def remove_all_invalid_non_speech(txt):
    """Remove all non-speech markers that are not in the valid_non_speech list.

    Only keeps valid non-speech markers like [breath], [sigh], etc.
    """
    # Find all text within square brackets
    bracket_pattern = r"\[([^\]]+)\]"
    brackets = re.findall(bracket_pattern, txt)

    # For each bracketed text, check if it's in our valid list
    for bracket in brackets:
        bracket_with_brackets = f"[{bracket}]"
        if bracket_with_brackets not in valid_non_speech and bracket != "pause":
            # If not valid, remove it from the text
            txt = txt.replace(bracket_with_brackets, "")

    return txt


def simple_clean(text):
    """Normalise text for TTS: expand numbers, strip special characters, add
    trailing pause."""
    text = re.sub(r"(\d+)am", r"\1 AM", text)
    text = re.sub(r"(\d+)pm", r"\1 PM", text)
    text = replace_numbers_with_words(text)
    text = ensure_spaces_around_tags(text)
    text = remove_all_invalid_non_speech(text)

    text = text.replace('"', "")
    text = text.replace("”", "")
    text = text.replace("“", "")
    text = text.replace("’", "'")
    text = text.replace("%", " percent")
    text = text.replace("*", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(";", "")
    text = text.replace("–", " ")
    text = text.replace("—", "")
    text = text.replace(":", "")
    text = text.replace("…", "...")
    text = text.replace("s...", "s")

    # replace repeating \n with just one \n
    text = re.sub(r"\n+", "\n", text)
    ntxt = re.sub(r" +", " ", text)

    # Ensure that ntxt ends with . or ? -- upstream's condition, kept verbatim
    # so prompts match the official inference path: it appends "." to every
    # text not ending in "." (including "?" and "!" endings).
    ntxt = ntxt.strip()
    if not ntxt.endswith(".") or ntxt.endswith("?"):
        ntxt += "."
    ntxt += " [pause]"
    return ntxt


def _prepare_prompt_codes(
    prompt_codes: Tensor | None,
    *,
    batch_size: int,
    n_quantizers: int,
    max_gen_len: int,
    device,
) -> Tensor:
    """Normalize optional codec prompts to the decoder's code layout."""
    if prompt_codes is None:
        return torch.zeros(
            (batch_size, n_quantizers, 0),
            dtype=torch.int64,
            device=device,
        )
    if not torch.is_tensor(prompt_codes):
        raise TypeError("`prompt_codes` must be a torch tensor or None.")
    if prompt_codes.ndim == 2:
        prompt_codes = prompt_codes.unsqueeze(0)
    if prompt_codes.ndim != 3:
        raise ValueError(
            "`prompt_codes` must have shape (quantizers, frames) or "
            "(batch, quantizers, frames).")
    if prompt_codes.shape[1] < n_quantizers:
        raise ValueError(
            f"`prompt_codes` contains {prompt_codes.shape[1]} quantizers, "
            f"but this Vui checkpoint requires {n_quantizers}.")
    if prompt_codes.shape[0] not in (1, batch_size):
        raise ValueError(
            f"`prompt_codes` batch size must be 1 or {batch_size}, "
            f"received {prompt_codes.shape[0]}.")
    if prompt_codes.shape[-1] >= max_gen_len:
        raise ValueError("`prompt_codes` must leave at least one frame available for "
                         "generation.")
    prompt_codes = prompt_codes[:, :n_quantizers].to(
        device=device,
        dtype=torch.int64,
    )
    if prompt_codes.shape[0] == 1 and batch_size > 1:
        prompt_codes = prompt_codes.repeat(batch_size, 1, 1)
    return prompt_codes.contiguous()


@torch.inference_mode()
def generate(
        self: Vui,
        text: str,
        prompt_codes: Tensor | None = None,
        temperature: float = 0.5,
        top_k: int | None = 150,
        top_p: float | None = None,
        max_gen_len: int = int(120 * 21.53),
):
    """Autoregressively generate multi-codebook audio codes from cleaned text.

    Args:
        self: The Vui model instance (bound externally).
        text: Input text to synthesise.
        prompt_codes: Optional codec codes for voice prompting.
        temperature: Sampling temperature.
        top_k: Top-k filtering threshold.
        top_p: Nucleus sampling threshold.
        max_gen_len: Maximum number of codec frames to generate.

    Returns:
        Tensor of shape ``(1, Q, T)`` containing generated codebook indices.
    """
    text = simple_clean(text)
    autocast_context, cache_dtype = _inference_precision(self)
    with autocast_context, sdpa_kernel([SDPBackend.MATH]):
        batch_size = 1
        device = self.device
        self.decoder.allocate_inference_cache(batch_size, device, cache_dtype)

        texts = [text]

        encoded = self.tokenizer(
            texts,
            padding="longest",
            return_tensors="pt",
        )

        input_ids = encoded.input_ids.to(device)
        text_embeddings = self.token_emb(input_ids)

        B = batch_size
        Q = self.config.model.n_quantizers

        prompt_codes = _prepare_prompt_codes(
            prompt_codes,
            batch_size=batch_size,
            n_quantizers=Q,
            max_gen_len=max_gen_len,
            device=device,
        )

        start_offset = prompt_codes.size(-1)

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1
        special_token_id = self.config.model.special_token_id

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        codes = torch.full((B, Q, max_gen_len), unknown_token, dtype=torch.int64, device=device)
        codes[:, :, :start_offset] = prompt_codes

        sequence, indexes, mask = pattern.build_pattern_sequence(codes, special_token_id)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        prev_offset = 0
        S = sequence.size(-1)

        do_prefill = True
        eos = self.config.model.audio_eos_id

        for offset in range(start_offset_sequence, S):
            # print(f"{prev_offset}:{offset}")
            curr_sequence = sequence[..., prev_offset:offset]
            audio_embeddings = (sum([self.audio_embeddings[q](curr_sequence[:, q]) for q in range(Q)]) / Q)

            if do_prefill:
                embeddings = torch.cat((text_embeddings, audio_embeddings), dim=1)
                T = embeddings.size(1)
                input_pos = torch.arange(0, T, device=device)
                do_prefill = False
            else:
                embeddings = audio_embeddings
                input_pos = torch.tensor([T], device=device)
                T += 1

            out = self.decoder(embeddings, input_pos)

            logits = torch.stack([self.audio_heads[q](out[:, -1]) for q in range(Q)], dim=1)

            repetition_penalty = 1.4
            history_window = 12

            # Get the history of generated tokens for each quantizer
            for q in range(Q):
                # Extract the history window for this quantizer
                history_start = max(0, offset - history_window)
                token_history = sequence[0, q, history_start:offset]

                # Only apply penalty to tokens that appear in the history
                unique_tokens = torch.unique(token_history)
                unique_tokens = unique_tokens[unique_tokens != special_token_id]
                unique_tokens = unique_tokens[unique_tokens != eos]
                unique_tokens = unique_tokens[unique_tokens != unknown_token]

                if len(unique_tokens) > 0:
                    # Apply penalty by dividing the logits for tokens that have appeared recently
                    logits[0, q, unique_tokens] = (logits[0, q, unique_tokens] / repetition_penalty)

            if offset < 24.53 * 4:
                logits[..., eos] = -float("inf")

            probs = F.softmax(logits / temperature, dim=-1)

            # print(probs.shape)
            if top_p is not None and top_k is not None:
                next_codes = sample_top_p_top_k(probs, top_p, top_k)
            elif top_p is not None and top_p > 0:
                next_codes = sample_top_p(probs, top_p)
            elif top_k is not None and top_k > 0:
                next_codes = sample_top_k(probs, top_k)
            else:
                next_codes = multinomial(probs, num_samples=1)

            next_codes = next_codes.repeat(batch_size, 1, 1)

            if (probs[..., eos] > 0.95).any():
                print("breaking at", offset)
                break

            valid_mask = mask[..., offset:offset + 1].expand(B, -1, -1)
            next_codes[~valid_mask] = special_token_id

            sequence[..., offset:offset + 1] = torch.where(
                sequence[..., offset:offset + 1] == unknown_token,
                next_codes,
                sequence[..., offset:offset + 1],
            )

            prev_offset = offset

        # print(sequence.shape)
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(
            sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        # assert (out_codes[..., :max_gen_len] != unknown_token).all()
        # assert (out_mask[..., :max_gen_len] == 1).all()
        out_codes = out_codes[..., prompt_codes.shape[-1]:offset]
        return out_codes[[0]]


@torch.inference_mode()
def render(
    self: Vui,
    text: str,
    prompt_codes: Tensor | None = None,
    temperature: float = 0.5,
    top_k: int | None = 100,
    top_p: float | None = None,
    max_secs: int = 100,
    max_chunk_retries: int = 3,
    vad_pipeline=None,
):
    """Render audio from text.

    Uses generate for text < 1000 characters, otherwise breaks text into
    sections and uses chunking with context.

    Like upstream, the text is cleaned here and again inside
    :func:`generate`; the second pass appends another ``". [pause]"``.
    ``vad_pipeline`` selects the trimming detector (see
    :func:`voicehub.models.vui.vad.detect_voice_activity`); ``None`` keeps
    the module default, upstream's pyannote VAD.
    """
    text = remove_all_invalid_non_speech(text)
    text = simple_clean(text)
    SR = self.codec.config.sample_rate
    HZ = self.codec.hz
    max_gen_len = int(HZ * max_secs)

    if len(text) < 1000:
        codes = generate(self, text, prompt_codes, temperature, top_k, top_p, max_gen_len)
        codes = codes[..., :-10]
        audio = self.codec.from_indices(codes)
        # torchaudio.functional.resample(audio[0], 22050, 16000) upstream.
        paudio = resample_waveform_hann(audio[0].float(), SR, 16_000)
        results = vad(paudio, vad_pipeline)

        if len(results):
            # Cut the audio based on VAD results, add 200ms silence at end
            s, e = results[0][0], results[-1][1]
            return audio[..., int(s * SR):int((e + 0.2) * SR)].cpu()

        raise Exception("Failed to render")

    # Otherwise we have to do some clever chaining!

    orig_codes = prompt_codes

    lines = text.split("\n")
    audios = []
    prev_codes = prompt_codes
    prev_text = ""

    for chunk_index, line in enumerate(lines):
        last_error = None
        for _ in range(max_chunk_retries):
            current_text = prev_text + "\n" + line if prev_text else line
            current_text = current_text.strip()
            current_text = current_text.replace("...", "")
            current_text = current_text + " [pause]"

            # Calculate max length based on text length
            estimated_seconds = max(1.0, 60 * len(current_text) / 500)
            maxlen = max(
                1,
                min(int(HZ * max_secs), round(HZ * estimated_seconds)),
            )

            try:
                autocast_context, _ = _inference_precision(self)
                with (
                        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
                        autocast_context,
                ):
                    codes = generate(
                        self,
                        current_text,
                        prompt_codes=prev_codes,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        max_gen_len=maxlen,
                    )

                codes = codes[..., :-10]
                audio = self.codec.from_indices(codes)
                # Resample for VAD
                paudio = resample_waveform_hann(audio[0].float(), SR, 16_000)

                results = vad(paudio, vad_pipeline)

                if len(results):
                    prev_text = line
                    # Cut the audio based on VAD results, add 200ms silence at end
                    s, e = results[0][0], results[-1][1]
                    codes = codes[..., int(s * HZ):int(e * HZ)]
                    prev_codes = codes
                    audio = audio[..., int(s * SR):int((e + 0.2) * SR)].cpu()
                    audios.append(audio)
                    break
                else:
                    prev_codes = orig_codes
                    prev_text = ""
            except KeyboardInterrupt:
                raise
            except RuntimeError as e:
                last_error = e
                prev_codes = orig_codes
                prev_text = ""
        else:
            message = (
                f"Vui failed to render chunk {chunk_index + 1} after "
                f"{max_chunk_retries} attempts.")
            if last_error is None:
                raise RuntimeError(message)
            raise RuntimeError(message) from last_error

    return torch.cat(audios, dim=-1)
