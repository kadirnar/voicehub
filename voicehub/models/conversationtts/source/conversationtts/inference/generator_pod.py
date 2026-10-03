''' The inference code for speech generation.
Author: Dongchao Yang. 2025
Code based on:
https://github.com/yangdongchao/UniAudio
https://github.com/pytorch/torchtune/tree/main
https://github.com/SesameAILabs/csm

The key features include:
(1) support KV-cache
(2) multiple segments inference

To do list:
Intergrating vLLM inference to further improve the inference efficiency
'''
from dataclasses import dataclass
from typing import List, Tuple
import torch
from voicehub.audio import load_audio
from voicehub.processing.waveform import resample_waveform_hann
from voicehub.processing.waveform import save_pcm_wave
from voicehub.models.conversationtts.source.conversationtts.models.model_new import Model, ModelArgs
from voicehub.models.conversationtts.source.conversationtts.tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
from voicehub.models.conversationtts.source.conversationtts.tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from voicehub.models.conversationtts.runtime import resume_for_inference


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_text_tokenizer(tokenizer_checkpoint_path):
    tokenizer = TextTokenizer(checkpoint_dir=tokenizer_checkpoint_path)
    # bos = tokenizer.bos_id
    # eos = tokenizer.eos_id
    return tokenizer

def load_audio_tokenizer(model_path, device):
    return MimiTokenizer(ckpt_path=model_path, device=device)


class Generator:
    def __init__(
        self,
        model: Model,
        text_tokenizer_path,
        audio_tokenizer_path
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_text_tokenizer(text_tokenizer_path)

        device = next(model.parameters()).device

        self._audio_tokenizer = load_audio_tokenizer(audio_tokenizer_path, device)

        self.sample_rate = 24000
        self.device = device

    def _tokenize_text_segment(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        text_tokens = self._text_tokenizer.tokenize(text)
        # print('text ', text)
        # print('text_tokens ', text_tokens)
        text_frame = torch.zeros(len(text_tokens), 33).long()
        # print('text_frame ', text_frame.shape)
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))
        #print('torch.cat(frame_tokens, dim=0) ', torch.cat(frame_tokens, dim=0))
        #assert 1==2
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        print('audio ', audio.shape)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))
        print('audio_tokens ', audio_tokens.shape)
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device) # eos is zero?

        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        # print('audio_frame ', audio_frame)
        # assert 1==2
        audio_frame_mask[:, :-1] = True # true denotes can be used

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(f'[{str(segment.speaker)}]'+segment.text)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        #
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate_v1(
        self,
        text: str,
        speaker: int = 0,
        max_audio_length_ms: float = 30_000,
        context: List[Segment] = [],
        temperature: float = 0.9,
        topk: int = 30,
    ) -> torch.Tensor:
        ''' the v1 version only supports input one text segment, and generate one audio segment
        '''
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 40) # the frame number of max prediction
        tokens, tokens_mask = [], []
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(f'[{str(speaker)}]'+text)

        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")
        print('curr_tokens ', curr_tokens.shape)
        for _ in range(max_audio_frames):

            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        print('samples ', len(samples))
        audio = self._audio_tokenizer.detokenize(torch.stack(samples).permute(1, 2, 0).squeeze(0))
        return audio

def load_prompt_audio(audio_path):
    # Upstream: torchaudio.load + torchaudio.functional.resample(..., 24000).
    # Decode natively, then resample with the bit-identical Hann sinc kernel;
    # VoiceHub's generic resampler changes the prompt's Mimi codes.
    decoded = load_audio(audio_path)
    return resample_waveform_hann(
        decoded.waveform,
        decoded.sampling_rate,
        24_000,
    )

def prepare_prompt(text, speaker, audio_path):
    audio_tensor = load_prompt_audio(audio_path)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)


if __name__ == '__main__':
    SPEAKER_PROMPTS = {
        "speaker_a": {
            "text": (
                "Perhaps she may have a relation who might suit us, and be glad of our place."
            ),
            "audio": "/mnt/users/hccl.local/jkzhao/projects/ConversationTTS/egs/pretraining/1688_142285_000059_000000.wav",
        },
        "speaker_b": {
            "text": (
                "The whole assembly wore an aspect of the most profound gravity-- the reflection, as it were, of the sombre countenance of the austere and relentless Grand Master."
            ),
            "audio": "/mnt/users/hccl.local/jkzhao/projects/ConversationTTS/egs/pretraining/8461_281231_000052_000000.wav",
        }
    }
    resume = '/mnt/users/hccl.local/jkzhao/ckpts/ckpt1.checkpoint'
    exp_dir = 'exp_data/speech_lm/exp_v3'
    device = 'cuda:0'
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128_256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(config)
    model.to(device=device, dtype=torch.bfloat16)
    resume_for_inference(resume, exp_dir, model, device) # init the model
    generator = Generator(model, text_tokenizer_path='/mnt/users/hccl.local/jkzhao/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062',
                          audio_tokenizer_path = '/mnt/users/hccl.local/jkzhao/projects/moshi/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors')

    # Prepare speaker prompts
    speaker_a_prompt = prepare_prompt(
        SPEAKER_PROMPTS["speaker_a"]["text"],
        0,
        SPEAKER_PROMPTS["speaker_a"]["audio"]
    )

    speaker_b_prompt = prepare_prompt(
        SPEAKER_PROMPTS["speaker_b"]["text"],
        1,
        SPEAKER_PROMPTS["speaker_b"]["audio"]
    )
    prompt_segments = [speaker_a_prompt, speaker_b_prompt]
    generated_segments = []
    conversation_lines = [
        "Have you heard about the recent advances in audio language models?",
        "Yeah, it's fascinating how they're starting to handle both understanding and generation tasks.",
        "Exactly, especially the way they tokenize audio into discrete units, like words.",
        "Right, soundstream and other neural codecs are playing a big role in that.",
        "And with LLMs like GPT handling the tokens, it's bridging the gap between speech and text.",
        "I saw a paper where they used semantic-rich audio tokens for emotion recognition too.",
        "Yes, semantic tokens retain meaning while compressing the signal.",
        "But training these models is tricky, especially aligning tokens with text labels.",
    ]
    for i, line in enumerate(conversation_lines):
        # Alternating speakers A and B, starting with A
        speaker_id = i % 2
        speaker_name = "A" if speaker_id == 0 else "B"

        print(f"Generating audio for Speaker {speaker_name}: {line[:50]}...")

        context = prompt_segments + generated_segments[-8:] if generated_segments else prompt_segments

        audio_tensor = generator.generate_v1(
            text=line,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=30_000,
        )
        # print('gene ', audio_tensor.shape)
        # assert 1==2
        generated_segments.append(Segment(text=line, speaker=speaker_id, audio=audio_tensor.squeeze(0)))

    audio_tensors = [segment.audio for segment in generated_segments]
    audio_tensor = torch.cat(audio_tensors, dim=0)
    # Save the full conversation
    output_file = "podcast_output_ep3.wav"
    save_pcm_wave(output_file, audio_tensor.cpu(), generator.sample_rate)
    print(f"Saved conversation audio to {output_file}")
