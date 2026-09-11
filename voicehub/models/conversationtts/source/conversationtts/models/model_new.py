''' Multi-scale audio language model using LLAMA3_2 as the backbone
Author: Dongchao Yang. 2025
Code based on:
https://github.com/yangdongchao/UniAudio
https://github.com/pytorch/torchtune/tree/main
https://github.com/SesameAILabs/csm

The key features:
(1) It is a causal model for both global and local transformer. So it can easily combine with any LLM
(2) We donot calculate the loss for text token, only focusing on audio tokens.
for text sequence, the audio place is set as 0.
for audio sequence, the text place is set as 0.
We use mask token to solve the influence caused by these tokens
'''
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.nn import functional as F

from voicehub.models.conversationtts.native.decoder import (
    ConversationDecoder,
    build_llama32_decoder,
)
from voicehub.optimization.protocols import OptimizationCompileTarget

def select_with_fixed_mask(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Args:
        x: (B, T) 输入张量
        mask: (B, T) 掩码张量（每行恰好有k个1）
        k: 每行选中的元素数量
    Returns:
        (B, k) 选中元素组成的张量
    """
    indices = torch.nonzero(mask, as_tuple=True)[1].reshape(-1, k)
    return x.gather(1, indices)


def CrossEntropyAndAccuracy_zero(logits, y, mask, ignore_id=0):
    '''
    The zero layer loss.
    logits: [B, T, K]
    y: (B, T), mask (B, T)
    '''
    y, mask = y.to(logits.device), mask.to(logits.device)
    #print('ignore_id ', ignore_id, y, logits.shape, mask, mask.shape)
    # assert 1==2
    loss = F.cross_entropy(logits.transpose(1, 2).contiguous(), y.contiguous(), ignore_index=ignore_id, reduction='none')

    #assert 1==2
    pred = logits.argmax(2)
    num_all_tokens = mask.int().sum()
    acc = torch.logical_and(pred.eq(y), mask).int().sum() / num_all_tokens
    loss = (loss*mask).sum() / num_all_tokens
    metrics = {'acc_0': acc, 'loss_0': loss.clone().detach()}
    return loss, metrics

def CrossEntropyAndAccuracy_residual(logits, y, loss_weights=[1, 1, 1], ignore_id=None):
    """
       The residual layer loss.
       mask: (B, T, N)
       logits: (B, T, N, k)
       reserved_mask: (B, T)
       y: (B, T, N)
    """
    y = y.to(logits.device)
    loss_dict = {}
    acc = {}
    loss_avg = 0
    for idx, w in enumerate(loss_weights):

        tmp_logit = logits[:,idx,:].contiguous()

        tmp_y = y[:,idx].contiguous()
        tmp_loss = F.cross_entropy(tmp_logit, tmp_y, ignore_index=ignore_id, reduction='none')

        tmp_pred = tmp_logit.argmax(1) #
        tmp_num_all_tokens = tmp_y.shape[0] # we only calculate the non-mask part

        tmp_acc_tk = tmp_pred.eq(tmp_y).int().sum()
        acc[f'acc_{idx+1}'] = tmp_acc_tk/tmp_num_all_tokens
        tmp_loss = tmp_loss.sum()/tmp_num_all_tokens
        loss_avg += tmp_loss*loss_weights[idx]
        loss_dict[f'loss_{idx+1}'] = tmp_loss.clone().detach()

    loss_avg = loss_avg/len(loss_weights)
    metrics = {}
    metrics.update(acc)
    metrics.update(loss_dict)
    return loss_avg, metrics


def llama3_2_1B() -> ConversationDecoder:
    return build_llama32_decoder(
        # ConversationTTS replaces both token I/O modules with identities.
        # Building a one-row placeholder avoids allocating an unused 1 GiB
        # vocabulary table before that replacement.
        vocabulary_size=1,
        number_of_layers=16,
        number_of_heads=32,
        number_of_kv_heads=8,
        embedding_dimension=2048,
        maximum_sequence_length=2048,
        intermediate_dimension=8192,
        attention_dropout=0.0,
        normalization_epsilon=1e-5,
        rope_base=500_000,
        rope_scale_factor=32,
    )


def llama3_2_100M() -> ConversationDecoder:
    return build_llama32_decoder(
        vocabulary_size=1,
        number_of_layers=4,
        number_of_heads=8,
        number_of_kv_heads=2,
        embedding_dimension=1024,
        maximum_sequence_length=2048,
        intermediate_dimension=8192,
        attention_dropout=0.0,
        normalization_epsilon=1e-5,
        rope_base=500_000,
        rope_scale_factor=32,
    )


FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    embed_dim = model.embedding_dimension
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """ get the expected causal mask based on the position input
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))
        nn.init.normal_(self.audio_head, mean=0.0, std=0.02)
        self.random_type = 'k_style' # or batch style

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        """Expose the execution boundary used by each public mode."""
        if mode == "training":
            attribute = "forward"
        elif mode == "inference":
            attribute = "generate_frame"
        else:
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        return (OptimizationCompileTarget(
            f"conversationtts.{attribute}",
            self,
            attribute,
        ), )

    def sequence_randomly_drop_based_batch(self, x: torch.Tensor, reserved_part=2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从输入 x 的每个序列中随机选择 1/B 的帧，并返回掩码。
        args:
            x (torch.Tensor): 输入张量，形状(B, T, audio_num_codebooks, D)，其中 B 是 batch size，T 是序列长度。
        返回:
            selected_frames (torch.Tensor): 选中的帧，形状为 (B, T_subset, 32, D)，其中 T_subset = T // B。
            mask (torch.Tensor): 掩码，形状为 (B, T)，1 表示该帧被选中，0 表示未被选中。
        """
        B, T, _, D = x.shape  # 获取输入的形状
        T_subset = T // reserved_part  # 每个序列中选择的帧数
        mask = torch.zeros(B, T, dtype=x.dtype, device=x.device) # 初始化掩码
        selected_frames = []
        for i in range(B):
            indices = torch.randperm(T)[:T_subset]  # 随机排列并取前 T_subset 个索引
            selected_frames.append(x[i, indices])  # 选择对应的帧
            mask[i, indices] = 1  # 更新掩码
        selected_frames = torch.stack(selected_frames, dim=0) # 将选中的帧堆叠成一个张量
        return selected_frames, mask


    def forward(self, tokens: torch.Tensor, labels: torch.Tensor, tokens_mask: torch.Tensor, input_pos=None):
        '''
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            labels: (batch_size, seq_len, audio_num_codebooks)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len
        '''
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask[:,:-1,:].unsqueeze(-1) #
        h = masked_embeds.sum(dim=2) # merge
        if input_pos is None:
            # Training mode - full sequence processing
            seq_len = h.size(1)
            # The native decoder applies the same causal semantics internally.
            # Omitting the materialized square mask also permits dense
            # FlashAttention-4 when that optional backend is selected.
            curr_backbone_mask = None
            g_input_pos = torch.arange(0, seq_len).unsqueeze(0).expand(b, seq_len).long().to(h.device)
        else:
            # Inference mode with caching
            curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
            g_input_pos = input_pos

        h = self.backbone(h, input_pos=g_input_pos, mask=curr_backbone_mask)
        c0_logits = self.codebook0_head(h) #

        audio_local_embed = self._embed_local_audio(labels[:,:,:-1]) # remove the text streaming and the last audio streaming
        # forward local
        curr_h = torch.cat([h.unsqueeze(2), audio_local_embed], dim=2) # B, seq_len, audio_num_codebooks, D
        # print('curr_h -1 ', curr_h.shape)
        # print('audio_local_embed ', audio_local_embed.shape)
        # print('tokens_mask ', tokens_mask)
        curr_h = curr_h[tokens_mask[:,1:,0].bool()] # transfer to (N, audio_num_codebooks, D)
        #print('curr_h 0 ', curr_h.shape)
        choosed_label = labels[tokens_mask[:,1:,0].bool()]
        #print('choosed_label 0 ', choosed_label.shape)
        # randomly dropout some frames during training
        if self.random_type == 'k_style':
            selected_count = max(1, curr_h.shape[0] // 2)
            indices = torch.randperm(
                curr_h.shape[0],
                device=curr_h.device,
            )[:selected_count]
            choosed_label = choosed_label[indices]
            curr_h = curr_h[indices]  # [audio_len//16, embed_dim]
        elif self.random_type == 'batch_style':
            # not implement now
            #self.sequence_randomly_drop_based_batch()
            assert 1==2
        else:
            # do not dropout
            pass
        if input_pos is None:
            # Training mode - full sequence processing
            seq_len = curr_h.size(1)
            curr_decoder_mask = None
            curr_pos = torch.arange(0, seq_len).unsqueeze(0).expand(curr_h.shape[0], seq_len).long().to(curr_h.device)
        else:
            # Inference mode with caching
            curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0)
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
        decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask) # B, 32, D
        ci_logits = torch.einsum("bsd,sdv->bsv", decoder_h[:, 1:, :], self.audio_head)
        return c0_logits, ci_logits, choosed_label[:,1:]

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)

        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device))

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1) #
        h = masked_embeds.sum(dim=2) # merge
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
        last_h = h[:, -1, :] # the last frame
        c0_logits = self.codebook0_head(last_h) # only predict the audio part
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample) # sampling audio

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)

        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        # Decoder caches must be reset every frame.
        self.decoder.reset_caches() # !!!!
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_local_audio(self, tokens):
        ''' the token from 0-30
        '''
        audio_tokens = tokens + (self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks-1, device=tokens.device))
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks-1, -1
        )
        return audio_embeds

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2) # the last layer is text token

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.backbone.layers)
