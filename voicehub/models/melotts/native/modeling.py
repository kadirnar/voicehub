"""VoiceHub-owned construction of the pinned MeloTTS VITS graph."""

from __future__ import annotations

from torch import nn

from voicehub.models.melotts.native.configuration import MeloTTSArchitectureConfig
from voicehub.models.melotts.native.metadata import MELOTTS_GENERATOR_COMPONENTS
from voicehub.models.melotts.source.melo.models import SynthesizerTrn

DEPLOYABLE_MELOTTS_COMPONENTS = MELOTTS_GENERATOR_COMPONENTS


def build_melotts_model(config: MeloTTSArchitectureConfig, ) -> SynthesizerTrn:
    """Build the exact graph described by an upstream or native JSON config."""
    if not isinstance(config, MeloTTSArchitectureConfig):
        raise TypeError("`config` must be a MeloTTSArchitectureConfig.")
    model = config.model
    data = config.data
    graph = SynthesizerTrn(
        config.vocab_size,
        data.n_fft // 2 + 1,
        config.segment_frames,
        model.inter_channels,
        model.hidden_channels,
        model.filter_channels,
        model.n_heads,
        model.n_layers,
        model.kernel_size,
        model.p_dropout,
        model.resblock,
        model.resblock_kernel_sizes,
        model.resblock_dilation_sizes,
        model.upsample_rates,
        model.upsample_initial_channel,
        model.upsample_kernel_sizes,
        n_speakers=data.n_speakers,
        gin_channels=model.gin_channels,
        use_sdp=True,
        n_flow_layer=model.n_flow_layer,
        n_layers_trans_flow=model.n_layers_trans_flow,
        flow_share_parameter=model.flow_share_parameter,
        use_transformer_flow=model.use_transformer_flow,
        use_vc=model.use_vc,
        num_languages=config.num_languages,
        num_tones=config.num_tones,
        use_spk_conditioned_encoder=model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=model.use_noise_scaled_mas,
        mas_noise_scale_initial=model.mas_noise_scale_initial,
        noise_scale_delta=model.noise_scale_delta,
    )
    _validate_component_inventory(graph)
    return graph


def _validate_component_inventory(model: nn.Module) -> None:
    prefixes = {name.split(".", 1)[0] for name in model.state_dict()}
    expected = set(DEPLOYABLE_MELOTTS_COMPONENTS)
    if prefixes != expected:
        raise RuntimeError(
            "MeloTTS graph component inventory is incomplete: "
            f"missing={sorted(expected - prefixes)!r}, "
            f"unexpected={sorted(prefixes - expected)!r}.")


__all__ = [
    "DEPLOYABLE_MELOTTS_COMPONENTS",
    "build_melotts_model",
]
