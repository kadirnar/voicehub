"""Configuration for the pyannote segmentation-3.0 VAD preset."""

from collections.abc import Mapping

from voicehub.models.vad_pyannote.configuration import PyannoteVADConfig


class PyannoteSegmentationVADConfig(PyannoteVADConfig):
    """Configure the native segmentation-3.0 powerset checkpoint."""

    model_type = "vad_pyannote_segmentation"

    def __init__(
        self,
        *,
        pipeline_kwargs: Mapping | None = None,
        inference_config=None,
        **kwargs,
    ):
        if pipeline_kwargs is not None and not isinstance(pipeline_kwargs, Mapping):
            raise TypeError("`pipeline_kwargs` must be a mapping or None.")
        if pipeline_kwargs is not None and "segmentation" in pipeline_kwargs:
            raise ValueError(
                "`pipeline_kwargs` cannot replace VoiceHub's managed "
                "segmentation checkpoint.")
        super().__init__(
            pipeline_kwargs=pipeline_kwargs,
            inference_config=({
                "threshold": 0.5,
                "onset": 0.5,
                "offset": 0.5,
                "min_speech_duration_ms": 0,
                "min_silence_duration_ms": 0,
                "speech_pad_ms": 0,
                "return_frames": False,
            } if inference_config is None else inference_config),
            **kwargs,
        )
