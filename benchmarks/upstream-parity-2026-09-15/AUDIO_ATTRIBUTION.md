# Public evaluation audio

The five excerpts in `audio/` are from the LibriSpeech ASR corpus, prepared by
Vassil Panayotov with assistance from Daniel Povey, using LibriVox readings.
The corpus is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
as recorded by [OpenSLR](https://www.openslr.org/12/).

The excerpts were obtained through the Hugging Face
[`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
validation fixture. VoiceHub selected five utterances and wrote mono 16 kHz
WAV files. The `-pcm16.wav` copies change the container encoding for readers
requiring PCM; their decoded sample values were verified to match the float
WAV fixtures exactly. `public-samples.json` records transcripts, hashes, source
location, durations, and archived paths. Selection and container conversion do
not imply endorsement by the corpus authors.

Reference: Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur,
“LibriSpeech: an ASR corpus based on public domain audio books,” ICASSP 2015.

The separately archived Kokoro and Dia listening outputs in `runs/` were
generated during this audit. Their checkpoint identities and input texts are
recorded in the corresponding requests and reports.

## Irodori Japanese example

The Irodori comparison uses the first Japanese demonstration text from the
[official model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3/blob/236c1e56591279fc24e3c1bf6609fc06e48dde28/README.md).
Its main checkpoint and codec revisions are recorded in `irodori-artifacts.json`
and the run metadata. Both repositories generated new audio from this text
without a reference voice. This public demonstration sentence is not a held-out
quality corpus. The original runtime applies SilentCipher watermarking; the
VoiceHub output lacks this unsupported stage.
