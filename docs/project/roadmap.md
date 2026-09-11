---
description: Evidence-first roadmap for the VoiceHub 0.3 release line.
release: 0.4.0
---

# Roadmap

!!! note "0.4 architecture rewrite"

    The current source is 0.4.0. See the [migration guide](migration-0.4.md).
    The 0.3 report below is historical evidence. Its benchmark and CI results
    do not validate this rewrite. Current release gates remain pending until
    evidence is recorded for the exact new candidate.


VoiceHub 0.3 is in release-hardening mode. The project is not accepting another
model family into the built-in registry until the current TTS, ASR, and VAD
surface has passed the [release-readiness gates](release-readiness.md).

## Current milestone: 0.3 release candidate

The milestone has four ordered outcomes:

1. Keep source, documentation, benchmark evidence, Git tags, and distribution
   metadata on one version.
2. Prove that the wheel and source distribution install in clean environments
   and contain every registered runtime and required data file.
3. Separate all-provider contract coverage from the smaller set of public
   checkpoints that have run end to end on recorded hardware.
4. Publish only the artifacts that passed the release workflow, using PyPI
   Trusted Publishing and a protected GitHub `pypi` environment.

New providers, unmeasured performance claims, and approximate optimizations as
defaults are outside this milestone.

## Completed scope discovery

The registry currently exposes 68 integrations: 34 TTS, 23 ASR, and 11 VAD.
Each has generated documentation, lazy construction, a normalized task output,
an explicit training boundary, and a default-installation contract.

The historical [GitHub roadmap issue #14](https://github.com/kadirnar/voicehub/issues/14)
listed LLaSA, Chatterbox, ConversationTTS, CosyVoice, F5-TTS, GPT-SoVITS,
Kokoro, MeloTTS, OpenVoice, OuteTTS, Parler-TTS, StyleTTS 2, Orpheus, Dia,
and Vui. Those families are now integrated. The issue is retained as historical
context; updating or closing it is an external maintainer action, not a release
automation step.

## After 0.3

Post-release work must start from user demand and reproducible evidence. A new
provider needs the same package, license, checkpoint, contract, and hardware
evidence as the current registry. A serving or optimization feature needs a
measured quality boundary before it can become an automatic default.
