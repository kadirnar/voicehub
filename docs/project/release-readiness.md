---
description: Release checklist, verification matrix, and current candidate report for VoiceHub 0.3.
release: 0.3.0
---

# VoiceHub 0.3 release readiness

This is the authoritative release-candidate checklist for VoiceHub 0.3.0.
It separates locally reproducible checks, cross-platform CI, real-checkpoint
evidence, and maintainer-controlled publication. A pending hardware or external
gate is never counted as a pass.

## Candidate report

Updated 2026-08-04. The source version is 0.3.0. PyPI still serves 0.1.6, so
installing `voicehub` from PyPI does not yet provide the current repository
contract.

| Gate | Current evidence | Status |
| --- | --- | --- |
| Source, docs, benchmark, and PyPI candidate alignment | Local check identified 0.3.0 as a new candidate over PyPI 0.1.6 | Passed |
| Full CPU-safe suite | Merge commit `566db682` passed the supported Linux, macOS, and Windows test matrix in [CI run 30749398917](https://github.com/kadirnar/voicehub/actions/runs/30749398917). Fresh exact-current macOS environments passed 2,582 tests and 4,657 subtests with 15 skips and 35 warnings on Python 3.10.19 in 226.26 seconds and Python 3.11.15 in 212.37 seconds. The Python 3.12.12 full-dependency environment executed three additional default-runtime paths and passed 2,585 tests and 4,795 subtests with 12 skips and 35 warnings in 181.72 seconds. All seven asset/oracle paths pass separately on this worktree; only three Triton and two CUDA-extension paths remain locally unpassed | Passed remotely for merge commit `566db682` and locally on exact-current macOS Python 3.10-3.12; exact-current Linux, Windows, and remote default-runtime CI remain pending |
| Formatting and lint | PR 71 exact head `8ea5e94` passed CI lint and `pre-commit.ci`; merge commit `566db682` passed CI lint. The exact current candidate set now passes every configured hook over 4,675 tracked and candidate untracked files, excluding the protected `uv.lock`, with identical before/after SHA-256 manifests | Passed locally for the exact current worktree and remotely for merge commit `566db682`; exact-current-worktree remote lint remains pending |
| Documentation | Merge commit `566db682` passed the strict build and deployment in [Docs run 30749398929](https://github.com/kadirnar/voicehub/actions/runs/30749398929); the current shell passed the strict eleven-language build, the eleven-route DOM validator, and a pinned Playwright 1.62.0 plus Axe 4.12.1 validator. Its 60 route/viewport/palette cases protect geometry, state, screenshot-derived signatures, automated detectable accessibility rules, and complete native sequential focus. Six cases each enforce the current Home, Installation, Quickstart, Pipeline, Auto Classes, SpeechT5, Trainer, Optimization, Contribution, and Models API route-specific structure and content inventories. Six activate Home page copy, six activate Installation first-code copy, six activate Installation page copy, six activate every Quickstart tab set by keyboard, six activate Quickstart page copy, six activate Pipeline code copy, and six each activate Auto Classes, SpeechT5, Trainer, Optimization, Contribution, and Models API page copy at every viewport and palette, verify exact clipboard text, visible success/idle states where required, and focus retention, and rerun Axe after activation. Twenty pointer and 20 Enter cases activate the final TOC target on every representative route in both palettes, require the exact hash, CSS target, one settled active link, visible heading alignment, zero overflow, keyboard focus retention where applicable, and rerun Axe. Forty desktop/tablet Ctrl+K cases and 20 mobile pointer cases open and close search on every representative route in both palettes, protect focus and ARIA/inert/tab-order state, rerun Axe while open, restore focus to the visible breakpoint-specific control after Escape, and allow native Tab to continue without delayed restoration. Thirty Enter and 30 pointer cases open and close the version menu across every route, viewport, and palette, require three exact destinations, one current item, visible in-viewport geometry, focus and ARIA state, zero overflow, and an open-state Axe pass, then restore focus to the summary after Escape. Twenty ArrowDown cases switch every representative desktop/tablet route to Turkish and 20 pointer/semantic cases switch them to Arabic; all 40 require the exact 11-locale inventory and production-base destinations, selected target, LTR/RTL direction, palette preservation, zero overflow, and a localized-route Axe pass. Twenty Enter cases and 20 pointer cases switch every representative desktop/tablet route to the opposite palette; all 40 require exact toggle labels and targets, route and locale stability, focus transfer to the newly visible toggle, preserved geometry, zero overflow, and a post-switch Axe pass. Twenty Enter and 20 pointer cases focus the exact source repository link on every representative desktop/tablet route, validate its name, target, geometry, outline, overflow, and Axe result, and perform deterministic browser navigation. Thirty settled interaction cases cover open search/results/empty, version, branch, and drawer states. The latest matrix traversed 4,745 visible focus stops, rejected inactive branches, reached `BODY`, and returned to the skip link in every base case; it recorded 348 keyboard cases in total. Auto Classes uses its exact API ancestry, while all 68 generated model guides use their exact Base classes ancestry. Thirty-six nested-branch cases activate and restore all nine visible Base classes/model disclosures on SpeechT5 at desktop and tablet widths in both palettes, retain the sticky rail through document scroll, require unique page-subsection landmark names, and rerun Axe. Thirty-two root-branch cases activate and restore every top-level disclosure, and two separate cases retain mobile drawer activation evidence. Sixty page-action cases cover all ten representative routes at all three viewports and both palettes: 30 Enter and 30 pointer paths perform 60 exact edit navigations, 114 exact previous/next navigations, and 60 Back to top activations with focus, palette, route, overflow, and Axe checks | Passed remotely for merge commit `566db682` and locally for the tested current slice; the uncommitted worktree is not covered remotely, automated Axe coverage is not a complete WCAG or manual audit, and direct raw-pixel equivalence with intentionally different upstream content, branding, and palette is not claimed |
| Wheel, sdist, and editable installs | Merge commit `566db682` passed [Package CI run 30749398914](https://github.com/kadirnar/voicehub/actions/runs/30749398914). A fresh current-slice probe produced a 57,194,978-byte wheel and a 55,468,722-byte source distribution and passed wheel, sdist, and editable installs with 68 models, 81 provenance manifests, 193 compliance files, required package data, zero dependency violations, and no eager PyTorch import. Earlier worktree fingerprint builds are retained below as historical evidence | Passed remotely for merge commit `566db682` and locally for the tested current slice; exact tagged-workflow artifact hashes remain pending |
| Python 3.10–3.12 on Linux, macOS, and Windows | Merge commit `566db682` passed all nine version/platform jobs, both runtime smokes, default runtime, training, and lint in [CI run 30749398917](https://github.com/kadirnar/voicehub/actions/runs/30749398917). Fresh exact-current environments pass the complete suite on macOS 26.5.2 arm64 with Python 3.10.19, 3.11.15, and 3.12.12, PyTorch 2.8.0, Transformers 5.14.1, and pytest 9.1.1; the Python 3.12 run also activates the full default runtime | Passed remotely for merge commit `566db682` and locally for the exact-current macOS worktree on Python 3.10-3.12; exact-current remote Linux, macOS, and Windows execution remains pending |
| Canonical AI guidance | Merge commit `566db682` passed the canonical guidance contract in every supported platform matrix job; the current tree's merged guidance change passed 13 local tests and 11 subtests | Passed cross-platform for merge commit `566db682` and locally for the recorded tree |
| Released-checkpoint TTS, ASR, and VAD evidence | Dated RTX 4090 JSON and guide; see matrix below | Passed for the listed representatives |
| Pinned small release assets | The official ESPNet configuration plus SenseVoice and SpeechBrain tokenizers at immutable revisions matched declared sizes, file fingerprints, extracted tokens, and published encoding vectors on the exact current worktree | Passed locally for the exact current worktree; Package CI and the tagged release build repeat all three opt-in online gates |
| TEN-VAD checkpoint oracle | The official 315,449-byte ONNX graph at immutable revision `22a3bcd4509d0faaa8eef4881e8af5f39c178950` matched its pinned digest, converted to native Safetensors, and matched ONNX Runtime across 25 recurrent steps on the exact current worktree | Passed locally with pinned ONNX Runtime 1.22.1; Package CI and the tagged build repeat the isolated development oracle |
| NVIDIA QuartzNet checkpoint conversion | The official 70,993,538-byte NGC `stt_en_quartznet15x5` 1.0.0rc1 archive matched its pinned digest and exact converted tensor fingerprint on the exact current worktree | Passed locally; Package CI and the tagged build repeat the isolated conversion without redistributing the NGC artifact |
| WeNet GigaSpeech checkpoint and tokenizer | The immutable `openspeech/wenet-models` mirror at revision `90acd57d17169a15d5ceab462c6e7db3bd003921` supplied the exact 503,845,602-byte upstream archive with pinned SHA-256 `061ccfa51d64ebe7ea091a5a13ae31e37d9c36f4eface5c7bafc80bd4a06b26e`; restricted conversion and published tokenizer vectors passed on the exact current worktree | Passed locally; Package CI and the tagged build now repeat both isolated gates without redistributing the pickle archive |
| Tokenless publication workflow | Source contract test verifies separate build/publish jobs, protected environment, and job-scoped OIDC | Passed locally; tagged run pending |
| Protected `pypi` environment and PyPI publisher | GitHub's environment inventory currently contains only `github-pages`; PyPI publisher settings require maintainer access | Pending maintainer configuration |
| Git tag, GitHub release, and PyPI publication | No local/remote `v0.3.0` tag or matching GitHub release exists; publication requires explicit maintainer approval | Pending |
| Shared diffusion-serving dispatch | Native and vLLM-Omni model sets are derived from registered architecture features; runtime extension and fail-closed tests pass | Passed; the registry-wide shared-layer policy covers the remaining provider-branch boundary |
| Shared task-factory defaults | TTS, ASR, and VAD no-argument defaults are unique `ModelSpec.default_for_task` declarations; modern and compatibility factories follow live metadata and contain no registered-model literals | Passed; the registry-wide shared-layer policy covers the remaining provider-branch boundary |
| Package-root public API inventory | A generated reference resolves all 261 unique `voicehub.__all__` exports, records canonical source and line, kind, version-stable signature or constant/type-alias marker, summary, and lazy state, and fails on duplicate, unresolved, undocumented, source-less, or stale entries. The exact focused contract passes on Python 3.10-3.12, and a fresh subprocess confirms package-root import remains PyTorch-free | Passed locally; exact-current remote platform execution remains pending |
| Model contribution completion gate | An activated package-local manifest now supplies the lazy `ModelSpec`, aliases, and honest inference-only training profile without editing either central catalog; inactive scaffolds stay undiscovered and duplicate or richer unsupported declarations fail explicitly | Passed for representative zero-central-edit, TTS/ASR/VAD, extension, mismatch, lazy-import, and no-mutation fixtures |
| Model-page source provenance | The generator resolves source records from model packages and every lazy native-architecture component without importing an implementation; 58 pages link existing `SOURCE.json` files and the remaining 10 explicitly report that no integration-specific record is bundled. All 68 pages use one registry-derived uppercase-first display label, the same nine-section detail contract, verified paper and upstream GitHub metadata, and direct links to their resolved configuration and model facade source while retaining canonical keys | Passed; registry-derived documentation tests reject missing files, false provenance links, unresolved facade source, missing upstream GitHub references, incomplete paper metadata, fewer than four parsed examples, absent limitations or evidence boundaries, stale generated pages, stale display names, and regressions across representative TTS, ASR, and VAD architectures |
| Checkpoint documentation provenance | A shared declarative contract distinguishes real Hugging Face repositories, external archives, verified mirrors, and local-only inputs; page, index, gallery, quickstart, and notebook generation consume the same metadata without provider-name branches | Passed; 59 real Hub notebooks remain, WeNet records its immutable byte-identical mirror without presenting the external archive as a native Hugging Face model repository, and integrations without a default no longer claim a registry default |
| Native dependency boundary | The policy derives 755 seed files from the stable core boundary, all 383 immediate model-package Python facades, literal architecture component references, and three runtime-generated vendored roots; the 1,304-file fixed-point closure has zero violations and does not import PyTorch | Passed; a new model facade joins the default audit without a central provider-list edit |
| Shared provider independence | A source-only AST policy checks all 202 shared Python files against 68 canonical model types and 102 live aliases; declarative metadata and model-local code remain valid, while comparisons and every supported condition form fail | Passed with zero shared behavior violations; runtime extensions join without a central list edit |
| Universal optimization lifecycle | The registry-derived CPU-safe contract executes all six public passes across all 68 models: 408 pairs cover application, validation, manifest reporting, strict JSON serialization, semantic/state preservation, restoration, and cleanup without silent skips. Pass-specific suites separately exercise configured and compiled application paths | Passed locally for all 408 pairs; three Triton and two compiled CUDA-extension checks remain hardware-limited and unpassed |

## Layered verification matrix

Contract coverage and checkpoint coverage answer different questions and stay
separate in release reports.

| Layer | Scope | Evidence | Interpretation |
| --- | --- | --- | --- |
| Registry and lazy construction | All 34 TTS, 23 ASR, and 11 VAD integrations | Full pytest suite plus the TTS and ASR/VAD registry audits | Proves discovery, configuration, lazy allocation, and normalized contract behavior; it does not prove every public weight file |
| Native graph and checkpoint shape | Every registered family | Model-specific CPU/meta tests and immutable inventory tests | Proves the implemented graph and declared checkpoint namespace under deterministic fixtures |
| Package surface | Every registered runtime | `scripts/check_distribution.py` and Package CI | Proves wheel, sdist, editable install, package data, import coverage, and declared dependencies |
| TTS real checkpoint | `vits` with `facebook/mms-tts-eng@c71de0f` | [`tts_vits_rtx4090_2026-07-31.json`](https://github.com/kadirnar/voicehub/blob/main/benchmarks/tts_vits_rtx4090_2026-07-31.json) | Complete tokenizer, acoustic graph, flow, and vocoder path on an RTX 4090 |
| ASR real checkpoints | Moonshine tiny, Wav2Vec2 base, and Whisper tiny through five adapters | [`asr_vad_rtx4090_2026-07-31.json`](https://github.com/kadirnar/voicehub/blob/main/benchmarks/asr_vad_rtx4090_2026-07-31.json) | Complete transcription paths with deterministic text and single-sample WER; not a corpus accuracy claim |
| ASR checkpoint conversion | NeMo QuartzNet15x5 character CTC | Pinned NGC 1.0.0rc1 archive, source/config/weights digests, tensor inventory, and strict native reload | Proves the official 639-tensor namespace converts to the implemented graph; it is not a corpus accuracy claim |
| VAD real checkpoints or algorithms | Silero, Sherpa/Silero, WebRTC, and Auditok | Same ASR/VAD JSON evidence | Complete segmentation paths with boundary/score comparisons |
| Hardware-limited remainder | Gated, restricted, multi-gigabyte, or unavailable checkpoints | Provider documentation and explicit coverage records in benchmark JSON | Pending by design; no claim that every public checkpoint was downloaded |

The readable methodology and results are in the
[RTX 4090 speech benchmark](../guides/rtx-4090-speech-benchmarks.md). Performance
numbers are checkpoint- and machine-specific and are not release thresholds on
other hardware.

## Local release gates

Run from a clean checkout with Python 3.12 after installing
`.[test,training,docs]`:

```bash
python scripts/check_release.py
python -m pytest -q
pre-commit run --all-files
mkdocs build --strict --clean
python scripts/check_documentation_dom.py site
python scripts/check_documentation_visual.py site
python scripts/check_distribution.py
```

Before the final capability-driven serving slice, the local working tree passed
the full suite in independent macOS environments on Python 3.10.19 (173.39
seconds), Python 3.11.15 (154.82 seconds), and Python 3.12.12 (106.98 seconds).
The Python 3.10 and 3.11 runs created isolated temporary environments,
installed `.[test]` with uv's CPU PyTorch backend, and did not use or modify the
repository lock file. Each run reported 2,400 passed, 25 skipped, 3,287
subtests, and 35 warnings.

The current tree then reran the complete Python 3.12 suite after replacing the
diffusion-serving model allowlist with architecture capability discovery. It
reported 2,401 passed, 25 skipped, 3,287 subtests, and 35 warnings in 102.48
seconds. The 16 changed-contract tests separately passed under Python 3.10.19,
3.11.15, and 3.12.12. Skipped paths are recorded as unverified, not passed.
Candidate CI separately passed the supported Python versions on Linux, macOS,
and Windows for commit `e2bfb4a`; that remote result does not prove later
uncommitted changes.

The next provider-branch slice moved ASR and VAD task defaults into unique
`ModelSpec.default_for_task` declarations. The current Python 3.12 tree then
reported 2,405 passed, 25 skipped, 3,287 subtests, and 35 warnings in 97.82
seconds. Its five changed registry/auto tests passed under Python 3.10.19,
3.11.15, and 3.12.12 without loading a heavy backend. An attempted broader
Python 3.10 run reused the Python 3.12 environment and failed on the expected
compiled-PyTorch ABI mismatch; it is invalid cross-version evidence and is not
counted as a pass.

The following compatibility slice declared the existing Orpheus TTS default in
the same registry metadata and removed the provider literal from
`AutoInferenceModel`. The final Python 3.12 tree reported 2,406 passed, 25
skipped, 3,289 subtests, and 35 warnings in 103.26 seconds. Its three changed,
dependency-light default-contract tests passed under Python 3.10.19, 3.11.15,
and 3.12.12. The runtime replacement regression proves both the modern TTS
factory and the compatibility factory follow a new process-local default
without editing either shared factory.

The next contribution slice replaced the scaffold checker's quoted-name search
with import-free AST validation of built-in `ModelSpec`, alias, and training
profile declarations. A quoted comment can no longer satisfy discovery;
wrong lazy modules, classes, config paths, checkpoints, tasks, aliases, and
training tasks produce separate errors. TTS, ASR, and VAD fixtures pass, and a
completed separately distributed extension still passes without either central
catalog file. The final Python 3.12 tree reported 2,411 passed, 25 skipped,
3,303 subtests, and 35 warnings in 95.12 seconds. Its five changed pure-Python
contract tests passed under Python 3.10.19, 3.11.15, and 3.12.12. The first
focused run exposed a shadowed checkpoint variable, and the first broad docs
run exposed a non-compiling alias example; both failed runs are excluded from
passing evidence, and both exact regressions passed after correction.

The following contribution slice added the read-only `catalog` command. It
derives the built-in `ModelSpec`, alias entries, task enum, and honest
inference-only training profile from `model-integration.json`, labels each
central insertion point, and never imports VoiceHub or a model backend. The
same output completed the checker fixtures for TTS, ASR, and VAD and remained
byte-for-byte deterministic without changing their file inventories. Its three
pure-Python tests passed on Python 3.10.19, 3.11.15, and 3.12.12. The focused
test's first fake-catalog assembly had invalid indentation, and the first
selected pre-commit run reformatted the new docstring; neither failed run is
counted. The corrected current tree reported 2,414 passed, 25 skipped, 3,307
subtests, and 35 warnings in 107.99 seconds.

The complete current executable and test tree then passed the same full suite
in independent macOS environments on Python 3.10.19 (166.51 seconds) and
Python 3.11.15 (153.60 seconds). Each direct `uv pip install -e ".[test]"`
used the CPU PyTorch backend in a new temporary virtual environment and did not
read or modify the repository lock file. Together with the Python 3.12.12 run,
all three supported interpreters reported exactly 2,414 passed, 25 skipped,
3,307 subtests, and 35 warnings. This strengthens supported-version evidence;
it does not substitute for Linux or Windows execution of the later local tree.

The next architecture-policy slice removed 230 model/provider paths from a
253-entry manually maintained native-runtime seed list. The audited boundary is
now derived from 23 stable core roots, every immediate Python file in each
model package, literal lazy component references in architecture registrations,
and only three narrow source roots needed by runtime-generated vendored imports.
The resulting 755 seeds reach a 1,304-file fixed-point closure with zero
external dependency violations and no eager PyTorch import. A temporary future
model facade importing `transformers` fails the default audit without any
central policy edit, while MeloTTS and OpenVoice regressions prove that active
internal closures stay covered and dormant vendored frontends stay excluded.
The pure policy probe reported the same 755 seeds, 1,304 closure files, zero
violations, and no PyTorch import on Python 3.10.19, 3.11.15, and 3.12.12. The
current Python 3.12.12 tree then reported 2,416 passed, 25 skipped, 3,307
subtests, and 35 warnings in 113.71 seconds. Two initial focused runs exposed
missing literal architecture references and direct-seed/closure test
semantics; the corrected tests pass, and those failed runs are not counted as
release evidence.

The following shared-architecture slice unified the boundary previously
covered by four partial provider-branch checks into one registry-wide policy.
It scans all 202 Python files outside model-local `models/` and
`architectures/` roots against 68 canonical
model types and 102 live aliases. Declarative catalogs, licensing, provenance,
and training metadata remain valid, while provider literals in comparisons,
`if` and conditional expressions, loop conditions, assertions, comprehension
filters, and `match` cases or guards fail with file and line evidence. A
runtime-registered model and alias join the default audit without editing a
policy list. The complete related slice reported 141 passes and 178 subtests;
the same synthetic `if` and `match` violations were detected on Python 3.10.19,
3.11.15, and 3.12.12. The current Python 3.12.12 tree then reported 2,420
passed, 25 skipped, 3,307 subtests, and 35 warnings in 113.34 seconds. The first
selected pre-commit run let docformatter update the new policy file and exited
nonzero; the formatted source passed the second run and the focused four-test
suite, so the first run is not counted as passing evidence.

The next contribution slice made `model-integration.json` the package-local
source of truth for a completed inference-only built-in. New scaffolds are
inactive by default and therefore cannot enter either registry while their
runtime, checkpoint revision, or evidence is incomplete. Once explicitly
activated, one strict source-only parse requires every package facade,
`IMPLEMENTATION_STATUS = "ready"`, an immutable checkpoint revision, and
bundled license text before it derives the lazy `ModelSpec`, aliases,
capabilities, components, checkpoint, and task; the same manifest produces an
honest inference-only `ModelTrainingSpec`. No model package or PyTorch module is
imported, and neither `voicehub/models/registry.py` nor
`voicehub/training/specs.py` needs a model entry. A richer training claim must
use an explicit profile, while simultaneous manifest and legacy central
declarations fail as three actionable duplicate errors. The focused scaffold
suite reported 19 passes and 35 subtests, and the related registry/training
slice reported 116 passes and 330 subtests. The same temporary model and
training profile were discovered without importing PyTorch or the model package
on Python 3.10.19, 3.11.15, and 3.12.12. The current Python 3.12.12 tree then
reported 2,424 passed, 25 skipped, 3,310 subtests, and 35 warnings in 114.23
seconds. The final activation-gate rerun reported the same counts in 111.58
seconds. The first documentation run exposed its obsolete requirement for a
central alias mapping, and the first two selected pre-commit runs applied
formatter changes and exposed one continuation-indent error. The corrected
documentation contract and formatted source pass; none of those failed runs is
counted as passing evidence.

The following CPU-safe evidence slice audited every skip reason. Seven tests in
the old mock-Transformers VAD class were permanently skipped because the native
Wav2Vec2 provider had replaced that runtime. The redundant mock loader,
windowing, training, and serialization fixtures were removed; four still-valid
speech-label, checkpoint-stride, and invalid-training-input edge contracts were
moved into the executable native provider suite. The two focused native VAD
files reported 21 passes and 20 subtests, and the related VAD, Wav2Vec2,
registry, task, and inference slice reported 137 passes and 235 subtests. The
complete Python 3.12.12 tree then reported 2,428 passed, 18 skipped, 3,310
subtests, and 35 warnings in 112.27 seconds. Every remaining skip names its
missing dependency, dedicated CI job, CUDA host, checkpoint, or release asset;
none is counted as passing evidence. The changed test sources also parsed on
Python 3.10.19, 3.11.15, and 3.12.12; syntax parsing is not reported as runtime
coverage on the two interpreters without installed test dependencies.

The next default-runtime evidence slice corrected the active codec import
inventory. Its Higgs entry still pointed at a dormant vendored tokenizer that
required `vector_quantize_pytorch`, even though the registered `higgstts`
wrapper executes VoiceHub's native PyTorch-only tokenizer. The gate now imports
the active native Higgs module beside the DAC, FishTTS, and Irodori codec paths
while explicitly rejecting `audiotools`, `loguru`, and
`vector_quantize_pytorch`. It executes under the installed test environment
without a dependency skip. The focused codec file reported six passes; the
related codec, Higgs, FishTTS, Llasa, dependency-policy, and optimization slice
reported 130 passes, 52 subtests, and four warnings. The complete Python
3.12.12 tree then reported 2,429 passed, 17 skipped, 3,310 subtests, and 35
warnings in 114.45 seconds. The changed test source parsed on Python 3.10.19,
3.11.15, and 3.12.12; parsing alone is not runtime evidence on the two
interpreters without installed PyTorch.

The following default-runtime slice removed the last two optional-dependency
skips. They exercised dormant MeloTTS Japanese and OuteTTS GGUF provider files,
while the registered built-ins use VoiceHub's native MeloTTS and OuteTTS
architectures. Their replacements import each native frontend, runtime, and
public wrapper in fresh processes while rejecting MeloTTS's legacy `MeCab`,
`pykakasi`, `unidic_lite`, and Transformers frontends and OuteTTS's legacy
`llama_cpp`, `loguru`, `polars`, and Transformers backends. The formatted
focused compatibility file reported seven passes. The related native provider,
inference, lifecycle, compile-target, and dependency-policy slice reported 144
passes and 61 subtests. The complete Python 3.12.12 tree then reported 2,431
passed, 15 skipped, 3,310 subtests, and 35 warnings in 114.57 seconds. Every
remaining skip now names a dedicated CI job, CUDA or toolkit requirement,
checkpoint, or release asset; none is counted as passing evidence. The first
selected pre-commit run let YAPF rewrite the new helper and exited nonzero; the
formatted focused suite and second hook run passed, so the first run is not
counted. The test source parsed on Python 3.10.19, 3.11.15, and 3.12.12;
parsing alone is not runtime evidence on interpreters without installed
dependencies.

The next release-asset slice exercised the remaining ESPNet token-list gate
against the official Hugging Face repository at immutable revision
`bc6bbd771cec698f070640ee677a66719181f0a2`. The downloaded configuration was
82,131 bytes and matched SHA-256
`16351b9bf79631d1df0a4645a858dc330c40434cf03470408c9c8fd446b6ea19`; its
extracted token list matched SHA-256
`48ec6eedbee6a22e2a9b51adeb425af3c39db23128086c015240f591601a3ea3`. The
opt-in test passed through VoiceHub's dependency-free Hub transport, while the
same test remained explicitly skipped in the default offline run. Package CI
and the tagged release build now require this online gate. The related ESPNet,
Hub transport, release-readiness, and packaging slice reported 56 passes, one
offline skip, and 81 subtests. An earlier unqualified `python` invocation could
not start pytest in the local shell and is not counted as a pass. The first
selected pre-commit run let YAPF format the changed test and exited nonzero;
the formatted tests and second hook run passed. The complete current Python
3.12.12 tree then reported 2,431 passed, 15 skipped, 3,310 subtests, and 35
warnings in 112.86 seconds. The online ESPNet pass remains separate from those
offline suite counts.

The following tagged-runtime slice closed a release-workflow discrepancy. Main
CI already ran the three opt-in default-runtime tests on Ubuntu and in macOS
and Windows smoke jobs, but both full-suite steps in the tagged release
workflow left them skipped. The tagged nine-job Python/platform matrix and the
dependent build job now set `VOICEHUB_FULL_RUNTIME_TEST=1`; the workflow source
contract requires both declarations. The focused file reported five passes and
138 subtests. The release-equivalent complete Python 3.12.12 macOS run then
reported 2,434 passed, 12 skipped, 3,448 subtests, and 35 warnings in 110.63
seconds. Those three default-runtime paths are locally verified, while remote
execution of the changed tagged matrix is still pending. The separate ESPNet
online gate accounts for one of the 12 remaining default-run skips, leaving 11
hardware, checkpoint, tokenizer, or ONNX paths unverified.

The next release-asset slice added the official SenseVoiceSmall tokenizer at
immutable revision `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` to the same
opt-in online gate as ESPNet. VoiceHub's Hub transport downloaded the
377,341-byte file, which matched SHA-256
`aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8`.
The native tokenizer reproduced the pinned English text IDs, control-token
labels, and semantic language, emotion, and event values. Package CI and the
tagged build now run both release-asset tests through one plural gate. Their
focused online run reported two passes; the same SenseVoice test remained an
explicit skip offline. The related SenseVoice, ESPNet, Hub, release, and
packaging slice reported 71 passes, two offline skips, and 81 subtests. The
first selected pre-commit run let YAPF format the SenseVoice test and exited
nonzero; the formatted online tests and second hook run passed. With the
default-runtime and both release-asset gates enabled, the complete Python
3.12.12 macOS suite reported 2,436 passed, 10 skipped, 3,448 subtests, and 35
warnings in 112.97 seconds.

The following real-checkpoint slice exercised the official TEN-VAD ONNX graph
from immutable source revision
`22a3bcd4509d0faaa8eef4881e8af5f39c178950`. The 315,449-byte source matched
SHA-256 `e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4`
before VoiceHub's standard-library ONNX reader converted it to native
Safetensors. The differential oracle used pinned ONNX Runtime 1.22.1 for 25
deterministic recurrent feature steps; native speech probability and four LSTM
states stayed within the existing `2e-7` and `2e-6` absolute-error thresholds.
The test now passes the expected official source digest into conversion and
requires the exported metadata to report an official source. Package CI and
the tagged build download the exact raw file, install ONNX Runtime only for
this isolated development gate, and leave the wheel runtime dependency-free.
The focused oracle passed and remained explicitly skipped without its source
environment variable. The related VAD, checkpoint, release, and packaging
slice reported 64 passes, one offline skip, and 115 subtests. With every local
opt-in gate enabled, the complete Python 3.12.12 macOS suite reported 2,437
passed, 9 skipped, 3,448 subtests, and 35 warnings in 112.68 seconds.

The next real-checkpoint slice exercised NVIDIA's official
`stt_en_quartznet15x5` archive from NGC release 1.0.0rc1. The 70,993,538-byte
source matched SHA-256
`1b9b7b87a9277e6fef164d8f99d1226f0511af154423bbf919b920421ac9602f`.
VoiceHub's restricted, `weights_only` converter then verified the embedded
configuration and weight digests, all 639 tensor names and shapes, the
19,018,554-value state count, and the native Safetensors strict reload. Package
CI and the tagged build now download the exact NGC version and run only this
isolated conversion gate; the NVIDIA-governed artifact remains outside the
distribution. The focused oracle passed, and the related NeMo, release,
packaging, and distribution slice reported 32 passes and 76 subtests. The
first all-opt-in run reported 2,437 passes and 9 skips because its temporary
TEN-VAD source was no longer present; that run is not evidence for an eight-skip
state. After redownloading and fingerprinting the pinned TEN-VAD graph, the
complete Python 3.12.12 macOS suite reported 2,438 passed, 8 skipped, 3,448
subtests, and 35 warnings in 114.03 seconds. The remaining paths are five
CUDA/Triton or CUDA-toolkit checks, one SpeechBrain tokenizer gate, and the
WeNet checkpoint and tokenizer gates.

The following documentation-provenance slice removed a false negative from the
model-page generator. It previously searched a registry architecture alias as
if it were a package directory and therefore claimed that 38 integrations had
no bundled source record. Source discovery now follows the lazy module paths
already declared by each native `ArchitectureSpec`, checks both package-root
and `source/` layouts without importing an implementation, and preserves
model-local precedence. Twenty-eight regenerated pages now point to real
manifests; 58 of 68 pages have an existing source link and the remaining 10
honestly report no integration-specific record. The focused documentation
contract reported 3 passes and 142 subtests, including exact TTS, ASR, and VAD
examples, path existence, deterministic 68-page regeneration, and a fresh
process that imports none of NeMo, Safetensors, SentencePiece, PyTorch, or
Transformers. The related
documentation, registry, optimization, scaffold, packaging-metadata, and
distribution-contract suite reported 102 passes and 1,839 subtests; strict
multilingual documentation and release-version alignment also passed. The
first clean distribution run passed the wheel probe but its sdist probe exited
nonzero without reproducible stderr and is not counted as a pass. The exact
sdist install and probe then passed in a preserved clean environment, and a
fresh complete distribution run passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, zero dependency
violations, and no eager PyTorch import.

The same evidence audit rechecked WeNet before selecting this bounded slice.
The pinned upstream README still lists the 20210728 GigaSpeech U2++ archive,
but the official HTTP artifact endpoint returned 404 on 2026-08-02, its HTTPS
variant failed certificate-hostname validation, and the apparent
`wenet/gigaspeech-u2pp-conformer` Hugging Face page returned 404. The
503,845,602-byte archive is therefore recorded as currently inaccessible, not
passed; its checkpoint and tokenizer opt-in tests remain explicit unverified
gates.

The next checkpoint-documentation slice removed the resulting false public
claim. A shared `CheckpointDocumentation` projection now derives provider,
authoritative URL, availability status, quickstart input, and limitation note
from architecture metadata. It is model-independent and preserves the lazy
boundary. The WeNet package declares an external archive, links the immutable
upstream README, records the 2026-08-02 HTTP 404 and certificate-hostname
failure in `SOURCE.json`, and shows only a local VoiceHub-native artifact in
its quickstart. Its false Hugging Face link, Colab badge, gallery row, and
generator-owned notebook were removed. The same generic path corrected
`styletts2` and `vad_transformers` from "registry default" to "no default".
The focused page/notebook contract reported 4 passes and 127 subtests. The
related WeNet, architecture, registry, optimization, scaffold,
provider-independence, documentation, and packaging suite reported 140 passes,
2 explicitly unverified WeNet skips, and 1,906 subtests. The first selected
pre-commit run applied YAPF and is not counted as passing; the formatted second
run passed. Strict multilingual documentation, release alignment, and fresh
wheel, sdist, and editable probes also passed.

The following release-asset slice exercised SpeechBrain's published
`tokenizer.ckpt` from immutable model revision
`979a53a7a3f6c9291c02c040fd8ebfb2471cf8a3`. VoiceHub's dependency-free Hub
transport downloaded 253,217 bytes with SHA-256
`37a6cba34cd520b33fd83612d5efc8ba7e351166541eb2726642bb3032234d31`.
The native SentencePiece implementation reproduced the pinned `HELLO WORLD`
encoding and decoding vectors. The common release-asset opt-in now covers this
test beside ESPNet and SenseVoice, and both Package CI and the tagged release
build require the three-test gate. The focused SpeechBrain/workflow contract
reported 3 passes, the combined online release-asset gate reported 3 passes,
and the related SpeechBrain, release, packaging, registry, documentation,
optimization, and scaffold suite reported 129 passes and 1,838 subtests. After
redownloading and fingerprinting the TEN-VAD and QuartzNet artifacts, the full
opt-in Python 3.12.12 macOS suite reported 2,442 passes, 7 skips, 3,521
subtests, and 35 warnings in 115.78 seconds. The remaining skips are five
CUDA/Triton or CUDA-toolkit gates plus the inaccessible WeNet checkpoint and
tokenizer; none is counted as passed. Strict multilingual documentation,
release-version alignment, and fresh wheel, source-distribution, and editable
probes also passed.

The next supported-version evidence slice installed the current working tree
from source into fresh temporary CPU environments with direct `uv pip install`
commands; the repository lock-file hash remained unchanged. The full
default-offline suite reported 2,434 passes, 15 explicit skips, 3,383 subtests,
and 35 warnings on both Python 3.10.19 in 179.77 seconds and Python 3.11.15 in
166.56 seconds. Together with the current Python 3.12.12 full opt-in result,
the complete tree has now executed on every supported interpreter on macOS.
This does not substitute for Linux or Windows execution of the uncommitted
tree; the latest remote nine-job matrix remains tied to `f2d6332`.

The following formatting-and-lint evidence slice ran the complete repository
pre-commit configuration against the current working tree. End-of-file,
trailing-whitespace, case-conflict, private-key, AWS-credential, pyupgrade,
isort, YAPF, Markdown formatting, Flake8, and docformatter hooks all passed
without modifying a file. This replaces older partial-hook evidence; it does
not substitute for the pending Linux and Windows matrix.

The release workflow first repeats the complete test suite on Linux, macOS,
and Windows with Python 3.10, 3.11, and 3.12. Its build job cannot begin unless
all nine tagged-source jobs pass. It then repeats the remaining local gates,
checks that tag `v0.3.0` points at the checked-out commit, verifies that 0.3.0
is not already on PyPI, builds one wheel/sdist pair, checks their embedded
metadata and size, and transfers those exact artifacts to a separate publish
job.

Fresh clean-install builds passed embedded name, version, and size checks and
remained below the 100 MB publication limit. Gzip timestamps and evidence-only
documentation edits can change archive bytes, so release sizes and hashes are
recorded from the exact workflow artifacts rather than copied from a local
build.

The current repository-wide pre-commit gate initially normalized imports in 15
files and applied YAPF and docformatter changes. That first formatter run exited
with failure and is not counted as a pass. The second `pre-commit run
--all-files` completed every hook successfully. A raw AST fingerprint changed
because docformatter edits docstring constants, so it is not used as behavior
evidence. The formatted Python 3.12 tree instead reran the complete suite and
reported 2,400 passed, 25 skipped, 3,287 subtests, and 35 warnings. The later
diffusion-serving slice passed its selected hooks after formatter changes and
the Python 3.12 full suite reported 2,401 passed with the same 25 skips, 3,287
subtests, and 35 warnings. The task-default slice also required one formatter
pass before its selected hooks succeeded; its full-suite count was 2,405. The
compatibility-default slice also required one isort pass before its selected
hooks succeeded; the second selected run passed and the final full-suite count
was 2,406. The scaffold-catalog checker slice passed its selected hooks without
a formatter rewrite, and its full-suite count was 2,411. The catalog-renderer
slice required one docformatter rewrite before its selected hooks passed; its
final full-suite count is 2,414. The native dependency-boundary slice required
one YAPF/docformatter rewrite before its selected hooks passed; the corrected
focused tests reported 19 passes, and its final full-suite count is 2,416. The
shared provider-independence slice required one docformatter rewrite before its
selected hooks passed; its final full-suite count is 2,420. The manifest
discovery slice required two formatter runs before its selected hooks passed;
its final full-suite count is 2,424. The later VAD skip-audit slice passed its
selected hooks without a rewrite and reduced the current unverified count from
25 to 18; its final full-suite count is 2,428. The default-runtime codec slice
also passed its selected hooks without a rewrite and reduced the current
unverified count to 17; its final full-suite count is 2,429. The native
MeloTTS/OuteTTS dependency slice required one YAPF rewrite before its selected
hooks passed and reduced the current unverified count to 15; its final
full-suite count is 2,431.

The diffusion-serving resolver no longer contains a provider-name allowlist.
Its native and vLLM-Omni verified model snapshots are derived from
`diffusion-serving-native` and `diffusion-serving-vllm-omni` features declared
beside each architecture integration. A runtime-registration regression proves
that a new native model can join the resolver without editing shared serving
code; unverified pairings and VibeVoice's incomplete high-level path still fail
closed.

The shared auto factories no longer embed registered TTS, ASR, or VAD model
names. The former `AutoInferenceModel` Orpheus default is now the TTS registry
declaration, preserving its no-argument behavior while sharing the same policy
with `AutoModelForTextToSpeech`.
The registry enforces at most one `default_for_task` declaration per task and
exposes the selection through `get_default_model_spec()`. A runtime extension
regression proves that the no-argument factory follows replaced registry
metadata without changing `voicehub/auto.py`; missing and ambiguous defaults
fail explicitly.

PR 68 was merged as main commit `679d5bd` before its required matrix was green.
PR 69 head `b6de7b5` subsequently repaired its canonical AI skill frontmatter
and normalized materialized root-guidance pointers. CI run 30741892984 then
passed every Linux and macOS Python job, both runtime smokes, default runtime,
training, and lint. Windows 3.10, 3.11, and 3.12 each failed only the same two
scaffold-checker assertions: actionable diagnostics rendered temporary
repository-relative paths with native `\` separators while the public
cross-platform contract expected stable `/` separators. Each Windows job
otherwise reported 2,437 passes, 16 explicit skips, 3,392 subtests, and 35
warnings.

The bounded correction centralizes repository-relative display through
`PurePath.as_posix()` and includes a dependency-free `PureWindowsPath`
regression. The focused scaffold file reported 20 passes and 35 subtests; the
related registry, model-page, optimization, release, and AI-guidance slice
reported 113 passes and 1,849 subtests; selected pre-commit hooks, all 68
generated model pages, all 59 notebooks, and release alignment also passed.
The complete Python 3.12.12 suite reported 2,439 passes, 15 explicit skips,
3,394 subtests, and 35 warnings in 115.67 seconds. That correction is now PR 69
head `3a3e224`; [CI run 30742766090](https://github.com/kadirnar/voicehub/actions/runs/30742766090)
passed every Linux, macOS, and Windows Python 3.10-3.12 job, both runtime
smokes, default runtime, training, and lint.

The following documentation-parity slices record the official Transformers
`main` commit and toctree fingerprint, map representative routes, and restore
the left navigation and right table of contents on all eleven localized home
sources. Rendered checks at 1440 x 900, 1024 x 768, and 390 x 844 verified the
desktop shell, mobile collapse, both VoiceHub palettes at all three viewports,
and the absence of horizontal overflow. The tablet slice then matched the
reference region behavior at 1024 x 768: a persistent 270-pixel left
navigation, hidden right table of contents, 739-pixel content region inside the
1,009-pixel main region, and no redundant drawer button. The mobile drawer and
desktop three-column shell remained structurally unchanged. The drawer backdrop now excludes the
242-pixel panel itself, leaving a 133 x 844 pointer target at 390 x 844. A real
backdrop click in both LTR and RTL layouts and Escape in the LTR layout cleared
the drawer checkbox and returned the panel off-canvas with no horizontal
overflow. Initial probes against
the full-width backdrop and the preexisting Escape behavior did not close the
drawer; those failed checks are not counted. The final 30 documentation
contracts reported 1,132 subtests, the related registry and optimization slice
reported 70 passes and 310 subtests, all 68 generated model pages and 59
notebooks were current, and the strict multilingual build completed. Release
alignment, 13 AI-guidance and release-readiness tests with 11
subtests, and selected hooks also passed. Fresh wheel, source-distribution, and
editable probes passed with the inventory recorded above. The earlier shell
slice's first selected hook run let YAPF format its regression and exited
nonzero; the formatted regression and second hook run passed, so the failed run
is not counted as passing evidence.

The distribution gate inventories every `SOURCE.json`, license, licence,
NOTICE, and COPYING file under the installed package. Each of the 81 source
manifests must contain a pinned revision/release and explicit license metadata;
all 193 compliance files plus the project-level Apache-2.0 license must survive
both wheel and source-distribution builds.

PR 69 exact head `3a3e224` passed every required job in CI run 30742766090.
Interim main head `6a75eda` subsequently passed Package CI and the strict
documentation build in [runs 30745818302](https://github.com/kadirnar/voicehub/actions/runs/30745818302)
and [30745818292](https://github.com/kadirnar/voicehub/actions/runs/30745818292).
The documentation-parity branch derives from `6a75eda`. PR 71 exact head
`8ea5e94` passed all nine platform/version jobs, both runtime smokes, default
runtime, training, lint, strict documentation, package build, and
`pre-commit.ci`. The documentation deployment job was skipped for the pull
request and is not counted as a pass. PR 71 was merged externally at
2026-08-02 13:11 UTC as `566db6822d47a335c720efb9ea66d7bcb22a1a82`.
That exact merge commit passed all nine platform/version jobs, both runtime
smokes, default runtime, training, lint, the strict documentation build and
deployment, and Package CI in runs 30749398917, 30749398929, and 30749398914.
The current uncommitted navigation and header worktree remains outside that
remote evidence.

The next left-navigation state slice maps the Transformers and VoiceHub
Installation routes. Exact rendered checks at 1440 x 900, 1024 x 768, and
390 x 844 now cover one visible active item, the expanded current branch,
visible keyboard focus, and zero horizontal overflow. VoiceHub rendered its
219 x 34-pixel desktop item and 212 x 34-pixel tablet item in both light and
dark themes. Its opened mobile drawer measured 242 pixels with a 133-pixel
backdrop and exposed one 251 x 48-pixel active row in both themes. The mapped
Transformers active item measured about 218 x 31 pixels at desktop and tablet
widths and 323 x 31 pixels in its opened mobile navigation; all three reference
states exposed visible keyboard focus without horizontal overflow. The focused
source regression first failed before the state styles existed. A rendered
keyboard probe then exposed Material's `focus-visible` compatibility class;
the first selector did not draw an outline, and that failed probe is not
counted. The compatibility selector and focused regression now pass; the exact
viewport rerun's three focused navigation and drawer tests also pass. The
documentation file reported 30
passes and 1,132 subtests, the related registry and universal-optimization
files reported 46 passes and 236 subtests, all 68 model pages and 59 model
notebooks were current, release alignment found all five benchmark files, the
strict multilingual documentation and fresh wheel, source-distribution, and
editable probes passed. The complete Python 3.12.12
suite reported 2,444 passes, 15 explicit skips, 3,406 subtests, and 35 warnings
in 116.03 seconds. The first selected hook run let YAPF format the new
regression and exited nonzero; the formatted second run passed, so the first is
not counted. The reference mobile light state and the remaining representative
page matrix remain pending and are not reported as passed.

The next bounded documentation slice replaced the five product tabs with the
Goal's eight-section hierarchy: Get started, Base classes, Inference, Training,
Quantization and optimization, Ecosystem integrations, Resources, and API.
The generated 68-model block now remains nested under `Base classes > Models`,
while all existing documentation routes remain unchanged. The source contract
first failed against the five-section configuration and then passed after the
navigation and all ten localized label inventories were updated. Rendered
checks at 1440 x 900, 1024 x 768, and 390 x 844 verified exact order, visible
keyboard focus, both VoiceHub themes, and zero horizontal overflow. These
later uncommitted worktree changes are not covered by PR 71 exact-head CI and
must not be treated as cross-platform or exact-head release evidence. On the
current Python 3.12.12 worktree, 43 documentation, release, and guidance tests
passed with 1,143 subtests; the related registry and universal-optimization
files passed 46 tests with 236 subtests; all 68 model pages and 59 notebooks
were current; release alignment and the strict eleven-language build passed;
and the wheel, source-distribution, and editable probes passed. The complete
suite reported 2,444 passes, 15 explicit skips, 3,406 subtests, and 35 warnings
in 114.71 seconds. The first selected hook run let YAPF format the regression
and exited nonzero; the formatted regression and second hook run passed, so
the failed hook run is not counted.

The next bounded header slice adds the missing version control and renders the
six product controls in the reference order: product, search, version,
language, theme, and source. The version menu distinguishes the current `main`
documentation and 0.3.0 release candidate from the published PyPI 0.1.6
package. Rendered desktop, tablet, mobile LTR, and mobile RTL checks exercised
pointer dismissal, Enter and Space activation, Escape dismissal with focus
restoration, `aria-expanded`, a two-pixel focus outline, both VoiceHub themes,
and zero horizontal overflow. Initial source, keyboard, focus-compatibility,
mobile-label, and RTL-overflow probes each failed before their corresponding
implementation or correction; none is counted as passed. This evidence covers
control order and version interaction only. The reference places its controls
in the documentation rail at desktop and tablet widths and includes a Hugging
Face corporate banner, while VoiceHub retains a Material top header. Exact
placement and geometry, the corporate-banner decision, remaining focus order,
other expanded control states, the reference mobile light render, and the
representative page matrix remain pending. These uncommitted changes are not
covered by merge-commit CI. The focused header regression and 31-test
documentation file passed; the combined documentation, release, and guidance
slice reported 44 passes and 1,143 subtests. The related registry and universal
optimization slice reported 46 passes and 236 subtests. All 68 model pages and
59 notebooks were current, release alignment found all five benchmark files,
the strict eleven-language build completed, and the fresh wheel,
source-distribution, and editable probes passed. The complete Python 3.12.12
suite reported 2,445 passes, 15 explicit skips, 3,406 subtests, and 35 warnings
in 111.61 seconds. Every applicable selected hook passed; the
Markdown-formatting hook found no matching files and is not counted as a pass.

The following bounded search-interaction slice matches the reference expanded
dialog's outer bounds at all three checked viewports: 500 x 72 pixels at
x = 470, y = 64 on desktop; 500 x 72 pixels at x = 262, y = 64 on tablet; and
358 x 72 pixels at x = 16, y = 64 on mobile. VoiceHub's named trigger opens by
pointer, Enter, Space, or Command/Ctrl+K, updates `aria-expanded`, focuses the
query field, locks body scrolling, and closes with Escape while restoring a
visible focus indicator. Light and dark desktop, tablet, and mobile states plus
English LTR and Arabic RTL mobile containment had zero overflow. The reference
Command+K and Escape paths were also exercised, and both reference mobile
themes were captured. Its checked mobile icon has no accessible name, so that
upstream limitation is not counted as a pass.

The initial 688 x 48-pixel local dialog, the first constrained 625 x 92/100-
pixel correction, the first Escape path without visible focus restoration, and
the first mobile state with 77 pixels of header overflow all failed and are not
passing evidence. After correction, the focused search contract passed, the
32-test documentation file and combined documentation/release/guidance slice
reported 45 passes and 1,143 subtests, and the registry and universal-
optimization slice reported 46 passes and 236 subtests. All 68 model pages and
59 notebooks were current, release alignment found all five benchmark files,
the strict eleven-language build completed, and fresh wheel, source-
distribution, and editable probes passed with 68 models, 81 provenance
manifests, and 193 compliance files. The complete Python 3.12.12 suite reported
2,446 passes, 15 explicit skips, 3,406 subtests, and 35 warnings in 114.70
seconds. Every applicable selected hook passed; the Markdown-formatting hook
found no matching files and is not counted as a pass. These uncommitted changes
remain outside merge-commit CI, and the collapsed reference-rail placement,
corporate-banner decision, other expanded header controls, and remaining
representative page matrix are still pending.

The next bounded language-control slice replaces Material's 40 x 40-pixel icon
and 125 x 200-pixel custom list with the reference's compact native select.
VoiceHub and Transformers now expose the same 48 x 26-pixel control dimensions
at 1440 x 900 and 1024 x 768; their x/y coordinates still differ because the
reference uses its documentation rail while VoiceHub retains the known top-
header placement. Both sites hide the control at 390 x 844. VoiceHub preserves
only its eleven built locales, labels the control accessibly, displays the
current uppercase locale code, and selected `TR` through the control to load a
Turkish document with the Turkish option still selected.

Pointer, Enter, and Space activation followed by Escape kept the current value
and focus on the select on both sites. An outside-pointer probe transferred
local focus to the target control, the local two-pixel focus indicator was
visible, and light/dark desktop and tablet plus English LTR and Arabic RTL
mobile states had zero overflow. Native option popups are browser chrome and
are not claimed as DOM-geometry or screenshot evidence. The focused source
test first failed because the override did not exist. The first rendered
native-select probe then exposed white text on white; its scoped regression
failed before the color correction. None of those failed checks is counted as
passing evidence.

After correction, the focused language contract and 33-test documentation file
passed. The combined documentation/release/guidance slice reported 46 passes
and 1,143 subtests; registry and universal optimization reported 46 passes and
236 subtests; 68 model pages and 59 notebooks were current; release alignment
found all five benchmark files; and the strict eleven-language build passed.
Fresh wheel, source-distribution, and editable probes passed with 68 models,
81 provenance manifests, 193 compliance files, zero dependency violations,
and no eager PyTorch import. The complete Python 3.12.12 suite reported 2,447
passes, 15 explicit skips, 3,406 subtests, and 35 warnings in 106.30 seconds.
Every applicable selected hook passed; the Markdown-formatting hook found no
matching files and is not counted. At that point, the uncommitted worktree was
outside merge-commit CI, and reference-rail placement, the corporate-banner
decision, theme/source exact responsive evidence, and the representative page
matrix were left for later slices.

The following bounded header-tail slice replaces Material's 40 x 40-pixel
theme control and 234 x 48-pixel repository block with the current reference's
compact forms. At the executed 1280 x 720 viewport, VoiceHub rendered a
34 x 24-pixel theme button, a 55 x 16-pixel source link, and a 12-pixel gap,
matching the measured reference geometry. The theme switched in both directions
through pointer, Enter, and Space while restoring focus to the newly visible
control with a two-pixel outline; both palettes retained zero overflow. The
source link gained an accessible name and pointer activation reached the
declared GitHub repository.

The focused source contract first failed because neither override existed. The
first combined documentation run then exposed a stale single-selector language
assertion after the mobile visibility rule was shared. An earlier
browser-driven Enter probe on the native source link did not navigate, and the
first embedded exact-viewport attempt was blocked by URL security policy. None
of those failed checks is counted.

The final bounded correction adds an explicit Enter path to the named source
link. Its focused regression first failed before the handler existed and then
passed after correction. Browser evidence verified Enter navigation to the
declared repository. Fresh 1440 x 900 and 1024 x 768 renders retained the
34 x 24-pixel theme button, 55 x 16-pixel source link, and 12-pixel gap in both
palettes. Fresh 390 x 844 light and dark renders hid language, theme, and source
with zero rendered width or height. Every state had zero document and header
overflow; keyboard theme switching restored focus with a two-pixel outline and
two-pixel offset. A separate attempt to move focus from the theme button with a
synthetic Tab key did not move focus and is not counted as focus-order evidence.

After correction, the focused regression and 34-test documentation file passed.
The combined documentation/release/guidance slice reported 47 passes and 1,143
subtests; the complete Python 3.12.12 suite reported 2,448 passes, 15 explicit
skips, 3,406 subtests, and 35 warnings in 117.70 seconds. The strict
eleven-language build completed; 68 model pages, 59 notebooks, and release
alignment passed. Fresh wheel, source-distribution, and editable
probes passed with 68 models, 81 provenance manifests, 193 compliance files,
zero dependency violations, and no eager PyTorch import; the wheel measured
57,186,791 bytes and the source distribution 55,427,491 bytes. The worktree
remains outside exact-commit remote CI.

The next bounded shell slice moves the six documentation controls out of the
Material top-header layout and into the reference's desktop and tablet rail.
The final exact-viewport renders use a 65-pixel global VoiceHub brand row, a
270-pixel left rail, and a 270 x 128-pixel rail-control block. At 1440 x 900 the
main regions measure 270/900/270 pixels; at 1024 x 768 they measure 270/754
pixels with the right table of contents hidden. The product, collapsed search,
and utility regions begin at x = 16 and y = 77, 112, and 150. The utility row
preserves the reference's 80 x 26 version, 48 x 26 language, 34 x 24 theme, and
55 x 16 source controls and 12-pixel final gap. All eight root navigation
sections are visible without horizontal tabs. The 1024-pixel Arabic route
mirrors the rail to x = 754 and leaves the 754-pixel content region at x = 0.
Light and dark desktop geometry is identical and every checked state has zero
horizontal overflow.

The 390 x 844 shell retains the existing 64-pixel mobile header and working
242-pixel drawer. Moving search into the shared control wrapper initially made
the opened mobile search region collapse to zero size; that failed probe is not
counted. The scoped mobile-open rule was corrected and the rerun restored the
expected 358 x 72 region at x = 16, y = 64, with Escape dismissal and zero
overflow. Version Enter/Escape state, Command/Ctrl+K search focus and dismissal,
theme switching, mobile drawer pointer/Escape behavior, LTR/RTL mirroring, and
both palettes were exercised. VoiceHub has no Hugging Face corporate ecosystem,
so the matched global row intentionally contains VoiceHub branding rather than
invented corporate links. Complete sequential focus order and the eight pending
representative page pairs remain explicit gates.

The focused shell contract and 35-test documentation file passed with 1,132
subtests. The registry and universal-optimization slice reported 46 passes and
236 subtests; all 68 model pages and 59 notebooks were current; release
alignment found all five benchmark files; the strict eleven-language build and
fresh wheel, source-distribution, and editable probes completed. These
uncommitted changes remain outside merge-commit CI and are not cross-platform
evidence. The complete Python 3.12.12 suite then reported 2,449 passes, 15
explicit skips, 3,406 subtests, and 35 warnings in 122.45 seconds.

The next bounded representative-page slice maps the Transformers and VoiceHub
Quickstart routes. The source regression first failed against the `First
generation` navigation label and the page's earlier task-example-only
structure. After correction, VoiceHub exposes the mapped setup, lazy discovery,
pretrained-model, inference, trainer, and next-step hierarchy, plus a page-copy
action. Exact browser measurements match the reference article at 804 pixels
wide on 1440 x 900, 658 pixels on 1024 x 768, and 342 pixels with a 24-pixel
gutter on 390 x 844. The title, section headings, body type, desktop code block,
and mobile 12/366-pixel code bounds also match. VoiceHub light and dark renders
at all three viewports had zero horizontal overflow, and pointer copy reported
`Copied`.

The first desktop gutter correction still resolved to Material's earlier
24-pixel physical margin because the theme selector had higher specificity;
that intermediate render is not counted. The later equal-specificity override
produced the measured 48-pixel gutter. The first copy-control render wrapped
its label and was likewise replaced before final evidence. A locator-driven
Enter/Tab probe did not expose copied or focus-visible state, so page-action
keyboard activation and sequential focus remain pending rather than passed.
An initial ad hoc optimization inventory also imported a nonexistent
`list_optimization_passes` symbol and failed; the corrected registry-object
query reported six passes and 408 model/pass pairs, while the failed command is
not evidence.

After correction, the focused contract passed; the documentation file reported
36 passes and 1,133 subtests; and the registry and universal-optimization slice
reported 46 passes and 236 subtests. All 68 model pages and 59 model notebooks
were current, release alignment found all five benchmark files and 68
documented providers, and the strict eleven-language build passed. Fresh wheel,
source-distribution, and editable probes passed with 68 models, 81 provenance
manifests, 193 compliance files, zero dependency violations, and no eager
PyTorch import; the wheel measured 57,186,791 bytes and the source distribution
55,428,271 bytes. The complete Python 3.12.12 suite reported 2,450 passes, 15
explicit skips, 3,407 subtests, and 35 warnings in 121.32 seconds. Every
applicable selected hook passed; the Markdown-formatting hook found no matching
files and is not counted. These uncommitted changes remain outside exact-commit
remote CI. Home, Installation, and Quickstart remain partial at their stated
interaction gates, and seven representative page pairs remain unexamined.

A follow-up iteration selected only the Quickstart page-copy keyboard boundary.
The first focused source regression failed because the action had no explicit
keyboard contract. The first correction intercepted `keydown` and disabled the
button while copying; the rendered Enter probe then reported `Copy failed` and
lost focus, so neither result was counted. A second regression rejected the
disabled state and required an `aria-busy` lifecycle, a selection fallback, and
focus restoration. Its rendered Enter probe preserved the two-pixel focus
treatment but still reported `Copy failed` because both clipboard methods were
rejected from the intercepted key event. The final bounded correction retains
the native button's one keyboard-generated `click`, avoids `preventDefault`,
and invokes the copy routine only once while restoring keyboard focus.

The focused source contract now passes. Pointer activation replaced a clipboard
sentinel with the complete 5,252-character Quickstart article, `aria-busy`
returned to `false`, and the exact 804/658/342-pixel article geometry remained
unchanged at 1440 x 900, 1024 x 900, and 390 x 844. The 390-pixel render kept the
12/366-pixel code bounds; light and dark renders had zero horizontal overflow.
Locator-driven Enter and Space both left the button focused with the solid
two-pixel outline and two-pixel offset, but that driver did not dispatch the
native button `click`; the seeded clipboard remained unchanged. Two bounded
Chrome attempts could not navigate to the localhost preview. Native keyboard
activation and sequential focus are therefore precise pending gates and are
not reported as passed.

After this correction, the documentation file reported 36 passes and 1,133
subtests; the registry and universal-optimization slice reported 46 passes and
236 subtests; all 68 model pages and 59 model notebooks were current; and
release alignment found all five benchmark files and 68 documented providers.
The strict eleven-language build and selected applicable pre-commit hooks
passed. Fresh wheel, source-distribution, and editable probes passed with 68
models, 81 provenance manifests, 193 compliance files, zero dependency
violations, and no eager PyTorch import; the wheel measured 57,186,791 bytes and
the source distribution 55,428,842 bytes. The complete Python 3.12.12 suite
reported 2,450 passes, 15 explicit skips, 3,407 subtests, and 35 warnings in
113.65 seconds. These uncommitted changes remain outside exact-commit remote CI.

The next bounded representative-page slice maps the Transformers Pipeline
tutorial to VoiceHub's speech tasks. The focused regression first failed at
import time because VoiceHub did not export a pipeline function or task
adapters. The implemented dependency-light contract now selects the registered
TTS, ASR, or VAD auto factory, preserves normalized outputs, retains lazy
loading, wraps configured model objects without changing runtime state, and
rejects cross-task or incomplete models. It deliberately does not report a
universal vectorized batch or chunking path.

The VoiceHub guide now follows the reference's task, parameter, device, batch,
task-specific parameter, chunking, large-input, and large-model hierarchy while
keeping speech-specific examples and limitations. Both articles measured 804
pixels from x = 318 at 1440 x 900, 658 pixels from x = 318 at 1024 x 768, and
342 pixels from x = 24 at 390 x 844. VoiceHub light and dark states retained
those bounds with zero document overflow. The active Pipeline navigation,
complete desktop TOC, mobile drawer, Tasks anchor, Command+K/Escape search, and
pointer page copy passed; the copied article contained 6,742 characters.

The synthetic Enter page-action probe restored named-button focus and the
focus-visible class but did not receive clipboard permission. Pointer copy in
the same harness did receive permission and reported `Copied`. Native keyboard
clipboard activation is therefore still inaccessible, and complete sequential
focus is pending; neither is counted as passed. An initial click selected a
hidden duplicate Tasks link and timed out, and the first reference geometry
selector targeted the page shell rather than its article. The corrected visible
link and article selector produced the recorded evidence; the failed probes are
not evidence.

After correction, the focused pipeline/documentation slice reported 9 passes
and 13 subtests. The registry, model-page, import, and documentation slice
reported 118 passes, 3 explicit default-runtime skips, and 1,459 subtests. The
optimization slice reported 168 passes and 759 subtests. The strict
eleven-language build passed, and fresh wheel, source-distribution, and
editable probes reported 68 models, 81 provenance manifests, 193 compliance
files, zero dependency violations, and no eager PyTorch import. The wheel was
57,189,063 bytes and the source distribution was 55,432,769 bytes. The complete
Python 3.12.12 suite reported 2,458 passes, 15 explicit skips, 3,417 subtests,
and 35 warnings in 115.32 seconds. The skips remain assigned to three
default-runtime CI jobs, five CUDA/Triton/toolkit paths, and seven inaccessible
release assets, checkpoints, or oracles. Every selected pre-commit hook passed,
including Markdown formatting. These uncommitted changes remain outside
exact-commit remote CI.

The next bounded shared-shell slice verifies the right table of contents on the
mapped Pipeline pair. The current official and local pages both use a 270-pixel
desktop rail at x = 1,170, keep their link geometry stable while the article
scrolls, hide the rail at 1,024 and 390 pixels, and retain zero horizontal
overflow. VoiceHub's tracking contract selected exactly one of Tasks,
Parameters, or Large models after the corresponding heading crossed the
fixed-header threshold. Its earlier inherited filled pill was heavier than the
reference, so the smallest correction scopes the secondary active state to a
transparent background, no box shadow, and the existing modern VoiceHub color
token in both light and dark themes.

The focused regression first failed because that scoped secondary state did
not exist, then passed after the correction. The first runner invocation used
an unavailable `python` command and is not evidence; the same test was executed
with the repository virtual environment for both the failing and passing runs.
Pointer anchors placed headings below the fixed shell, and focused TOC links
rendered a solid two-pixel outline with a two-pixel offset. Locator-driven
Enter and a native browser keypress both left the target link focused without
navigating, so keyboard anchor activation remains inaccessible and is not
counted as passed.

After the correction, the complete documentation contract file reported 38
passes and 1,135 subtests, and the strict eleven-language site build passed.
Every applicable selected pre-commit hook passed; the Markdown-formatting hook
found no matching files and is not counted. This slice changes no registry,
runtime, optimization, or packaging contract, so the immediately preceding
successful evidence for those surfaces remains applicable. The working tree is
still outside exact-commit remote CI.

The next bounded shared-shell slice closes the left documentation rail's
desktop and tablet scroll-offset mismatch. The official Pipeline product label
and navigation region move upward by the 65-pixel global row as the article
scrolls. VoiceHub now applies the same bounded transition: at 1,440 x 900 its
product controls move from y = 65 to y = 0, its product label from y = 77 to
y = 12, its primary rail from y = 65 to y = 0, and its independent navigation
region from y = 193 / 707 pixels to y = 128 / 772 pixels. At 1,024 x 768, the
scrolled primary rail has a 640-pixel navigation region below the 128-pixel
control block and the secondary rail remains hidden. Light and dark renders at
both widths retain the same geometry and zero horizontal overflow. The
390 x 844 shell retains its fixed 64-pixel mobile header and working 242-pixel
drawer without overflow. Keyboard focus on the desktop search input renders a
solid two-pixel indigo form outline with a two-pixel offset.

The focused source regression first failed because the shell did not track the
article offset. Intermediate header-transform and header-position attempts
changed fixed descendants or obscured the rail controls and are not counted.
Material's inline primary-rail dimensions and tablet stacking state required
the final narrowly scoped overrides. The resulting focused regression passes;
the documentation contract reports 39 passes and 1,135 subtests; JavaScript
syntax, whitespace validation, the strict eleven-language build, and every
applicable selected pre-commit hook pass. Markdown formatting found no matching
files and is not counted. One earlier full documentation run failed only
because its source assertion still required the superseded fixed top offset;
the assertion was updated to the tracked contract before the passing complete
run. This documentation-only slice does not change registry, model,
optimization, runtime, or package contents, so the immediately preceding
successful evidence for those surfaces remains applicable. The worktree is
still outside exact-commit remote CI.

The next bounded representative-page slice maps the Transformers Auto Classes
reference to VoiceHub's model index. The refreshed inventories still contain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, 68 generated pages, 59
generated notebooks, six public optimization passes, and 408 model/pass pairs.
The first source regression failed because the index still began with `Model
guides`. A new immutable `ModelSpec.display_name` projection now derives the
presentation label from the public class name without changing any canonical
key. The generator uses it consistently in every model-page heading, index
link, and navigation label and exposes a compact AutoConfig, AutoProcessor,
task-specific auto-model, discovery, and registered-model hierarchy.

The local Auto Classes page rendered in both VoiceHub palettes at 1,440 x 900,
1,024 x 768, and 390 x 844. Its article measured 804, 658, and 342 pixels; the
desktop left and right rails measured 270 pixels; the right rail collapsed at
the smaller widths; and every state retained one active `Auto Classes` item
and zero document overflow. The four mobile tables scroll inside their own
wrappers. Keyboard focus on the active desktop item rendered a solid two-pixel
outline and inset cue. The official reference was rendered at its available
1,280 x 720 viewport and exposed the mapped auto-class hierarchy with zero
overflow. Its product banner, community panel, NLP families, and exhaustive
generated API inventory are intentional product/content differences.

The focused index and guide checks reported two passes and 136 subtests, and
the all-task display-name regression reported one pass and 68 subtests. The
broader registry, documentation, and optimization slice reported 141 passes
and 1,597 subtests. The complete documentation file reported 40 passes and
1,204 subtests; the strict eleven-language build passed; and candidate release
alignment found five benchmark files and 68 documented providers. The full
Python 3.12.12 suite reported 2,462 passes, 15 explicit skips, 3,554 subtests,
and 35 warnings in 126.54 seconds. Fresh wheel, source-distribution, and
editable probes reported 68 models, 81 provenance manifests, 193 compliance
files, zero dependency violations, and no eager PyTorch import. The wheel was
57,189,200 bytes and the source distribution was 55,434,554 bytes. Every
applicable selected pre-commit hook passed after YAPF's first invocation
reformatted the changed Python files; that modifying invocation is not counted
as a pass. Markdown formatting found no matching files and is not counted.

One combined focused-test command named three nonexistent test files and
collected no tests; the corrected command used the repository's actual
registry and optimization filenames and produced the 141-pass result above.
An ad hoc inventory that attempted to load MkDocs' Python-specific YAML tag
with `safe_load` also failed and is not evidence. Complete sequential focus is
still unverified: a locator Tab probe did not advance, and the native desktop
fallback was safety-blocked. These uncommitted changes remain outside
exact-head remote CI, so neither that interaction gate nor cross-platform
coverage for this worktree is reported as passed.

The next bounded representative-page slice maps the official Transformers
SpeechT5 model detail to VoiceHub's generated model guides. The refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, 68
generated pages, 59 generated notebooks, six public optimization passes, and
408 model/pass pairs. The first focused SpeechT5 regression failed because the
old guide did not provide the mapped Usage, configuration, processing,
limitations, and source-linked API hierarchy. The generator now applies one
nine-section detail contract to every registered model without importing its
implementation. It resolves paper and upstream GitHub metadata alongside the
stable configuration and model facades, requires at least four parseable Python
examples, separates checkpoint defaults from execution evidence, and states
optional-dependency, hardware, optimization-failure, and real-checkpoint
limitations explicitly.

The local SpeechT5 page rendered with 804-, 658-, and 342-pixel articles at
1,440 x 900, 1,024 x 768, and 390 x 844. Every state had zero document
overflow; all eight desktop tables fit their wrappers, and all eight mobile
tables scrolled inside their wrappers. The default palette passed all three
widths and the dark palette passed desktop. The desktop primary and secondary
rails were visible, the right table of contents exposed the complete generated
hierarchy, the smaller widths collapsed the right rail, and the mobile drawer
showed one visible active SpeechT5 item. Pointer copy reported `Copied`, the
edit action targeted the SpeechT5 documentation source, both public-facade
links targeted their resolved local source paths, and desktop active-link focus
rendered a solid two-pixel outline with a two-pixel offset. The official page
was rendered at the controller's available 1,280 x 720 viewport and exposed
the mapped configuration, processor, task-model, and vocoder API hierarchy,
expandable parameters, copy controls, and zero overflow.

The complete documentation file reported 41 passes and 1,272 subtests. The
corrected registry, latest-ASR, speech-task, and universal-optimization slice
reported 101 passes and 427 subtests. All 68 model pages and all 59 model
notebooks were generator-current, the strict eleven-language build passed, and
every applicable selected pre-commit hook passed after YAPF's first invocation
reformatted the changed Python files; that modifying invocation is not counted
as a pass. The first post-implementation representative run expected double
quotes while the generator correctly emitted Python `repr` with single quotes,
and an intermediate all-page run used an obsolete 210-line bound; both failed
assertions were corrected before the complete passing run. One broader command
named nonexistent `test_latest_asr_models.py` and `test_speech_task.py` files
and collected no tests; the corrected command used
`test_latest_asr_registry.py` and `test_speech_task_registry.py` and produced
the 101-pass result. Two attempted mobile dark-theme clicks exceeded the
browser selector deadline and are not evidence.

Exact reference renders at the three target viewports, dark local tablet and
mobile renders, and complete sequential focus remain pending and are not
counted as passed. This documentation-generation slice does not change package
contents or model runtime behavior, so distribution probes and the full suite
were not rerun; their immediately preceding evidence remains applicable. The
worktree remains outside exact-commit remote CI.

The next bounded representative-page slice maps the official Transformers
Trainer overview to VoiceHub. The pinned upstream snapshot remains commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722` with toctree SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`;
the 30-line Trainer source has SHA-256
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`.
The refreshed repository inventories remain 68 models (34 TTS, 23 ASR, and
11 VAD), 102 aliases, 68 complete model pages, 59 generated notebooks, six
public optimization passes, 408 model/pass pairs, five benchmark records, and
eight model-contribution steps.

The first focused regression failed because Training still linked its only
`Get started` item directly to the long fine-tuning guide. VoiceHub now exposes
one nested `Trainer overview` followed by `Fine-tuning`, matching the reference
mental model without discarding the speech-specific workflow. The concise
overview documents the shared `Trainer` and `TrainingArguments` loop, the
model-owned objective boundary, fail-closed support validation, and four
speech-relevant next steps. It contains the same title and single `Next steps`
heading depth as the reference and adds no copied upstream prose, generic loss,
or unsupported training claim.

The local overview rendered at 1,440 x 900, 1,024 x 768, and 390 x 844 with
804-, 658-, and 342-pixel articles. Light and dark states had identical
geometry, one visible active overview item, the expected desktop TOC and
responsive rail/drawer behavior, and zero document overflow at every width.
Pointer copy reported `Copied`; the edit action targeted the new source; the
next footer link targeted Fine-tuning; and focused navigation rendered a solid
two-pixel outline with a two-pixel offset. The official page rendered at the
controller's available 1,280 x 720 viewport with a 644-pixel article, one
active overview item, one `Next steps` heading, copy and source actions, and
zero document overflow. Its main-version banner, community panel, and Trainer
video remain intentional product-content differences.

The complete documentation file reported 42 passes and 1,279 subtests. The
focused Trainer, training-contract, adapter, runtime, and speech-training slice
reported 127 passes and 180 subtests; the registry and universal-optimization
slice reported 47 passes and 304 subtests. All 68 model pages and all 59 model
notebooks were generator-current, release alignment found five benchmark files
and 68 documented providers, the strict eleven-language build passed, and all
applicable selected pre-commit hooks passed without modifying a file. Markdown
formatting found no matching files and is not counted.

The first post-implementation focused rerun used a line-sensitive prose
assertion and failed on normal Markdown wrapping; the corrected regression
normalizes whitespace before checking the contract. The first strict preview
rebuild also rejected a nonexistent `#trainingarguments` API anchor; the link
now targets the existing `#training-arguments` anchor and the complete strict
build passes. A screenshot attempt waited on a drawer button after the
responsive state had changed and timed out; the corrected desktop capture is
the visual evidence. An exact reference viewport override remained fixed at
1,280 x 720, two reference theme-control clicks did not change the palette, and
the local short-page TOC click did not update the URL hash. Exact responsive
reference renders, reference dark mode, that anchor interaction, and complete
sequential focus remain pending and are not counted as passed.

This documentation-only navigation slice does not change model, training,
optimization, or package runtime code. The full suite and distribution probes
were therefore not rerun; their immediately preceding evidence remains
applicable. The untracked `uv.lock` stayed unchanged, and the worktree remains
outside exact-commit remote CI.

The next bounded representative-page slice replaces the stale Transformers
`perf_infer_gpu_one` mapping with the current official
`optimization_overview`. The pinned upstream commit remains
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the current overview source has
SHA-256
`19622667a7299f258f5c9a72940c9f26492619636f35d9bd592701c02745b620`,
and the toctree remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The refreshed local inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD),
102 aliases, 68 complete model pages, 59 generated notebooks, six public
optimization passes, 408 model/pass pairs, five benchmark records, and eight
model-contribution steps.

The first focused regression failed because the fifth navigation group still
linked its Overview label to the detailed TTS workflow. VoiceHub now exposes a
concise Optimization overview before that preserved workflow and the existing
codec, diffusion, and VITS-family guides. Its table covers `compile`,
`flash-attention-4`, `custom-kernels`, `codec-kernels`, `diffusion-cache`, and
`diffusion-sampling` exactly once. A single copyable helper demonstrates
discovery, application, manifest reporting, and restoration. The prose keeps
support evidence honest: compatibility is decided by runtime validation, not
registry discovery; failed reversible passes roll back; no registry-wide public
quantization pass exists; parallelism is a training or serving topology; and
continuous batching belongs to the serving scheduler.

The local page rendered at 1,440 x 900, 1,024 x 768, and 390 x 844 with 804-,
658-, and 342-pixel articles. Light and dark states retained the same geometry,
one active Overview item, the expected desktop TOC and responsive rail/drawer
behavior, internal mobile table scrolling, copy/edit actions, previous/next
navigation, and zero document overflow. The official page also rendered at all
three target viewports with its current heading hierarchy, active Overview,
desktop TOC, responsive rail, technique table, version banner, community panel,
copy action, and zero document overflow. Reference dark mode and complete
sequential focus remain pending and are not counted as passed.

The corrected focused regression passed with six subtests. The complete
documentation file reported 43 passes and 1,292 subtests. Registry and all
documented public optimization suites reported 138 passes and 839 subtests;
one PyTorch decomposition warning was emitted and is not reported as a pass or
failure. All 68 model pages and 59 model notebooks remained generator-current,
release alignment found five benchmark files and 68 documented providers, and
the strict eleven-language documentation build passed. The earlier failing
focused run remains a development failure, not evidence.

This navigation and documentation slice does not change package contents or
model runtime behavior, so the full suite and distribution probes were not
rerun; their immediately preceding evidence remains applicable. Exact-commit
remote CI, reference dark mode, complete sequential focus, protected publisher
configuration, and publication remain open. The untracked `uv.lock` stayed
unchanged.

The next bounded representative-page slice corrects the stale Transformers
contribution mapping. The pinned upstream commit remains
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the current modular guide has
SHA-256
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
the legacy contribution guide has SHA-256
`4d7e7066deeefde340c3e0460eae540f343cdfe642d690a600cba0a90441cb03`,
and the pinned toctree remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The current upstream hierarchy places the modular guide under Base classes,
Models, Contribute and labels the former route Legacy model contribution.

VoiceHub now mirrors that nested contribution group and uses the generic Add a
model label in every configured locale. The guide maps the upstream
reduced-boilerplate and standalone-file outcome to speech-specific composition:
provider configuration, runtime, task wrapper, conversion, audio, codec, and
streaming boundaries stay explicit instead of being generated through model
inheritance. Its eight cards and matching file table cover Create, Audit,
Configure, Wrap, Register, Support, Test, and Document, including optional
owned architecture, provenance and legal files, the manifest, training and
optimization factories, CPU-safe tests, generated provider documentation, and
navigation.

The local page rendered at 1,440 x 900, 1,024 x 768, and 390 x 844 with 804-,
658-, and 342-pixel articles. Light and dark states retained zero document
overflow. The process renders as two 392-pixel desktop columns, two 319-pixel
tablet columns, and one 342-pixel mobile column. The mobile drawer exposes the
active Contribute and Add a model path, and all three wide tables scroll inside
their 374-pixel wrappers. The official modular guide rendered at the same
three viewports with its current active route, desktop right TOC, version
banner, community panel, copy action, table, internally scrollable mobile code,
and zero document overflow. Reference dark mode and complete sequential focus
remain pending and are not counted as passed.

The initial focused regression failed on the legacy route and seven-card
process. The stylesheet assertion then failed before the Material list-rule
specificity fix, and the first corrected mobile render exposed a second
specificity mismatch before the one-column override was aligned. Two
exploratory inventory commands also failed because they assumed nonexistent
optimization and model-spec attributes. None of these failed checks is counted
as passed.

After correction, the focused contribution, navigation, and stylesheet slice
reported three passes and 38 subtests. The complete documentation contract
reported 44 passes and 1,310 subtests; the model scaffold contract reported 20
passes and 35 subtests; and the registry plus public optimization slice
reported 76 passes and 786 subtests. All 68 model pages and 59 model notebooks
remained generator-current, release alignment found five benchmark files and
68 documented providers, and the strict eleven-language documentation build
passed. The full suite and distribution probes were not rerun for this
navigation, prose, test, and CSS slice; their preceding evidence remains
applicable. Exact-commit remote CI, reference dark mode, complete sequential
focus, protected publisher configuration, publication approval, and the API
reference representative-page mapping remain open. The untracked `uv.lock`
stayed unchanged.

The next bounded representative-page slice maps the current Transformers
Models API. The pinned upstream commit remains
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the 44-line Models source has
SHA-256
`a4899c758b5d621075b2e2f39f0aa79671010c88b08d1508f7af5731375c2871`,
and the toctree remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
All 24 current upstream Main Classes entries now map to a VoiceHub route or an
explicit unsupported or out-of-domain record.

VoiceHub now exposes `API > Main Classes > Models` while retaining the existing
Full API reference. The representative page documents the shared
`PreTrainedSpeechModel` marker; the TTS, audio, ASR, and VAD task bases; lazy
loading, saving, and training transitions; normalized outputs; public source
links; and portable versus provider-native artifacts. It records that no
public `push_to_hub()` contract exists rather than treating local saving as a
registry-wide sharing pass.

The local page rendered at 1,440 x 900, 1,024 x 768, and 390 x 844 with 804-,
658-, and 342-pixel articles in both VoiceHub palettes. It retained the active
nested Models route, the desktop TOC, responsive drawer, contained code, four
internally scrolling mobile tables, adjacent page navigation, and zero document
overflow. Pointer copy wrote the complete 5,106-character page and all three
declared source targets resolved. The official page rendered at the same three
viewports with its current Models, PreTrainedModel, ModuleUtilsMixin, and
Pushing to the Hub hierarchy, generated signature controls, responsive
navigation, internally scrollable mobile code, and zero document overflow.

Three official theme-control clicks did not change the reference palette. The
initial local Model outputs TOC click reached y = 68 but left the preceding link
active; that failed probe is not a pass. The later shared-shell correction
aligns the target and observer thresholds and closes this pointer transition.
A local Enter copy probe retained the two-pixel focus outline without changing
the clipboard. Reference dark mode, native keyboard copy activation, and
complete sequential focus are therefore pending and are not counted as passed.

The initial focused regression failed because API was still a flat entry and
the Models page did not exist. After correction it reported one pass and 50
subtests. The complete documentation contract reported 45 passes and 1,379
subtests; the base API, speech-core, and registry slice reported 49 passes and
216 subtests; and the public optimization slice reported 59 passes and 593
subtests. All 68 model pages and 59 model notebooks remained generator-current,
release alignment found five benchmark files and 68 documented providers, and
the strict eleven-language build passed. An exploratory optimization inventory
failed after treating its registry as iterable; the corrected public `list()`
query reported six passes and 408 model/pass pairs. The failed query is not
evidence.

The full suite and distribution probes were not rerun for this navigation,
documentation, and test slice; their preceding evidence remains applicable.
Exact-commit remote CI, reference dark mode, native keyboard copy activation,
complete sequential focus, protected publisher configuration, publication
approval, and the other representative-page interaction gates remain open.
The untracked `uv.lock` stayed unchanged.

The next bounded shared-shell slice closes the Models API pointer TOC
transition without introducing a competing observer. Measurement showed that
Material's default hash target stopped at y = 68, while its existing navigation
tracker switched to the new heading only at approximately y = 64. The shared
target margin now derives from the 65-pixel VoiceHub header and Material's own
heading offset, resolving to 64 pixels. The first focused regression failed
because that target contract did not exist; the corrected regression passes.

At 1,440 x 900, all four Models API TOC links retained their requested hashes,
positioned their headings between y = 63.8 and y = 64.1, and became the sole
active link in both VoiceHub palettes with zero document overflow. The mapped
Pipeline route preserved the same result for Tasks, Parameters, and Large
models, and manual scroll states at the threshold retained one matching active
link. At 1,024 x 768 and 390 x 844, the right rail remained hidden with zero
width, zero height, and zero document overflow. The official Models reference
retained its current hash navigation and zero overflow at the pinned upstream
commit. Its inaccessible dark palette remains unpassed.

The complete documentation contract reported 45 passes and 1,379 subtests;
the registry slice reported 52 passes and 316 subtests; and the public
optimization slice reported 161 passes, 751 subtests, and four warnings. The
13 release and distribution-source contracts passed. All 68 model pages and
59 model notebooks remained generator-current, release alignment found five
benchmark files and 68 documented providers, and the strict eleven-language
build passed. Fresh wheel, source-distribution, and editable probes reported
68 models, 81 provenance manifests, 193 compliance files, zero dependency
violations, and no eager PyTorch import; the wheel measured 57,189,200 bytes
and the source distribution 55,439,250 bytes. This CSS and regression slice did
not rerun the complete Python suite. Exact-commit remote CI, native keyboard
activation, complete sequential focus, reference dark mode, protected
publisher configuration, and publication approval remain open.

The next bounded evidence slice closes the responsive SpeechT5 model-detail
comparison shared by all 68 generated provider pages. The pinned Transformers
commit remains `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the current 87-line
SpeechT5 source has SHA-256
`71bba8a2921cf637383fb8d6f2fd66df9cd95deb59118b9f49e1362485c27eb5`,
and the toctree remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.

The local and official pages rendered at 1,440 x 900, 1,024 x 768, and
390 x 844 with matching 804-, 658-, and 342-pixel article widths. Both palettes
were verified on both sites at all three sizes. VoiceHub retained its active
SpeechT5 navigation, thirteen-link desktop TOC, collapsed responsive right
rail, source and edit targets, eight contained tables, and zero document
overflow. Its mobile drawer exposed Base classes, Models, Text to speech, and
SpeechT5 in both palettes. Public API and SpeechT5ForTextToSpeech pointer
transitions kept their requested hashes, landed at y = 64, and left exactly
one matching active TOC link.

The official page retained its active SpeechT5 route, generated API hierarchy,
page-copy action, and twelve collapsed signature controls. Activating the first
60-parameter control exposed its full signature. The previously unverified
reference dark state is now closed for this page pair: the visible theme menu
produced verified dark desktop, tablet, and mobile renders. The official mobile
page itself measured a 594-pixel document scroll width and a 570-pixel article
scroll width because long generated code and API spans are not contained.
VoiceHub intentionally retains internal overflow at 390 pixels instead of
copying that upstream defect.

The first focused pytest invocation used the wrong test-class name and
collected no tests; the corrected SpeechT5 contract passed. The first inventory
invocation used an unavailable `python` shim; a later exploratory query assumed
a nonexistent `ModelSpec.model_id` attribute and a nonexistent notebook-script
name. The corrected virtual-environment inventory reports 68 models (34 TTS,
23 ASR, and 11 VAD), 102 aliases, no invalid display names, 68 provider pages,
no missing or orphaned navigation entries, six public optimization passes, and
408 model/pass pairs. The page and notebook generators report 68 and 59 current
artifacts, and release alignment finds five benchmark files and 68 documented
providers. None of the failed exploratory commands is counted as evidence.

A direct mobile theme-label click timed out because that label is intentionally
hidden at the mobile breakpoint. The verified path used the visible tablet
theme control and then returned to the exact mobile viewport; this does not
establish native mobile keyboard access. After the evidence update, the focused
SpeechT5 contract reported one pass; the complete documentation contract
reported 45 passes and 1,379 subtests; and the registry contract reported 17
passes and 193 subtests. The 68 model pages and 59 model notebooks remained
generator-current, release alignment found five benchmark files and 68
documented providers, and the strict eleven-language documentation build
passed.

No source, stylesheet, registry, runtime, or package change was needed. The
public optimization suites, distribution probes, and complete Python suite were
therefore not rerun; their immediately preceding evidence remains applicable.
Native keyboard activation, complete sequential focus, exact-commit remote CI,
protected publisher configuration, and publication approval remain open. Other
representative pages still retain their own unverified reference-theme states.
The untracked `uv.lock` stayed unchanged.

The next bounded evidence slice closes the exact responsive Auto Classes model
index comparison that fronts all 68 registered integrations. The pinned
Transformers commit remains `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`;
the current 301-line Auto Classes source has SHA-256
`557f5836c0722fef6a484c46805dfab0eb69a387b028a914b132350edf09f167`,
and the toctree remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.

The local and official pages rendered at 1,440 x 900, 1,024 x 768, and
390 x 844 with matching 804-, 658-, and 342-pixel articles. Light and dark were
verified on both sites at all three sizes. VoiceHub retained the active Auto
Classes route, eight-link desktop TOC, collapsed responsive right rail, four
contained tables, source/edit controls, Quickstart/Bark adjacent navigation,
and zero document overflow. Its mobile drawer exposed Base classes, Models,
and Auto Classes in both palettes. Pointer page copy replaced a sentinel with
the complete 6,200-character article. Registered models and Voice activity
detection pointer transitions kept their requested hashes, landed at y = 64,
and left exactly one matching active TOC link.

The official page retained its active Auto Classes route, generated
configuration/processor/model hierarchy, desktop TOC, page-copy control,
responsive navigation, and zero document overflow. Its
`#transformers.AutoConfig` pointer transition retained the hash and placed the
target at y = 36. The community panel, product banner, non-speech class
families, and 324,041-pixel generated API article are upstream content
differences that VoiceHub does not fabricate. The official copy button did not
replace a clipboard sentinel in this harness; that failed interaction is not a
pass.

The focused model-index contract reported one pass and 68 subtests before and
after the evidence edit. The complete documentation contract reported 45
passes and 1,379 subtests, while the registry contract reported 17 passes and
193 subtests. All 68 model pages and 59 model notebooks remained
generator-current, release alignment found five benchmark files and 68
documented providers, and the strict eleven-language documentation build
passed.

No source, stylesheet, generator, registry, runtime, or package change was
needed. The public optimization suites, distribution probes, and complete
Python suite were therefore not rerun; their immediately preceding evidence
remains applicable. Native keyboard activation, complete sequential focus,
exact-commit remote CI, protected publisher configuration, and publication
approval remain open. The untracked `uv.lock` stayed unchanged.

The next bounded release slice closes the exact responsive and reference-theme
Trainer comparison. The pinned Transformers commit remains
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its 30-line Trainer source has
SHA-256
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and the toctree remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.

The local and official Trainer pages rendered at 1,440 x 900, 1,024 x 768,
and 390 x 844 with matching 804-, 658-, and 342-pixel articles. Light and dark
were verified on both sites at all three sizes. Both retained zero document
overflow, the concise Trainer and Next steps hierarchy, one active Trainer
overview route when navigation was visible, responsive navigation, page
actions, and a Fine-tuning destination. The local mobile drawer and official
mobile documentation menu retained their active Trainer overview entries in
both palettes.

Local page copy replaced a sentinel with the complete 1,257-character article.
The official copy action left its 25-character sentinel unchanged, so it is
recorded as failed rather than passed. The official `#next-steps` pointer
transition retained its hash in both palettes but bottom-clamped its short-page
target at y = 417 without an active TOC item. The first VoiceHub pointer probe
also scrolled but then lost its hash and active item when Material's observer
reclassified the short page. A bounded header-control repair now waits for the
pointer scroll to settle and preserves the requested hash and sole active TOC
link without intercepting navigation. The corrected local transition retains
`#next-steps` in both palettes, marks exactly one matching active link, and
bottom-clamps at y = 329. Direct anchored loading remains valid.

The first focused pytest invocation used the wrong test-class name and
collected no tests. The corrected pre-change Trainer contract passed, and the
post-change Trainer plus TOC contracts reported two passes. Exact responsive
and dark evidence did not reveal any local geometry or overflow change. Native
keyboard activation and complete sequential focus remain inaccessible in the
current driver and are not counted as passed. The official copy failure,
exact-commit remote CI, protected publisher configuration, and publication
approval remain open.

The complete documentation contract then reported 45 passes and 1,379
subtests; the registry contract reported 17 passes and 193 subtests. All 68
model pages and 59 model notebooks remained generator-current, release
alignment found five benchmark files and 68 documented providers, and the
strict eleven-language documentation build passed. The selected code,
credential, whitespace, and file-integrity pre-commit hooks passed; the
Markdown hook had no matching files and is not counted as executed evidence.
Public optimization suites, distribution probes, packaging checks, and the
complete Python suite were not rerun because the change is confined to the
documentation shell and its contract test. Their immediately preceding
evidence remains applicable. The untracked `uv.lock` stayed unchanged.

The next bounded evidence slice closes the exact responsive and reference-dark
Optimization comparison. The pinned Transformers commit remains
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its 178-line Optimization source
has SHA-256
`19622667a7299f258f5c9a72940c9f26492619636f35d9bd592701c02745b620`,
and the toctree remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.

The local and official pages rendered at 1,440 x 900, 1,024 x 768, and
390 x 844 with matching 804-, 658-, and 342-pixel articles. Light and dark were
verified on both sites at all three sizes. Both retained zero document
overflow, an active Overview route when navigation was visible, desktop right
TOCs, collapsed responsive rails, page actions, and mobile documentation
navigation. The local mobile technique table remained contained by a
374-pixel wrapper with a 752-pixel internal scroll width.

Local page copy replaced a sentinel with the complete 3,730-character article.
Its Compilation and Diffusion sampling pointer transitions retained their
hashes, landed between y = 64.1 and y = 64.2, and left exactly one matching TOC
link active in both palettes. The official Compilation and Caching transitions
retained their hashes at y = 35.9 and y = 36.2. The official page-copy control
left its 21-character sentinel unchanged, so it remains failed rather than
passed. Native keyboard activation and complete sequential focus remain
unverified and are not counted as passed. No source, stylesheet, generator,
registry, optimization runtime, or package change was needed.

The focused Optimization contract reported one pass and six subtests. The
complete documentation contract reported 45 passes and 1,379 subtests, while
the registry and all documented public optimization suites reported 178 passes
and 944 subtests. One platform-specific PyTorch decomposition warning and three
weight-normalization deprecation warnings remain warnings rather than passes or
failures. All 68 model pages and 59 model notebooks remained
generator-current, release alignment found five benchmark files and 68
documented providers, and the strict eleven-language documentation build
passed. The selected file-integrity, whitespace, credential, and case-conflict
pre-commit hooks passed; code and Markdown hooks had no matching files and are
not counted as executed evidence. Distribution probes and the complete Python
suite were not rerun for this evidence-only change. Exact-commit remote CI,
protected publisher configuration, publication approval, native keyboard
activation, and complete sequential focus remain open. The untracked `uv.lock`
stayed unchanged.

The next bounded evidence slice closes the exact responsive and reference-dark
Contribution comparison. The pinned Transformers commit remains
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its 500-line modular guide has
SHA-256
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The current registry-derived contribution inventory remains eight ordered
steps: Create, Audit, Configure, Wrap, Register, Support, Test, and Document.

The local and official contribution pages rendered at 1,440 x 900, 1,024 x
768, and 390 x 844 with matching 804-, 658-, and 342-pixel articles. Light and
dark were verified on both sites at all three sizes, and every render retained
zero document overflow. The local process retained two 392-pixel desktop
columns, two 319-pixel tablet columns, and one 342-pixel mobile column. The
mobile drawer preserved the active Contribute and Add a model path, and all
three local tables remained contained by their 374-pixel scrolling wrappers.
The official mobile documentation menu preserved its active modular guide.

Local pointer copy replaced a sentinel with the complete 18,197-character
article. Its Register once transition retained `#5-register-once`, placed the
heading at y = 64, and left exactly one matching right-TOC link active. The
official copy action replaced its sentinel with the complete 24,443-character
Markdown source. Its Generate the modeling files transition retained
`#generate-the-modeling-files`, focused the requested link, and placed the
target at y = 36; the official TOC did not expose a separate active-link state.
Native keyboard activation and complete sequential focus remain unverified and
are not counted as passed. No source, stylesheet, generator, registry,
optimization runtime, or package change was needed.

The focused contribution contract passed before the evidence edit with 18
subtests. The post-edit contribution and model-scaffold slice reported 21
passes and 53 subtests. The complete documentation contract reported 45 passes
and 1,379 subtests. Registry and all documented public optimization suites
reported 305 passes and 1,049 subtests. Five native-kernel checks were skipped:
three require `VOICEHUB_TEST_TRITON_KERNELS=1` on a Triton CUDA host, and two
require `VOICEHUB_TEST_CUDA_EXTENSIONS=1` on a CUDA-toolkit host. They remain
hardware-limited gates and are not counted as passes. One platform-specific
PyTorch decomposition warning and seven weight-normalization deprecation
warnings remain warnings rather than passes or failures. All 68 model pages and
59 model notebooks remained generator-current, release alignment found five
benchmark files and 68 documented providers, and the strict eleven-language
documentation build passed. The selected file-integrity, whitespace,
credential, and case-conflict pre-commit hooks passed; code and Markdown hooks
had no matching files and are not counted as executed evidence. Distribution
probes and the complete Python suite were not rerun for this evidence-only
change. Exact-commit remote CI, protected publisher configuration, publication
approval, native keyboard activation, complete sequential focus, and the five
native-kernel hardware gates remain open. The untracked `uv.lock` stayed
unchanged.

The next bounded evidence slice closes the exact responsive and reference-dark
Models API comparison. The pinned Transformers commit remains
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its 44-line Models source retains
SHA-256
`a4899c758b5d621075b2e2f39f0aa79671010c88b08d1508f7af5731375c2871`,
and the current Main Classes inventory remains 24 entries with mapped or
explicitly non-applicable VoiceHub dispositions.

The local and official Models pages rendered at 1,440 x 900, 1,024 x 768, and
390 x 844 with matching 804-, 658-, and 342-pixel articles. Light and dark were
verified on both sites at all three sizes, and every render retained zero
document overflow. The local page preserved its active Models path, four-link
desktop TOC, edit/copy/adjacent navigation, three distinct public source
targets, and four mobile tables inside 374-pixel scrolling wrappers. The mobile
drawer retained API, Main Classes, Models, and Full API reference in both
palettes.

Local pointer copy replaced a sentinel with the complete 5,106-character
article. Its Model outputs transition retained `#model-outputs`, placed the
heading at y = 64, and left exactly one matching TOC link active. Official
pointer copy replaced its sentinel with the complete 49,539-character Markdown
source. Its PreTrainedModel transition retained
`#transformers.PreTrainedModel`, focused the requested link, and placed the
target at y = 36. The first collapsed generated signature exposed all 14 hidden
parameters, and the light and dark official mobile menus preserved their
active Models item. Native keyboard copy activation and complete sequential
focus remain unverified and are not counted as passed. No source, stylesheet,
generator, registry, runtime, optimization, or package change was needed.

The focused Models API contract passed before and after the evidence edit with
50 subtests. The complete documentation contract reported 45 passes and 1,379
subtests, while the base API, speech-core, and registry slice reported 49
passes and 216 subtests. All 68 model pages and 59 model notebooks remained
generator-current, release alignment found five benchmark files and 68
documented providers, and the strict eleven-language documentation build
passed. The selected file-integrity, whitespace, credential, and case-conflict
pre-commit hooks passed; code and Markdown hooks had no matching files and are
not counted as executed evidence. The public optimization suites, distribution
probes, packaging checks, and complete Python suite were not rerun for this
evidence-only change; their immediately preceding evidence remains applicable.
Exact-commit remote CI, protected publisher configuration, publication
approval, native keyboard activation, complete sequential focus, and the five
native-kernel hardware gates remain open. The untracked `uv.lock` stayed
unchanged.

The next bounded evidence slice closes the stale Goal-level universal
optimization lifecycle audit. The live public registry remains six passes—
`codec-kernels`, `compile`, `custom-kernels`, `diffusion-cache`,
`diffusion-sampling`, and `flash-attention-4`—across 68 registered models, for
408 model/pass pairs. The audit derives both dimensions from the public
registries; it contains no provider or pass allowlist in runtime code.

Every pair executed through `apply_optimization_plan()` on a CPU-safe normalized
TTS, ASR, or VAD semantic runtime. The contract checks validation before
mutation, the resolved architecture context, an explicit non-skipped outcome,
reason text for universal fallbacks, manifest reporting, strict JSON
serialization, output type and value preservation, checkpoint-state
preservation, deterministic restoration, result cleanup, and post-restore
semantic/state equality. All 68 registered classes separately retain the five
public lifecycle methods from `BaseSpeechModel` rather than overriding them
with provider-specific behavior.

The executed inventory reported 68 reasoned `eager-fallback` outcomes for the
CPU `compile` path and 340 reasoned `not-applicable` outcomes for the five
architecture-specific passes. No pair reported `skipped`, and no fallback
omitted its reason. These universal fallbacks are lifecycle support rather than
acceleration claims. The separate pass-specific suites exercise real
configured or compiled paths, state-key safety, semantic behavior, failure
rollback, manifest data, and restoration.

The narrow 408-pair contract reported one pass and 408 subtests; the shared
68-model lifecycle-owner contract reported one pass and 68 subtests. The full
native optimization lifecycle file reported 29 passes and 482 subtests with no
skips. The registry and all public pass-specific CPU suites reported 305
passes and 1,049 subtests. Five opt-in checks were skipped and remain unpassed:
three require `VOICEHUB_TEST_TRITON_KERNELS=1` on a Triton CUDA host, and two
require `VOICEHUB_TEST_CUDA_EXTENSIONS=1` on a CUDA-toolkit host. One
platform-specific PyTorch decomposition warning and seven weight-normalization
deprecation warnings remain warnings rather than passes or failures.

The complete documentation contract reported 45 passes and 1,379 subtests.
All 68 model pages and 59 model notebooks remained generator-current, release
alignment found five benchmark files and 68 documented providers, and the
strict eleven-language documentation build passed. The selected
file-integrity, whitespace, credential, and case-conflict pre-commit hooks
passed; code hooks had no matching files and are not counted as executed
evidence. No model, runtime, optimization, registry, generator, package, or test
code changed, so distribution probes, packaging checks, and the complete Python
suite were not rerun; their immediately preceding evidence remains applicable.
Exact-commit remote CI, protected publisher configuration, publication
approval, native keyboard activation, complete sequential focus, and the five
native-kernel hardware gates remain open. The untracked `uv.lock` stayed
unchanged.

The next bounded packaging-evidence slice replaces a stale single source-
distribution size with two fresh current-worktree builds. The focused
distribution and release contracts reported 13 passes. The complete
`scripts/check_distribution.py` probe built an isolated wheel and source
distribution, installed the wheel, source distribution, and editable checkout
into separate dependency-free virtual environments, and reported version
0.3.0, 68 models, 81 provenance manifests, 193 compliance files, all required
representative package data, zero runtime dependency violations, and no eager
PyTorch import in every installation mode. That build produced a 57,189,200-
byte wheel and a 55,439,867-byte source distribution.

A second isolated build passed `scripts/check_release.py --dist-dir` with the
same source, documentation, five-file benchmark, 34 TTS, 23 ASR, 11 VAD, and
68-provider alignment. It produced the same 57,189,200-byte wheel and a
55,438,790-byte source distribution. The source-distribution byte difference
is expected from gzip build timestamps and is not presented as reproducible
artifact identity; the tagged workflow must retain the exact published pair
and record its hashes. Both fresh artifact pairs remained below PyPI's 100 MB
per-file limit. Neither build used `--with-dependencies`, so dependency
resolution and runtime execution are not claimed by this slice; the supported-
platform CI and complete CPU-safe suite remain the applicable runtime evidence.

No source, packaging configuration, runtime, model, test, or generated file
changed in this slice. Existing repository-local `dist/` artifacts were not
overwritten. Exact-commit CI for the uncommitted worktree, the protected PyPI
environment and trusted publisher, tag and publication approval, five native-
kernel hardware gates, seven inaccessible asset/checkpoint/oracle paths,
native documentation keyboard activation, and complete sequential focus remain
open. The untracked `uv.lock` stayed unchanged.

The next bounded public-lifecycle slice corrects
`AutoProcessor.from_pretrained()` option routing. Previously, configuration
loader values such as `revision` and `local_files_only` were omitted from
`AutoConfig.from_pretrained()` and then leaked into the processor constructor
or restored `processor_config.json`. Explicit `model_type` construction had the
same leak and ignored configuration overrides. The factory now accepts the
same explicit `config_kwargs` separation as VoiceHub's auto-model factories,
validates the mapping before loading, rejects ambiguous `config` plus
`config_kwargs`, and reserves ordinary keyword arguments for the processor.

The first focused regression executed before the implementation and reported
three failures: remote configuration options were omitted, explicit-model-type
configuration overrides were ignored, and local artifact restoration retained
the leaked `config_kwargs`. After correction, all three regressions passed; the
complete focused file reported 10 passes. The broader auto, base API,
speech-core, task-registry, pipeline, and registry slice reported 90 passes and
298 subtests. The two registry-derived optimization lifecycle contracts
reported two passes and 476 subtests. A fresh subprocess constructed the
registered Kokoro processor without importing PyTorch, and the public root
inventory remained 261 exports with no eager PyTorch import.

The complete Python 3.12.12 suite reported 2,471 passes, 15 explicit skips,
3,729 subtests, and 35 warnings in 112.60 seconds. Those skips remain unpassed.
The documentation contract reported 45 passes and 1,379 subtests; the combined
documentation, release, and canonical-guidance slice reported 58 passes and
1,390 subtests; all 68 model pages remained generator-current; and the strict
eleven-language build passed. The distribution and release source contracts
reported 23 passes and 76 subtests, and release alignment found five benchmark
records plus all 68 documented providers. The fresh wheel, source-distribution,
and editable probe reported version 0.3.0, 68 models, 81 provenance manifests,
193 compliance files, all required representative package data, zero runtime
dependency violations, and no eager PyTorch import. The wheel measured
57,189,217 bytes and the source distribution 55,440,131 bytes.

Every applicable selected pre-commit hook passed. Markdown formatting found no
matching files and is not counted. The official Transformers `main` reference
remained `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, its navigation SHA-256
remained `f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the documentation endpoint returned HTTP 200. The refreshed inventories
remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, 68 provider pages,
six public optimization passes, 408 model/pass pairs, five benchmark records,
and no missing, orphaned, or lowercase-first model pages. Exact-commit remote
CI for the uncommitted worktree, the protected PyPI environment and publisher,
tag and publication approval, native documentation keyboard activation,
complete sequential focus, five native-kernel hardware gates, and seven
inaccessible asset/checkpoint/oracle paths remain open. The untracked `uv.lock`
stayed unchanged.

The following bounded public-lifecycle slice completes the processor artifact
half of `AutoProcessor.from_pretrained()`. The pinned Transformers
`AutoProcessor` source delegates the original local directory or Hub identifier
to the selected processor class after resolving its configuration. VoiceHub
previously delegated only an existing local directory containing
`processor_config.json`; Hub identifiers, direct processor configuration files,
and missing-optional-artifact fallback never reached the processor loader.

VoiceHub now resolves the registered processor class without constructing a
temporary instance, delegates every source form to that class's
`from_pretrained()` method, and reuses `subfolder`, `cache_dir`, `revision`,
`token`, and `local_files_only` from configuration loading when the processor
call does not override them. The base processor consumes those loader-only
values before either its direct-file or resolved-file branch, so credentials
and cache settings cannot become serialized processor state. A missing optional
base `processor_config.json` remains an explicit constructor fallback.

The first focused regression reported three failures before implementation:
the remote processor loader was never called, a missing remote processor
artifact never exercised fallback, and a direct processor configuration file
was ignored. An expanded direct-file regression then failed because its token
reached a temporary processor constructor. Neither failed run is counted as a
pass. The corrected focused file reported 13 passes. The broader auto, base
API, speech-core, task-registry, registry, and Hub-transport slice reported 105
passes and 295 subtests. The two registry-derived optimization lifecycle
contracts reported two passes and 476 subtests.

The complete Python 3.12.12 suite reported 2,474 passes, 15 explicit skips,
3,729 subtests, and 35 warnings in 112.17 seconds; the skips remain unpassed.
The combined documentation, release, canonical-guidance, distribution-source,
and packaging-metadata slice reported 72 passes and 1,466 subtests. All 68 model
pages remained generator-current, release alignment found five benchmark
records and all 68 documented providers, and the strict eleven-language build
passed. A fresh wheel, source-distribution, and editable probe reported version
0.3.0, 68 models, 81 provenance manifests, 193 compliance files, all required
representative package data, zero runtime dependency violations, and no eager
PyTorch import. The wheel measured 57,189,342 bytes and the source distribution
55,440,749 bytes.

Every applicable selected pre-commit hook passed; Markdown formatting found no
matching files and is not counted. Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. The pinned `processing_auto.py`
SHA-256 is `c1e82fbd511b4fb3838cae570a29716d54903c0658c4e37de10e17fba8583cad`,
the pinned `processing_utils.py` SHA-256 is
`77386cb128481c3bc5bd330f6a6687d69822e9eacb8072d2f3c3fc930f426483`, and
the navigation SHA-256 remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, 68 provider pages, six public optimization passes, 408 model/pass
pairs, five benchmark records, and no missing, orphaned, or lowercase-first
model pages. Exact-commit remote CI for the uncommitted worktree, the protected
PyPI environment and publisher, tag and publication approval, native
documentation keyboard activation, complete sequential focus, five
native-kernel hardware gates, and seven inaccessible asset/checkpoint/oracle
paths remain open. The untracked `uv.lock` stayed unchanged.

The next bounded lazy-discovery slice removes model-wrapper imports from
registry-wide processor selection. The first fresh-process inventory
constructed and restored all 68 processors successfully, but 13 registrations
imported PyTorch merely to discover their processor class. That inventory
exited nonzero and is not counted as a pass. `ModelSpec` now records a validated
lazy processor import target, supplies the shared TTS or audio processor by
task when no override is declared, and preserves a custom importable
`processor_class` through the recommended extension registrar.
`AutoProcessor` resolves that metadata directly instead of importing a model
wrapper.

The focused processor-metadata contracts reported three passes and 68
subtests. The broader task-registry, registry, auto-configuration, base API,
and speech-core slice reported 88 passes and 358 subtests. The two
registry-derived optimization lifecycle contracts reported two passes and 476
subtests. A post-change fresh-process inventory constructed and offline-loaded
the task-appropriate processor for all 68 registrations, exercised the text or
audio input envelope, and imported none of PyTorch, Transformers, or the named
optional speech backends. The 68 generated pages and navigation remained
current. Documentation, release, distribution-source, and packaging contracts
reported 68 passes and 1,455 subtests, and the strict eleven-language build
passed. A local 1,280-by-720 browser render showed the new registry metadata on
the light API page and the eight-step contribution guide in dark mode with no
document overflow. The in-app surface offered no viewport-resize control, so
the 1,440-, 1,024-, and 390-pixel comparisons were not rerun and no new
responsive-parity claim is made.

The complete Python 3.12.12 suite reported 2,476 passes, 15 explicit skips,
3,797 subtests, and 35 warnings in 118.53 seconds; the skips remain unpassed.
The fresh wheel, source-distribution, and editable probe reported version
0.3.0, 68 models, 81 provenance manifests, 193 compliance files, all required
representative package data, zero runtime dependency violations, and no eager
PyTorch import. The wheel measured 57,189,672 bytes and the source distribution
55,442,882 bytes. Every applicable selected pre-commit hook passed; Markdown
formatting found no matching files and is not counted.

A staged fresh-process probe intentionally combined `AutoConfig.for_model()`
with `AutoProcessor.from_config()` and still exited nonzero because seven
configuration facades import PyTorch: `llasa`, `f5tts`, `parlertts`, `zonos`,
`zonos2`, `csm`, and `asr_seamless_m4t_v2`. The processor-only phase of that
probe imported no additional heavy backend, so those seven failures are a
separate configuration-facade gap rather than processor discovery evidence.
They remain unpassed. Exact-commit remote CI for the uncommitted worktree, the
protected PyPI environment and publisher, tag and publication approval, native
documentation keyboard activation, complete sequential focus, five
native-kernel hardware gates, and seven inaccessible asset/checkpoint/oracle
paths also remain open. The untracked `uv.lock` stayed unchanged.

The next bounded lazy-configuration slice closes that seven-model gap. The
first registry-wide regression and per-model fresh-process inventory both
exited nonzero because `AutoConfig.for_model()` imported PyTorch for `llasa`,
`f5tts`, `parlertts`, `zonos`, `zonos2`, `csm`, and
`asr_seamless_m4t_v2`; neither failing check is counted as a pass. Four model
packages exported runtime objects eagerly, five public config classes lived in
Torch-backed inference modules, and the SeamlessM4T-v2 config reached its
language table through a Torch-backed tokenizer.

The five config classes now live in their dependency-light configuration
modules, the affected package exports resolve lazily, and the SeamlessM4T-v2
language table lives in a framework-free metadata module shared with its
tokenizer. Runtime wrappers import those same config class objects, so model
construction, serialization, checkpoint loading, and public compatibility
paths retain one class identity. This follows the pinned Transformers lazy
configuration mapping rather than adding a provider branch to `AutoConfig`.

The corrected registry-wide config regression reported one pass after its
initial failure. A separate post-change process for each of all 68 registered
models reported zero PyTorch, Transformers, or named optional speech-backend
imports. The affected LLaSA, F5-TTS, Parler-TTS, Zonos, ZONOS2, CSM, and
SeamlessM4T-v2 suites reported 168 passes, 15 subtests, and eight warnings. The
broader task-registry, registry, auto-configuration, base API, and speech-core
slice reported 89 passes and 358 subtests. Native dependency and shared
provider-independence policies plus the two registry-derived optimization
lifecycle contracts reported 25 passes and 476 subtests.

The complete Python 3.12.12 suite reported 2,477 passes, 15 explicit skips,
3,797 subtests, and 35 warnings in 109.21 seconds; the skips remain unpassed.
All 68 model pages and navigation entries remained generator-current. The
documentation, release, distribution-source, and packaging contracts reported
68 passes and 1,455 subtests, and the strict eleven-language build passed. A
fresh wheel, source-distribution, and editable probe reported version 0.3.0, 68
models, 81 provenance manifests, 193 compliance files, all required
representative package data, zero runtime dependency violations, and no eager
PyTorch import. The wheel measured 57,192,256 bytes and the source distribution
55,444,113 bytes.

Every applicable selected pre-commit hook passed; Markdown formatting found no
matching files and is not counted. The official Transformers `main` revision
remained `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its
`configuration_auto.py` SHA-256 was
`5eaeb741896b05e852fcca8d316b329e42fa3788e2017c560d2e95002b84f0e9`,
and the official Auto Classes endpoint returned HTTP 200. Exact-commit remote
CI for the uncommitted worktree, the protected PyPI environment and publisher,
tag and publication approval, native documentation keyboard activation,
complete sequential focus, five native-kernel hardware gates, and seven
inaccessible asset/checkpoint/oracle paths remain open. The untracked
`uv.lock` stayed unchanged.

The next bounded documentation-shell slice closes the shared page-copy
keyboard gap. Before correction, a rendered Enter press left its clipboard
sentinel unchanged. The focused source contract then failed after being
strengthened to require an explicit, single keyboard copy path. The first
focused command used an unavailable bare `python` executable and did not run;
the corrected virtual-environment invocation produced the expected pre-change
failure. Neither result is counted as a pass.

The shared action now handles Enter and Space directly, prevents the browser's
second native activation, and invokes the existing copy lifecycle exactly
once. It retains the selection fallback, `aria-busy` and live success state,
focus restoration, and pointer path without adding a page- or provider-specific
branch. The focused contract passed after correction.

At the available 1280 x 720 browser viewport, Enter and Space each replaced a
fresh sentinel with the complete 5,252-character Quickstart article. Both
activations returned `aria-busy` to `false`, reported `Copied`, and left the
button focused with its focus-visible class. Enter repeated the same result in
the dark palette with zero horizontal overflow. Pointer activation still
copied all 5,252 characters and removed the keyboard-only focus class. The
official Quicktour at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722` exposed its corresponding action
as a button, but Enter left its 18-character sentinel unchanged, so that probe
is recorded as failed rather than passed. Its pinned source SHA-256 remains
`ecfb99781204bcaea1ca63bcb4ad9ef70c99812e5f965a49b29de23cedd25bd7`.

The in-app browser did not expose viewport resizing. This JavaScript-only
correction changes no markup or stylesheet, so the immediately preceding exact
1440 x 900, 1024 x 768, and 390 x 844 geometry remains applicable; no fresh
exact-viewport keyboard claim is made. The complete documentation contract
reported 45 passes and 1,379 subtests. All 68 model pages and navigation
entries remained generator-current, release alignment found five benchmark
records and all 68 documented providers, and the strict eleven-language build
passed. Every applicable selected pre-commit hook for the changed script and
contract test passed; Markdown formatting had no matching files and is not
counted.

Registry runtime, optimization, package, and complete-suite checks were not
rerun because the bounded change is confined to the shared documentation
action and its source/evidence contracts. Their immediately preceding evidence
remains applicable: 68 models, six public optimization passes, 408 model/pass
pairs, five benchmark records, a 2,477-pass complete suite, and the fresh
57,192,256-byte wheel plus 55,444,113-byte source distribution. Exact-commit
remote CI for the uncommitted worktree, native Enter activation of focused
right-TOC links, complete sequential focus, the protected PyPI environment and
publisher, tag and publication approval, five native-kernel hardware gates,
and seven inaccessible asset/checkpoint/oracle paths remain open. The untracked
`uv.lock` stayed unchanged.

The next bounded documentation-shell slice closes the shared right-table-of-
contents Enter gap. Before correction, Enter left the local Parameters link
focused but retained an empty hash, no active TOC link, and a Parameters
heading more than 2,700 pixels below the viewport. Pointer activation already
preserved `#parameters`, one matching active link, and the y = 63.8 heading
offset. The official Pipeline link likewise stayed focused without navigating
under the same Enter probe, so neither pre-change keyboard result is a pass.

The strengthened source regression failed before the handler existed and
passed after correction. The shared secondary rail now handles only
unmodified Enter, prevents a second native activation, delegates to its
existing click path, and restores focus without scrolling. Modified Ctrl+Enter
remains unintercepted. No page, provider, heading, or route allowlist was
added.

At the available 1280 x 720 desktop viewport, Enter on Parameters in the dark
palette retained `#parameters`, settled the heading at y = 63.8, left exactly
that link active and focused, and rendered its solid two-pixel outline with a
two-pixel offset. Enter on Large models repeated the result in the light
palette at y = 63.9. Pointer activation of Tasks still produced `#tasks`, one
matching active link, and y = 64.0. All checked states retained zero horizontal
overflow. The first 900-millisecond post-change probe captured an intermediate
y = 128.8 state and the preceding active link before the existing settling
timers completed; it is not counted as final evidence. The later settled probes
above are the passing state.

The in-app browser again exposed no viewport resizing. This shared JavaScript
change alters no markup or CSS, and the right rail is already verified as
hidden at 1024 and 390 pixels, so the preceding exact responsive geometry
remains applicable; no fresh 1440-pixel Enter claim is made. The official
Transformers `main` revision remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, its navigation SHA-256 remained
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
the Pipeline source SHA-256 was
`80a95da29bb1bc960d7570065322a267c47052b132b7a5f8130b77e809d3914f`,
and the rendered documentation endpoint returned HTTP 200.

The focused contract passed; the complete documentation contract reported 45
passes and 1,379 subtests. All 68 model pages and navigation entries remained
generator-current, release alignment found five benchmark records and all 68
documented providers, and the strict eleven-language build passed. Every
applicable selected pre-commit hook for the changed script and contract test
passed; Markdown formatting had no matching files and is not counted.

Registry runtime, optimization, package, and complete-suite checks were not
rerun because this bounded change is confined to one shared documentation
interaction and its source/evidence contracts. Their immediately preceding
evidence remains applicable: 68 models, 102 aliases, six public optimization
passes, 408 model/pass pairs, five benchmark records, a 2,477-pass complete
suite, and the fresh 57,192,256-byte wheel plus 55,444,113-byte source
distribution. Complete sequential focus is now the remaining shared
documentation keyboard gate. Exact-commit remote CI for the uncommitted
worktree, the protected PyPI environment and publisher, tag and publication
approval, five native-kernel hardware gates, and seven inaccessible
asset/checkpoint/oracle paths remain open. The untracked `uv.lock` stayed
unchanged.

The next bounded documentation-shell slice removes inactive left-navigation
descendants from the desktop focus inventory. Before correction, the unchecked
Base classes branch alone exposed 88 focusable descendants, including all 68
model pages, before the current Pipeline article. The strengthened source
contract first failed because the desktop shell had no collapse rule. Moving
the existing grid, opacity, and visibility treatment to desktop made that
source contract pass, but the rendered inventory still exposed every inactive
descendant because descendant visibility rules overrode the parent. The
contract was strengthened again to require actual display removal and failed
against that intermediate implementation. A later accessibility strengthening
also failed before section labels became operable controls. None of those
failed or intermediate results is counted as a pass.

Unchecked desktop navigation branches now use `display: none`; checked and
indeterminate branches use `display: block`. Section labels are pointer-
operable controls, expose `aria-controls`, synchronize `aria-expanded`, and
handle unmodified Enter and Space through one shared initializer. The focused
regression passed after correction. At the available 1280 x 720 desktop
viewport, both VoiceHub palettes exposed 63 visible focusable elements,
including the version summary, and zero focusable descendants inside inactive
branches. All eight root controls remained visible; the checked Inference and
Pipeline API branches retained Pipeline, Speech recognition, and VAD, while
unchecked Serving descendants remained hidden. Pointer activation expanded
Base classes, synchronized `aria-expanded`, revealed exactly Models,
Preprocessors, and Architecture, and left nested Models collapsed. The local
article began at x = 318 and measured 644 pixels, matching the official
Pipeline page at this viewport, and every checked state retained zero
horizontal overflow.

Native Tab could not be replayed: the in-app browser, DOM keyboard surface, and
locator surface did not advance focus. Locator-driven Enter and Space on a
branch label also failed because the browser reported that its focused input
target no longer matched the resolved locator. The source-tested handler is
therefore not rendered keyboard evidence; complete sequential focus and
rendered branch-key activation remain pending. The in-app surface exposed no
viewport resizing, so no fresh exact 1440 x 900, 1024 x 768, or 390 x 844
claim is made. The complete documentation contract reported 45 passes and
1,379 subtests. Every applicable selected pre-commit hook for the stylesheet,
shared script, contract test, and evidence passed. All 68 model pages remained
generator-current, release alignment found five benchmark records and all 68
documented providers, and the strict eleven-language build passed.

Registry runtime, optimization, package, and complete-suite checks were not
rerun because this bounded change is confined to the shared documentation
shell and its source/evidence contracts. Their immediately preceding evidence
remains applicable: 68 models, 102 aliases, six public optimization passes,
408 model/pass pairs, five benchmark records, a 2,477-pass complete suite, and
the fresh 57,192,256-byte wheel plus 55,444,113-byte source distribution.
Exact-commit remote CI for the uncommitted worktree, native sequential focus,
rendered branch-key activation, the protected PyPI environment and publisher,
tag and publication approval, five native-kernel hardware gates, and seven
inaccessible asset/checkpoint/oracle paths remain open. The untracked
`uv.lock` stayed unchanged.

The next bounded documentation-validation slice turns the representative-route
navigation matrix into a tagged-candidate gate. Native Tab replay was attempted
first, but the browser remained focused on `BODY`; complete sequential focus
therefore remains unavailable and unpassed. A rendered audit then checked Home,
Installation, Quickstart, Pipeline, Auto Classes, SpeechT5, Trainer,
Optimization overview, Add a model, and Models API at the available 1280 x 720
viewport. In both VoiceHub palettes, every route exposed exactly one expected
active primary-navigation link, its exact checked ancestors, all eight root
controls, zero focusable descendants inside inactive branches, and zero
horizontal overflow.

The initial browser audit failed because its evaluation used an unavailable
`HTMLElement` constructor; the corrected node-type check produced the route
matrix. The focused source contract then failed before the post-build checker
and workflow steps existed. The checker's first site execution failed because
it treated Material's active-page `__toc` control as a product branch and
expected a branch tabindex. The corrected implementation scopes branch checks
to `__nav_`, retaining every product-navigation toggle while excluding the
separate page-TOC control. The first selected pre-commit run also failed after
YAPF modified the new script and docformatter exited nonzero. None of these
failed or intermediate runs is counted as passing evidence.

The dependency-free checker parses the ten generated English pages and requires
their expected H1, exact eight-root navigation order, sole active anchor, exact
checked ancestor sequence, focusable branch labels, and panel
`aria-expanded` state synchronized with each toggle. Documentation CI and the
tagged release workflow now run it immediately after the strict build. The
corrected focused contract passed, and the post-build validator reported ten
representative routes and all eight roots. The complete documentation contract
reported 46 passes and 1,379 subtests, the strict eleven-language build plus
the post-build validator passed, the release-workflow contract reported nine
passes, and every applicable selected hook passed after formatting. Markdown
formatting had no matching files and is not counted.

All 68 model pages remained generator-current, release alignment found five
benchmark records and all 68 documented providers, and the refreshed
inventories remain 34 TTS, 23 ASR, 11 VAD, 102 aliases, six public optimization
passes, 408 model/pass pairs, and eight contribution steps. The official
Transformers `main` revision remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, its navigation SHA-256 remained
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
the current Installation source SHA-256 was
`d050e3e0e1c89d543c71c25367a455a9b49ea89c92f8c2376bb58294a4a4cf3b`,
and the rendered endpoint returned HTTP 200.

Runtime, optimization-lifecycle, distribution, and complete-suite checks were
not rerun because this slice changes only documentation validation, its source
contract, and the two workflows that consume the already-built site. Their
immediately preceding evidence remains applicable: a 2,477-pass complete suite
and fresh 57,192,256-byte wheel plus 55,444,113-byte source distribution. The
validator proves rendered static structure and ARIA state; it does not prove
pixel geometry or native keyboard order. Exact-commit remote CI for the current
uncommitted worktree, native sequential focus, rendered branch-key activation,
fresh exact 1440/1024/390 checks, protected publisher configuration, tag and
publication approval, five native-kernel hardware gates, and seven inaccessible
asset/checkpoint/oracle paths remain open. The untracked `uv.lock` stayed
unchanged.

The next bounded documentation-validation slice adds executable responsive
geometry and palette evidence for the same ten representative routes. The
checker pins Playwright 1.62.0 and Chromium 151.0.7922.34 revision 1234, then
renders each route in both VoiceHub palettes at 1,440 x 900, 1,024 x 768, and
390 x 844. The resulting 60 cases require exact article, header, primary-rail,
secondary-rail, and closed mobile-drawer geometry; the expected colors, H1,
active link, checked ancestors, and eight navigation roots; zero horizontal
overflow; and zero inactive desktop/tablet focusables. Documentation CI and
the tagged release workflow install the pinned Chromium runtime and execute
the checker after the strict build and static DOM validator.

The pre-implementation focused contract failed because the visual checker did
not exist. Its first post-implementation run failed on an over-specific test
fragment, the first selected hook run exited nonzero after YAPF formatted the
new script, and a scratch forced palette-checkbox action failed because the
hidden input was outside the viewport. The corrected checker uses the theme
control's change-event path. None of these failed or intermediate results is
counted as passing evidence.

The final focused contract and every applicable selected hook passed. The
strict eleven-language build completed; the structural validator reported ten
routes and eight roots; and the responsive validator passed all 60 cases. The
complete documentation contract reported 47 passes and 1,379 subtests. The
related documentation, release, distribution, and packaging slice reported 70
passes and 1,455 subtests. The complete Python 3.12.12 suite reported 2,479
passes, 15 skipped, 3,797 subtests, and 35 warnings in 112.83 seconds. The
fresh wheel, source-distribution, and editable probe reported 68 models, 81
provenance manifests, 193 compliance files, required package data, zero runtime
dependency violations, and no eager PyTorch import; the wheel measured
57,192,274 bytes and the source distribution 55,444,958 bytes.

All 68 model pages and 59 notebooks remained generator-current, release
alignment found five benchmark records and all 68 documented providers, and
the refreshed inventories remain 34 TTS, 23 ASR, 11 VAD, 102 aliases, six
public optimization passes, 408 model/pass pairs, five benchmark records,
eight contribution steps, and ten representative routes. The official
Transformers revision remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, its navigation SHA-256 remained
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official Pipeline endpoint returned HTTP 200.

Computed geometry and palette validation is not screenshot pixel comparison or
native keyboard evidence. At that point native sequential Tab and rendered
Enter/Space branch activation remained open; the following slice closes those
two executable gates. Exact-commit remote CI, protected publisher
configuration, tag and publication approval, five native-kernel hardware
gates, and seven inaccessible asset/checkpoint/oracle paths remain open. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

The next bounded documentation-accessibility slice converts the two remaining
shared keyboard gaps into a tagged-candidate gate. The first native Chromium
probe reached a zero-height search result region and invisible palette radio,
skipped all eight root branch labels because their generated `tabindex` was
empty, and left Base classes collapsed after Enter. Those product failures are
not counted as passing evidence.

The shared correction removes the search result viewport and palette radios
from sequential focus, normalizes branch labels to `tabindex="0"`, adds their
two-pixel visible focus outline, and replaces click delegation with one capture-
phase Enter/Space toggle plus the existing bubbling change lifecycle. The
Playwright checker now traverses the full Pipeline page from document start in
both palettes at desktop and mobile widths. Four cycles contain 228 stable
native Tab stops, enter no invisible element or inactive branch, visit the exact header,
root navigation, expanded Inference descendants, right table of contents,
article, previous and next, and footer order, reach `BODY`, and return to the
skip link. A second complete replay contained 230 stops because Material's
scroll-dependent Back to top control became focusable in two cycles; both
conditional stops were visible and passed the same checks.

Separate default-palette Enter and slate-palette Space cases reach Base classes
through nine native Tabs. Each key opens and closes the branch while retaining
focus and the visible outline, synchronizing checked state, `aria-expanded`,
and `block`/`none` panel display, preserving the sole active Pipeline route,
and retaining zero overflow. An inspected 1,440 x 900 slate screenshot shows
the focused control inside the unchanged three-column shell. The official
Transformers Pipeline page returned HTTP 200, and its first 25 native Tab stops
at the same viewport were all visible.

The responsive continuation makes the mobile menu label keyboard-operable and
synchronizes its accessible name, controls, and expanded state. Closed mobile
navigation is inert and the closed search input leaves sequential focus. Enter
and Space open the drawer in separate palettes, wait until the first navigation
control is fully on canvas, and retain zero overflow; Escape closes it, restores
inert state, and returns focus to the trigger. The opened 390 x 844 screenshot
was inspected with the Pipeline drawer visible.

The pre-implementation focused regression failed against the old markup and
control contract. A first corrected source run then over-counted one Jinja loop
declaration, the first selected hook run exited nonzero after pyupgrade rewrote
the JavaScript evaluation string, and two rendered runs started after the
theme control because the palette setup intentionally restored focus there.
The final checker reloads the persisted palette before beginning from document
start. None of those failed or intermediate runs is counted as a pass.

The mobile audit then failed on the old off-canvas drawer and search sequence,
and its focused source regression failed before the inert and trigger lifecycle
existed. Early rendered checks applied viewport containment to long article
stops that the automation did not scroll and sampled the opening transition
before the drawer reached x = 0. These harness failures are not passes; the
final check scopes viewport containment to fixed shell controls and waits for
the transition to finish.

The two focused source regressions, all applicable selected hooks, strict
eleven-language build, ten-route DOM check, 60 responsive cases, eight keyboard
cases, and 228 stable focus stops passed; a repeat also validated two
conditional Back to top stops. The complete documentation and release-
contract slice reported 57 passes and 1,379 subtests. The complete Python
3.12.12 suite reported 2,480 passes, 15 skipped, 3,797 subtests, and 35 warnings
in 113.79 seconds. All 68 model pages and 59 notebooks remained generator-
current, release alignment found five benchmark records and all 68 documented
providers, and the refreshed inventories remain 34 TTS, 23 ASR, 11 VAD, 102
aliases, six public optimization passes, 408 model/pass pairs, five benchmark
records, eight contribution steps, and ten representative routes.

This slice changes no package runtime or metadata, so distribution probes were
not rerun; the immediately preceding wheel, source-distribution, and editable
evidence remains applicable. At that point screenshot pixel comparison, exact-
commit remote CI, protected publisher configuration, tag and publication approval, five
native-kernel hardware gates, and seven inaccessible asset/checkpoint/oracle
paths remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Screenshot-derived visual regression gate

The next bounded documentation slice replaces the local screenshot-pixel gap
with a reproducible regression gate for all ten representative routes. The
official reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, with navigation SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Pipeline endpoint returned HTTP 200 when the reference was
refreshed.

Each of the existing 60 route, viewport, and palette cases now captures an
in-memory PNG after fonts settle and animations are disabled. Pillow 12.3.0
verifies the exact 1,440 x 900, 1,024 x 768, or 390 x 844 raster dimensions,
applies a two-pixel Gaussian blur, and derives an aspect-preserving 64-pixel-
wide difference hash plus mean RGB values. The checked-in schema-one JSON
manifest contains exactly 60 signatures generated by Playwright 1.62.0 and
Chromium 151.0.7922.34. Default validation rejects missing, orphaned, malformed,
dimension-mismatched, more than 8%-different perceptual hashes, or mean channel
deltas above 6.0. Baseline generation is an explicit CLI mode that prints JSON
for review; neither documentation nor tagged-release CI can update it.

The pre-implementation focused regression failed because the baseline artifact
did not exist. The first selected hook run then let YAPF format the Python
source and exited nonzero; its formatted rerun passed. An initial baseline-
application attempt failed before changing the tree because the generated
patch lacked its final newline. These failed tooling runs are not passing
evidence.

The corrected source contract passed. The rebuilt site then passed all 60 DOM-
geometry cases, all 60 screenshot comparisons, eight keyboard cases, and four
complete focus cycles. Two deliberate negative controls prove the gate is
sensitive: replacing the first expected perceptual hash with zeroes failed at
566 changed bits, or 22.109%, against the 8% limit; replacing its expected mean
RGB with zeroes failed at channel deltas of 229.142, 231.456, and 239.484
against the 6.0 limit. Neither intentional negative control is a product pass.

The complete documentation and release-contract slice reported 58 passes and
1,379 subtests. The complete Python 3.12.12 suite reported 2,481 passes, 15
skipped, 3,797 subtests, and 35 warnings in 110.41 seconds. The fresh
distribution probe passed wheel, source-distribution, and editable installs,
including dependency and lazy-import checks, with a 57,192,282-byte wheel and
55,446,461-byte source distribution. Release alignment still finds five
benchmark records and all 68 documented providers; all 68 model pages and 59
notebooks remain generator-current.

Fresh official Pipeline screenshots returned HTTP 200 for light desktop,
tablet, and mobile plus dark desktop and tablet. The first dark-mobile request
was rate-limited with HTTP 429 and is not a pass; a later isolated retry
returned HTTP 200 and its screenshot was inspected. The local desktop
light/dark and mobile light/dark screenshots were visually inspected.
The new manifest protects VoiceHub against local raster regressions; it does
not claim raw-pixel identity with upstream's intentionally different Hugging
Face branding, community panel, prose, or VoiceHub's approved palette. Exact-
commit remote CI, protected publisher configuration, tag and publication
approval, five native-
kernel hardware gates, and seven inaccessible asset/checkpoint/oracle paths
remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Representative-route native focus gate

The next bounded documentation-accessibility slice expands the native keyboard
contract from the Pipeline sample to every representative page. The official
reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, and the current navigation file
retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Pipeline endpoint returned HTTP 200 during the refresh.

A pre-edit Chromium probe established three real viewport prefixes: desktop
starts with skip, logo, search, version, language, theme, and source; tablet
omits the compacted logo; mobile starts with skip, drawer, search trigger, and
version. The same probe found 239 focus stops on the desktop model index, so
the previous fixed 200-step ceiling could not represent a complete cycle. The
checker now derives a safe bound from each rendered DOM and executes a full
cycle inside all 60 route, viewport, and palette cases. Every stop must be
rendered, every inactive navigation branch must remain unreachable, fixed
prefix controls must stay on canvas, branch, drawer, and content-tab controls
must expose the two-pixel focus indicator, each cycle must reach `BODY`, and a
forward Tab from that boundary must return to Skip to content.

The first full render resumed after the theme control because a plain blur did
not reset Chromium's sequential-navigation cursor. A document-start focus
reset corrected that harness boundary. The next run reached a zero-sized
Installation content-tab radio whose visible label had no focus indicator.
The shared tab initializer now maps focused radios to their rendered labels,
and the checker validates the label as the visible focus proxy. A later Models
API tablet cycle reached `BODY` but Chromium resumed at a nearby scrolled
article link. The shared document boundary now routes an unmodified forward
Tab from body focus to the skip link. These failed runs are regression evidence
and are not counted as passes. An initial focused test invocation also named
the wrong test class and collected no tests; its corrected pre-implementation
run failed as intended.

After correction, the two focused source contracts and every applicable
selected pre-commit hook passed. Markdown formatting had no matching selected
file and is not counted. The strict eleven-language build and ten-route DOM
validator passed. The Playwright 1.62.0 and Chromium 151.0.7922.34 gate passed
all 60 geometry cases, all 60 screenshot comparisons, all 60 complete focus
cycles, and four separate Enter/Space activation cases. The latest run visited
4,075 focus stops; this is an observed count rather than a fixed expectation
because Material may conditionally expose Back to top.

The documentation and release-contract slice reported 59 passes and 1,379
subtests. The registry and public-optimization slice reported 231 passes,
1,203 subtests, and four recorded warnings. The complete Python 3.12.12 suite
reported 2,482 passes, 15 skipped, 3,797 subtests, and 35 warnings in 110.60
seconds. The fresh distribution probe passed wheel, source-distribution, and
editable installs with 68 models, 81 provenance manifests, 193 compliance
files, no eager Torch import, and zero dependency violations. Its wheel was
57,192,282 bytes and its source distribution was 55,446,859 bytes.

Exact-commit remote CI for the uncommitted worktree, protected publisher
configuration, tags, publication approval, five native-kernel hardware gates,
and seven inaccessible asset, checkpoint, or oracle paths remain open. The
untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Rendered semantic-accessibility gate

The next bounded documentation slice closes the missing automated rendered
accessibility contract without claiming a complete WCAG or manual audit. The
official structural reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the navigation fingerprint remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The checker pins `axe-playwright-python` 0.1.8 and reports its bundled Axe Core
4.12.1 engine beside the existing Playwright matrix.

The pre-edit 60-case inventory found five shared failure classes: invalid
button roles on labels, insufficient code-token and footer contrast,
non-unique code-action landmarks, links distinguished only by color, and
closed search results that still exposed an unfocusable scroll region. Drawer,
search, theme, and runtime navigation toggles now use native buttons; closed
search output is both inert and hidden from the accessibility tree; code-action
landmarks receive stable per-page names; paragraph links remain underlined;
and the affected light and dark color tokens exceed the automated contrast
threshold. The first corrected matrix then found a Quickstart table that was
scrollable only by pointer at mobile width. All rendered table wrappers now
have a named keyboard stop and a visible two-pixel focus indicator.

The focused source contract failed before those changes and passed afterward.
Two early full-browser attempts timed out before a product assertion while
waiting for the first local `networkidle` state, and the first completed Axe
run failed on the Quickstart table; none is counted as a pass. A deliberate
unnamed-button injection triggered Axe's `button-name` rule, demonstrating that
the new gate fails observably. The final current-code run passed all 60 Axe
cases, all 60 geometry cases, all 60 screenshot comparisons, all 60 complete
focus cycles, and four separate Enter/Space activations. Its 4,213 observed
focus stops are not a fixed expectation because Material conditionally exposes
Back to top.

The documentation source suite reported 51 passes and 1,379 subtests. The
registry and all public-optimization suites reported 231 passes, 1,203 subtests,
and four warnings. The combined documentation, release, guidance,
distribution-source, and packaging slice reported 78 passes and 1,466
subtests. The full Python 3.12.12 suite reported 2,483 passes, 15 skips, 3,797
subtests, and 35 warnings in 113.36 seconds. The fresh distribution probe
passed wheel, source-distribution, and editable installs with 68 models, 81
provenance manifests, 193 compliance files, no eager Torch import, and zero
dependency violations; its wheel was 57,192,297 bytes and its source
distribution was 55,446,839 bytes. The first selected pre-commit run reformatted
the Python source and returned nonzero; the final applicable selected hooks
passed. Markdown formatting had no matching file in that pre-evidence run and
is not counted there.

Exact-commit remote CI, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and seven inaccessible asset,
checkpoint, or oracle paths remain open. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Expanded-state semantic-accessibility gate

The next bounded documentation-accessibility slice extends the automated Axe
contract from settled page loads to settled shared-shell interaction states.
The official reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the current `_toctree.yml`
fingerprint remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`, and
the official Pipeline endpoint returned HTTP 200 during the refresh.

The new 30-case inventory contains six search-open, six search-results, six
search-empty, six version-open, four desktop/tablet branch-open, and two mobile
drawer-open states across both VoiceHub palettes and every applicable viewport.
Before editing, 12 cases failed. Dynamically generated search-result code
regions were scrollable but not keyboard-focusable; the version disclosure
mixed list and menu roles; nested navigation landmarks repeated their names;
and the opened mobile drawer exposed an insufficiently contrasting nested
title. Search-result code regions now receive stable names, keyboard stops,
and visible focus through a mutation observer. The version selector uses native
`details`, `summary`, list, and link semantics. Runtime navigation panels
receive unique labels, and the mobile nested title uses the shared readable
foreground token.

The checker also waits for the search dialog's opacity transition to finish
before auditing it. This is necessary because an intermediate mobile result
probe measured the search container at opacity `0.666036`; its apparent result
count contrast was 4.24:1 even after the final foreground token was correct.
The settled state exceeds the automated threshold. The initial inventory, a
malformed branch-state JavaScript predicate, a palette focus-restoration
timeout, missing shell `python`, a stale pytest node ID, the 3.2:1, 4.2:1, and
4.24:1 contrast probes, and the YAPF/docformatter rewrite conflict are all
recorded failures or harness errors, not passing evidence. Ordinary
concatenated JavaScript literals made the two Python formatters converge, and
the final selected hook run passed every applicable hook.

The exact current-code Playwright 1.62.0 and Chromium 151.0.7922.34 run passed
60 base Axe cases, 30 expanded-state Axe cases, 60 geometry checks, 60
screenshot comparisons, 60 complete focus cycles with 4,229 observed focus
stops, and four independent Enter/Space activation cases. The source contract
reported two focused passes. The documentation suite reported 52 passes and
1,379 subtests; registry and public-optimization suites reported 231 passes,
1,203 subtests, and four warnings; and the focused guidance, release,
distribution-source, and pipeline slice reported 24 passes and 19 subtests.
The complete Python 3.12.12 suite reported 2,484 passes, 15 skips, 3,797
subtests, and 35 warnings in 110.14 seconds.

Fresh wheel, source-distribution, and editable probes passed with 68 models,
81 provenance manifests, 193 compliance files, all required package data, no
eager Torch import, and zero dependency violations. The artifacts measured
57,192,297 and 55,446,883 bytes. The refreshed inventories remain 34 TTS, 23
ASR, 11 VAD, 102 aliases, 68 current model pages, 59 current notebooks, six
public optimization passes, 408 model/pass pairs, five benchmark records,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This evidence covers automated detectable rules in the enumerated settled
states; it is not a complete WCAG or manual accessibility audit. Exact-current-
worktree remote CI, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and seven inaccessible asset,
checkpoint, or oracle paths remain open. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Installation route parity gate

The next bounded documentation slice maps the current official Transformers
Installation page to VoiceHub's speech-specific installation and offline
workflow. The refreshed reference is Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its `_toctree.yml` SHA-256 is
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official Installation endpoint returned HTTP 200.

The official rendered page currently uses `Installation`, `Virtual
environment`, `Python`, `Source install`, `Editable install`, `conda`, `Set
up`, `Cache directory`, and `Offline mode` in that order, with no content-tab
component. Its article measured 804 pixels at x = 318 on desktop, 658 pixels
at x = 318 on tablet, and 342 pixels at x = 24 on mobile, with zero document
overflow. VoiceHub's previous source instead used numbered sections and an OS
tab set. The pre-edit focused source contract failed on that stale hierarchy
and is not counted as a pass.

The rewritten VoiceHub page now matches the current heading and TOC hierarchy
while retaining original speech-library instructions: uv-first environment
management, published, source, and editable installs, an honest conda-owned
environment boundary, the exact shared Hub cache precedence, and global or
call-level offline operation. It does not invent a conda-forge VoiceHub
package, report checkpoint-specific offline inference as verified, or imply
that the development contract is already available from the older PyPI
release. Six rendered route cases enforce the exact headings, TOC, absence of
stale tabs, required cache/offline markers, shared geometry, both palettes,
three viewports, Axe checks, complete native focus, and reviewed screenshot
signatures.

The first focused rendered run exposed a real light-theme syntax-token failure:
Pygments variable names rendered at 4.486:1 against the code background, below
the 4.5:1 Axe threshold. The shared light-theme variable token is now
`#686a72`, which measures 4.948:1 against `#f4f5f8`. That failed run is not
passing evidence. The first selected hook run also let YAPF format the Python
checker and returned nonzero; the formatted rerun passed every applicable
hook.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base accessibility, geometry, screenshot, and complete-focus cases, six
Installation structure cases, 30 expanded-state accessibility cases, and four
independent Enter/Space activation cases. The matrix traversed 4,250 visible
focus stops. The documentation suite reported 53 passes and 1,380 subtests;
the registry and public-optimization slice reported 231 passes, 1,203
subtests, and four warnings; and the focused guidance, release, distribution-
source, and pipeline slice reported 24 passes and 19 subtests. The complete
Python 3.12.12 suite reported 2,485 passes, 15 skips, 3,798 subtests, and 35
warnings in 110.86 seconds.

Package code and metadata did not change, so wheel, source-distribution, and
editable probes were not rebuilt for this documentation-only slice. The
immediately preceding successful artifacts remain 57,192,297 and 55,446,883
bytes and are not new evidence here. The refreshed inventories remain 34 TTS,
23 ASR, 11 VAD, 102 aliases, 68 current model pages, 59 current notebooks, six
public optimization passes, 408 model/pass pairs, five benchmark records,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

Exact-current-worktree remote CI, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and seven
inaccessible asset, checkpoint, or oracle paths remain open. The untracked
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Quickstart route parity gate

The bounded release slice mapped VoiceHub's Quickstart route to the current
official Transformers Quickstart at commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. The upstream `_toctree.yml`
retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the rendered endpoint returned HTTP 200. Its current hierarchy is
`Quickstart`, `Set up`, `Agent skills`, `Pretrained models`, `Pipeline`,
`Trainer`, and `Next steps`; its articles measured 804, 658, and 342 pixels at
the matched desktop, tablet, and mobile viewports with zero overflow.

VoiceHub's stale `Discover models` and `Inference` hierarchy had no current
tab or callout contract. The pre-edit source regression failed on that
mismatch. The corrected route keeps VoiceHub instructions original and
speech-specific while mapping the current page structure: uv/pip setup,
source-checkout agent skills, shared pretrained contracts, TTS/ASR/VAD
pipelines, and an explicit Trainer support boundary. Source and rendered tests
require the exact hierarchy, three tab sets with seven options, two tips, one
model table, at least twelve code blocks, required content markers, and parsed
Python examples.

Rendered negative evidence remained explicit. The iteration found and fixed an
off-canvas keyboard target, tab state persisted between viewports, an unnamed
keyboard-inaccessible code scroller, eight pixels of mobile overflow, an
unnamed scrollable option strip, and transient low-contrast Back to top text.
The final shared behavior gives overflowing code and option strips named
keyboard stops, keeps mobile tabs inside the document, resets tab state for
deterministic baselines, and removes the opacity transition that produced the
low-contrast intermediate state. None of the failed runs is counted as a
pass.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base accessibility, geometry, screenshot, and complete-focus cases, six
Quickstart structure cases, six post-activation Quickstart tab cases, 30
expanded-state accessibility cases, and four independent branch/drawer
activations. The matrix traversed 4,297 visible focus stops. The documentation
suite reported 53 passes and 1,380 subtests; the registry and
public-optimization slice reported 231 passes, 1,203 subtests, and four
warnings; and the focused guidance, release, distribution-source, and pipeline
slice reported 24 passes and 19 subtests. The complete Python 3.12.12 suite
reported 2,485 passes, 15 skips, 3,798 subtests, and 35 warnings in 107.12
seconds.

The first combined selected-file pre-commit runs returned docformatter exit 3.
The failure was reproducible only after the preceding hooks because
docformatter 1.7.8 misclassified the multiline JavaScript callback used to
reset Quickstart tabs and proposed damaging whitespace inside the string. A
serial-execution experiment did not fix the observable command and was
discarded. Replacing only that callback with equivalent concatenated strings
made docformatter's direct check, its hook, and the full selected-file sequence
exit zero with identical before/after hashes. The failed runs are not passes.
Package code and metadata did not change during the Quickstart slice. A
subsequent exact-current-worktree distribution iteration rebuilt and validated
the wheel, source distribution, and editable install; its fresh evidence is
recorded in the next section.

The refreshed inventories remain 34 TTS, 23 ASR, 11 VAD, 102 aliases, 68
current model pages, 59 current notebooks, six public optimization passes, 408
model/pass pairs, five benchmark records, eight top-level navigation roots,
ten representative routes, and eight contribution steps. Exact-current-
worktree remote CI, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and seven inaccessible asset,
checkpoint, or oracle paths remain open. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree distribution evidence

The focused distribution and release contracts reported 13 passes. The source
alignment check found version 0.3.0, 34 TTS, 23 ASR, 11 VAD, all 68 documented
providers, and five benchmark records. The full temporary
`scripts/check_distribution.py` run then built the current worktree and probed
wheel, source-distribution, and editable installations in three isolated
environments without installing runtime dependencies.

All three installation modes reported version 0.3.0, 68 models, 193 compliance
files, every required representative data file, zero runtime-dependency
violations, and no eager PyTorch import. The build contained 81 pinned and
licensed provenance manifests. The validated wheel was 57,192,297 bytes and
the validated source distribution was 55,447,546 bytes, both below the 100 MB
publication limit. The temporary environments and artifacts were removed after
the probe; nothing was copied into the repository or published.

A second temporary build passed `scripts/check_release.py --dist-dir` and
recorded exact-run fingerprints:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `voicehub-0.3.0-py3-none-any.whl` | 57,192,297 | `83b32ea12255f7829d8e152069a32f85f5eb786d53a31bb01d458676426b2a74` |
| `voicehub-0.3.0.tar.gz` | 55,448,121 | `eb7284f989b2c3f8365959c1f790c3071c7e43f58853f1852e86cc5fb2c1fbe1` |

The two successful sdist builds differed by 575 bytes, so this report does not
claim byte-for-byte reproducible archives; the fingerprints identify only the
second exact build. Tagged-workflow artifacts and their authoritative hashes
remain a separate release gate. Exact-current-worktree remote CI, protected
publisher configuration, tags, publication approval, five native-kernel
hardware gates, and seven inaccessible asset, checkpoint, or oracle paths also
remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree supported-Python evidence

The next bounded release iteration exercised the complete current worktree on
the two supported interpreters whose full-suite records predated the latest
documentation and packaging contracts. Python 3.10.19 and Python 3.11.15 each
created a new temporary virtual environment. uv 0.11.21 then used its CPU
PyTorch backend to install the current source directly with `uv pip install`;
the install target was `-e ".[test]"`. Project synchronization was not used,
and the repository lock file was neither read nor modified.

Both environments first passed the 13 focused release and distribution
contracts. The corrected Hub transport regression separately reported 19
passes and five subtests on each interpreter. The complete default-offline
suite, with all opt-in asset variables left unset, then reported identical
coverage:

| Interpreter | Passed | Skipped | Subtests | Warnings | Duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python 3.10.19 | 2,485 | 15 | 3,798 | 35 | 108.21 seconds |
| Python 3.11.15 | 2,485 | 15 | 3,798 | 35 | 98.60 seconds |

Those runs did not count any of the 15 skipped paths as passed. The next
default-runtime iteration executes the three import gates separately. An
initial command incorrectly forced `VOICEHUB_OFFLINE=1`, which prevented 12
mocked Hub transport tests from reaching their patched URL opener. Those two
executions failed with 12 failures, 2,473 passes, 15 skips, 3,798 subtests, and
35 warnings in 213.83 seconds on Python 3.10 and 200.08 seconds on Python 3.11.
They are recorded as failed command evidence and are not counted as supported-
version passes.

The successful Python 3.10 and 3.11 results join the fresh Python 3.12.12
current-worktree result above, so the exact current source has now passed its
complete CPU-safe suite on every supported interpreter on macOS. This does not
substitute for exact-current-worktree Linux or Windows CI. At that point,
tagged-workflow artifact hashes, publisher configuration, publication approval,
and the explicit hardware and asset skip boundaries remained open; the next
section refreshes their current status. The temporary environments were removed
after verification, and the untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree default-runtime evidence

The next bounded release iteration refreshed the official parity reference
before selecting a gate. Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, its documentation toctree retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official documentation endpoint remained available. No parity mapping
changed in this release-evidence iteration.

Python 3.12.12 created a fresh temporary virtual environment. uv 0.11.21 used
its CPU PyTorch backend to install the exact current source with `.[test]` and
without project synchronization. With `VOICEHUB_FULL_RUNTIME_TEST=1`, the
focused default-runtime file reported five passes and 138 registry subtests in
16.58 seconds. The same isolated environment then completed the entire suite
with 2,488 passes, 12 skips, 3,936 subtests, and 35 warnings in 157.75 seconds.
The three default-runtime tests that the ordinary suite skips all executed and
passed.

The remaining 12 skips retain separate boundaries:

- Three Triton and two compiled CUDA-extension tests require unavailable CUDA
  hardware or a CUDA toolkit and remain unpassed.
- ESPNet, SenseVoice, SpeechBrain, TEN-VAD, and QuartzNet are opt-in in the
  default suite but retain their separately executed successful evidence.
- The WeNet checkpoint and tokenizer remain inaccessible and unpassed.

The ignored local `dist/` directory was inspected without modification. Its
54,572,319-byte wheel has SHA-256
`d37fee52843a2a8ac654894509199aaccc5493830b6d19b6f3ccaddf252352f4`, and its
52,863,267-byte source distribution has SHA-256
`d0224a0d39c58ca7a0398fad356a4e8c40577131f250c58b459ed2d3902c1b42`.
Those files do not match the exact current-worktree fingerprints recorded
above, were preserved as user-owned local artifacts, and are not release-
candidate evidence.

This closes the current-worktree default-runtime gate locally; execution of
the same uncommitted source in remote Linux CI remains pending. Exact-current-
worktree Linux and Windows matrix coverage, tagged-workflow artifact hashes,
protected publisher configuration, tags, publication approval, five native-
kernel hardware gates, and two inaccessible WeNet asset paths remain open. The
temporary environment was removed after verification, and the untracked
`uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Pipeline interaction evidence

The next bounded documentation iteration refreshed the official task-guide
pair before selecting the remaining local shell gap. Transformers `main`
remained `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, and `_toctree.yml`
retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The mapped routes remain `/docs/transformers/main/en/pipeline_tutorial` and
`/voicehub/guides/inference/`.

The focused source test first failed on the absence of a rendered Pipeline
contract. The initial exact render then found that all six Pipeline code blocks
had zero copy buttons despite the configured `content.code.copy` feature; that
failure is not passing evidence. The shared page-action runtime now creates
native, labelled code-copy buttons with Clipboard API and selection-fallback
paths, explicit busy/success states, and retained keyboard focus. The first
implementation copied syntax-highlighted visual line breaks rather than exact
source text and failed its clipboard comparison. That failure is also
excluded; the corrected action copies the code element's exact text content.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 then passed 60 representative
route/viewport/palette cases, including six exact Pipeline structure/content
cases and six Pipeline keyboard-copy interaction cases. The copy cases use
Enter and Space across desktop, tablet, mobile, light, and dark states; they
verify exact clipboard text, visible success state, focus retention, zero
overflow, and a second zero-violation Axe result. The full matrix recorded 60
base Axe checks, 60 reviewed screenshot signatures, 60 complete focus cycles
with 4,572 stops, 30 expanded-state Axe checks, four branch/drawer activation
cases, six Quickstart tab cases, and six Pipeline copy cases. The ten-route DOM
validator also passed.

The complete documentation suite reported 53 passes and 1,380 subtests. A
risk-proportional registry, universal-optimization, distribution-compliance,
and release-readiness slice reported 97 passes and 452 subtests. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, 11 VAD), 102 aliases, zero
invalid display names, 68 current provider pages, 59 current model notebooks,
six public optimization passes, and 408 model/pass pairs.

The first combined final-file sequence exposed a YAPF line split that broke
one source-level CI fragment assertion; that run reported one failure, 52
passes, and 1,380 subtests and is not passing evidence. Computing the aggregate
keyboard-case count before constructing the result mapping restored stable
formatting. The complete selected-file pre-commit sequence and all 53
documentation tests then passed together.

This closes the local Pipeline structure and code-copy interaction gate. The
complete Python suite and physical package builds were not rerun for this
documentation-runtime slice; their exact preceding current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths remain
open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Auto Classes interaction evidence

The next bounded documentation iteration refreshed the official Auto Classes
pair before selecting the remaining local shell gap. Transformers `main`
remained `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, `_toctree.yml`
retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
the official route returned HTTP 200, and the pinned 301-line Auto Classes
source has SHA-256
`557f5836c0722fef6a484c46805dfab0eb69a387b028a914b132350edf09f167`.
The mapped routes remain `/docs/transformers/main/en/model_doc/auto` and
`/voicehub/models/providers/`.

The focused source contract intentionally failed before implementation with
one failure and 68 passing subtests because no model-index-specific rendered
contract existed. That run is not passing evidence. The first selected-file
pre-commit run later failed when docformatter rewrote a multiline JavaScript
predicate; it is also excluded. The predicate now uses formatter-safe
concatenated strings, and the complete selected-file sequence passes.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Auto Classes structure and
content cases. The contract protects nine headings, eight right-TOC entries,
four registry-derived tables with 3/34/23/11 rows, 68 unique uppercase-first
provider links, three code blocks and copy actions, and the required
configuration, processing, task-model, registry-discovery, and lazy-loading
markers.

Six additional cases used Enter in the default palette and Space in slate at
desktop, tablet, and mobile widths to activate page copy. They verified exact
clipboard text, visible success and idle state, focus retention, zero
overflow, and a second zero-violation Axe result. The complete matrix recorded
60 base Axe checks, 60 reviewed screenshot signatures, 60 complete focus
cycles with 4,620 stops, 30 expanded-state Axe checks, four branch/drawer
activation cases, six Quickstart tab cases, six Pipeline copy cases, and six
Auto Classes page-copy cases, for 82 keyboard cases total. The ten-route DOM
validator also passed.

The complete documentation suite reported 53 passes and 1,380 subtests. A
risk-proportional registry, speech-task, universal-optimization,
distribution-compliance, and release-readiness slice reported 97 passes and
452 subtests. All 68 generated provider pages and 59 generated model notebooks
remain current, and the five-record release-alignment check passed.

This closes the local Auto Classes structure and page-copy interaction gate.
The complete Python suite and physical package builds were not rerun for this
documentation-only slice; their exact preceding current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths remain
open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree SpeechT5 interaction evidence

The next bounded documentation iteration refreshed the official SpeechT5
model-detail pair. Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
the route returned HTTP 200, and the pinned 87-line source retained SHA-256
`71bba8a2921cf637383fb8d6f2fd66df9cd95deb59118b9f49e1362485c27eb5`.
The mapped routes remain `/docs/transformers/main/en/model_doc/speecht5` and
`/voicehub/models/providers/speecht5/`.

The first focused command named a nonexistent unittest class, collected no
tests, and exited 4; it is not passing evidence. The corrected focused test
then intentionally failed because no SpeechT5-specific rendered contract
existed. The first full render later failed because the new table inventory
expected ten checkpoint rows while the source correctly renders nine. Neither
failure is counted as a pass. Correcting the expectation required no generated
page change.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact SpeechT5 structure and
content cases. The contract protects 14 headings, 13 right-TOC entries, eight
generated tables with 6/3/4/2/6/1/9/8 rows, seven code blocks and copy actions,
two local facade-source links, and required auto-model, processor,
normalized-output, checkpoint, real-evidence, and fail-closed optimization
markers.

Six additional cases used Enter in the default palette and Space in slate at
desktop, tablet, and mobile widths to activate page copy. They verified exact
clipboard text, visible success and idle state, focus retention, zero
overflow, and a second zero-violation Axe result. The complete matrix recorded
60 base Axe checks, 60 reviewed screenshot signatures, 60 complete focus
cycles with 4,637 stops, 30 expanded-state Axe checks, four branch/drawer
activation cases, six Quickstart tab cases, six Pipeline copy cases, six Auto
Classes page-copy cases, and six SpeechT5 page-copy cases, for 88 keyboard
cases total. The ten-route DOM validator also passed.

The complete documentation suite reported 53 passes and 1,380 subtests. A
risk-proportional registry, speech-task, universal-optimization,
distribution-compliance, and release-readiness slice reported 97 passes and
452 subtests. All 68 generated provider pages and 59 generated model notebooks
remain current, and the five-record release-alignment check passed.

This closes the local SpeechT5 model-detail structure and page-copy interaction
gate. The complete Python suite and physical package builds were not rerun for
this documentation-only slice; their exact preceding current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths remain
open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Trainer interaction evidence

The next bounded documentation iteration refreshed the official Trainer
overview pair. Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and the `Trainer overview` then `Fine-tuning` order, the route returned HTTP
200, and the pinned 30-line source retained SHA-256
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`.
The mapped routes remain `/docs/transformers/main/en/trainer` and
`/voicehub/guides/trainer/`.

The focused source contract intentionally failed before implementation because
no Trainer-specific rendered contract existed. That run is not passing
evidence. The new contract protects the exact two-heading and one-entry TOC
inventory, zero tables and code blocks, one page-copy action, four exact
next-step destinations, the edit and Fine-tuning footer targets, and the
Trainer, TrainingArguments, model-owned objective, exact-resume, and
fail-closed speech-training markers.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Trainer structure and
content cases. Six additional cases used Enter in the default palette and
Space in slate at desktop, tablet, and mobile widths to activate page copy.
They verified exact clipboard text, visible success and idle state, focus
retention, zero overflow, and a second zero-violation Axe result. The complete
matrix recorded 60 base Axe checks, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,771 stops, 30 expanded-state Axe checks, four
branch/drawer activation cases, six Quickstart tab cases, six Pipeline copy
cases, six Auto Classes page-copy cases, six SpeechT5 page-copy cases, and six
Trainer page-copy cases, for 94 keyboard cases total. The ten-route DOM
validator also passed.

The complete documentation suite reported 53 passes and 1,380 subtests. A
risk-proportional registry, speech-task, universal-optimization,
distribution-compliance, and release-readiness slice reported 97 passes and
452 subtests. All 68 generated provider pages and 59 generated model notebooks
remain current, and the five-record release-alignment check passed.

This closes the local Trainer structure and page-copy interaction gate. The
complete Python suite and physical package builds were not rerun for this
documentation-only slice; their exact preceding current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths remain
open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Optimization interaction evidence

The next bounded documentation iteration refreshed the official Optimization
overview pair. Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and the `optimization_overview` title `Overview`, the route returned HTTP 200,
and the pinned 178-line source retained SHA-256
`19622667a7299f258f5c9a72940c9f26492619636f35d9bd592701c02745b620`.
The mapped routes remain `/docs/transformers/main/en/optimization_overview`
and `/voicehub/guides/optimization-overview/`.

The focused source contract intentionally failed before implementation with
one failure and six passing subtests because no Optimization-specific rendered
contract existed. That run is not passing evidence. The new contract protects
the exact eight-heading and seven-entry TOC inventory, one six-row technique
table, one code block and copy action, one page-copy action, all six registered
optimization pass names, five exact workflow destinations, the edit and
previous/next footer targets, and the application, validation, manifest,
restoration, unsupported-quantization, parallelism, and continuous-batching
boundaries.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Optimization structure and
content cases. Six additional cases used Enter in the default palette and
Space in slate at desktop, tablet, and mobile widths to activate page copy.
They verified exact clipboard text, visible success and idle state, focus
retention, zero overflow, and a second zero-violation Axe result. The complete
matrix recorded 60 base Axe checks, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,603 stops, 30 expanded-state Axe checks, four
branch/drawer activation cases, six Quickstart tab cases, six Pipeline copy
cases, six Auto Classes page-copy cases, six SpeechT5 page-copy cases, six
Trainer page-copy cases, and six Optimization page-copy cases, for 100
keyboard cases total. The ten-route DOM validator also passed.

The complete documentation suite reported 53 passes and 1,380 subtests. A
risk-proportional registry, speech-task, universal-optimization,
distribution-compliance, and release-readiness slice reported 97 passes and
452 subtests. All 68 generated provider pages and 59 generated model notebooks
remain current, and the five-record release-alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the local Optimization overview structure and page-copy
interaction gate. The complete Python suite and physical package builds were
not rerun for this documentation-only slice; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Contribution interaction evidence

The next bounded documentation iteration refreshed the official modular-
contribution pair. Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and the modular then legacy contribution order, the route returned HTTP 200,
and the pinned 500-line source retained SHA-256
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The mapped routes remain `/docs/transformers/main/en/modular_transformers` and
`/voicehub/project/adding-a-model/`.

The focused source contract intentionally failed before implementation with
one failure and 18 passing subtests because no Contribution-specific rendered
contract existed. That run is not passing evidence. Two read-only pre-edit
HTML inventory probes also failed because Beautiful Soup is not installed and
the first dependency-free parser selector assumed the content root was a
`div`; neither is verification evidence. The corrected built-in parser
established the exact local inventory without changing the site.

The new contract protects the exact 10-heading and nine-entry TOC inventory,
the eight ordered process labels, three tables with 8/3/7 rows, 13 code blocks
and copy actions, one page-copy action, two final contribution-guide targets,
the edit and adjacent footer destinations, and the scaffold, registry, task
bases, training, optimization, generation, distribution, and unverified-
evidence boundaries.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Contribution structure and
content cases. Six additional cases used Enter in the default palette and
Space in slate at desktop, tablet, and mobile widths to activate page copy.
They verified exact clipboard text, visible success and idle state, focus
retention, zero overflow, and a second zero-violation Axe result. The complete
matrix recorded 60 base Axe checks, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,572 stops, 30 expanded-state Axe checks, four
branch/drawer activation cases, six Quickstart tab cases, six Pipeline copy
cases, six Auto Classes page-copy cases, six SpeechT5 page-copy cases, six
Trainer page-copy cases, six Optimization page-copy cases, and six
Contribution page-copy cases, for 106 keyboard cases total. The ten-route DOM
validator also passed.

The complete documentation suite reported 53 passes and 1,380 subtests. The
model-scaffold contract reported 20 passes and 35 subtests. A
risk-proportional registry, speech-task, universal-optimization,
distribution-compliance, and release-readiness slice reported 97 passes and
452 subtests. All 68 generated provider pages and 59 generated model notebooks
remain current, and the five-record release-alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the local Contribution structure and page-copy interaction gate.
The complete Python suite and physical package builds were not rerun for this
documentation-only slice; their exact preceding current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths remain
open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Models API interaction evidence

The next bounded documentation iteration refreshed the official Models API
pair. Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`, `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and the Models entry between Logging and Text Generation, the route returned
HTTP 200, and the pinned 44-line source retained SHA-256
`a4899c758b5d621075b2e2f39f0aa79671010c88b08d1508f7af5731375c2871`.
The mapped routes remain `/docs/transformers/main/en/main_classes/model` and
`/voicehub/reference/models/`.

The focused source contract intentionally failed before implementation with
one failure and 26 passing subtests because no Models-API-specific rendered
contract existed. That run is not passing evidence. The first full render then
failed because the initial lifecycle-link selector included code-line number
anchors. That run is also excluded. The corrected selector ignores links
inside `pre` and protects only the declared user-facing content links.

The new contract protects the exact five-heading and four-entry TOC inventory,
four tables with 6/3/3/5 rows, two code blocks and copy actions, one page-copy
action, four exact facade-source targets, three exact internal lifecycle
targets, the edit and Model audit/Full API footer destinations, and the
configuration, factory, lazy-loading, task-base, normalized-output, training-
output, portable-state, and unsupported-sharing boundaries.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Models API structure and
content cases. Six additional cases used Enter in the default palette and
Space in slate at desktop, tablet, and mobile widths to activate page copy.
They verified exact clipboard text, visible success and idle state, focus
retention, zero overflow, and a second zero-violation Axe result. The complete
matrix recorded 60 base Axe checks, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,571 stops, 30 expanded-state Axe checks, four
branch/drawer activation cases, six Quickstart tab cases, six Pipeline copy
cases, six Auto Classes page-copy cases, six SpeechT5 page-copy cases, six
Trainer page-copy cases, six Optimization page-copy cases, six Contribution
page-copy cases, and six Models API page-copy cases, for 112 keyboard cases
total. The ten-route DOM validator also passed.

The complete documentation suite reported 53 passes and 1,380 subtests. The
speech-core, AutoConfig, and registry slice reported 54 passes and 216
subtests. A risk-proportional registry, speech-task, universal-optimization,
distribution-compliance, and release-readiness slice reported 97 passes and
452 subtests. All 68 generated provider pages and 59 generated model notebooks
remain current, and the five-record release-alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the local Models API structure and page-copy interaction gate. The
complete Python suite and physical package builds were not rerun for this
documentation-only slice; their exact preceding current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths remain
open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Home interaction evidence

The next bounded documentation iteration refreshed the official Home pair.
Transformers `main` remained
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the 65-line `index.md`
source has SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and `_toctree.yml` retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
with Transformers, Installation, and Quickstart at the start of Get started.
The official route returned HTTP 200. The mapped routes remain
`/docs/transformers/main/en/index` and `/voicehub/`.

The focused source contract intentionally failed before implementation because
the local Home exposed only `What is VoiceHub?` instead of the current
Features, Design, and Learn hierarchy. An intermediate raw-source assertion
also failed on an existing line-wrapped bold registry label. Neither run is
passing evidence. Normalizing the Markdown source corrected the assertion
without weakening rendered checks. The first rendered run then passed the
ten-route DOM validator but failed the stale desktop-light Home screenshot
baseline by 289 bits (11.289%); that run is also excluded.

The replacement candidate differed materially above threshold only for the six
Home viewport/palette signatures. Only those six reviewed entries were
accepted, and no non-Home baseline was replaced. Rendered review also moved the
long wrapping H1 to the compact product name `VoiceHub`, while retaining the
original one-lifecycle message in the VoiceHub-specific tagline.

The final contract protects the VoiceHub, Features, Design, and Learn heading
order; three TOC entries; Pipeline, Trainer, and speech-generation links; one
tip; two design principles; 13 resource cards; four exact badge targets; six
images with two decorative and four named status images; zero tables and code
blocks; one page-copy action; the edit target; no previous destination;
Installation as next; exact live registry counts; and the lazy-checkpoint,
training-extra, and third-party-license boundaries.

The strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Home structure/content cases
and six Home keyboard-copy interaction cases. The copy cases use Enter and
Space across desktop, tablet, mobile, light, and dark states; they verify exact
clipboard text, visible success and idle state, focus retention, zero overflow,
and a second zero-violation Axe result. The complete matrix recorded 60 base
Axe checks, 60 reviewed screenshot signatures, 60 complete focus cycles with
4,713 stops, 30 expanded-state Axe checks, four branch/drawer activation
cases, six Home page-copy cases, six Quickstart tab cases, six Pipeline copy
cases, and six page-copy cases each for Auto Classes, SpeechT5, Trainer,
Optimization, Contribution, and Models API, for 118 keyboard cases total. The
ten-route DOM validator also passed.

A successful fresh official render confirmed the exact Transformers,
Features, Design, and Learn hierarchy, an 804-pixel desktop article, and zero
overflow. Subsequent six-state remote retries returned an unhydrated shell and
were interrupted; those retries are not counted as passed. The preceding
official six-state Home shell evidence remains tied to the same upstream
commit, while the current source, HTTP, and local six-state evidence support
this bounded slice.

The complete documentation suite reported 54 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes the local Home structure and page-copy interaction gate. The
complete Python suite and physical package builds were not rerun for this
documentation-only slice; their exact preceding current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths remain
open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Installation interaction evidence

The next bounded documentation iteration refreshed the official Installation
pair at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its 164-line
`docs/source/en/installation.md` source has SHA-256
`d050e3e0e1c89d543c71c25367a455a9b49ea89c92f8c2376bb58294a4a4cf3b`,
and the 1,576-line `_toctree.yml` retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
with Transformers, Installation, and Quickstart at the start of Get started.
The official route returned HTTP 200. Its rendered hierarchy remains
Installation, Virtual environment, Python, Source install, Editable install,
conda, Set up, Cache directory, and Offline mode, with code-copy controls, an
edit action, and Transformers/Quickstart footer actions. The mapped local route
remains `/voicehub/getting-started/installation/`.

The observable contract was fixed before implementation: six viewport/palette
states must enforce the exact nine headings and eight-entry TOC, zero stale tab
sets, 15 code blocks and code-copy buttons, one page-copy button, four exact
external package-manager destinations, three exact internal workflow
destinations, edit and previous/next targets, and the installation, cache,
offline, and evidence-safety markers. Six Enter/Space cases must copy the first
command, while six more must copy the readable page. Both action families must
verify exact clipboard content, visible success and idle states, focus
retention, zero overflow, and a post-action zero-violation Axe result.

The focused source contract intentionally failed before implementation because
the checker had no Installation destination constants, copy helpers, or result
counters. The first rendered run also failed because an assertion searched for
the narrower phrase `without importing PyTorch` rather than the existing
`without downloading a checkpoint or importing PyTorch` sentence. The exact
phrase correction passed; neither failed run is counted. The Installation
source itself required no change.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all 60
representative route/viewport/palette cases, including six exact Installation
structure/destination cases, six Installation first-code copy cases, and six
Installation page-copy cases. The matrix recorded 60 base Axe checks, 60
reviewed screenshot signatures, 60 complete focus cycles with 4,613 observed
stops, 30 expanded-state Axe checks, four branch/drawer activation cases, and
130 keyboard cases. Installation used Enter in the default palette and Space
in slate at desktop, tablet, and mobile widths; both actions returned from the
success state to idle without losing focus or introducing overflow.

The complete documentation suite reported 54 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. Selected pre-commit hooks passed in one invocation.
All 68 generated provider pages and 59 generated model notebooks remain
current. The five-record release-alignment check passed, and refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Installation structure, destination, code-copy,
and page-copy gate. The complete Python suite and physical package builds were
not rerun for this documentation-checker-only slice; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Quickstart interaction evidence

The next bounded documentation iteration refreshed the official Quickstart
pair at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its 312-line
`docs/source/en/quicktour.md` source has SHA-256
`ecfb99781204bcaea1ca63bcb4ad9ef70c99812e5f965a49b29de23cedd25bd7`,
and the 1,576-line `_toctree.yml` retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
with Transformers, Installation, and Quickstart at the start of Get started.
The official route returned HTTP 200. Its rendered hierarchy remains
Quickstart, Set up, Agent skills, Pretrained models, Pipeline, Trainer, and
Next steps, with content tabs, code-copy controls, page actions, and
previous/next navigation. The mapped local route remains
`/voicehub/getting-started/quickstart/`. All three declared GitHub skill
destinations also returned HTTP 200 in the refreshed link probe.

The observable contract was fixed before implementation: six viewport/palette
states must enforce the exact seven headings and six-entry TOC, three tab sets
with seven options, two tips, one three-row table, 12 code blocks and code-copy
buttons, one page-copy button, three external skill destinations, nine internal
workflow-link occurrences, and exact edit and previous/next targets. The six
existing tab cases must still activate the final option in every set. Six
Enter/Space cases must then copy the readable active-tab state with exact
clipboard content, visible success and idle states, focus retention, zero
overflow, and a post-action zero-violation Axe result.

The focused source contract intentionally failed before implementation because
the checker had no Quickstart destination constants, page-copy helper, or
result counter. The first rendered run also failed because Material's seven
generated tab-control hashes entered the initial content-link inventory. The
corrected inventory excludes those component controls while their dedicated
tab-activation contract remains intact. Neither failed run is counted. The
Quickstart source and screenshot baselines required no change.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all 60
representative route/viewport/palette cases, including six exact Quickstart
structure/destination cases, six Quickstart tab cases, and six Quickstart
page-copy cases. The matrix recorded 60 base Axe checks, 60 reviewed screenshot
signatures, 60 complete focus cycles with 4,644 observed stops, 30 expanded-
state Axe checks, four branch/drawer activation cases, and 136 keyboard cases.
Quickstart page copy used Enter in the default palette and Space in slate at
desktop, tablet, and mobile widths and returned from success to idle without
losing focus or introducing overflow.

The complete documentation suite reported 54 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. Selected pre-commit hooks passed in one invocation.
All 68 generated provider pages and 59 generated model notebooks remain
current. The five-record release-alignment check passed, and refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Quickstart structure, destination, tab, and
page-copy gate. The complete Python suite and physical package builds were not
rerun for this documentation-checker-only slice; their exact preceding current-
worktree evidence remains separate. Exact-current-worktree remote CI, exact
tagged artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree representative TOC activation evidence

The next bounded documentation iteration refreshed the shared
table-of-contents reference at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its 65-line
`docs/source/en/index.md` source has SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. No new remote rendered interaction
was executed, so prior official shell evidence remains separate.

The observable contract was fixed before implementation. At the 1,440-pixel
desktop viewport, all ten representative routes must activate their final TOC
target by pointer and by unmodified Enter in both palettes. Each of the 40
cases must require the current hash to identify the CSS `:target`, exactly one
matching active link after smooth scrolling settles, visible heading alignment
beneath the header, zero overflow, and a post-action zero-violation Axe result.
The 20 Enter cases must retain focus on the activated link.

The focused source contract intentionally failed before implementation because
the checker had no representative-route activation matrix, helper, or result
counters. The first full rendered run then failed on the intentional Home
observer alignment at 63.703 pixels beneath the 65-pixel header. A focused
rendered probe also failed by inspecting Installation before smooth scrolling
and Material's observer settled. The corrected helper applies the existing
two-pixel observer tolerance, waits for the exact hash and active link, and
then requires another 600 milliseconds of stable state. Those failed runs are
not counted as passing evidence.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all 60
base route/viewport/palette cases and 60 reviewed screenshot signatures. It
completed 60 focus cycles with 4,607 stops, 30 expanded-state Axe cases, four
branch/drawer activation cases, 20 pointer TOC cases, and 20 Enter TOC cases.
Every TOC interaction received a second Axe pass, and the total keyboard
inventory increased from 136 to 156 cases.

The complete documentation suite reported 55 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. Selected pre-commit hooks passed. All 68 generated
provider pages and 59 generated model notebooks remain current. The five-record
release-alignment check passed, and refreshed inventories remain 68 models (34
TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display names, six public
optimization passes, 408 model/pass pairs, eight top-level navigation roots,
ten representative routes, and eight contribution steps.

This closes representative-route TOC activation in the local rendered matrix.
Non-representative routes and future templates remain outside that matrix. The
complete Python suite and physical package builds were not rerun for this
documentation-checker-only slice; their exact preceding current-worktree
evidence remains separate. Exact-current-worktree remote CI, exact tagged
artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree representative search activation evidence

The next bounded documentation iteration refreshed the shared header-search
reference at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its 65-line
`docs/source/en/index.md` source has SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. No new remote rendered search
interaction was executed, so prior official shell evidence remains separate.

The observable contract was fixed before implementation. All ten
representative routes must activate and close search at desktop, tablet, and
mobile widths in both palettes. Forty desktop/tablet cases use Ctrl+K, and 20
mobile cases use the visible pointer trigger. Each opened state must expose the
checked toggle, expanded trigger, open body state, focused and visible input,
active input/result tab stops, non-hidden and non-inert output, zero overflow,
and a zero-violation Axe result. Escape must close the dialog, restore the
hidden/inert/tab-order state, preserve zero overflow, and return focus to the
visible inline input on desktop/tablet or the search trigger on mobile.

The focused source contract intentionally failed before implementation because
the checker lacked the route/viewport/palette search matrix, helper, counters,
and breakpoint-aware close-focus behavior. The implementation corrected the
hidden desktop trigger as Escape's universal focus destination by selecting the
visible inline input outside the mobile breakpoint. A six-case Home breakpoint
probe passed. The first complete documentation-suite run after the rendered
matrix then failed one legacy static assertion that still required
`trigger.focus()`; the updated assertion now protects the breakpoint-aware
helper. Neither failed run is passing evidence.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60 base
route/viewport/palette cases, 60 reviewed screenshot signatures, 60 complete
focus cycles with 4,654 stops, 30 expanded-state Axe cases, four branch/drawer
activation cases, 60 search activation/closure cases, and 40 TOC activation
cases. Every opened search state received another Axe pass. The complete
keyboard inventory increased from 156 to 196 cases.

The complete documentation suite reported 56 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current. The five-record release-alignment check passed, and
refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes search activation and Escape restoration in the representative
local route/viewport/palette matrix. Version, language, theme, and source
route-specific activation remain outside this slice. The complete Python suite
and physical package builds were not rerun; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree representative version-control activation evidence

The next bounded documentation iteration refreshed the shared header-version
reference at Transformers commit
`af0993dda925a8cac0a590f6e43a239933cc6d5b`. Its 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. No new remote rendered version-menu
interaction was executed, so prior official shell evidence remains separate.

The observable contract was fixed before implementation. All ten
representative routes must activate and close the version control at desktop,
tablet, and mobile widths in both palettes. The 30 default-palette cases use
unmodified Enter and the 30 slate cases use the pointer. Each opened state must
expose the exact three labels and destinations, one current item, the open and
expanded state, summary focus, a visible menu inside the viewport, zero
overflow, and a zero-violation Axe result. Escape must close and hide the menu,
restore the collapsed ARIA state, preserve zero overflow, and return focus to
the version summary.

The focused source contract intentionally failed before implementation because
the checker had no version activation matrix, helper, or counters. The first
rendered probe then exposed a real geometry failure: the desktop and tablet
menu was right-aligned to a 64-pixel left-rail control and extended outside the
viewport. The minimal responsive CSS correction left-aligns the LTR menu and
right-aligns the RTL menu inside that rail while preserving mobile placement.
A fresh six-case Home breakpoint/palette probe passed. Neither failed run is
counted as passing evidence.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,704 stops, 30 expanded-state Axe cases, four
branch/drawer activation cases, 60 version activation/closure cases, 60 search
activation/closure cases, and 40 TOC activation cases. Every opened version
state received another Axe pass. The complete keyboard inventory increased
from 196 to 226 cases.

The complete documentation suite reported 57 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current. The five-record release-alignment check passed, and
refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes version activation and Escape restoration across the
representative local route/viewport/palette matrix. Language, theme, and source
route-specific activation remain outside this slice. The complete Python suite
and physical package builds were not rerun; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree representative language-control activation evidence

The next bounded documentation iteration refreshed the shared header-language
reference at Transformers commit
`af0993dda925a8cac0a590f6e43a239933cc6d5b`. Its 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. No new remote rendered
language-switch interaction was executed, so prior official shell evidence
remains separate.

The observable contract was fixed before implementation. The language control
is intentionally hidden at the shared mobile breakpoint, so all ten
representative routes must switch locale at desktop and tablet widths in both
palettes. Each of the 40 cases must expose one visible, focusable select with
the exact ordered EN, TR, ES, FR, DE, PT, ZH, JA, KO, RU, and AR inventory and
production-base destination for the current route. The 20 default-palette
cases use unmodified ArrowDown to select Turkish; the 20 slate cases use
pointer activation and semantic native-option selection to select Arabic. The
target must retain the palette, route, selected locale, LTR or RTL direction,
zero overflow, and a zero-violation Axe result.

The focused source contract intentionally failed before implementation because
the checker had no language activation matrix, helper, or counters. Rendered
probes then exposed that the local server did not mount `/voicehub/`, headless
Chromium did not surface native picker key events, and a locale navigation
reset slate to default. The test server now serves the production base, the
native select explicitly commits unmodified ArrowUp and ArrowDown changes, and
a one-navigation `sessionStorage` transfer restores the chosen palette before
removing its temporary key. Follow-up failures corrected exact native-option
selection, body-owned direction, and nested-route mounting. The first full
matrix failed on that nested mount before the corrected four-case Installation
probe passed. None of the failed or timed-out runs is counted as passing
evidence.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,752 stops, 30 expanded-state Axe cases, four
branch/drawer activation cases, 40 language switches, 60 version
activation/closure cases, 60 search activation/closure cases, and 40 TOC
activation cases. Every localized destination received another Axe pass. The
complete keyboard inventory increased from 226 to 246 cases.

The complete documentation suite reported 58 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current. The five-record release-alignment check passed, and
refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps. The selected pre-commit sequence exited zero for
every applicable hook; its Markdown hook reported no matching files and is not
counted as a pass.

This closes language switching across the visible representative local
desktop/tablet matrix. The mobile-hidden state remains covered by all 20 mobile
base cases. Theme and source route-specific activation remain outside this
slice. The complete Python suite and physical package builds were not rerun;
their exact preceding current-worktree evidence remains separate.
Exact-current-worktree remote CI, exact tagged artifact hashes, protected
publisher configuration, tags, publication approval, five native-kernel
hardware gates, and two inaccessible WeNet asset paths remain open. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree representative theme-control activation evidence

The next bounded documentation iteration refreshed the shared theme reference
at Transformers commit `b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`.
Its 65-line `docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. Later repeated rendered probes were
rate-limited with HTTP 429 and are excluded rather than reported as passing
comparison evidence.

The observable contract was fixed before implementation. Because the theme
control is intentionally hidden at the shared mobile breakpoint, all ten
representative routes must switch palette at desktop and tablet widths from
both starting palettes. Twenty default-palette cases use Enter to switch to
`slate`; 20 slate cases use pointer activation to switch to `default`. Each
case requires the exact two toggle labels and target inputs, one visible native
tab stop at the reference-sized 34 by 24-pixel geometry, unchanged route and
English locale, focus on the newly visible toggle, exact target colors, zero
overflow, and a zero-violation Axe result after activation.

The focused source contract intentionally failed before implementation because
the checker had no theme activation matrix, helper, or counters. The first
four-case rendered probe rejected a one-pixel y-coordinate error, and its
corrected rerun passed. The first full matrix then exposed a two-animation-
frame focus race between programmatic palette setup and native Tab traversal.
The setup helper now waits for those exact frames. The previously failing
Models tablet/slate route then passed all 239 focus stops, and a fresh complete
matrix passed. None of the failed or interrupted commands is counted as
passing evidence. The production theme template, JavaScript, and CSS required
no change.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,709 stops, 30 expanded-state Axe cases, four
branch/drawer activation cases, 40 theme switches, 40 language switches, 60
version activation/closure cases, 60 search activation/closure cases, and 40
TOC activation cases. Every switched theme received another Axe pass. The
complete keyboard inventory increased from 246 to 266 cases.

The complete documentation suite reported 59 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current. The five-record release-alignment check passed, and
refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps. The selected pre-commit sequence exited zero for
every applicable hook; its Markdown hook reported no matching files and is not
counted as a pass.

This closes palette switching across the visible representative local
desktop/tablet matrix. The mobile-hidden state remains covered by all 20 mobile
base cases. Source-link route-specific activation remains outside this slice.
The complete Python suite and physical package builds were not rerun; their
exact preceding current-worktree evidence remains separate. Exact-current-
worktree remote CI, exact tagged artifact hashes, protected publisher
configuration, tags, publication approval, five native-kernel hardware gates,
and two inaccessible WeNet asset paths remain open. The untracked `uv.lock`
stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree representative source-link activation evidence

The next bounded documentation iteration refreshed the shared source-control
reference at Transformers commit
`b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`. Its 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200.

The observable contract was fixed before implementation. Because the source
link is intentionally hidden at the shared mobile breakpoint, all ten
representative routes must activate it at desktop and tablet widths in both
palettes. Twenty default-palette cases use Enter and 20 slate cases use a
pointer. Each requires exactly one native tab stop named `Open VoiceHub source
repository`, the exact `https://github.com/kadirnar/voicehub` target, 55 by
16-pixel geometry at x = 198 and y = 155, a two-pixel focus outline, route and
palette stability before activation, zero overflow, and a zero-violation Axe
result. The checker then requires exact browser navigation while intercepting
the declared external target with a deterministic fixture; GitHub availability
and GitHub-page accessibility are not reported as local passes.

The focused contract first failed because no source activation matrix, helper,
or counters existed. A later focused assertion rejected a quote-sensitive
checker fragment and was corrected before the contract passed. The four-case
Home render then passed both palettes and visible viewports. The first two full
matrix attempts exposed a real sequential-focus race: closing Material search
after native Tab could asynchronously restore focus to the search input and
skip the source link. Neither run is counted. The production header control now
cancels only that delayed restoration when Tab leaves the desktop input;
Escape closure still restores focus. A 20-cycle focused probe and a fresh full
matrix passed after the strict rebuild.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,621 stops, 30 expanded-state Axe cases, four
branch/drawer activation cases, 40 source navigations, 40 theme switches, 40
language switches, 60 version activation/closure cases, 60 search
activation/closure cases, and 40 TOC activation cases. Every focused source
link received its own local-page Axe pass. The complete keyboard inventory is
now 286 cases.

The complete documentation suite reported 60 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current. The five-record release-alignment check passed, and
refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes source-link activation across the visible representative local
desktop/tablet matrix; all 20 mobile base cases retain the intentional hidden
state. Non-representative routes and future shared controls remain outside the
matrix. The complete Python suite and physical package builds were not rerun;
their exact preceding current-worktree evidence remains separate. Exact-
current-worktree remote CI, exact tagged artifact hashes, protected publisher
configuration, tags, publication approval, five native-kernel hardware gates,
and two inaccessible WeNet asset paths remain open. The untracked `uv.lock`
stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree root left-navigation activation evidence

The next bounded documentation-shell iteration refreshed the official
navigation reference at Transformers commit
`b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`. Its 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official rendered Home route returned HTTP 200 and presents the same eight
ordered top-level disclosures as the VoiceHub rail.

The observable contract was fixed before implementation. On the Pipeline
route, every root branch must activate and restore at desktop and tablet widths
in both palettes. Sixteen default-palette cases use Enter and 16 slate cases
use a pointer. Each case requires the exact eight-label order, native checked
state and disclosure ARIA, correctly labeled panel visibility, the active
Pipeline link and unchanged route, palette stability, visible reference rail
geometry, zero horizontal overflow, and an Axe pass whenever the branch is
expanded. Keyboard activation also requires the two-pixel visible focus ring;
pointer activation retains DOM focus without incorrectly displaying that
keyboard-only ring. The mobile drawer remains covered separately.

The focused contract first failed because no root activation matrix, helper,
or counters existed. Three rendered probes then rejected, respectively, a
desktop coordinate applied to tablet, an expectation that ignored the final
API control's legitimate eight-pixel focus scroll, and an incorrect demand for
a keyboard-only focus ring after pointer input. Those expectations were fixed
without changing production markup or styling, and none of the failed probes
is counted. The complete 32-case rerun passed with Axe Core 4.12.1.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,737 stops, 30 expanded-state Axe cases, 32
root-branch activation/restoration cases, two mobile drawer activation cases,
40 source navigations, 40 theme switches, 40 language switches, 60 version
activation/closure cases, 60 search activation/closure cases, and 40 TOC
activation cases. The complete keyboard inventory is now 300 cases.

The complete documentation suite reported 61 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current. The five-record release-alignment check passed, and
refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps. The selected pre-commit sequence exited zero for
every applicable hook; its Markdown hook reported no matching files and is not
counted as a pass.

This closes root-branch activation on the representative Pipeline route across
the visible desktop/tablet matrix. Nested branch activation and sticky behavior
outside that recorded route remain pending. The complete Python suite and
physical package builds were not rerun; their preceding exact-current-worktree
evidence remains separate. Exact-current-worktree remote CI, exact tagged
artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree API model-navigation hierarchy evidence

The next bounded documentation iteration refreshed Transformers `main` at
commit `b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`. The 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
That source places Auto Classes under `API → Main Classes` and model guides,
including SpeechT5, under `API → Models`. The official Home and Auto Classes
routes returned HTTP 200. The SpeechT5 shell rendered its selected model link,
but its content pane was rate-limited with a visible 429. That current content
capture is unavailable and is not counted as a page-content pass.

The shared generator writes Auto Classes exactly once under `API → Main
Classes` and all 68 unique model paths under `Base classes → Models`, grouped
as 34 TTS, 23 ASR, and 11 VAD guides. Base classes retains lifecycle and support
material, contribution, preprocessors, and architecture. Auto Classes expands
`API → Main Classes`; SpeechT5 expands `Base classes → Models → Text to speech`.
Model routes remain stable, and the generator is idempotent.

The first complete visual run rejected a stale Quickstart footer expectation
after the correct navigation move; that failed run is not counted. The strict
eleven-language build and ten-route DOM validator then passed. Reviewed local
light and dark captures covered Auto Classes and SpeechT5 at desktop, tablet,
and mobile widths. The complete Playwright 1.62.0, Chromium 151.0.7922.34, and
Axe Core 4.12.1 matrix passed 60 base cases, all 60 existing screenshot
signatures, 60 complete focus cycles with 4,745 stops, 36 nested-branch
activation/restoration cases, and 348 keyboard cases. The final model-route
slice revalidated exact Base classes ancestry, screenshots, Axe, and sticky
navigation behavior after formatting and the final strict build.

The complete documentation suite reported 62 passes and 1,476 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 model pages and 59 model notebooks remain
current. The five-record release-alignment check passed, and refreshed
inventories remain 68 models, 102 aliases, zero invalid display names, six
public optimization passes, 408 model/pass pairs, eight top-level navigation
roots, ten representative routes, and eight contribution steps. The selected
pre-commit sequence passed every applicable hook on its second invocation; the
first invocation changed formatting and is not counted as a pass, while the
Markdown hook found no matching files in either run.

This closes the model-navigation placement gap. The current official SpeechT5
content capture remains rate-limited and unpassed. Nested branch activation and
sticky behavior outside the existing representative matrix remain pending. The
complete Python suite and physical package builds were not rerun for this
navigation-only slice; their preceding exact-current-worktree evidence remains
separate. Exact-current-worktree remote CI, exact tagged artifact hashes,
protected publisher configuration, tags, publication approval, five native-
kernel hardware gates, and two inaccessible WeNet asset paths remain open. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree nested navigation and sticky-rail evidence

The next bounded shell iteration refreshed Transformers `main` at commit
`339c18c08bf0c143b8307c255004506e358984f2`. The 65-line
`docs/source/en/index.md` source remains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` remains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official SpeechT5 route returned HTTP 200 and rendered current content at
desktop, tablet, and mobile widths without the earlier 429 response. The
upstream rail still selects SpeechT5 inside the API model inventory; VoiceHub
retains original speech-specific content and its approved color palette.

The observable contract was fixed before implementation on VoiceHub's deepest
representative route. Its nine visible nested disclosure paths are `Base
classes → Models`, `Base classes → Models → Catalogs and support`, `Base classes
→ Models → Text to speech`, `Base classes → Models → Text to speech → SpeechT5`,
`Base classes → Models → Automatic speech recognition`, `Base classes → Models
→ Voice activity detection`, `Base classes → Models → Contribute`, `Base classes
→ Preprocessors`, and `Base classes → Architecture`. Each must activate and
return to its
exact initial checked, `aria-expanded`, controlled-panel, focus, route, active-
link, palette, and overflow state at desktop and tablet widths in both
palettes. Eighteen default-palette cases use Enter and 18 slate cases use a
pointer. Every target also scrolls the document to 320 pixels and requires the
270-pixel sticky rail to move from top 65 with viewport-minus-65 height to top
0 with full viewport height while the shell offset becomes 65 pixels. Axe runs
in all 24 sticky target states.

The focused source test first failed because no nested matrix, helper, or
counters existed. The first rendered case then rejected a checker query that
included the active `Usage` table-of-contents link after scrolling; that run is
not a pass. The corrected query excludes embedded secondary navigation. The
first complete nested matrix reached the active SpeechT5 disclosure and then
failed Axe with four `landmark-unique` violations because its expanded page
table of contents duplicated unnamed or identically named subsection
landmarks. That run is also failed evidence. The shared navigation initializer
now gives every nested page-section landmark a deterministic SpeechT5-scoped
accessible name before the panel opens. The corrected 24-case focused render
passed with Axe Core 4.12.1.

The strict eleven-language build and ten-route DOM validator passed. The full
Playwright 1.62.0, Chromium 151.0.7922.34, and Axe Core 4.12.1 matrix passed 60
base cases, all 60 existing screenshot signatures, 60 complete native focus
cycles with 4,628 stops, 30 expanded-state Axe cases, 32 root-branch cases, 24
nested-branch/sticky cases, two mobile drawer cases, 40 source navigations, 40
theme switches, 40 language switches, 60 version cases, 60 search cases, and
40 TOC cases. The complete keyboard inventory is now 312 cases.

The complete documentation suite reported 63 passes and 1,476 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated model pages and 59 notebooks remain
current, and the five-record release-alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes nested disclosure activation and sticky-rail behavior on the
deepest representative model route. Non-representative navigation structures,
remaining page-action states, the object-by-object public API audit, and exact
comparison with the current Modular Transformers contribution path remain
open. The complete Python suite and physical package builds were not rerun for
this documentation-only slice; their preceding exact-current-worktree evidence
remains separate. Exact-current-worktree remote CI, exact tagged artifact
hashes, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and two inaccessible WeNet asset paths also
remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree representative page-action evidence

The page-action iteration refreshed Transformers `main` at commit
`179a24360e55f1daff1bc20f0d11f4e8f47a6f44`. The official 65-line
`docs/source/en/index.md` and 1,576-line `_toctree.yml` retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b` and
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The current SpeechT5 page returned HTTP 200 and its rendered source contains
the shared Copy page control and `DocFooterNav` adjacency for Speech2Text and
UniSpeech. That is current structural reference evidence, not a claim that an
external page or destination passed VoiceHub's local accessibility checks.

The observable local contract covers all ten representative routes at desktop,
tablet, and mobile widths in both palettes. Thirty default-palette cases use
Enter and 30 slate cases use a pointer. Each case requires exactly one native
edit link, the route's exact optional previous link and exact next link, and
one native Back to top button. Edit activation must navigate to the declared
GitHub URL through a deterministic local intercept. Footer activation must
perform real local browser navigation to every declared destination. Back to
top must appear after the documented down-then-up scroll direction, expose a
visible two-pixel focus outline, return to scroll position zero, preserve the
route and palette, introduce no horizontal overflow, and pass Axe in its
visible focused state.

The focused source contract first failed because no page-action inventory,
helper, matrix, or counters existed. A bare `python` invocation was unavailable
in the shell and is not test evidence; the repository virtual environment then
produced the intended failing regression. The first rendered case timed out
because Material reveals Back to top only after upward scroll. The corrected
probe then exposed a real missing focus outline on that button. After the
shared edit/footer/top focus rule was added, Axe caught a transient 2.05:1
contrast state during the declared 125-millisecond color/background
transition. None of those runs is counted as passing. The final checker models
the directional reveal and waits 150 milliseconds for the transition to
settle before the accessibility audit.

The complete Playwright 1.62.0, Chromium 151.0.7922.34, and Axe Core 4.12.1
matrix passed 60 base cases, all 60 screenshot signatures, 60 complete focus
cycles with 4,576 stops, 30 expanded-state Axe cases, 32 root disclosures, 24
nested sticky disclosures, two mobile drawer activations, and the existing
source, theme, language, version, search, and TOC matrices. The new slice
passed 60 page-action cases, 60 exact edit activations, 114 exact footer
navigations, 60 Back to top activations, and 60 focused interaction-state Axe
audits. Its 30 keyboard cases raise the aggregate keyboard inventory to 342.

The strict eleven-language build, ten-route DOM validator, 64 documentation
tests with 1,476 subtests, and the 97-test release-risk slice with 522 subtests
passed. All 68 generated model pages and 59 generated notebooks remain current,
and the five-record release-alignment check passed. A fresh distribution check
passed wheel, sdist, and editable installs with 68 models, 81 provenance
manifests, 193 compliance files, required package data, zero runtime dependency
violations, and no eager PyTorch import. It produced a 57,192,297-byte wheel
and a 55,450,920-byte sdist. A separate local fingerprint build produced:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `voicehub-0.3.0-py3-none-any.whl` | 57,192,297 | `d6fc30aa0d037d6624425e877fac6d5d356ad07bc901adccdc72b8c930304d98` |
| `voicehub-0.3.0.tar.gz` | 55,450,952 | `2a9becc4014c7e487eb1a60d1cd1009977dfd8df2f493397b8c7296d522326a7` |

The first selected pre-commit sequence is not a pass: docformatter exited 3
after YAPF collapsed three multiline JavaScript object callbacks. A standalone
docformatter run restored them, proving a formatter oscillation rather than a
source failure. Those callbacks now use equivalent concatenated strings. The
complete selected sequence and `git diff --check` then exited zero; its
Markdown hook had no matching files and is not counted as a pass.

This closes actual page-action behavior on the complete representative local
matrix. Non-representative routes and future controls remain outside that
matrix, and the deterministic edit intercept does not make any claim about
GitHub's page content or availability. Exact-current-worktree remote Linux and
Windows CI, exact tagged-workflow artifacts, protected publisher configuration,
tags, publication approval, five native-kernel hardware gates, and two
inaccessible WeNet asset paths remain open. The untracked `uv.lock` stayed
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree AutoConfig subfolder evidence

This bounded public-lifecycle iteration used Transformers `main` commit
`179a24360e55f1daff1bc20f0d11f4e8f47a6f44`. Its current
`AutoConfig.from_pretrained()` delegates configuration discovery to
`PreTrainedConfig.get_config_dict()`, whose `_get_config_dict()` consumes
`subfolder` and passes it to the cached-file resolver. VoiceHub's concrete
configuration loader already followed that contract, but automatic model-type
discovery omitted `subfolder` and searched only the repository root.

The observable regression uses a local artifact whose only `config.json` is
`nested/config.json`. `AutoConfig.from_pretrained(root, subfolder="nested")`
must discover and restore its canonical `vits` type, and
`AutoModelForTextToSpeech.from_pretrained()` must preserve the same path through
`config_kwargs` without loading the model runtime. Both tests first failed at
the root lookup with `FileNotFoundError`; after the discovery resolver received
the existing loader option, both passed. A bare `python` invocation failed with
exit 127 because that shell alias is absent and is not counted as test
evidence; the repository virtual environment produced the intended red and
green runs.

The focused auto-configuration file passed 16 tests. The configuration, Hub,
pipeline, inference, task-registry, and registry slice passed 141 tests and 364
subtests. The registry-related optimization slice passed 55 tests and 204
subtests. Documentation passed 64 tests and 1,476 subtests, the strict
eleven-language build, the ten-route DOM validator, all 68 model-page and 59
notebook freshness checks, and the complete Playwright/Axe matrix. That matrix
reported 60 base and screenshot cases, 60 focus cycles with 4,583 stops, 342
keyboard cases, and all existing copy, search, version, language, theme,
source, TOC, root, nested, and page-action interaction inventories. The first
DOM invocation used an unsupported `--site-dir` option, exited 2, and is not a
pass; the positional invocation then passed.

The complete Python 3.12.12 suite passed 2,499 executed tests and 3,894
subtests with 35 warnings in 105.59 seconds. Its 15 skips are not passes:

- three complete-dependency import checks remain assigned to the
  default-runtime CI job;
- three Triton and two compiled CUDA-extension checks require their declared
  CUDA hosts;
- ESPNet, NeMo QuartzNet, SenseVoice, SpeechBrain, and TEN-VAD each require
  their declared opt-in release asset or checkpoint; and
- WeNet conversion and tokenizer validation require two currently unavailable
  asset paths.

Some opt-in assets have separate earlier successful evidence in the candidate
matrix; this run does not relabel its skips as successes. The release,
packaging-metadata, and distribution-compliance slice passed 23 tests and 76
subtests. A fresh physical distribution check passed wheel, sdist, and editable
installs with 68 models, 81 provenance manifests, 193 compliance files,
required package data, zero runtime dependency violations, and no eager
PyTorch import. It produced a 57,192,308-byte wheel and a 55,451,111-byte
sdist.

The selected-file pre-commit sequence passed every applicable source hook in
one invocation. Its Markdown hook found no matching files and is not counted
as a pass.

This closes automatic configuration discovery from local or Hub subfolders and
its task-factory propagation. The broader object-by-object public API audit
remains open. Exact-current-worktree Linux, Windows, Python 3.10, Python 3.11,
default-runtime, and tagged-workflow evidence; protected publisher
configuration; tags; publication approval; the five CUDA/Triton gates; and the
listed opt-in asset paths remain pending. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree configuration-secret boundary evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`179a24360e55f1daff1bc20f0d11f4e8f47a6f44`; the 1,576-line navigation and
500-line Modular Transformers source retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a` and
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The upstream revision is reference context. VoiceHub's safe checkpoint and
runtime-credential policy is the controlling speech-domain requirement for
this slice.

The observable contract requires `VoiceHubConfig` construction to reject
top-level and nested credential-shaped fields, rejects secrets attached after
construction before dictionary or file serialization, and validates the final
payload returned by a subclass before diff, JSON, representation, or checkpoint
output. A concrete loader's top-level Hub `token` remains runtime-only and is
omitted, while model fields such as `pad_token_id` remain serializable. A
failed checkpoint write must not create `config.json`.

The first focused run proved the gap: two base tests and 35 of 68 registry
subtests failed, while 33 configurations passed only because they carried
their own local protection. The shared constructor and dictionary boundary
then made all 68 subtests pass. The first broader base/registry run is not a
pass: four TTS configs intentionally accepted runtime Hub tokens and attempted
to remove them after the new base check, producing seven failures. The common
boundary now omits only the top-level runtime `token` before validating all
remaining state; no provider branch was added. A separate malicious-subclass
regression first showed that final serialization and `repr` could still expose
a secret added after the base call. Final-payload validation closed those
paths. The focused result is five tests with 68 registry subtests passed.

The configuration, processing, inference, and provider security slice passed
225 tests and 348 subtests. The registry, universal-optimization, packaging,
distribution-compliance, and release-policy slice passed 95 tests and 473
subtests. The complete Python 3.12.12 suite passed 2,504 executed tests and
3,962 subtests with 35 warnings in 106.84 seconds. Its 15 skips remain
unpassed: three complete-dependency default-runtime CI checks, three Triton
and two CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/
TEN-VAD asset or oracle checks, and two inaccessible WeNet paths.

Documentation passed 64 tests and 1,476 subtests, all 68 model-page and 59
model-notebook freshness checks, the strict eleven-language build, and the
ten-route DOM validator. The complete Playwright 1.62.0, Chromium
151.0.7922.34, and Axe Core 4.12.1 matrix passed all 60
route/viewport/palette cases, 60 screenshots, and 60 focus cycles with 4,610
visible focus steps and 342 keyboard cases. Its complete interaction inventory
passed search, version, language, theme, source, table-of-contents, disclosure,
page-action, copy, and accessibility paths. A fresh physical distribution check
passed wheel, sdist, and editable installs with 68 models, 81 provenance
manifests, 193 compliance files, required package data, zero runtime dependency
violations, and no eager PyTorch import. It produced a 57,192,400-byte wheel
and a 55,451,994-byte sdist.

The first selected-file pre-commit invocation is not a pass because YAPF
rewrote the new code. The focused contract passed after that rewrite, and the
second invocation passed every applicable source hook. The Markdown hook found
no matching files and is not counted as a pass.

This closes the shared configuration credential boundary for the complete
registry. The broader object-by-object public API audit remains open.
Exact-current-worktree Linux, Windows, Python 3.10, Python 3.11,
default-runtime, tagged-workflow, and publisher evidence; publication
approval; five CUDA/Triton gates; and the seven opt-in asset paths remain
pending. No protected action was taken. The untracked `uv.lock` stayed
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree generation-configuration secret boundary evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`d79d0c1ed3e8fc3f4742cc34aee46d92abd0ea44`. Its 1,576-line navigation and
500-line Modular Transformers source retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a` and
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The upstream revision is structural context; VoiceHub's safe speech-checkpoint
boundary controls this slice.

The observable contract requires standalone `TTSGenerationConfig`
construction and untrusted checkpoint loading to reject top-level or nested
credential-shaped fields. Secrets attached later must fail before dictionary,
representation, or file serialization, and an unsafe subclass payload must be
revalidated by representation and checkpoint output. A failed write must not
create `generation_config.json`. The Hub loader `token` remains runtime-only,
while legitimate model settings such as `pad_token_id`, `eos_token_id`, and
`max_new_tokens` remain serializable. Embedded generation defaults must enter
the same configuration constructor boundary for every registered model.

A direct pre-implementation probe confirmed that standalone generation
configuration serialized a nested `api_key`. The first focused contract is not
a pass: four tests failed on embedded construction, standalone construction,
post-construction serialization, and final subclass output, while the existing
runtime-only Hub token test passed. A separate pre-fix registry inventory found
18 configurations that still accepted the embedded secret at construction;
39 rejected it through the existing shared boundary and 11 through local
credential-specific checks. The common explicit-field check now rejects all 68
at construction without a provider branch.

After correction, the focused contract passed seven tests and 68 registry
subtests. The broader generation, configuration, auto-loading, pipeline,
task-registry, and pretrained TTS slice passed 135 tests and 314 subtests. The
dedicated registry and public-optimization slice passed 169 tests and 486
subtests with four recorded framework warnings. The complete Python 3.12.12
suite passed 2,510 executed tests and 4,030 subtests with 35 warnings in 106.64
seconds. Its 15 skips remain unpassed: three complete-dependency
default-runtime checks, three Triton and two CUDA-extension checks, five opt-in
ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks, and two
inaccessible WeNet paths.

Documentation and scaffold validation passed 93 tests and 1,511 subtests; all
68 model pages, 68 model navigation entries, 59 model notebooks, and five
benchmark records remained current. The strict eleven-language build and the
ten-route DOM validator passed. The unchanged documentation shell retains the
immediately preceding complete Playwright/Axe evidence; the visual matrix was
not rerun for this API-safety and prose-only slice and is not presented as a
new pass.

A fresh physical distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required package
data, zero runtime dependency violations, and no eager PyTorch import. It
produced a 57,192,472-byte wheel and a 55,453,207-byte sdist. The
first selected-file pre-commit invocation is not a pass because YAPF rewrote
the new test formatting. The formatted focused contract passed, and the second
invocation passed every applicable source hook. The Markdown hook found no
matching files and is not counted as a pass.

This closes credential handling for both embedded and standalone generation
configuration. The broader object-by-object public API audit remains open.
Exact-current-worktree Linux, Windows, Python 3.10, Python 3.11,
default-runtime, tagged-workflow, and publisher evidence; publication approval;
five CUDA/Triton gates; and the seven opt-in asset paths remain pending. No
protected action was taken. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree native-manifest secret boundary evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`d79d0c1ed3e8fc3f4742cc34aee46d92abd0ea44`. Its 1,576-line navigation and
500-line Modular Transformers source retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a` and
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The upstream revision is structural context; VoiceHub's safe portable-artifact
boundary controls this slice.

The observable contract requires `VoiceHubManifest.metadata` construction and
untrusted manifest loading to reject top-level or nested credential-shaped
fields. Because the frozen manifest contains a mutable metadata mapping,
`to_dict()` and `save()` must revalidate post-construction changes. Save must
also validate the final payload returned by a subclass before creating the
artifact directory, manifest, or temporary file. Safe descriptive fields such
as `token_count` must continue to round-trip.

The first focused contract is not a pass: all four new tests failed because
construction, untrusted loading, post-construction dictionary output, and a
malicious subclass save accepted secrets. The shared secret detector was moved
to the shallow serialization layer while remaining available from the existing
configuration module for compatibility. Manifest construction and dictionary
output now validate metadata, and save validates the complete final payload
before any filesystem mutation. The corrected focused contract passed four
tests with 12 deselected. The checkpoint, configuration, inference, and task
slice passed 132 tests and 294 subtests. The registry, public-optimization,
checkpoint, distribution-compliance, and release-policy slice passed 269 tests
and 1,339 subtests with four framework warnings.

Documentation passed 64 tests and 1,476 subtests; all 68 model pages and 59
model notebooks remained current. The strict eleven-language build and
ten-route DOM validator passed. The unchanged documentation shell retains the
immediately preceding complete Playwright/Axe evidence; the visual matrix was
not rerun for this API-safety and prose-only slice and is not presented as a
new pass. An initial notebook command referenced a nonexistent
`scripts/check_model_notebooks.py`, exited 2, and is not a pass; the repository's
actual `generate_model_notebooks.py --check` command then passed.

The complete Python 3.12.12 suite passed 2,514 executed tests and 4,030
subtests with 35 warnings in 105.83 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two
CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD
asset or oracle checks, and two inaccessible WeNet paths. A fresh physical
distribution check passed wheel, sdist, and editable installs with 68 models,
81 provenance manifests, 193 compliance files, required package data, zero
runtime dependency violations, and no eager PyTorch import. It produced a
57,192,624-byte wheel and a 55,452,570-byte sdist. The selected-file pre-commit
invocation passed every applicable source hook; its Markdown hook found no
matching files and is not counted as a pass.

This closes credential handling for native manifest metadata and keeps the
credential detector in one serialization contract. The broader
object-by-object public API audit remains open. Exact-current-worktree Linux,
Windows, Python 3.10, Python 3.11, default-runtime, tagged-workflow, and
publisher evidence; publication approval; five CUDA/Triton gates; and the
seven opt-in asset paths remain pending. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree ASR/VAD inference-configuration secret boundary evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`d79d0c1ed3e8fc3f4742cc34aee46d92abd0ea44`. Its 1,576-line navigation and
500-line Modular Transformers source retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a` and
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The upstream revision is structural context; VoiceHub's safe public task-
configuration boundary controls this slice.

The observable contract requires ASR and VAD inference configuration
construction and untrusted checkpoint loading to reject top-level or nested
credential-shaped fields. Mutable state must be rechecked, and representation
and checkpoint output must validate the final payload returned by a subclass.
A failed write must not create `transcription_config.json`, `vad_config.json`,
or their artifact directory. The Hub loader `token` remains runtime-only,
while legitimate task fields such as `max_new_tokens` continue to round-trip.

The pre-implementation focused run is not a pass: two tests failed because a
malicious subclass payload reached representation and checkpoint output; the
untrusted-load test passed with two task subtests. Final-payload validation was
added to the shared `SpeechInferenceConfig` representation and save boundaries
without a task or provider branch. The corrected focused contract passed five
tests and two subtests. The speech-core, base-API, inference-contract, and task-
registry slice passed 120 tests and 319 subtests. The registry, native-artifact,
universal-optimization, documentation, scaffold, release-policy, and
distribution-compliance slice passed 173 tests and 1,934 subtests.

The complete Python 3.12.12 suite passed 2,517 executed tests and 4,032
subtests with 35 warnings in 120.34 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two CUDA-
extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset
or oracle checks, and two inaccessible WeNet paths.

All 68 model pages and 59 model notebooks remained current. The strict eleven-
language build and ten-route DOM validator passed. The unchanged documentation
shell retains the immediately preceding complete Playwright/Axe evidence; the
visual matrix was not rerun for this API-safety and prose-only slice and is not
presented as a new pass. A fresh physical distribution check passed wheel,
sdist, and editable installs with 68 models, 81 provenance manifests, 193
compliance files, required package data, zero runtime dependency violations,
and no eager PyTorch import. It produced a 57,192,647-byte wheel and a
55,453,587-byte sdist. The selected-file pre-commit invocation passed every
applicable source hook; its Markdown hook found no matching files and is not
counted as a pass.

This closes final-payload credential handling for public ASR and VAD inference
configuration. The broader object-by-object public API audit remains open.
Exact-current-worktree Linux, Windows, Python 3.10, Python 3.11,
default-runtime, tagged-workflow, and publisher evidence; publication approval;
five CUDA/Triton gates; and the seven opt-in asset paths remain pending. No
protected action was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree processor secret boundary evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. The rendered Home and Modular
Transformers routes returned current content. The 1,576-line navigation and
500-line Modular Transformers source retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a` and
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The upstream revision is structural context; VoiceHub's safe processor-
artifact boundary controls this slice. An initial read-only refresh command
was rejected before execution because it included a disallowed cleanup form;
it is not evidence. The corrected command used a temporary directory without
that cleanup and produced the revision and fingerprints above.

The observable contract requires `VoiceHubProcessor` and `AudioProcessor`
construction and untrusted artifact loading to reject top-level or nested
credentials. Dictionary and checkpoint serialization must recheck mutable
state, and save must validate the final mapping returned by a subclass before
creating `processor_config.json` or its artifact directory. A Hub loader
`token` remains runtime-only, while safe construction fields such as
`normalization` continue to round-trip.

The pre-implementation focused run is not a pass: four tests plus four task
subtests passed, while the malicious-subclass save test failed because its
final payload reached the file writer. The shared processor save boundary now
validates that complete mapping before filesystem mutation, without a task or
provider branch. The corrected focused contract passed five tests and four
subtests. The first selected pre-commit invocation is not a pass because YAPF
rewrote the new test; the formatted focused contract and the second invocation
then passed every applicable source hook. Its Markdown hook found no matching
files and is not counted as a pass.

The base API, auto-loading, speech-core, task-registry, inference, pipeline,
and registry slice passed 165 tests and 524 subtests. The universal-
optimization, documentation, scaffold, release-policy, and distribution-
compliance slice passed 156 tests and 2,104 subtests. The complete Python
3.12.12 suite passed 2,522 executed tests and 4,036 subtests with 35 warnings
in 122.47 seconds. Its 15 skips remain unpassed: three complete-dependency
default-runtime checks, three Triton and two CUDA-extension checks, five opt-in
ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks, and two
inaccessible WeNet paths.

All 68 model pages and 59 model notebooks remained current. The strict eleven-
language build and ten-route DOM validator passed. The unchanged documentation
shell retains the immediately preceding complete Playwright/Axe evidence; the
visual matrix was not rerun for this API-safety and prose-only slice and is not
presented as a new pass. A fresh physical distribution check passed wheel,
sdist, and editable installs with 68 models, 81 provenance manifests, 193
compliance files, required package data, zero runtime dependency violations,
and no eager PyTorch import. It produced a 57,192,663-byte wheel and a
55,453,948-byte sdist.

This closes final-payload credential handling for public processor artifacts.
The broader object-by-object public API audit remains open. Exact-current-
worktree Linux, Windows, Python 3.10, Python 3.11, default-runtime, tagged-
workflow, and publisher evidence; publication approval; five CUDA/Triton
gates; and the seven opt-in asset paths remain pending. No protected action was
taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree TrainerState secret boundary evidence

This bounded public-API safety iteration refreshed Transformers `main` at
commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 1,576-line navigation
and 500-line Modular Transformers sources retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and `a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The current 784-line upstream `trainer_callback.py` has SHA-256
`a9703b60f3d585627054ed976eca3dbea69fddcfd4eb01b1cc19bff67bccf00a`.
That dataclass and JSON shape are structural context; VoiceHub's safe portable-
checkpoint boundary controls this slice.

The observable contract requires nested credential-shaped values to fail at
`TrainerState` construction and untrusted JSON loading. Because `log_history`
is mutable, the final state write must revalidate post-construction changes and
a subclass's complete dataclass payload before creating the output path.
`Trainer.log()` must reject the same values before state mutation or callback
dispatch. Safe metric names such as `loss`, `step`, and `token_count` must
continue to round-trip.

The pre-implementation focused contract is not a pass: all five new tests
failed because construction and loading accepted nested secrets, mutated and
subclass payloads reached the file writer, and unsafe live logs reached state
and callbacks. The shared serialization detector now validates state at
construction and immediately before output; the live log path validates the
complete normalized record before mutation. No task or provider branch was
added. The corrected focused contract passed five tests with 19 deselected,
and the complete trainer file passed 24 tests and 14 subtests. The training,
checkpoint, callback, and integration slice passed 185 tests and 188 subtests.
The registry, speech-task, universal-optimization, packaging, distribution-
compliance, and release-policy slice passed 99 tests and 658 subtests.

The complete Python 3.12.12 suite passed 2,527 executed tests and 4,036
subtests with 35 warnings in 129.02 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two CUDA-
extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset
or oracle checks, and two inaccessible WeNet paths.

Documentation passed 64 tests and 1,476 subtests. All 68 model pages and 59
model notebooks remained current, the five-record release-alignment check
passed, and the strict eleven-language build completed in 38.60 seconds. The
ten-route DOM validator retained all eight ordered navigation roots. The
unchanged documentation shell retains the immediately preceding complete
Playwright/Axe evidence; the visual matrix was not rerun for this API-safety
and prose-only slice and is not presented as a new pass.

A fresh physical distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime dependency violations, and no eager PyTorch import.
It produced a 57,192,782-byte wheel and a 55,454,411-byte sdist. The first
selected-file pre-commit invocation is not a pass because YAPF rewrote the new
test formatting. The corrected focused contract and second invocation passed
every applicable source hook. The Markdown hook found no matching files and is
not counted as a pass.

Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, 68 provider pages with no missing or
orphaned page, six public optimization passes, 408 model/pass pairs, eight
top-level navigation roots, ten representative routes, eight contribution
steps, and 261 public exports. The public root import still avoids PyTorch.

This closes credential handling for public trainer logs and state artifacts.
The broader object-by-object public API audit remains open. Exact-current-
worktree Linux, Windows, Python 3.10, Python 3.11, default-runtime, tagged-
workflow, and publisher evidence; publication approval; five CUDA/Triton
gates; and the seven opt-in or inaccessible asset paths remain pending. No
protected action was taken. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree checkpoint-manifest secret boundary evidence

This bounded public-API safety iteration refreshed Transformers `main` at
commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 1,576-line navigation
and 500-line Modular Transformers sources retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and `a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The pinned 4,460-line upstream `trainer.py` has SHA-256
`6ea719d4f225c6fcbccfcd2137cdfb304426c93c4dd65983ae4186f6e8fb3d8d`.
That Trainer lifecycle is structural context; VoiceHub's safe exact-resume
artifact boundary controls this slice.

The observable contract requires normalized dataset, collator, stateful-
callback, optimizer, and scheduler fingerprints to reject nested credential-
shaped fields without reading or reporting their values. The complete
checkpoint manifest, including output added by a `Trainer` subclass, must be
revalidated immediately before its atomic write. An untrusted manifest must
fail before model or runtime restoration. A rejected save must leave neither a
final checkpoint nor an incomplete checkpoint directory, while safe identity
and metric fields such as `dataset_id` and `token_count` remain serializable.

The pre-implementation focused command is not a pass: two tests and five
credential-bearing subtests failed because unsafe fingerprints, subclass
manifest output, and an untrusted manifest were accepted. Its safe fingerprint
assertion completed, but the failed command is excluded from passing evidence.
The shared fingerprint normalizer, final manifest write boundary, and
untrusted manifest read boundary now apply the same credential detector without
a task or provider branch. The corrected focused contract passed three tests
and five subtests with 33 deselected. The complete training-runtime file passed
36 tests and five subtests. The training, checkpoint, callback, and integration
slice passed 294 tests and 205 subtests with four warnings.

The registry, public-API, optimization, documentation, scaffold, release-
policy, and distribution-compliance slice passed 326 tests and 2,737 subtests
with four warnings. The complete Python 3.12.12 suite passed 2,530 executed
tests and 4,041 subtests with 35 warnings in 127.62 seconds. Its 15 skips remain
unpassed: three complete-dependency default-runtime checks, three Triton and
two CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-
VAD asset or oracle checks, and two inaccessible WeNet paths.

All 68 model pages and 59 model notebooks remained current, and the five-
record release-alignment check passed. The strict eleven-language build
completed in 37.39 seconds. The ten-route DOM validator retained all eight
ordered navigation roots. An initial DOM invocation used an unsupported
`--site-dir` option and is not evidence; the corrected positional invocation
passed. The unchanged visual shell retains its preceding Playwright/Axe
evidence; no fresh visual-parity claim is made for this API-safety and prose-
only slice.

A fresh physical distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime dependency violations, and no eager PyTorch import.
It produced a 57,192,833-byte wheel and a 55,455,064-byte sdist. The selected-
file pre-commit invocation passed every applicable source hook. Its Markdown
hook found no matching files and is not counted as a pass.

Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, 68 provider pages with no missing or
orphaned page, six public optimization passes, 408 model/pass pairs, eight
top-level navigation roots, ten representative routes, eight contribution
steps, and 261 public exports. The public root import still avoids PyTorch.

This closes credential handling for exact-resume checkpoint manifests. The
broader object-by-object public API audit remains open. Exact-current-worktree
Linux, Windows, Python 3.10, Python 3.11, default-runtime, tagged-workflow, and
publisher evidence; publication approval; five CUDA/Triton gates; and the
seven opt-in or inaccessible asset paths remain pending. No protected action
was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree binary checkpoint-state secret boundary evidence

This bounded public-API safety iteration refreshed Transformers `main` at
commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 1,576-line navigation,
500-line Modular Transformers guide, and 4,460-line Trainer source retain
SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
and `6ea719d4f225c6fcbccfcd2137cdfb304426c93c4dd65983ae4186f6e8fb3d8d`.
The official Home and Modular Transformers routes returned HTTP 200. That
Trainer lifecycle is structural context; VoiceHub's exact-resume binary-state
boundary controls this slice.

The observable contract requires optimizer, scheduler, random-generator,
gradient-scaler, callback, sampler, and strategy state mappings to reject
nested credential-shaped fields at their final binary write boundary. A
rejected write must leave neither a final nor an incomplete checkpoint, and
the error must expose only the owning state and key path. Loaded binary state
must be checked before any model or runtime state is restored. Safe metric
fields such as `token_count` must still round-trip through exact resume. The
post-deserialization check is explicitly not a safe-unpickling claim: Python
pickle input must come from a trusted VoiceHub checkpoint with intact manifest
integrity.

The first command used an unavailable `python` executable and ran no tests; it
is not evidence. The corrected pre-implementation command is also not a pass:
it reported three failures alongside two passing results because unsafe
callback and optimizer state completed a checkpoint and loaded optimizer state
was reached only after model restoration. The shared final-payload helper now
validates every Trainer-owned binary state mapping immediately before
`torch.save`. The load path deserializes and validates all non-model state,
including `TrainerState`, before applying the model or any runtime state. No
task, provider, optimizer, callback, or strategy allowlist was added.

The corrected focused contract passed three tests and two subtests with 36
deselected. The complete training-runtime file passed 39 tests and seven
subtests. The training, checkpoint, callback, and integration slice passed 306
tests and 207 subtests with four warnings. The registry, public-API,
optimization, scaffold, packaging, distribution-compliance, and release-
policy slice passed 319 tests and 1,454 subtests with four warnings.

The first documentation run is not a pass: its new warning pushed the concise
training guide to 253 lines and one contract failed. The warning was condensed
without weakening it; the corrected documentation file passed 64 tests and
1,476 subtests with the guide restored to its 250-line limit. The complete
Python 3.12.12 suite then passed 2,533 executed tests and 4,043 subtests with
35 warnings in 113.29 seconds. Its 15 skips remain unpassed: three complete-
dependency default-runtime checks, three Triton and two CUDA-extension checks,
five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks,
and two inaccessible WeNet paths.

All 68 model pages and 59 model notebooks remained current, and the five-
record release-alignment check passed. The strict eleven-language build
completed in 32.93 seconds. The ten-route DOM validator retained all eight
ordered navigation roots. The unchanged documentation shell retains its
preceding Playwright/Axe evidence; no fresh visual-parity claim is made for
this runtime-safety and prose-only slice.

A fresh physical distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime dependency violations, and no eager PyTorch import.
It produced a 57,193,103-byte wheel and a 55,455,792-byte sdist. The selected-
file pre-commit invocation passed every applicable source hook. Its Markdown
hook found no matching files and is not counted as a pass.

Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, 68 provider pages with no missing or
orphaned page, six public optimization passes, 408 model/pass pairs, eight
top-level navigation roots, ten representative routes, eight contribution
steps, and 261 public exports. The public root import still avoids PyTorch.

This closes credential handling for exact-resume binary state mappings. The
broader object-by-object public API audit remains open. Exact-current-worktree
Linux, Windows, Python 3.10, Python 3.11, default-runtime, tagged-workflow, and
publisher evidence; publication approval; five CUDA/Triton gates; and the
seven opt-in or inaccessible asset paths remain pending. No protected action
was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree TrainingArguments secret boundary evidence

This bounded public-API safety iteration refreshed Transformers `main` at
commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 1,576-line navigation
and 500-line Modular Transformers sources retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and `a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The current 2,906-line upstream `training_args.py` has SHA-256
`cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home and Modular Transformers routes returned HTTP 200. The
upstream argument object and its dictionary/JSON lifecycle are the structural
reference; VoiceHub's credential-safe portable artifact boundary controls this
slice.

The observable contract requires nested credential-shaped subclass fields to
fail during normal construction and before an untrusted JSON payload can
construct a subclass. Inherited dictionary, JSON-string, and file writers must
revalidate the complete current dataclass payload even when construction
validation was bypassed. `Trainer.save_model()` must reject the same unsafe
arguments before creating its destination or writing model state. Errors may
report only the owning object and key path. Safe metadata such as `dataset_id`
and `token_count` must continue to round-trip.

The pre-implementation focused command is not a pass: three tests failed while
the safe round-trip case passed because unsafe construction, untrusted loading,
inherited serialization, and trainer export all accepted credential fields.
The shared arguments normalizer now checks its complete dataclass payload at
construction and every inherited persistence boundary. The untrusted loader
checks the payload before subclass construction, and trainer export performs
the same preflight before importing the training backend or creating the
destination. No reporting backend, task, provider, or subclass allowlist was
added.

The corrected focused contract passed four tests and two subtests with 24
deselected. The complete training, checkpoint, W&B, adapter, and optimization
slice passed 180 tests and 212 subtests. The registry, public-API, universal-
optimization, scaffold, packaging-compliance, and release-policy slice passed
151 tests and 621 subtests.

The complete Python 3.12.12 suite passed 2,537 executed tests and 4,045
subtests with 35 warnings in 109.04 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two CUDA-
extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset
or oracle checks, and two inaccessible WeNet paths.

Documentation passed 64 tests and 1,476 subtests. All 68 model pages and 59
model notebooks remained current, the five-record release-alignment check
passed, and the strict eleven-language build completed in 35.53 seconds. The
ten-route DOM validator retained all eight ordered navigation roots. The
unchanged documentation shell retains its preceding Playwright/Axe evidence;
no fresh visual-parity claim is made for this API-safety and prose-only slice.

A fresh physical distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime dependency violations, and no eager PyTorch import.
It produced a 57,193,213-byte wheel and a 55,456,837-byte sdist. The first
selected-file pre-commit invocation is not a pass because YAPF rewrote the new
test formatting. The corrected focused contract and second invocation passed
every applicable source hook. The Markdown hook found no matching files and is
not counted as a pass.

Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, 68 provider pages with no missing or
orphaned page, six public optimization passes, 408 model/pass pairs, eight
top-level navigation roots, ten representative routes, eight contribution
steps, and 261 public exports. The public root import still avoids PyTorch.

This closes credential handling for portable `TrainingArguments` artifacts.
The broader object-by-object public API audit remains open. Exact-current-
worktree Linux, Windows, Python 3.10, Python 3.11, default-runtime, tagged-
workflow, and publisher evidence; publication approval; five CUDA/Triton
gates; and the seven opt-in or inaccessible asset paths remain pending. No
protected action was taken. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree optimization-manifest secret boundary evidence

This bounded public-API safety iteration refreshed Transformers `main` at
commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home and Modular Transformers routes returned HTTP 200. Upstream
is structural context; VoiceHub's safe optimization-artifact boundary controls
this slice.

The observable contract requires credential-shaped pass configuration to fail
before validation or model mutation. Credential-shaped application metadata
must fail before publishing the transformed graph and trigger deterministic
rollback. Because runtime status can evolve after application, the final
manifest call must revalidate it. Errors may report the owning manifest and
key path, but not a credential value. Descriptive fields such as `token_count`
must continue to round-trip.

A direct pre-implementation probe confirmed the gap: both nested `api_key`
configuration and later nested `auth_token` runtime status reached the public
optimization manifest, and the pass applied once. The focused regression is
not a pass: it failed one test with 29 deselected because unsafe configuration
did not raise. The shared canonical strict-JSON tree now applies the repository
credential detector before normalization. That one boundary covers pass
declarations, resolver-owned declaration snapshots, application metadata,
mutable runtime status, generic and TTS optimization manifests, and every
built-in or extension pass without a provider, pass, or architecture allowlist.

The corrected focused contract passed three tests with 27 deselected. The
native, universal TTS, compiler, accelerator, codec, diffusion-cache,
diffusion-sampling, and TTS optimization slice passed 190 tests and 684
subtests with one warning. The remaining codec, diffusion, serving, native-
codec, compile-target, report, VITS, Trainer, and training-runtime slice passed
204 tests and 218 subtests with eight warnings. The registry, public API,
speech-task, checkpoint, distribution-compliance, and release-policy slice
passed 147 tests and 508 subtests. An earlier registry command named a
nonexistent package-metadata test file, collected no tests, and is not passing
evidence.

The complete Python 3.12.12 suite passed 2,538 executed tests and 4,045
subtests with 35 warnings in 102.32 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two CUDA-
extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset
or oracle checks, and two inaccessible WeNet paths.

Documentation passed 64 tests and 1,476 subtests. The strict eleven-language
build completed in 32.60 seconds, and the ten-route DOM validator retained all
eight ordered navigation roots. The unchanged representative shell retains its
preceding Playwright/Axe evidence; no fresh visual-parity claim is made for
this API-safety and reference-prose slice. A fresh physical distribution check
passed wheel, sdist, and editable installs with 68 models, 81 provenance
manifests, 193 compliance files, required package data, zero runtime dependency
violations, and no eager PyTorch import.

All 68 generated model pages and 59 generated model notebooks remained
current. The five-record release-alignment check passed. Refreshed inventories
remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid
display names, 68 provider pages with no missing or orphaned page, six public
optimization passes, 408 model/pass pairs, eight top-level navigation roots,
ten representative routes, eight contribution steps, and 261 public exports.
The public root import still avoids PyTorch.

The first selected-file pre-commit invocation is not a pass because YAPF
rewrote the new test formatting. The corrected focused contract and second
selected-file invocation passed every applicable source hook; the Markdown
hook found no matching files and is not counted as a pass. This closes
credential handling for public optimization manifests. The broader object-by-
object public API audit remains open. Exact-current-worktree Linux, Windows,
Python 3.10, Python 3.11, default-runtime, tagged-workflow, and publisher
evidence; publication approval; five CUDA/Triton gates; and the seven opt-in or
inaccessible asset paths remain pending. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree training-recipe manifest secret boundary evidence

This bounded public-artifact safety iteration refreshed Transformers `main`
at commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home and Modular Transformers routes returned HTTP 200. Upstream
is structural context; VoiceHub's portable training-recipe boundary controls
this slice.

The observable contract requires the exact mapping returned by a training-
adapter subclass to reject nested credential-shaped fields before importing
the training backend, reading model state, creating the artifact destination,
or starting a native export. The final mapping must be checked again before
`training_recipe.json` is written. Errors may report the owning artifact and
field path, but not a credential value, while safe descriptive fields such as
`dataset_id` and `token_count` must continue to round-trip.

A direct pre-implementation probe confirmed the gap: a nested `api_key` from a
custom adapter was persisted in `training_recipe.json`, and the destination
was created. The focused regression is not a pass: it failed one test with 39
deselected because no error was raised. `Trainer.save_model()` now validates a
mapping snapshot before backend import or artifact mutation and reuses that
same validated payload for output. It rechecks the payload after adding the
bounded native-export path and immediately before the deterministic JSON
write. No task, provider, adapter, or recipe allowlist was added.

The corrected focused contract passed one test with 39 deselected. The
training-runtime, Trainer, adapter, checkpoint, and native-optimization slice
passed 132 tests and 575 subtests. The documentation, registry, speech-task,
universal-optimization, distribution-compliance, and release-readiness slice
passed 153 tests and 2,058 subtests. The complete Python 3.12.12 suite passed
2,539 executed tests and 4,045 subtests with 35 warnings in 104.51 seconds.
Its 15 skips remain unpassed: three complete-dependency default-runtime checks,
three Triton and two CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/
SpeechBrain/TEN-VAD asset or oracle checks, and two inaccessible WeNet paths.

Documentation passed 64 tests and 1,476 subtests. All 68 generated model pages
and 59 generated model notebooks remained current, and the five-record release
alignment check passed. The strict eleven-language build completed in 32.86
seconds, and the ten-route DOM validator retained all eight ordered navigation
roots. The unchanged representative shell retains its preceding Playwright/
Axe evidence; no fresh visual-parity claim is made for this persistence and
reference-prose slice.

A fresh physical distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime dependency violations, and no eager PyTorch import.
It produced a 57,193,356-byte wheel and a 55,458,161-byte sdist. The first
selected-file pre-commit invocation is not a pass because YAPF rewrote the new
test formatting. The formatted focused contract and second invocation passed
every applicable source hook; the Markdown hook found no matching files and is
not counted as a pass.

Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, 68 provider pages with no missing or
orphaned page, six public optimization passes, 408 model/pass pairs, eight
top-level navigation roots, ten representative routes, eight contribution
steps, and 261 public exports. The public root import still avoids PyTorch.

This closes credential handling for adapter-supplied training-recipe
manifests. The broader object-by-object public API audit remains open. Exact-
current-worktree Linux, Windows, Python 3.10, Python 3.11, default-runtime,
tagged-workflow, and publisher evidence; publication approval; five CUDA/
Triton gates; and the seven opt-in or inaccessible asset paths remain pending.
No protected action was taken. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree model-artifact state secret boundary evidence

This bounded public-artifact safety iteration refreshed Transformers `main`
at commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home and Modular Transformers routes returned current content.
Upstream is structural context; VoiceHub's safe binary model-artifact boundary
controls this slice.

The observable contract requires the exact state returned by a model or
portable optimization result to reject nested credential-shaped fields before
artifact destination creation or a model-owned save. Because a model-owned
save can mutate a previously returned mapping, the exact state must be checked
again immediately before `model_state.pt` is written. Errors may report the
owning artifact and field path but not a credential value. Tensors and safe
descriptive metadata such as `dataset_id` and `token_count` must continue to
round-trip.

A direct pre-implementation probe confirmed the gap: a nested `api_key` was
persisted in `model_state.pt`, and the destination was created. The focused
regression is not a pass: it failed one test with 40 deselected because no
error was raised. `Trainer.save_model()` now applies one shared model-artifact
state validator before filesystem mutation and again at the final binary write
boundary. This covers ordinary model state and the universal portable-
optimization state path without a task, provider, model, or optimization-pass
allowlist.

The corrected focused contract passed one test with 40 deselected. The complete
training-runtime file passed 41 tests and seven subtests. The training,
checkpoint, optimization, compilation, and reporting slice passed 241 tests
and 673 subtests with one platform warning. The registry, public API,
inference, documentation, scaffold, packaging, distribution-compliance, and
release-policy slice passed 264 tests and 2,089 subtests with one platform
warning. Selected pre-commit hooks passed in one invocation; the Markdown hook
found no matching files and is not counted as a pass.

The complete Python 3.12.12 suite passed 2,540 executed tests and 4,045
subtests with 35 warnings in 108.73 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two CUDA-
extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset
or oracle checks, and two inaccessible WeNet paths.

Documentation passed 64 tests and 1,476 subtests. All 68 generated model pages
and 59 generated model notebooks remained current, and the five-record release
alignment check passed. The strict eleven-language build completed in 35.02
seconds, and the ten-route DOM validator retained all eight ordered navigation
roots. The unchanged representative shell retains its preceding Playwright/
Axe evidence; this persistence and reference-prose slice makes no new visual-
parity claim.

A fresh physical distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, all required
package data, zero runtime-dependency violations, and no eager PyTorch import.
It produced a 57,193,419-byte wheel and a 55,458,392-byte sdist. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, 68 provider pages with no missing or orphaned page, six
public optimization passes, 408 model/pass pairs, eight top-level navigation
roots, ten representative routes, and eight contribution steps.

This closes credential handling for Trainer-owned binary model artifacts. The
broader object-by-object public API audit remains open. Exact-current-worktree
Linux, Windows, Python 3.10, Python 3.11, default-runtime, tagged-workflow, and
publisher evidence; publication approval; five CUDA/Triton gates; and the seven
opt-in or inaccessible asset paths remain pending. No protected action was
taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree model-state load boundary evidence

This bounded public-artifact safety iteration refreshed Transformers `main`
at commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, Modular Transformers, and Trainer routes returned current
content. Upstream is structural context; VoiceHub's credential-safe exact-
resume boundary controls this slice.

The observable contract requires credential-shaped values in deserialized
`model_state.pt` to fail before `load_state_dict()` can mutate the model. The
error may report only the owning artifact and key path, while ordinary model
state must still restore exactly once. This is a post-deserialization
credential check, not a safe-unpickling claim; Python pickle input must still
come from a trusted VoiceHub checkpoint with intact manifest integrity.

The pre-implementation focused regression is not a pass. It reached the
tracking model's `load_state_dict()` and raised a PyTorch missing/unexpected-key
error instead of the credential-policy error. The shared model-artifact state
validator now runs immediately after deserialization and before model
mutation, without a task, provider, model, or optimization-pass allowlist. The
corrected focused regression passed, and the complete training-runtime file
passed 42 tests and seven subtests.

The proportional training and checkpoint slice passed 199 tests and 201
subtests. The optimization, registry, public-API, and speech-task slice passed
242 tests and 1,239 subtests with one platform warning. The documentation,
release-policy, distribution-compliance, scaffold, pipeline, and automatic-
configuration slice passed 120 tests and 1,519 subtests. The complete Python
3.12.12 suite passed 2,541 executed tests and 4,045 subtests with 35 warnings
in 108.46 seconds. Its 15 skips remain unpassed: three complete-dependency
default-runtime checks, three Triton and two CUDA-extension checks, five
opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks, and
two inaccessible WeNet paths.

The release-alignment check passed with five benchmark records and all 68
documented providers. All 68 generated model pages and 59 generated model
notebooks remained current. Refreshed inventories remain 68 models (34 TTS,
23 ASR, and 11 VAD), 102 aliases, zero missing or orphaned provider pages, six
public optimization passes, 408 model/pass pairs, eight top-level navigation
roots, and 261 public exports. The public root import still avoids PyTorch.

The strict eleven-language build completed in 32.84 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this
Trainer runtime and reference-prose slice makes no new visual-parity claim. A
fresh isolated distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, all required
package data, zero runtime-dependency violations, and no eager PyTorch import.
It produced a 57,193,429-byte wheel and a 55,458,434-byte sdist.

The first selected-file pre-commit invocation is not a pass because YAPF
rewrote the new test. The formatted focused contract and second invocation
then passed every applicable source hook; the Markdown hook found no matching
files and is not counted as a pass. A direct release check against the
repository `dist/` directory also failed because that directory contains its
tracked `.gitignore`; it is not distribution evidence. The isolated
distribution checker above used a clean temporary artifact directory and
passed.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` has successful
remote Continuous Integration, Documentation, and Package CI runs, including
the supported Python and operating-system matrix, but those runs do not
contain the current dirty worktree and are not exact-candidate evidence. The
broader object-by-object public API audit, exact-current-worktree remote
platform and tagged-workflow evidence, publisher configuration, publication
approval, five CUDA/Triton gates, and seven opt-in or inaccessible asset paths
remain open. No protected action was taken. The untracked `uv.lock` stayed
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree public portable model-state load boundary evidence

This bounded public-lifecycle iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 65-line Home, 1,576-line
navigation, 500-line Modular Transformers, 30-line Trainer, and 2,906-line
TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, Modular Transformers, and Trainer routes returned current
content. Upstream is structural context; VoiceHub's shared public artifact
load boundary controls this slice.

The observable contract requires every public TTS, ASR, and VAD pretrained
base to reject nested credential-shaped fields after deserializing portable
`model_state.pt` and before either a training adapter or runtime can mutate a
fresh model. Errors may expose the owning artifact and field path but not the
credential value. A rejected state must remain pending for retry, while safe
ordinary state must load exactly once and restore inference mode. This is a
post-deserialization credential boundary, not a safe-unpickling claim; Python
pickle artifacts must still come from a trusted source.

The pre-implementation focused regression is not a pass: both public base
families accepted the credential mapping and failed two tests because no
policy error was raised. An earlier shell invocation also failed before test
collection because `python` was not on `PATH`; it is not evidence. One shared
task-neutral validator now runs immediately after `torch.load()` and before
the adapter/runtime branch. The final focused contract passed two tests. Its
TTS case proves the adapter-marked path cannot call adapter `setup()`, and its
audio case proves ordinary state cannot call runtime `load_state_dict()`.
Both cases prove redacted errors, retry preservation, and one successful safe
restore. The complete inference-lifecycle and speech-core files passed 76
tests and 41 subtests.

The proportional training and checkpoint slice passed 168 tests and 189
subtests. The optimization, registry, public-API, automatic-model, pipeline,
and speech-task slice passed 333 tests and 1,275 subtests with four warnings.
The documentation, release-policy, distribution-compliance, packaging, and
model-scaffold slice passed 107 tests and 1,587 subtests. The complete Python
3.12.12 suite passed 2,543 executed tests and 4,045 subtests with 35 warnings
in 107.12 seconds. Its 15 skips remain unpassed: three complete-dependency
default-runtime checks, three Triton and two CUDA-extension checks, five
opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks, and
two inaccessible WeNet paths.

The release-alignment check passed with five benchmark records and all 68
documented providers. All 68 generated model pages and 59 generated model
notebooks remained current. Refreshed inventories remain 68 models (34 TTS,
23 ASR, and 11 VAD), 102 aliases, zero missing or orphaned provider pages, six
public optimization passes, 408 model/pass pairs, eight top-level navigation
roots, and 261 public exports. The public root import still avoids PyTorch.

The strict eleven-language build completed in 34.21 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. Its first command
used an unsupported option and did not run; only the corrected positional
invocation is counted. The unchanged representative shell retains its
preceding Playwright/Axe evidence; this lifecycle and reference-prose slice
makes no new visual-parity claim. A fresh isolated distribution check passed
wheel, sdist, and editable installs with 68 models, 81 provenance manifests,
193 compliance files, all required package data, zero runtime-dependency
violations, and no eager PyTorch import. It produced a 57,193,552-byte wheel
and a 55,458,754-byte sdist. Every applicable selected-file pre-commit hook
passed; the Markdown hook found no matching files and is not counted.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` has successful remote
Continuous Integration, Documentation, and Package CI runs, including the
supported Python and operating-system matrix, but those runs do not contain
the current dirty worktree and are not exact-candidate evidence. The broader
object-by-object public API audit, exact-current-worktree remote platform and
tagged-workflow evidence, publisher configuration, publication approval, five
CUDA/Triton gates, and seven opt-in or inaccessible asset paths remain open.
No protected action was taken. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree speech dataset manifest safety evidence

This bounded public-artifact iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Revision-pinned revalidation now
shows its 65-line Home, 1,576-line navigation, 500-line Modular Transformers,
30-line Trainer, and 2,906-line TrainingArguments sources with SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, Modular Transformers, and Trainer routes returned current
content. The earlier 54-line Trainer response recorded here was not
reproducible against the pinned revision and is excluded rather than carried
forward. Upstream is structural context; VoiceHub's portable speech-dataset
boundary controls this slice.

The observable contract requires both public TTS and ASR manifest readers and
JSON Lines writers to reject nested credential-shaped fields. Reads must fail
before dataset construction. Writes must validate and serialize every record
before creating a parent directory, creating a file, or truncating an existing
destination. Errors may expose the task, record index, and field path but not a
credential value. Ordinary descriptive metadata such as `token_count` must
continue to round-trip.

The corrected four-test pre-implementation regression is not a pass: all four
selected writer/reader contracts accepted credentials, with two additional
writer subtest failures because no policy error was raised. The first command
used an incorrect ASR test class and collected no tests; it is also excluded.
Both task modules now invoke the shared credential detector after portable path
normalization and after parsing untrusted JSON, JSON Lines, CSV, or TSV
records. TTS export now precomputes every line before filesystem mutation,
matching the existing ASR preflight structure without introducing a task-
provider allowlist.

The final focused contract passed five tests. It proves credential rejection
for both writers and both readers, redacted errors, preservation of an existing
destination, absence of a rejected destination's parent directory, TTS
preservation on an ordinary JSON serialization failure, and safe
`token_count` round-trips. The complete TTS and ASR dataset files passed 79
tests and 116 subtests.

The proportional training and dataset slice passed 244 tests and 317 subtests.
The registry, optimization, public-API, automatic-model, pipeline, inference,
and speech-task slice passed 227 tests and 1,117 subtests. The documentation,
release-policy, distribution-compliance, packaging, and model-scaffold slice
passed 107 tests and 1,587 subtests. The complete Python 3.12.12 suite passed
2,548 executed tests and 4,045 subtests with 35 warnings in 110.21 seconds. Its
15 skips remain unpassed: three complete-dependency default-runtime checks,
three Triton and two CUDA-extension checks, five opt-in
ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks, and two
inaccessible WeNet paths.

The release-alignment check passed with five benchmark records and all 68
documented providers. All 68 generated model pages and 59 generated model
notebooks remained current. Refreshed inventories remain 68 models (34 TTS,
23 ASR, and 11 VAD), 102 aliases, zero invalid display names, 68 provider pages
with no missing or orphaned page, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, and 261 public exports.
The public root import still avoids PyTorch.

The strict eleven-language build completed in 35.42 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this data
contract and reference-prose slice makes no new visual-parity claim. A fresh
isolated distribution check passed wheel, sdist, and editable installs with 68
models, 81 provenance manifests, 193 compliance files, all required package
data, zero runtime-dependency violations, and no eager PyTorch import. It
produced a 57,193,721-byte wheel and a 55,459,521-byte sdist. Every applicable
selected-file pre-commit hook passed; the Markdown hook found no matching files
and is not counted.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, including
the supported Python and operating-system matrix, but those runs do not
contain the current dirty worktree and are not exact-candidate evidence. The
broader object-by-object public API audit, exact-current-worktree remote
platform and tagged-workflow evidence, publisher configuration, publication
approval, five CUDA/Triton gates, and seven opt-in or inaccessible asset paths
remain open. No protected action was taken. The untracked `uv.lock` stayed
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree shared JSON artifact atomicity evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources have SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, Modular Transformers, navigation, and Trainer routes
returned current content. Upstream is structural context; VoiceHub's shared
artifact writer controls this slice.

The observable contract requires shared model and Trainer JSON writers to
encode finite deterministic JSON before creating a parent directory or
touching a destination. A successful write must flush a temporary sibling and
atomically replace the destination. Serialization or replacement failure must
preserve an existing artifact and remove temporary files. Safe metadata such
as `token_count` must continue to round-trip with the existing trailing-newline
format. This is a per-document guarantee rather than a claim that a backend's
multi-file native export is transactional without its staging contract.

The pre-implementation focused command exited one: nine subcases demonstrated
parent creation, partial truncation, permissive non-finite output, or absence
of an atomic replacement boundary. Its three reported top-level passes and
three other subtest passes do not make the failed run passing evidence. The
shared writer now precomputes strict JSON, writes and fsyncs a temporary sibling,
uses `os.replace()`, and removes the temporary path on every failure. Trainer's
JSON helper delegates to the same implementation, so configuration, processor,
native-export helper, Trainer state, training arguments, and checkpoint
manifest call sites no longer maintain divergent document-write behavior.

The corrected focused contract passed three tests and 12 subtests. The first
selected-file pre-commit invocation is excluded because YAPF reformatted the
new code; after formatting, the focused contract passed again and every
applicable source hook passed. The Markdown hook found no applicable files and
is not counted. A first dependency-free Python 3.10/3.11 probe is also excluded
because its harness checked for the literal characters `\n` rather than a
newline. The corrected exact-slice probe passed on Python 3.10.19 and 3.11.15.

The public API, configuration, processor, pipeline, training, checkpoint, and
inference slice passed 207 tests and 76 subtests. The registry, optimization,
documentation-policy, packaging-policy, distribution-policy, and release slice
passed 193 tests and 2,616 subtests. The complete Python 3.12.12 suite passed
2,551 executed tests and 4,057 subtests with 35 warnings in 107.63 seconds. Its
15 skips remain unpassed: three complete-dependency default-runtime checks,
three Triton and two CUDA-extension checks, five opt-in
ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks, and two
inaccessible WeNet paths.

Documentation and final release-policy validation passed 87 tests and 1,552
subtests. The strict eleven-language build completed successfully, and the
ten-route DOM validator retained all eight ordered navigation roots. The
unchanged representative shell retains its preceding Playwright/Axe evidence;
this data-safety and reference-prose slice makes no new visual-parity claim. A
fresh isolated distribution check passed wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, all required
package data, zero runtime-dependency violations, and no eager PyTorch import.
It produced a 57,193,870-byte wheel and a 55,459,554-byte sdist.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remained current. Refreshed inventories
remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, six public
optimization passes, 408 model/pass pairs, eight top-level navigation roots,
eight contribution steps, and 261 public exports. The public root import still
avoids PyTorch.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. The broader object-by-object public API audit, exact-current-worktree
complete Python 3.10/3.11, Linux, Windows, default-runtime, tagged-workflow, and
publisher evidence, publication approval, five CUDA/Triton gates, and seven
opt-in or inaccessible asset paths remain open. No protected action was taken.
The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree shared JSON artifact read-safety evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources have SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, and Trainer routes returned current content.
The Modular Transformers route returned HTTP 429 and is unavailable evidence,
not a pass; its revision-pinned raw source was still fetched and matched the
recorded fingerprint. Upstream is structural context, while VoiceHub's shared
artifact loader controls this slice.

The observable contract requires duplicate object keys to fail at every
nesting level and requires `NaN`, positive or negative `Infinity`, and exponent
overflow to fail before configuration construction or automatic model
dispatch. Errors must identify the source and duplicate key or numeric path
without printing discarded values. Ordinary finite JSON, including nested
`token_count` metadata, must keep round-tripping.

The pre-implementation focused command exited one with seven failures; it
accepted both duplicate-key documents, all four non-finite forms, and an
ambiguous duplicate `model_type` before `AutoConfig` dispatch. Its three passes
are not transferred out of the failed run. The first post-implementation
command is also excluded: rejection behavior worked, but three message-case
assertions failed. After the diagnostic wording was corrected, the focused
contract passed seven tests and 18 subtests. It covers top-level and nested
duplicates, redacted discarded values, all non-finite forms, numeric-path
reporting for exponent overflow, pre-dispatch `AutoConfig` rejection, and a
safe nested round-trip.

Dependent public API, automatic-configuration, inference, pipeline, speech-
core, and native-checkpoint contracts passed 137 tests and 53 subtests. The
registry and optimization selection passed 234 tests and 1,339 subtests. The
complete Python 3.12.12 suite passed 2,555 executed tests and 4,063 subtests
with 35 warnings in 106.93 seconds. Its 15 skips remain unpassed: three
complete-dependency default-runtime checks, three Triton and two CUDA-extension
checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle
checks, and two inaccessible WeNet paths. Focused dependency-free probes passed
on Python 3.10.19 and 3.11.15; complete exact-current-worktree execution on
those interpreters remains pending.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display names,
68 provider pages with no missing or orphaned page, six public optimization
passes, 408 model/pass pairs, eight top-level navigation roots, eight
contribution steps, and 261 public exports. The public root import still avoids
PyTorch.

Release, distribution-policy, and documentation-policy validation passed 77
tests and 1,476 subtests. A fresh isolated distribution check passed wheel,
sdist, and editable installs with 68 models, 81 provenance manifests, 193
compliance files, all required package data, zero runtime-dependency
violations, and no eager PyTorch import. It produced a 57,194,351-byte wheel
and a 55,460,094-byte sdist. Every applicable selected source/test pre-commit
hook passed before this evidence update.

After the evidence update, the focused reader plus release, distribution, and
documentation-policy selection passed 84 tests and 1,494 subtests. The strict
eleven-language build completed successfully, and the ten-route DOM validator
retained all eight ordered navigation roots. The unchanged representative
shell retains its preceding Playwright/Axe evidence; this data-safety and
reference-prose slice makes no new visual-parity claim.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. The broader object-by-object public API audit, exact-current-worktree
complete Python 3.10/3.11, Linux, Windows, default-runtime, tagged-workflow,
publisher, and publication gates, five CUDA/Triton checks, and seven opt-in or
inaccessible asset paths remain open. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Trainer JSON read-safety evidence

This bounded exact-resume iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources have SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, Modular Transformers, Trainer, documentation
specification, and TrainingArguments routes returned current content. Upstream
supplies the structural training mental model; VoiceHub's speech-specific
artifact safety controls this slice.

The observable contract requires `TrainingArguments`, `TrainerState`,
checkpoint discovery, and exact-resume validation to reject duplicate object
keys and `NaN`, positive or negative `Infinity`, or exponent overflow through
the shared artifact reader. Object construction and model or runtime state
restoration must not occur first. Invalid checkpoint candidates must be
ignored rather than selected. Errors must identify the source and duplicate
key or numeric path without printing discarded values, while valid saved
objects and exact-resume behavior remain unchanged.

The pre-implementation focused command exited one with 11 failures. Both
public object loaders accepted duplicates or constructed before rejecting
overflow, discovery selected both invalid latest checkpoints, and checkpoint,
optimization, and Trainer-state documents reached later validation rather than
the parse boundary. Its three reported top-level passes are not transferred
out of the failed run. TrainingArguments, TrainerState, checkpoint discovery,
and all three exact-resume JSON reads now delegate to the same strict loader;
no new abstraction, provider branch, or artifact format was introduced.

The corrected focused contract passed three tests and 11 subtests. Its first
selected-file pre-commit run is excluded because YAPF reformatted the added
test data; the formatted focused contract passed again, followed by every
applicable selected source/test hook. The Markdown hook had no matching files
and is not counted. The complete Trainer and training-runtime files passed 73
tests and 34 subtests. The proportional training, checkpoint, and optimization
selection passed 285 tests and 692 subtests; the registry and lifecycle
selection passed 139 tests and 499 subtests.

The complete Python 3.12.12 suite passed 2,558 executed tests and 4,074
subtests with 35 warnings in 105.99 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two
CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD
asset or oracle checks, and two inaccessible WeNet paths. Focused
dependency-free probes passed on Python 3.10.19 and 3.11.15; complete exact-
current-worktree execution on those interpreters remains pending.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports. The public root import still avoids PyTorch.

Release, distribution-policy, and documentation-policy validation passed 77
tests and 1,476 subtests. A fresh isolated distribution check passed wheel,
sdist, and editable installs with 68 models, 81 provenance manifests, 193
compliance files, all required package data, zero runtime-dependency
violations, and no eager PyTorch import. It produced a 57,194,294-byte wheel
and a 55,460,694-byte sdist.

After the evidence update, the focused Trainer readers plus release,
distribution, and documentation-policy selection passed 80 tests and 1,487
subtests. The strict eleven-language build completed successfully, and the
ten-route DOM validator retained all eight ordered navigation roots. The
unchanged representative shell retains its preceding Playwright/Axe evidence;
this data-safety and reference-prose slice makes no new visual-parity claim.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. The broader object-by-object public API audit, exact-current-worktree
complete Python 3.10/3.11, Linux, Windows, default-runtime, tagged-workflow,
publisher, and publication gates, five CUDA/Triton checks, and seven opt-in or
inaccessible asset paths remain open. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree native checkpoint JSON read-safety evidence

This bounded data-safety iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line Home,
1,576-line navigation, 500-line Modular Transformers, 30-line Trainer, and
2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned current content. Upstream supplies
the structural artifact-loading mental model; VoiceHub's bounded native
checkpoint parsers control this slice.

The observable contract requires `VoiceHubManifest`, bounded Safetensors
headers, and sharded Safetensors indexes to reject duplicate object keys at
every nesting level and reject `NaN`, positive or negative `Infinity`, and
exponent overflow. Rejection must occur before manifest construction, tensor
materialization, or shard lookup. Diagnostics must identify the source and
duplicate key or numeric path without exposing a discarded value. Valid
deterministic manifests, headers, indexes, and descriptive `token_count`
metadata must retain their existing formats.

The pre-implementation focused command exited one with eight failing subcases;
its three reported top-level passes are not transferred out of that failed
run. The shared decoder now accepts an in-memory JSON document as well as a
file and applies one duplicate-key and finite-number policy. All three native
checkpoint readers delegate to it and translate failures into their existing
`CheckpointFormatError` boundary. No provider, model, task, optimization-pass,
or shard allowlist was introduced.

The corrected focused contract passed three tests and eight subtests. The
complete native-checkpoint and shared-JSON files passed 26 tests and 26
subtests. The proportional checkpoint, configuration, public API, Trainer,
training-runtime, and speech-core selection passed 165 tests and 89 subtests;
the registry, speech-task, inference, and optimization selection passed 124
tests and 969 subtests. Every applicable selected source/test pre-commit hook
passed. The Markdown hook had no selected Markdown file and is not counted.
Dependency-light exact-slice probes passed on Python 3.10.19 and 3.11.15
without importing PyTorch; complete exact-current-worktree runs on those
interpreters remain pending.

A first final policy command named the nonexistent
`tests/test_package_imports.py` path and collected no tests; it is excluded.
The corrected focused reader plus release, distribution, and documentation-
policy selection passed 87 tests and 1,502 subtests.

The complete Python 3.12.12 suite passed 2,561 executed tests and 4,082
subtests with 35 warnings in 105.91 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two
CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD
asset or oracle checks, and two inaccessible WeNet paths.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports. The public root and checkpoint-metadata imports
remain PyTorch-free.

The strict eleven-language build completed in 33.26 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this
checkpoint data-safety slice makes no new visual-parity claim. A fresh isolated
distribution check passed wheel, sdist, and editable installs with 68 models,
81 provenance manifests, 193 compliance files, all required package data,
zero runtime-dependency violations, and no eager PyTorch import. It produced a
57,194,195-byte wheel and a 55,460,853-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. The broader object-by-object public API audit, exact-current-worktree
complete Python 3.10/3.11, Linux, Windows, default-runtime, tagged-workflow,
publisher, and publication gates, five CUDA/Triton checks, and seven opt-in or
inaccessible asset paths remain open. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree portable-state config read-safety evidence

This bounded public-lifecycle iteration refreshed Transformers `main` at
commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line
Home, 1,576-line navigation, 500-line Modular Transformers, 30-line Trainer,
and 2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned current content. Upstream provides
the structural pretrained-model workflow; VoiceHub's portable speech-state
boundary controls this slice.

The observable contract requires both public pretrained base loaders to apply
the shared strict JSON policy to the sibling `config.json` when restoring a
local `model_state.pt`, including when the caller supplied `config=` and the
ordinary configuration factory was bypassed. Duplicate object keys, `NaN`,
positive or negative `Infinity`, and exponent overflow must fail before TTS,
ASR, or VAD wrapper construction. Diagnostics must identify the artifact and
duplicate key or numeric path without exposing a discarded value. A valid
saved `name_or_path` and descriptive `token_count` metadata must retain the
existing lazy restoration behavior.

The pre-implementation focused command exited one with six failing subcases:
both base loaders accepted every ambiguous document and reached the guarded
constructor. Its two reported top-level passes are not transferred out of the
failed run. The two permissive reads now delegate to the existing shared
reader; no new abstraction, provider branch, task allowlist, or artifact format
was introduced.

The first expanded post-implementation command is also excluded: all six
malformed subcases passed, but two valid-artifact assertions compared macOS
`/var` and `/private/var` spellings without resolving them. After correcting
only that test expectation, the focused contract passed two tests and six
subtests. The proportional public base, configuration, pipeline, inference,
training, Trainer, and portable-state selection passed 221 tests and 181
subtests. Registry, speech-task, and universal-optimization coverage passed 76
tests and 953 subtests. Every applicable selected source/test pre-commit hook
passed; the Markdown hook had no selected Markdown file and is not counted.
The public TTS, ASR, and VAD base imports remain PyTorch-free.

Direct pytest execution was unavailable in the standalone Python 3.10.19 and
3.11.15 installations because neither interpreter has pytest; those two
failed commands collected no tests and are not evidence. Corrected dependency-
light scripts then exercised duplicate, constant, and overflow rejection for
both base loaders before construction on both interpreters and confirmed that
PyTorch remained unimported. Complete exact-current-worktree suites on Python
3.10 and 3.11 remain pending.

The complete Python 3.12.12 suite passed 2,563 executed tests and 4,088
subtests with 35 warnings in 105.16 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two
CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD
asset or oracle checks, and two inaccessible WeNet paths.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports.

The strict eleven-language build completed in 33.88 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this
public data-safety slice makes no new visual-parity claim. A fresh isolated
distribution check passed wheel, sdist, and editable installs with 68 models,
81 provenance manifests, 193 compliance files, all required package data,
zero runtime-dependency violations, and no eager PyTorch import. It produced a
57,194,160-byte wheel and a 55,460,813-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. The broader object-by-object public API audit, exact-current-worktree
complete Python 3.10/3.11, Linux, Windows, default-runtime, tagged-workflow,
publisher, and publication gates, five CUDA/Triton checks, and seven opt-in or
inaccessible asset paths remain open. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree Hub JSON trust-boundary evidence

This bounded public-artifact iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line
Home, 1,576-line navigation, 500-line Modular Transformers, 30-line Trainer,
and 2,906-line TrainingArguments sources retain the release-ledger SHA-256
fingerprints. The official Home, navigation, documentation specification,
Modular Transformers, and Trainer routes returned current content. Upstream
provides the structural pretrained-model workflow; VoiceHub owns the native
Hub transport and cache policy exercised here.

The observable contract requires remote Hub API payloads to reject duplicate
object keys, `NaN`, positive or negative `Infinity`, and exponent overflow
before a commit or repository tree is interpreted. Ambiguous VoiceHub file
metadata and snapshot manifests must be treated as cache misses in offline
mode. Diagnostics must identify the source context and duplicate key or
numeric path without exposing a discarded value or authentication token.
Valid online and offline resolution, redirects, list-valued repository trees,
and the dependency-light import boundary must retain their behavior.

The pre-implementation focused command exited one with five failures and one
reported pass: all three malformed remote payloads reached the guarded tree
request, the ambiguous file metadata returned a commit, and the ambiguous
snapshot manifest was reused. No result from that failed command is counted.
The strict decoder now lives in a dependency-light shared module used by local
artifacts, Safetensors metadata, Hub API payloads, and native Hub cache
metadata. Remote ambiguity becomes `HubDownloadError`; ambiguous cached
objects become misses. No provider, model, task, or optimization allowlist was
introduced.

The corrected focused contract passed three tests and three subtests. Hub,
strict-artifact, and native-checkpoint coverage passed 48 tests and 34
subtests. Public configuration, base, speech-core, inference, and pipeline
lifecycle coverage passed 123 tests and 59 subtests. Registry, speech-task,
and universal-optimization coverage passed 76 tests and 953 subtests. Every
applicable selected source/test pre-commit hook passed. Dependency-light exact
probes on Python 3.10.19 and 3.11.15 rejected duplicate, constant, and overflow
Hub JSON without leaking the discarded value or runtime token and preserved a
PyTorch-free import. Complete exact-current-worktree suites on those two
interpreters remain pending.

The complete Python 3.12.12 suite passed 2,566 executed tests and 4,091
subtests with 35 warnings in 105.91 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two
CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD
asset or oracle checks, and two inaccessible WeNet paths.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports.

The strict eleven-language build completed in 35.30 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this
artifact-safety slice makes no new visual-parity claim. A fresh isolated
distribution check passed wheel, sdist, and editable installs with 68 models,
81 provenance manifests, 193 compliance files, all required package data,
zero runtime-dependency violations, and no eager PyTorch import. It produced a
57,194,648-byte wheel and a 55,461,196-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. The broader object-by-object public API audit, exact-current-worktree
complete Python 3.10/3.11, Linux, Windows, default-runtime, tagged-workflow,
publisher, and publication gates, five CUDA/Triton checks, and seven opt-in or
inaccessible asset paths remain open. No protected action was taken. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree tokenizer JSON trust-boundary evidence

This bounded public-artifact iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line
Home, 1,576-line navigation, 500-line Modular Transformers, 30-line Trainer,
and 2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned HTTP 200. Upstream supplies the
pretrained tokenizer workflow; VoiceHub owns the bounded declarative tokenizer
parsers exercised here.

The observable contract requires both shared tokenizer JSON entry points—the
Hugging Face Byte-BPE loader and the SentencePiece-BPE loader—to reject
duplicate object keys, `NaN`, positive or negative `Infinity`, and exponent
overflow before interpreting the model graph, vocabulary, merges, or added
tokens. Diagnostics must identify the source path and duplicate key or numeric
path without exposing a discarded value. Valid tokenizer loading and saving,
byte/token/merge/nesting/node limits, registered consumers, and PyTorch-free
imports must retain their behavior.

The pre-implementation focused command exited one with six failing subcases
and two reported top-level passes. The Byte-BPE loader accepted a duplicate
model and exponent overflow and omitted source context for `NaN`; the
SentencePiece-BPE loader omitted source context for its existing duplicate and
constant rejection and accepted exponent overflow. No result from that failed
command is counted. Both loaders now delegate decoding to the existing shared
strict JSON boundary before applying their tokenizer-specific graph and size
validators. The redundant SentencePiece duplicate/constant hooks and the
Byte-BPE constant hook were removed. No provider, model, task, or optimization
allowlist was introduced.

The corrected focused contract passed two tests and six subtests. Core
tokenizer, Moonshine, VoxCPM2, and shared-artifact coverage passed 59 tests and
27 subtests with four warnings. Eleven additional native consumer suites
passed 166 tests and 27 subtests. Registry, speech-task, universal-optimization,
model-page/navigation, and native dependency-policy coverage passed 159 tests
and 2,429 subtests. The first selected-file pre-commit command exited one after
the end-of-file hook repaired one missing final newline and is excluded; the
complete corrected selected-file command then passed every applicable hook.
The Markdown hook found no selected Markdown file and is not counted.

The first standalone Python 3.10.19 and 3.11.15 probe imported a test helper
that requires unavailable PyTorch, so both invocations failed before exercising
the contract and are excluded. Corrected dependency-light probes constructed
both valid tokenizer graphs directly, loaded them successfully, rejected the
three ambiguous variants for each loader with source-aware diagnostics, and
confirmed PyTorch-free imports on both interpreters. Complete exact-current-
worktree suites on Python 3.10 and 3.11 remain pending.

The complete Python 3.12.12 suite passed 2,568 executed tests and 4,097
subtests with 35 warnings in 110.46 seconds. Its 15 skips remain unpassed:
three complete-dependency default-runtime checks, three Triton and two
CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD
asset or oracle checks, and two inaccessible WeNet paths.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports.

The strict eleven-language build completed in 35.92 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this
artifact-safety slice makes no new visual-parity claim. A fresh isolated
distribution check passed wheel, sdist, and editable installs with 68 models,
81 provenance manifests, 193 compliance files, all required package data,
zero runtime-dependency violations, and no eager PyTorch import. It produced a
57,194,432-byte wheel and a 55,462,589-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. Newer successful runs for unrelated head `8fee3d7` likewise do not
cover this worktree. The broader object-by-object public API audit, exact-
current-worktree complete Python 3.10/3.11, Linux, Windows, default-runtime,
tagged-workflow, publisher, and publication gates, five CUDA/Triton checks,
and seven opt-in or inaccessible asset paths remain open. No protected action
was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree speech manifest JSON trust-boundary evidence

This bounded public-dataset iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line
Home, 1,576-line navigation, 500-line Modular Transformers, 30-line Trainer,
and 2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned HTTP 200. VoiceHub owns the speech
manifest parsers exercised here.

The observable contract requires public TTS and ASR `.json`, JSON Lines, and
embedded CSV/TSV JSON values to reject duplicate keys, `NaN`, positive or
negative `Infinity`, and exponent overflow before dataset construction.
Diagnostics must name the file and, where available, line and field without
exposing a discarded value. Valid object/list manifests, NeMo JSON Lines
fallback, tabular scalar coercion, credential checks, portable round trips,
and lazy imports must retain their behavior.

The pre-implementation focused command exited one with 18 failing subcases
and two reported top-level passes; none of that command is counted. Both
dataset readers now use the shared strict JSON decoder for whole JSON files,
JSON Lines records, and JSON-shaped tabular fields. Tabular syntax errors keep
the established string fallback. ASR `.json` falls back to NeMo JSON Lines
only for a syntax error, never for a duplicate or non-finite value. No provider,
model, task, or optimization allowlist was introduced.

The corrected focused contract passed two tests and 18 subtests. Complete TTS
and ASR dataset coverage passed 81 tests and 134 subtests. Proportional core
training, Trainer, adapter, collator, and speech-training coverage passed 168
tests and 212 subtests. Registry, speech-task, universal-optimization,
model-page/navigation, model-scaffold, and native dependency-policy coverage
passed 179 tests and 2,464 subtests. The first selected-file pre-commit command
exited one after YAPF changed the files and is excluded; the corrected command
then passed every applicable hook. The Markdown hook found no applicable file
and is not counted.

Dependency-light probes on Python 3.10.19 and 3.11.15 each rejected all 18
task/format/ambiguity paths and retained PyTorch-free public imports. Complete
exact-current-worktree suites on Python 3.10 and 3.11 remain pending. The
complete Python 3.12.12 suite passed 2,570 executed tests and 4,115 subtests
with 35 warnings in 110.64 seconds. Its 15 skips remain unpassed: three
complete-dependency default-runtime checks, three Triton and two CUDA-extension
checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle
checks, and two inaccessible WeNet paths.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports.

The strict eleven-language build passed, and the ten-route DOM validator
retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this
dataset-safety slice makes no new visual-parity claim. A fresh isolated
distribution check passed wheel, sdist, and editable installs with 68 models,
81 provenance manifests, 193 compliance files, all required package data,
zero runtime-dependency violations, and no eager PyTorch import. It produced a
57,194,588-byte wheel and a 55,464,587-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. Newer successful runs for unrelated head `8fee3d7` likewise do not
cover this worktree. The broader object-by-object public API audit, exact-
current-worktree complete Python 3.10/3.11, Linux, Windows, default-runtime,
tagged-workflow, publisher, and publication gates, five CUDA/Triton checks,
and seven opt-in or inaccessible asset paths remain open. No protected action
was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree external LLM JSON response trust-boundary evidence

This bounded public-serving iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned 65-line
Home, 1,576-line navigation, 500-line Modular Transformers, 30-line Trainer,
and 2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned current content. VoiceHub owns the
external speech-serving transport exercised here.

The observable contract requires bounded vLLM and SGLang JSON responses to
reject duplicate keys, `NaN`, positive or negative `Infinity`, and exponent
overflow before token IDs, usage metadata, or other protocol fields are
interpreted. Diagnostics must identify the backend route and offending key or
numeric path without exposing a discarded value. Valid responses, request
serialization, response-size limits, redirect rejection, credential
redaction, backend capability dispatch, and lazy imports must retain their
behavior.

The first pre-implementation command is excluded because a test-control-flow
error produced a secondary `AttributeError` after the expected duplicate-key
failure. After correcting only that test structure, the valid red command
exited one with all three ambiguity subcases failing and one reported top-level
pass; none of that command is counted. The HTTP response boundary now delegates
decoding to the shared strict JSON parser and wraps its source-aware failure in
the existing sanitized `LLMBackendRequestError`. No provider, model, task, or
optimization allowlist was introduced.

The corrected focused contract passed one test and three subtests. The complete
LLM-serving suite passed 53 tests and 57 subtests. Proportional diffusion-
serving, inference, base-model, speech-core, registry, speech-task,
universal-optimization, native-optimization, and native-architecture coverage
passed 237 tests and 1,185 subtests. Release, distribution, and documentation
policy coverage passed 77 tests and 1,476 subtests. The selected-file
pre-commit command passed every applicable hook; the Markdown hook found no
applicable file and is not counted.

Dependency-light probes on Python 3.10.19 and 3.11.15 each rejected all three
ambiguity forms, accepted one valid response, and retained PyTorch-free public
imports. Complete exact-current-worktree suites on Python 3.10 and 3.11 remain
pending. The complete Python 3.12.12 suite passed 2,571 executed tests and
4,118 subtests with 35 warnings in 109.45 seconds. Its 15 skips remain
unpassed: three complete-dependency default-runtime checks, three Triton and
two CUDA-extension checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/
TEN-VAD asset or oracle checks, and two inaccessible WeNet paths.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports.

The strict eleven-language build and ten-route DOM validator passed and
retained all eight ordered navigation roots. The unchanged representative
shell retains its preceding Playwright/Axe evidence; this serving-safety slice
makes no new visual-parity claim. A fresh isolated distribution check passed
wheel, sdist, and editable installs with 68 models, 81 provenance manifests,
193 compliance files, all required package data, zero runtime-dependency
violations, and no eager PyTorch import. It produced a 57,194,624-byte wheel
and a 55,465,049-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. Newer successful runs for unrelated head `8fee3d7` likewise do not
cover this worktree. The broader object-by-object public API audit, exact-
current-worktree complete Python 3.10/3.11, Linux, Windows, default-runtime,
tagged-workflow, publisher, and publication gates, five CUDA/Triton checks,
and seven opt-in or inaccessible asset paths remain open. No protected action
was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree model-integration JSON trust-boundary evidence

This bounded contribution-path iteration refreshed Transformers `main` at
commit `d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned
65-line Home, 1,576-line navigation, 500-line Modular Transformers, 30-line
Trainer, and 2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned current content. VoiceHub owns the
model-contribution metadata boundary exercised here.

The observable contract requires activated package-local
`model-integration.json` files and their required `source/SOURCE.json` records
to reject duplicate keys, `NaN`, positive or negative `Infinity`, and exponent
overflow before registry or training-profile construction. The standalone
scaffold checker and catalog renderer must enforce the same ambiguity boundary
without importing VoiceHub or PyTorch. Diagnostics must identify the file and
offending key or numeric path without exposing a discarded value. Inactive
work-in-progress manifests must remain undiscovered, and valid zero-central-
edit TTS, ASR, and VAD discovery must remain unchanged.

The pre-implementation command exited one with six failures, two reported
passes, and three reported subtests; that entire command is excluded. Activated
runtime discovery now strictly reparses its manifest and provenance record,
while the initial bounded parse retains the documented inactive-draft escape
hatch. The standalone checker and catalog renderer use an equivalent local
strict decoder to preserve their no-package-import boundary. No provider,
model, task, or optimization allowlist was introduced.

The corrected focused contract passed two tests and nine subtests; the complete
scaffold suite passed 22 tests and 44 subtests. Proportional scaffold,
registry, speech-task, training, documentation, release, universal-
optimization, native-optimization, and distribution coverage passed 241 tests
and 2,659 subtests. The selected-file pre-commit command passed every
applicable hook; the Markdown hook found no applicable file and is not counted.

Dependency-light probes on Python 3.10.19 and 3.11.15 each rejected all six
active artifact/ambiguity combinations, accepted one valid activated manifest,
and retained VoiceHub- and PyTorch-free standalone-tool imports. Complete
exact-current-worktree suites on Python 3.10 and 3.11 remain pending. The
complete Python 3.12.12 suite passed 2,573 executed tests and 4,127 subtests
with 35 warnings in 109.63 seconds. Its 15 skips remain unpassed: three
complete-dependency default-runtime checks, three Triton and two CUDA-extension
checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle
checks, and two inaccessible WeNet paths.

The five-record release-alignment check passed. All 68 generated model pages
and 59 generated model notebooks remain current. Refreshed inventories remain
68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 provider pages with no missing or orphaned page, 10 representative
page pairs, six shared component/state groups, six public optimization passes,
408 model/pass pairs, eight top-level navigation roots, eight contribution
steps, and 261 public exports.

The strict eleven-language build and ten-route DOM validator passed and
retained all eight ordered navigation roots. The unchanged representative
shell retains its preceding Playwright/Axe evidence; this contribution-safety
slice makes no new visual-parity claim. A fresh isolated distribution check
passed wheel, sdist, and editable installs with 68 models, 81 provenance
manifests, 193 compliance files, all required package data, zero runtime-
dependency violations, and no eager PyTorch import. It produced a
57,194,699-byte wheel and a 55,466,554-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. Newer successful runs for unrelated head `8fee3d7` likewise do not
cover this worktree. The broader object-by-object public API audit, exact-
current-worktree complete Python 3.10/3.11, Linux, Windows, default-runtime,
tagged-workflow, publisher, and publication gates, five CUDA/Triton checks,
and seven opt-in or inaccessible asset paths remain open. No protected action
was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree shared JSON artifact byte-bound evidence

This bounded public-artifact iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned
65-line Home, 1,576-line navigation, 500-line Modular Transformers, 30-line
Trainer, and 2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned current content. VoiceHub owns the
shared local and downloaded artifact boundary exercised here.

The observable contract requires every JSON object routed through
`read_json_file()` to have a validated positive integer byte ceiling before
decoding. The default ceiling is 64 MiB, and an owning internal loader may
choose a smaller limit. A document already over the limit, or one that grows
past it between metadata inspection and reading, must fail before JSON parsing.
Diagnostics must identify the source and actual size or configured limit
without exposing document content. Exact-limit finite JSON, existing strict
duplicate/non-finite rejection, and lazy dependency behavior must remain
unchanged.

The pre-implementation focused command exited one with six failures and one
reported pass; that entire command is excluded. The shared reader now validates
the caller's limit, rejects an already oversized file from metadata, and reads
at most one byte beyond the ceiling so concurrent growth also fails closed.
Configuration, auto-loading, processing, training, checkpoint, and native-
artifact callers inherit the default without provider branches or duplicated
reader logic.

The corrected focused contract passed four tests and four subtests; the complete
shared JSON artifact suite passed 11 tests and 22 subtests. Proportional public
configuration, auto-loading, pipeline, inference, speech-core, checkpoint,
Trainer, registry, task, optimization, documentation, release, and
distribution coverage passed 409 tests and 2,663 subtests. The first selected-
file pre-commit run is excluded because YAPF reformatted the new code. The
formatted rerun passed every applicable hook; the Markdown hook found no
applicable file and is not counted.

Dependency-light probes on Python 3.10.19 and 3.11.15 each passed four exact-
slice checks and retained PyTorch-free imports. Complete exact-current-
worktree suites on Python 3.10 and 3.11 remain pending. The complete Python
3.12.12 suite passed 2,577 executed tests and 4,131 subtests with 35 warnings
in 114.47 seconds. Its 15 skips remain unpassed: three complete-dependency
default-runtime checks, three Triton and two CUDA-extension checks, five opt-in
ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle checks, and two
inaccessible WeNet paths.

The inventory suite passed 198 tests and 2,608 subtests. The five-record
release-alignment check passed. All 68 generated model pages and 59 generated
model notebooks remain current. Refreshed inventories remain 68 models (34
TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display names, 68 provider
pages with no missing or orphaned page, 10 representative page pairs, six
shared component/state groups, six public optimization passes, 408 model/pass
pairs, eight top-level navigation roots, eight contribution steps, and 261
public exports.

The strict eleven-language build completed in 34.41 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding Playwright/Axe evidence; this
artifact-safety slice makes no new visual-parity claim. A fresh isolated
distribution check passed wheel, sdist, and editable installs with 68 models,
81 provenance manifests, 193 compliance files, all required package data,
zero runtime-dependency violations, and no eager PyTorch import. It produced a
57,194,971-byte wheel and a 55,467,121-byte sdist.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. Newer successful runs for unrelated head `8fee3d7` likewise do not
cover this worktree. The broader object-by-object public API audit, exact-
current-worktree complete Python 3.10/3.11, Linux, Windows, default-runtime,
tagged-workflow, publisher, and publication gates, five CUDA/Triton checks,
and seven opt-in or inaccessible asset paths remain open. No protected action
was taken. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree supported-Python macOS evidence

This bounded platform iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned
65-line Home, 1,576-line navigation, 500-line Modular Transformers, 30-line
Trainer, and 2,906-line TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, navigation, documentation specification, Modular
Transformers, and Trainer routes returned current content. VoiceHub's
supported-Python declaration and current test corpus control this slice.

The observable contract requires complete exact-worktree pytest execution on
each supported interpreter. Each non-default interpreter must run in a fresh
temporary virtual environment, import VoiceHub from this checkout, use the
declared `test` extra, and avoid every lockfile operation. Dependency setup,
test execution, platform, versions, duration, warnings, and skips must be
recorded separately. A setup failure, partial run, or skip is not a pass.

uv 0.11.21 created isolated environments with CPython 3.10.19 and 3.11.15.
`UV_TORCH_BACKEND=cpu uv pip install --python <environment> -e '.[test]'`
resolved and installed the declared test dependencies without using
`uv.lock`. Both environments imported
`/Users/kadirnar/Documents/voicehub/voicehub/__init__.py`. They used macOS
26.5.2 arm64, PyTorch 2.8.0, Transformers 5.14.1, and pytest 9.1.1. The Python
3.10 environment resolved 93 packages and the Python 3.11 environment resolved
90 packages under their respective compatibility constraints.

The complete Python 3.10.19 suite passed 2,577 executed tests and 4,131
subtests with 35 warnings in 173.72 seconds. The complete Python 3.11.15 suite
passed the same 2,577 executed tests and 4,131 subtests with 35 warnings in
163.92 seconds. The existing exact-current Python 3.12.12 suite passed those
same counts in 114.47 seconds with the same PyTorch, Transformers, and pytest
versions. Each interpreter reported 15 skips, which remain unpassed: three
complete-dependency default-runtime checks, three Triton and two CUDA-extension
checks, five opt-in ESPNet/NeMo/SenseVoice/SpeechBrain/TEN-VAD asset or oracle
checks, and two inaccessible WeNet paths.

Only this evidence record changed after the complete Python 3.10 and 3.11
runs. The release, distribution, and documentation-policy selection then
passed 77 tests and 1,476 subtests independently on Python 3.10.19, 3.11.15,
and 3.12.12. The strict eleven-language build completed in 35.55 seconds, and
the ten-route DOM validator retained all eight ordered navigation roots. A
fresh distribution check passed isolated wheel, sdist, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime-dependency violations, and no eager PyTorch import.
It produced a 57,194,971-byte wheel and a 55,466,862-byte sdist.

No runtime, test, workflow, package, generated page, or generated notebook
source changed in this platform-only slice. The preceding inventory evidence
therefore remains 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, 68 current provider pages, 59 current notebooks, 10
representative page pairs, six shared component/state groups, six public
optimization passes, 408 model/pass pairs, eight top-level navigation roots,
eight contribution steps, 261 public exports, and five benchmark records.

Committed HEAD `8ea5e941fcbcc0b93e5a4dd180b7a4c15c235930` retains successful
remote Continuous Integration, Documentation, and Package CI runs, but those
runs do not contain the current dirty worktree and are not exact-candidate
evidence. Newer successful runs for unrelated head `8fee3d7` likewise do not
cover this worktree. Exact-current-worktree Linux, Windows, default-runtime,
tagged-workflow, publisher, and publication gates, five CUDA/Triton checks,
seven opt-in or inaccessible asset paths, and the broader object-by-object
public API audit remain open. No protected action was taken. The untracked
`uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact current-worktree full-dependency default-runtime evidence

This bounded release iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments sources
retained the fingerprints recorded in the parity ledger. The official
navigation, documentation specification, Modular Transformers, and Trainer
routes returned current content; the documentation Home endpoint returned an
HTTP 429 during this retrieval and is not reported as passed.

The observable gate requires a fresh Python 3.12 environment to install the
checkout directly with all three declared extras, keep the lock file untouched,
pass dependency validation, execute the focused packaging and default-runtime
contracts, and complete the entire suite with
`VOICEHUB_FULL_RUNTIME_TEST=1`. A setup failure, skip, or partial run is not a
pass.

The first direct `uv pip install -e ".[test,training,docs]"` resolved the broad
librosa transitive range to numba 0.53.1 and failed because that release rejects
Python 3.12.12. That setup is recorded as failed evidence. The test extra now
declares the Python-3.12-compatible lower bound `numba>=0.59` while keeping
numba outside the default inference installation. A focused metadata contract
protects that boundary.

In the same still-empty temporary environment, uv 0.11.21 then installed all
135 packages directly from the current checkout with its CPU PyTorch resolver
and without synchronization or a lockfile operation. `uv pip check` reported
all packages compatible. VoiceHub imported from this checkout on macOS 26.5.2
arm64 with Python 3.12.12, PyTorch 2.8.0, Transformers 5.14.1, and pytest 9.1.1.
The focused metadata file passed 11 tests and 76 subtests, and the activated
default-runtime file passed five tests and 138 subtests.

The complete activated suite passed 2,581 tests and 4,269 subtests with 35
warnings in 176.53 seconds. Its 12 skips remain unpassed:

- Three Triton and two compiled CUDA-extension checks require unavailable CUDA
  hardware or a CUDA toolkit.
- The ESPNet, NeMo QuartzNet, SenseVoice, SpeechBrain, and TEN-VAD artifact or
  oracle checks remain opt-in and were not executed in this run.
- The WeNet checkpoint and tokenizer artifacts remain inaccessible.

Proportional registry, task, public-API, optimization, model-scaffold,
documentation, release, and packaging coverage passed 205 tests and 2,556
subtests. The release inventory retained five benchmark records and 68
documented providers: 34 TTS, 23 ASR, and 11 VAD. All 68 generated model pages
and 59 generated model notebooks remain current. The strict eleven-language
build completed in 35.90 seconds, and the ten-route DOM validator retained all
eight ordered navigation roots. This slice changes no rendered component and
makes no new visual-parity claim.

A fresh distribution check passed isolated wheel, source-distribution, and
editable installs with 68 models, 81 provenance manifests, 193 compliance
files, required package data, zero runtime-dependency violations, and no eager
PyTorch import. It produced a 57,194,978-byte wheel and a 55,467,389-byte
source distribution. The release report itself changed after this build, so
the source-distribution size is execution evidence rather than a final tagged-
artifact fingerprint.

This closes the locally executable full-dependency/default-runtime gate. It
does not substitute for exact-current-worktree Linux, Windows, or remote
default-runtime CI, and it does not refresh any asset, hardware, tagged-
workflow, publisher, or publication gate. No protected action was taken. The
untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current generated package-root public API evidence

This public-contract iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. The revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments sources
retain the fingerprints recorded in the parity ledger. The official
Transformers Models main-class page was rendered as the structural reference
at 1440 x 900, 1024 x 768, and 390 x 844.

The observable contract requires every unique name in `voicehub.__all__` to
resolve and appear exactly once in a generated public reference. Every entry
must record its kind, canonical repository module, source file and line,
callable signature or explicit constant/type-alias marker, summary, and root-
level lazy state. Duplicate, unresolved, undocumented, source-less, stale, or
version-dependent inventory output must fail. A fresh root import must remain
PyTorch-free.

The pre-implementation focused test failed during collection because the
generator did not exist and is excluded. Two intermediate generator runs then
failed while locating a facade constant and tuple assignment; those failures
led to generic facade re-export and assignment-target resolution and are also
excluded. Before the slice, 148 of 261 root exports were not individually
named in the two hand-written API pages.

`scripts/generate_public_api.py` now statically validates root export/source
metadata, follows direct and declarative facade re-exports, resolves canonical
repository definitions, and produces `docs/reference/public-api.md`. Its 261
records comprise 131 classes, 73 callables, 39 enums, 11 exceptions, four
constants, and three type aliases. The seven speech-domain groups contain 94
training, 81 optimization/codec, 29 configuration/factory/model, 28 inference/
serving, 20 input/output, eight policy/error/utility, and one package-metadata
entry. Every row has a local source line and a usable signature or explicit
marker; no entry uses an unavailable-signature fallback.

The first fresh Python 3.10 focused run exposed 39 Enum signatures whose
standard-library introspection differs from Python 3.11/3.12. It reported 66
passes, two failures, and 1,741 subtests and is not a pass. Enum rows now use
the stable public recovery contract `(value)`. The corrected public-API and
documentation selection passed 68 tests and 2,002 subtests independently in
fresh Python 3.10.19, 3.11.15, and Python 3.12.12 environments. The final
Python 3.12.12 complete suite passed 2,582 tests and 4,657 subtests with 15
skips and 35 warnings in 123.52 seconds. The skips remain unpassed: three
default-runtime checks, three Triton and two CUDA-extension checks, five opt-
in asset/oracle checks, and two inaccessible WeNet paths.

Proportional registry, task, optimization, model-scaffold, documentation,
release, packaging, and distribution-compliance coverage passed 190 tests and
3,075 subtests. All 68 generated model pages and 59 generated notebooks remain
current. The release inventory retained five benchmark records and 68
documented providers: 34 TTS, 23 ASR, and 11 VAD. The inventory remains 102
aliases, zero invalid display names, six public passes over 408 model/pass
pairs, eight top-level navigation roots, ten representative page pairs, six
shared component/state groups, eight contribution steps, and 261 public
exports.

The final strict eleven-language build passed. The ten-route DOM
validator retained all eight navigation roots, and a focused rendered check
found the new page's seven tables, 261 repository-source links, active
navigation, and inventory statement. The complete Playwright 1.62.0/Axe 4.12.1
matrix then passed 60 base cases, 60 screenshot signatures, 342 keyboard cases,
and 4,613 focus steps. Its first run failed because the Models page's expected
next footer still named Bark; this failed run is excluded, and the corrected
contract names Public exports. A separate corrected comparison passed the new
page in both palettes at all three mapped viewports with zero overflow and
rendered the official Transformers Models main-class page at those same
viewports. Three setup probes that used a missing script path, a nonexistent
upstream group route, and an incorrect heading selector did not reach the
comparison and are excluded.

A fresh isolated distribution check passed wheel, source-distribution, and
editable installs with 68 models, 81 provenance manifests, 193 compliance
files, required package data, zero runtime-dependency violations, and no eager
PyTorch import. It produced a 57,194,978-byte wheel and a 55,468,189-byte
source distribution.

This closes the broader object-by-object package-root public API audit. Exact-
current complete Python 3.10/3.11, Linux, Windows, remote default-runtime,
tagged-workflow, publisher, and publication gates remain pending. Five CUDA/
Triton checks and seven opt-in or inaccessible asset paths remain explicitly
unpassed. No protected action was taken. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current final supported-Python macOS evidence

This evidence-only platform iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. Its 65-line Home, 1,576-line
navigation, 500-line Modular Transformers, 30-line Trainer, and 2,906-line
TrainingArguments sources retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`,
and `cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`.
The official Home, documentation specification, Modular Transformers, and
Trainer routes returned HTTP 200.

The observable contract requires a fresh direct installation and complete
CPU-safe suite on every supported interpreter after the final generated-public-
API slice. Each environment must import the checkout rather than an installed
copy, pass dependency validation, retain a PyTorch-free package-root import,
and avoid every lockfile synchronization path. A partial run or skip is not a
pass.

uv 0.11.21 created fresh CPython 3.10.19 and 3.11.15 environments. Direct
`UV_TORCH_BACKEND=cpu uv pip install --python <environment> -e '.[test]'`
commands installed 93 and 90 compatible packages respectively without a
lockfile argument or synchronization; `uv pip check` passed in both environments. Both
resolved VoiceHub to
`/Users/kadirnar/Documents/voicehub/voicehub/__init__.py`, and isolated root-
import probes resolved all 261 public exports without importing PyTorch. The
host was macOS 26.5.2 arm64 with PyTorch 2.8.0, Transformers 5.14.1, and pytest
9.1.1.

The complete Python 3.10.19 suite passed 2,582 tests and 4,657 subtests with 15
skips and 35 warnings in 229.03 seconds. The complete Python 3.11.15 suite
passed the same counts in 214.94 seconds. Together with the exact-current
Python 3.12.12 result of the same counts in 123.52 seconds, all three supported
interpreters now pass the complete CPU-safe suite on the exact current macOS
worktree.

The 15 skips reproduced independently and remain unpassed:

- Three default-runtime import checks require the explicit full-runtime CI
  environment.
- Three Triton kernel checks require `VOICEHUB_TEST_TRITON_KERNELS=1` on a
  Triton CUDA host.
- Two compiled CUDA-extension checks require
  `VOICEHUB_TEST_CUDA_EXTENSIONS=1` on a CUDA toolkit host.
- ESPNet, NeMo QuartzNet, SenseVoice, SpeechBrain, and TEN-VAD require their
  pinned opt-in configuration, archive, tokenizer, or ONNX oracle assets.
- The WeNet conversion checkpoint and tokenizer assets remain inaccessible.

After the complete runs, the exact public-API, documentation, release,
packaging, and distribution-policy selection passed 92 tests and 2,078
subtests independently on Python 3.10.19, 3.11.15, and 3.12.12. The refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display
names, 68 current provider pages, 59 current notebooks, six public optimization
passes over 408 model/pass pairs, ten representative page pairs, eight top-
level navigation roots, eight contribution steps, 261 public exports, and five
benchmark records.

The final strict eleven-language build completed in 36.87 seconds, and the ten-
route DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding complete visual evidence; this
platform-only slice makes no new visual-parity claim. A fresh isolated
distribution check passed wheel, source-distribution, and editable installs
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime-dependency violations, and no eager PyTorch import.
It produced a 57,194,978-byte wheel and a 55,468,722-byte source distribution.
This evidence record changed afterward, so the sizes are execution evidence
rather than final tagged-artifact fingerprints.

No runtime, test, workflow, generated page, or generated notebook source
changed in this evidence-only slice. Exact-current Linux, Windows, remote
default-runtime, tagged-workflow, publisher, and publication gates remain
pending. The five CUDA/Triton gates and seven opt-in or inaccessible asset paths
remain explicitly unpassed. No protected action was taken. The untracked
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current pinned release-asset evidence

This checkpoint iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned reference
fingerprints remain unchanged. The observable contract requires every pinned
release asset to match its immutable source revision, declared byte size, and
SHA-256 before its behavioral, conversion, or differential test can pass. A
download failure, digest mismatch, skipped test, partial conversion, or missing
oracle dependency is not a pass.

The three small Hugging Face assets passed in one exact-current command:

- ESPNet repository
  `espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best`
  at revision `bc6bbd771cec698f070640ee677a66719181f0a2` supplied the
  82,131-byte `config.yaml` with SHA-256
  `16351b9bf79631d1df0a4645a858dc330c40434cf03470408c9c8fd446b6ea19`;
  its extracted token list matched
  `48ec6eedbee6a22e2a9b51adeb425af3c39db23128086c015240f591601a3ea3`.
- `FunAudioLLM/SenseVoiceSmall` revision
  `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` supplied the 377,341-byte
  tokenizer with SHA-256
  `aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8`;
  the published text/control vectors and semantic labels matched.
- `speechbrain/asr-crdnn-rnnlm-librispeech` revision
  `979a53a7a3f6c9291c02c040fd8ebfb2471cf8a3` supplied the 253,217-byte
  `tokenizer.ckpt` with SHA-256
  `37a6cba34cd520b33fd83612d5efc8ba7e351166541eb2726642bb3032234d31`;
  its published encode/decode vector matched.

That focused command passed three tests with 49 deselections in 1.90 seconds.
Each cached file was independently re-read and reproduced its declared URL,
revision, size, and digest.

The official TEN-VAD ONNX graph came from immutable source revision
`22a3bcd4509d0faaa8eef4881e8af5f39c178950`. The downloaded file was 315,449
bytes with SHA-256
`e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4`.
On macOS 26.5.2 arm64 with Python 3.12.12, PyTorch 2.8.0, and pinned ONNX
Runtime 1.22.1, the converter wrote the native Safetensors artifact, strict-
loaded it, and matched ONNX Runtime across 25 recurrent steps. The focused test
passed once with seven deselections in 0.91 seconds.

The official NVIDIA NGC `stt_en_quartznet15x5` 1.0.0rc1 archive downloaded
with identity encoding at 70,993,538 bytes and SHA-256
`1b9b7b87a9277e6fef164d8f99d1226f0511af154423bbf919b920421ac9602f`.
The exact-current converter reproduced tensor fingerprint
`47c098414f58e8380868692db82cf0e4cde3b2777be1cdfd557cb7c5865ef37e`;
the focused test passed once with eight deselections in 1.91 seconds.

One combined command then activated all five paths simultaneously and passed
five tests with 64 deselections in 2.80 seconds. Broader native-architecture,
registry, task, optimization, release, packaging, and distribution-compliance
coverage passed 169 tests and 1,029 subtests with all five assets active. No
runtime, test, workflow, generated page, or generated notebook source changed
in this evidence-only slice.

The release-policy selection passed 92 tests and 2,078 subtests. The strict
eleven-language build completed in 38.12 seconds, and the ten-route DOM
validator retained all eight ordered navigation roots. The unchanged rendered
shell retains its preceding complete visual evidence; this checkpoint-only
slice makes no new visual-parity claim. Fresh isolated wheel, source-
distribution, and editable installs passed with 68 models, 81 provenance
manifests, 193 compliance files, required package data, zero dependency
violations, and no eager PyTorch import. They produced a 57,194,978-byte wheel
and a 55,468,269-byte source distribution. This evidence record changed after
the build, so those sizes are execution evidence rather than final tagged-
artifact fingerprints.

This closes the five pinned opt-in release-asset/oracle gates for the exact
current macOS worktree. The two WeNet asset paths remain inaccessible and are
not passed. Three Triton and two compiled CUDA-extension gates remain hardware-
limited. Exact-current Linux, Windows, remote default-runtime, tagged-workflow,
publisher, and publication gates also remain pending. No protected action was
taken, and the untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current WeNet release-asset evidence

This release slice refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; all five pinned reference
fingerprints remain unchanged. Its observable contract requires an accessible
immutable source for the exact audited WeNet archive, independent size and
SHA-256 verification before parsing, successful restricted conversion and
tokenizer behavior, reproducible Package CI and tagged-workflow gates, and no
weakening of the pickle trust or checkpoint-license boundaries.

The original UCloud HTTP object still returns 404, its HTTPS host still fails
certificate-hostname validation, and the current WeNet download portal has an
expired certificate and returns `NoSuchKey` for the documented route. These
endpoints remain failed evidence. The public `openspeech/wenet-models`
Hugging Face mirror at immutable revision
`90acd57d17169a15d5ceab462c6e7db3bd003921` exposes
`gigaspeech_u2pp_conformer_exp.tar.gz` with linked size 503,845,602 bytes and
linked SHA-256
`061ccfa51d64ebe7ea091a5a13ae31e37d9c36f4eface5c7bafc80bd4a06b26e`.
An independent download reproduced both values and contained the five exact
audited source members. This makes the mirror byte-identical to the previously
fingerprinted upstream archive; it does not make the mirror the model's
upstream source or establish new checkpoint license terms.

With the downloaded archive and its extracted tokenizer members activated, the
real checkpoint conversion and tokenizer selection passed two tests with 18
deselections in 6.69 seconds. The converter verified the archive and all five
member digests, used PyTorch's restricted `weights_only=True` reader behind the
existing explicit trust flag, matched the 670-tensor, 136,225,077-value
namespace fingerprint, wrote native Safetensors, and strict-loaded the result.
The tokenizer reproduced the 4,999-entry unit inventory and the pinned `HELLO
WORLD` encode/decode behavior.

Runtime metadata now separates the failed original URL from the immutable
mirror repository, revision, filename, URL, size, and digest. `SOURCE.json`
records both the failed upstream routes and verified mirror without changing
the undeclared checkpoint-license status. The generated WeNet page keeps a
local converted-artifact quickstart and explains the trust-gated conversion.
Package CI and the tagged release build now download the exact mirror revision,
check its SHA-256 before extraction, and run only the two isolated WeNet gates;
the source pickle remains outside VoiceHub distributions.

The focused WeNet and workflow regression passed 22 tests in 7.11 seconds. The
activated WeNet, architecture, registry, task, optimization, documentation,
scaffold, release, packaging, and distribution-policy suite passed 224 tests
and 2,621 subtests in 17.89 seconds. The selected pre-commit sequence passed
every applicable hook. A post-evidence selection passed 88 documentation,
release, packaging, and distribution-policy tests plus 1,556 subtests; all
public-API, model-page, notebook, benchmark, version, and provenance
inventories remained current. Both workflow files parsed successfully as YAML.

The strict eleven-language build completed in 38.98 seconds, and the ten-route
DOM validator retained all eight ordered navigation roots. The unchanged
representative shell retains its preceding complete visual evidence; this
release-asset slice makes no new visual-parity claim. Fresh isolated wheel,
source-distribution, and editable installs passed with 68 models, 81 provenance
manifests, 193 compliance files, required package data, zero dependency
violations, and no eager PyTorch import. They produced a 57,195,308-byte wheel
and a 55,470,162-byte source distribution. This evidence record changed after
the build, so those sizes are execution evidence rather than final tagged-
artifact fingerprints.

This closes both previously inaccessible WeNet asset paths on the exact current
macOS worktree. Three Triton and two compiled CUDA-extension paths remain
hardware-limited and unpassed. Exact-current Linux, Windows, remote default-
runtime, tagged-workflow, publisher, and publication gates remain pending. No
protected action was taken, and the untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current full-dependency runtime refresh

This release-evidence iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the five pinned reference
fingerprints remain unchanged. The observable gate requires a fresh Python
3.12 environment installed directly from this worktree with all declared test,
training, and documentation extras, a clean dependency check, all focused
default-runtime tests executed rather than skipped, and a complete activated
suite. It does not transfer a macOS result to another interpreter or platform.

The repository `.venv` is not release evidence: `uv pip check` found an
unrelated installed `trainer` distribution that declares Python `<3.12`, so the
first command stopped before executing any tests. That failed command is not a
pass, and the existing environment was preserved without modification.

uv 0.11.21 created a fresh CPython 3.12.12 environment and installed 135
compatible packages directly from the current checkout through its CPU PyTorch
resolver without synchronizing or reading the repository lock file. `uv pip
check` passed. VoiceHub resolved to this checkout on macOS 26.5.2 arm64 with
PyTorch 2.8.0, Transformers 5.14.1, and pytest 9.1.1.

With `VOICEHUB_FULL_RUNTIME_TEST=1`, the focused default-runtime file passed all
five tests and 138 registry subtests in 9.34 seconds. The same environment then
passed the complete activated suite: 2,585 tests, 4,795 subtests, 12 skips, and
35 warnings in 181.72 seconds. The three default-runtime tests that the normal
suite skips all executed and passed.

The 12 residual skips remain separately classified and are not counted by this
command:

- Three Triton and two compiled CUDA-extension checks require unavailable CUDA
  hardware or a CUDA toolkit and remain unpassed.
- ESPNet, NeMo QuartzNet, SenseVoice, SpeechBrain, TEN-VAD, and the two WeNet
  checks require their opt-in assets or oracle environment. All seven have
  separate exact-current passing evidence in the immediately preceding release
  sections, but they were not executed in this full-runtime command.

The post-evidence documentation, release, packaging, and distribution-policy
selection passed 88 tests and 1,556 subtests. All public-API, model-page,
notebook, benchmark, version, and provenance inventories remained current, and
the selected pre-commit sequence passed. The strict eleven-language build
completed in 39.99 seconds, and the ten-route DOM validator retained all eight
ordered navigation roots. This evidence-only slice changes no rendered
component and makes no new visual-parity claim.

Fresh isolated wheel, source-distribution, and editable installs passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero dependency violations, and no eager PyTorch import. They produced a
57,195,308-byte wheel and a 55,471,307-byte source distribution. This evidence
record changed after the build, so those sizes are execution evidence rather
than final tagged-artifact fingerprints.

This closes the exact-current local Python 3.12 full-dependency/default-runtime
gate after the WeNet slice. It does not refresh Python 3.10 or 3.11 after those
source/test changes and does not substitute for exact-current Linux, Windows,
or remote default-runtime CI. Tagged-workflow artifacts, publisher
configuration, publication approval, and all five hardware gates remain open.
No protected action was taken, and the untracked `uv.lock` remained unchanged
at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current final supported-Python refresh

This release-evidence iteration refreshed Transformers `main` at commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; all five pinned reference
fingerprints remain unchanged. The observable gate requires fresh Python 3.10
and 3.11 environments to install this exact worktree directly, pass dependency
validation and the changed WeNet/workflow/documentation contracts, keep the
package-root import PyTorch-free, and complete the CPU-safe suite. A result from
the source state before the WeNet slice is not transferred.

uv 0.11.21 created CPython 3.10.19 and 3.11.15 environments and installed 93
and 90 compatible packages respectively through its CPU PyTorch resolver. Both
installs used direct `uv pip install` commands without synchronization or a
lockfile argument, and both passed `uv pip check`. Each resolved VoiceHub to
this checkout and resolved all 261 package-root exports without importing
PyTorch. The host was macOS 26.5.2 arm64 with PyTorch 2.8.0, Transformers
5.14.1, and pytest 9.1.1.

The focused WeNet metadata, Package CI, tagged-workflow, checkpoint-
documentation, and packaging contracts passed 15 tests and 76 subtests on each
interpreter: Python 3.10 in 18.03 seconds and Python 3.11 in 17.83 seconds. The
complete Python 3.10 suite then passed 2,582 tests and 4,657 subtests with 15
skips and 35 warnings in 226.26 seconds. Python 3.11 passed the same counts in
212.37 seconds.

After this evidence update, the public-API, documentation, release, packaging,
and distribution-policy selection passed 92 tests and 2,078 subtests on both
interpreters: Python 3.10 in 16.64 seconds and Python 3.11 in 15.65 seconds.
All 261 exports, 68 model pages, 59 notebooks, five benchmark records, version
metadata, and source-provenance inventories remained current.

The selected pre-commit sequence passed. The strict eleven-language
documentation build completed in 38.06 seconds, and the DOM validator retained
all ten representative routes and eight ordered navigation roots. Fresh
isolated wheel, source-distribution, and editable installs passed with 68
models, 81 provenance manifests, 193 compliance files, all required package
data, zero dependency violations, and no eager PyTorch import. They produced a
57,195,308-byte wheel and a 55,469,928-byte source distribution. This report
changed after the build, so those sizes are execution evidence rather than
final tagged-artifact fingerprints.

The 15 skips remain explicit and are not counted by either complete command:

- Three complete-dependency default-runtime imports require
  `VOICEHUB_FULL_RUNTIME_TEST=1`; the exact-current Python 3.12 runtime section
  above executes and passes them separately.
- Three Triton and two compiled CUDA-extension paths require unavailable CUDA
  hardware or a CUDA toolkit and remain unpassed.
- ESPNet, NeMo QuartzNet, SenseVoice, SpeechBrain, TEN-VAD, and the two WeNet
  paths require their opt-in assets or oracle variables. All seven retain
  separate exact-current passing evidence, but neither supported-version
  command executed them.

Together with the exact-current Python 3.12.12 full-dependency result, the
current macOS worktree now passes the complete suite on every supported Python
version. This does not substitute for exact-current remote Linux, macOS, or
Windows execution. Tagged-workflow artifacts, publisher configuration,
publication approval, and all five hardware gates remain open. No protected
action was taken, and the untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current repository-wide lint refresh

This release-evidence iteration retained Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd` and the five pinned reference
fingerprints. Its observable gate is stricter than `pre-commit run --all-files`
on a dirty candidate: every tracked file and every candidate untracked file
must enter one configured-hook invocation, while the protected untracked
`uv.lock` must not enter the argument set or change.

The resulting candidate set contained 4,675 files after explicitly excluding
`uv.lock`. End-of-file, trailing-whitespace, case-conflict, private-key,
AWS-credential, pyupgrade, isort, YAPF, Markdown-formatting, Flake8, and
docformatter hooks all passed. SHA-256 manifests captured before and after the
complete invocation were identical, so no hook rewrote a checked file. An
initial audit-wrapper attempt stopped before invoking pre-commit because its
NUL-delimited file list retained an empty sentinel; it changed no repository
file and is not counted as lint evidence.

The post-lint registry, public-API, documentation, optimization, scaffold,
release, and packaging selection passed 157 tests and 2,797 subtests in 20.25
seconds before the evidence edit and again in 21.62 seconds afterward. All 261
exports, 68 model pages, 59 notebooks, five benchmark records, version
metadata, and source-provenance inventories remained current. The strict
eleven-language build completed in 39.31 seconds, and the DOM validator retained
all ten representative routes and eight ordered navigation roots. Fresh
isolated wheel, source-distribution, and editable installs passed with 68
models, 81 provenance manifests, 193 compliance files, all required package
data, zero dependency violations, and no eager PyTorch import. They produced a
57,195,308-byte wheel and a 55,471,136-byte source distribution. This report
changed after the build, so those sizes are execution evidence rather than
final tagged-artifact fingerprints. The slice changes no runtime or rendered
component and makes no new visual-parity claim.

This closes the exact-current local repository-wide formatting-and-lint gate.
It does not substitute for exact-current remote lint or Linux, macOS, and
Windows execution. Tagged-workflow artifacts, publisher configuration,
publication approval, and the five hardware gates remain open. No protected
action was taken, and the untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Pull-request release-candidate verification

This final candidate iteration retrieved Transformers `main` at commit
`ff2421c67f35cc83a0fbabbc2633c96734685918` on 2026-08-04; the official
Transformers documentation route returned HTTP 200. VoiceHub implementation
commit `aead0611b9eafa0e20d32900568c063073976741` is the reviewed head of
[pull request 73](https://github.com/kadirnar/voicehub/pull/73) on
`codex/voicehub-0.3-rc`. The worktree contained only the protected untracked
`uv.lock`, which retained SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

The local focused Quickstart source regression passed two tests, three
subtests, and 62 deselections. The complete documentation source suite passed
64 tests and 1,480 subtests. The final Playwright 1.62.0/Axe 4.12.1 run passed
60 rendered cases, 60 screenshot signatures, 60 accessibility cases, 342
keyboard cases, and 4,591 focus steps across ten representative routes, three
viewports, and two palettes. Every applicable pre-commit hook passed for the
two changed checker files. Two preceding complete visual commands timed out in
the Quickstart interaction after exposing state left by the preceding full
focus cycle; they are failed diagnostic runs and are not counted. The passing
checker reloads the route before its independent content-tab interaction,
uses native radio-group ArrowRight activation, waits for focus and selection
to settle, and retains a one-CSS-pixel fractional viewport tolerance.

Exact-head GitHub Actions then passed every required pull-request job:

- [Documentation run 30875378385](https://github.com/kadirnar/voicehub/actions/runs/30875378385)
  passed in 45 minutes 34 seconds. Its Python 3.12.13 Linux job passed 64
  documentation-source tests, the strict eleven-language build in 74.81
  seconds, the ten-route DOM validator with all eight ordered navigation
  roots, 60 Linux screenshot signatures, 60 accessibility cases, 342 keyboard
  cases, and 4,587 focus steps. The Pages deployment job was skipped on the
  pull request and is not counted as passed.
- [Package run 30875378415](https://github.com/kadirnar/voicehub/actions/runs/30875378415)
  passed in 2 minutes 43 seconds. Wheel, source-distribution, and editable
  validation each found 68 models, 81 provenance manifests, 193 compliance
  files, all required package data, zero runtime-dependency violations, and no
  eager PyTorch import. The exact artifacts were a 57,195,308-byte wheel and a
  55,457,609-byte source distribution. All seven pinned asset/oracle gates,
  including both WeNet paths, executed and passed separately in this run.
- [Continuous Integration run 30875378456](https://github.com/kadirnar/voicehub/actions/runs/30875378456)
  passed lint, training, full-dependency runtime, both cross-platform runtime
  smokes, and the full suite on Python 3.10, 3.11, and 3.12 on Ubuntu, macOS,
  and Windows. Every ordinary matrix job executed 2,581 tests and 4,657
  subtests with 16 explicit skips. The full-dependency job executed 2,584 tests
  and 4,795 subtests with 13 explicit skips. The separate pre-commit.ci check
  also passed on the exact head.

The refreshed generated inventories remain 68 models (34 TTS, 23 ASR, and 11
VAD), 102 aliases, zero invalid display names, 68 model pages, 59 model
notebooks, six public optimizations over 408 model/pass pairs, ten
representative page pairs, eight top-level navigation roots, eight contribution
steps, 261 public exports, and five benchmark records. The direct model-page
and model-notebook generator checks passed. A bare local public-API generator
probe lacked the installed VoiceHub/PyTorch runtime and failed before
validation; it is not a pass and is superseded by the exact-head full-runtime,
package, and complete-suite jobs above.

Five native-kernel paths remain explicitly unpassed: three Triton tests require
`VOICEHUB_TEST_TRITON_KERNELS=1` on a Triton CUDA host, and two compiled
CUDA-extension tests require `VOICEHUB_TEST_CUDA_EXTENSIONS=1` on a CUDA
toolkit host. No tag or tagged-release workflow was created, the pull request
was not merged, publisher configuration was not changed, and no GitHub release
or PyPI publication was attempted. Those maintainer-controlled actions remain
outside release-candidate verification and require explicit approval.

## One-time publisher configuration

Before the first 0.3 publication:

1. In PyPI project settings, add a GitHub Trusted Publisher for repository
   `kadirnar/voicehub`, workflow `release.yml`, and environment `pypi`.
2. In GitHub, create the `pypi` environment and require a maintainer reviewer.
3. Do not add a long-lived PyPI API token. Only the publish job receives
   `id-token: write`.

PyPI's default per-file limit is 100 MB. `scripts/check_release.py --dist-dir`
rejects oversized or unexpected files before the protected publish job begins.

## Publish and post-publish verification

After every local and cross-platform gate is green, create the signed tag and
manually dispatch the release workflow with its publish confirmation enabled.
Approve the protected `pypi` environment only after reviewing the build job and
artifact hashes.

When PyPI finishes indexing the release, verify external parity:

```bash
python scripts/check_release.py \
  --tag v0.3.0 \
  --require-tag-at-head \
  --pypi-policy published
```

Only then create or finalize the matching GitHub release. If any external gate
fails, leave the candidate unpublished and record the exact blocker here.
