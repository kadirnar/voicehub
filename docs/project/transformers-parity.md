---
description: Versioned route mappings and rendered evidence for VoiceHub documentation parity with Transformers.
---

# Transformers documentation parity

This inventory records evidence instead of treating visual similarity as a
claim. A mapped route is not complete until its structure, interactions,
responsive states, accessibility, and screenshots have been checked.

## Reference snapshot

| Property | Value |
| --- | --- |
| Retrieved | 2026-08-05 |
| Transformers branch | `main` |
| Transformers commit | `838763bf4372a5d0e5643fbd76f88294fb66277f` |
| `_toctree.yml` SHA-256 | `9081408c4dc6b97cdcfb940f49868cddd51bd565aa56400f88d12a111dd485ea` |
| Rendered reference | `https://huggingface.co/docs/transformers/index` |
| VoiceHub route | `/voicehub/` |

The modern VoiceHub color tokens and speech-specific content are intentional
differences. Other shell, geometry, navigation, and interaction differences
remain gaps until the table below records executed evidence.

## Top-level navigation inventory

| Transformers section | Current VoiceHub destination | Current placement |
| --- | --- | --- |
| Get started | `/voicehub/` | `Get started`, first |
| Base classes | `/voicehub/models/providers/` | `Models`, second |
| Inference | `/voicehub/guides/inference/` | `Get started → Pipeline` |
| Training | `/voicehub/guides/trainer/` | `Train`, third |
| Quantization | `/voicehub/guides/optimization-overview/` | `Optimize`, fourth |
| Ecosystem integrations | Existing guide routes | Omitted from the primary navigation |
| Resources | Existing guide routes | Omitted from the primary navigation |
| API | `/voicehub/reference/models/` | `Models → Models API` |

VoiceHub intentionally condenses the upstream hierarchy into four user-facing
sections: `Get started`, `Models`, `Train`, and `Optimize`. Existing detailed
routes remain available without adding more root groups. All 68 generated model
pages remain grouped by TTS, ASR, and VAD under `Models`.

## Top-level navigation evidence

The current source and rendered DOM contracts require the four labels in that
exact order. Desktop and tablet expose the same concise left rail; mobile uses
the same hierarchy in its drawer. The longer eight-section comparison remains
historical evidence for the earlier parity layout, not the current navigation.

## Product header control evidence

VoiceHub now renders its product controls in the same semantic order as the
reference: product, search, version, language, theme, and source. The version
menu identifies the current `main` documentation as release candidate 0.3.0,
links the release-candidate status, and identifies PyPI 0.1.6 as the published
package. Pointer interaction, outside-pointer dismissal, Enter and Space
activation, Escape dismissal with focus restoration, `aria-expanded`, and a
two-pixel keyboard-focus outline were exercised in the rendered site.

At 1440 x 900 and 1024 x 768, VoiceHub now uses the reference shell geometry:
a 65-pixel global brand row, a 270-pixel documentation rail, and a 270 x
128-pixel rail-control block beginning below the brand row. The product region
measures 237 x 23 pixels at x = 16, y = 77; collapsed search measures
237 x 30 at x = 16, y = 112; and the 237 x 26 utility row begins at x = 16,
y = 150. Version, language, theme, and source measure 80 x 26, 48 x 26,
34 x 24, and 55 x 16 pixels respectively, including the reference's 12-pixel
theme/source gap. All four root sections are visible in the rail. At desktop
width the resulting regions are 270/900/270 pixels; at tablet width the right
rail is hidden and the regions are 270/754 pixels.

At 390 x 844, the product label becomes a compact `VoiceHub` label, search and
version become icon controls, and language is hidden like the reference. The
tail-control contract also hides theme and source at the same breakpoint;
fresh light and dark renders confirm that all three controls have zero rendered
width and height. The
opened 252 x 122-pixel version menu
remained inside the viewport in English LTR and Arabic RTL layouts. Both
VoiceHub themes were checked at desktop, tablet, and mobile widths with zero
horizontal overflow.

The reference's 65-pixel global row contains Hugging Face ecosystem navigation.
VoiceHub has no equivalent corporate ecosystem, so its matched global row
contains only the VoiceHub brand instead of inventing unrelated links. This is
an intentional product-content difference; the global-row and documentation-
rail structure and geometry are now covered. Complete sequential focus order
remains a separate interaction gate.

## Search interaction evidence

The expanded search dialog matches the reference outer bounds at all three
checked viewports: 500 x 72 pixels at x = 470, y = 64 on 1440 x 900; 500 x 72
pixels at x = 262, y = 64 on 1024 x 768; and 358 x 72 pixels at x = 16, y = 64
on 390 x 844. Its collapsed desktop and tablet form now also matches the
reference documentation-rail bounds at 237 x 30 pixels, x = 16, y = 112. The
two sites retain their own inner-field spacing and color systems.

VoiceHub's search opens through pointer input, Enter, Space, or Command/Ctrl+K,
focuses the query input, locks body scrolling, and closes with Escape. Escape
restores a visible two-pixel focus indicator in the collapsed search region;
at mobile width it restores focus to the named trigger itself. The trigger
reports `aria-expanded` and names its controlled dialog. The corresponding
reference Command+K and Escape behavior was executed. Its mobile search icon
opens the same 358 x 72-pixel dialog but has no accessible name in the checked
snapshot, so that upstream limitation is recorded rather than counted as an
accessibility pass.

The final VoiceHub state was exercised in light and dark themes at desktop,
tablet, and mobile widths, plus English LTR and Arabic RTL mobile layouts. Page
and header overflow remained zero. Intermediate probes are not passes: the
initial local dialog measured 688 x 48 pixels, the first geometry correction
was constrained to 625 x 92/100 pixels, the first Escape path did not restore a
visible focus indicator, and the first mobile overlay caused 77 pixels of
header overflow before the bounded fixes.

## Language interaction evidence

Transformers exposes language as a native select measuring 48 x 26 pixels in
its desktop and tablet documentation rail. VoiceHub now uses the same native
control dimensions and the same x = 100, y = 150 placement at both widths,
presents the current locale as an uppercase language code, and retains only the
eleven locales the site actually builds. Unsupported upstream locales are
content availability, not controls that VoiceHub claims to provide.

Pointer, Enter, and Space activation followed by Escape left the selected
language unchanged and returned focus to the native select on both sites. An
outside pointer moved local focus to the target control, and selecting `TR`
navigated from `/voicehub/` to `/voicehub/tr/` with the resulting document and
selected option both reporting Turkish. VoiceHub adds a `Select language`
accessible name and a visible two-pixel focus outline; the checked reference
select has no accessible name. The local light and dark states retained
readable foreground/background contrast and zero overflow at desktop and
tablet widths.

At 390 x 844, both sites hide the language select. VoiceHub verified that state
in English LTR and Arabic RTL routes with the correct hidden selected value and
zero overflow. Native dropdown contents are browser chrome and therefore are
not represented as DOM geometry or screenshot evidence. The replaced local
state was a 40 x 40-pixel icon plus a 125 x 200-pixel custom list with no
`aria-expanded`; the first native-select render was white text on a white
background. Those failed probes are not passes.

## Theme and source interaction evidence

At the 1280 x 720 reference viewport, the official Transformers theme button
measured 34 x 24 pixels and its adjacent GitHub link measured about 55 x 16
pixels with a 12-pixel gap. VoiceHub now renders those same outer bounds and
gap at 1440 x 900 and 1024 x 768. VoiceHub keeps its modern color tokens,
reports the next theme action by accessible name, and gives the source link an
`Open VoiceHub source repository` name. The checked reference theme button had
no accessible name, while its GitHub link exposed only the star count as its
name.

VoiceHub's theme button switched to dark and light by pointer, Enter, and Space.
After every switch, focus moved to the newly visible button and retained a
two-pixel outline with a two-pixel offset. The final light and dark desktop,
tablet, and mobile states had zero document and header overflow. Pointer and
Enter activation of the compact source link both navigated to the declared
VoiceHub GitHub repository.

The source contract hides language, theme, and source below the shared
59.984375-em breakpoint, matching the checked reference mobile visibility.
Fresh 390 x 844 light and dark renders measured zero width and height for all
three hidden regions. The earlier embedded-viewport attempt was blocked by URL
security policy, and the first native-link Enter probe did not navigate; neither
failed check is counted. The final checks used the browser viewport override
at exactly 1440 x 900, 1024 x 768, and 390 x 844, captured screenshots in both
palettes, and verified the explicit Enter path after correction.

## Representative page inventory

| Page type | Transformers route | VoiceHub route | Structural shell | Responsive and visual status |
| --- | --- | --- | --- | --- |
| Home | `/docs/transformers/main/en/index` | `/voicehub/` | The current product introduction, Features, Design, and Learn hierarchy, speech-specific hero, resource discovery, active navigation, right TOC, page actions, and Installation next navigation are mapped | Current local contract: all six viewport/palette cases pass the exact four-heading and three-entry TOC order, three feature targets, one tip, two design principles, 13 resource cards, four status badges, six images with correct decorative/alternative-text boundaries, exact registry and artifact-safety markers, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate page copy by keyboard, verify exact clipboard text and visible success state, retain focus and zero overflow, and rerun Axe |
| Installation | `/docs/transformers/main/en/installation` | `/voicehub/getting-started/installation/` | The current environment, source install, editable checkout, verification, cache, and offline hierarchy is mapped with Linux, macOS, and Windows tabs | Current local contract: all six viewport/palette cases pass the exact six-heading and five-entry TOC order, two tab sets with six options, 12 code blocks and code-copy actions, one page-copy action, one external destination, edit and previous/next targets, installation and offline-safety markers, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate the first visible code copy and six activate page copy by keyboard, verify exact clipboard text plus visible success and idle states, retain focus and zero overflow, and rerun Axe |
| Quickstart | `/docs/transformers/main/en/quicktour` | `/voicehub/getting-started/quickstart/` | The current Set up, Pretrained models, Pipeline, Trainer, and Next steps hierarchy is mapped with platform and task tabs, two tips, one model table, code blocks, active navigation, and the exact right TOC | Current local contract: all six viewport/palette cases pass the exact six-heading and five-entry TOC order, two tab sets with six options, two tips, one three-row table, 11 code blocks and code-copy actions, one page-copy action, seven internal content destinations, edit and previous/next targets, required workflow markers, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six cases activate the last option in every tab set; six more copy the active readable page by keyboard, verify exact clipboard text plus visible success and idle states, retain focus and zero overflow, and rerun Axe |
| Task guide | `/docs/transformers/main/en/pipeline_tutorial` | `/voicehub/guides/inference/` | The task/parameter hierarchy, active Pipeline navigation, exact right TOC, six runnable code examples, parameter table, and explicit large-input/model sections are mapped | Current local contract: all six viewport/palette cases pass exact rendered structure, content and component inventories, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate the first code-copy action by keyboard, preserve focus and overflow, verify exact clipboard content and visible success state, and rerun Axe |
| Model index | `/docs/transformers/main/en/model_doc/auto` | `/voicehub/models/providers/` | The registry example and complete 68-model inventory are mapped under `Models → Model list`; VoiceHub adds speech-specific discovery filters because the upstream Auto Classes page does not need to compare TTS, ASR, and VAD checkpoints | Current local contract: all six viewport/palette cases protect the generated model inventory, seven select facets, 11 capability filters, two resource filters, searchable language names, URL-restorable filter state, parameter and language sorting in both directions, model-type sorting, empty/reset states, three/two/one initial viewport card density, zero overflow, Axe, screenshots, and complete focus traversal. Audited learned-parameter totals are used where the registered native graph can be reconstructed; serialized totals are labelled explicitly, and missing totals remain visibly unreported and sort last |
| Model detail | `/docs/transformers/main/en/model_doc/speecht5` | `/voicehub/models/providers/speecht5/` | Usage, overview, paper and GitHub references, configuration, processing, inference, training and optimization, limitations, source-linked public facades, `Models → Text to speech → SpeechT5` active navigation, and right TOC are mapped through the shared generated-page contract | Current local contract: all six viewport/palette cases pass the exact 16-heading and 15-entry TOC order, eight generated table inventories, seven code blocks and copy actions, verified upstream and local source links, required checkpoint/optimization markers, exact Models ancestry, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate page copy by keyboard, verify exact clipboard text and visible success state, retain focus and zero overflow, and rerun Axe |
| Training | `/docs/transformers/main/en/trainer` | `/voicehub/guides/trainer/` | Trainer overview, speech-specific orchestration boundary, Next steps, active nested navigation, right TOC, source/edit action, copy-page action, and Fine-tuning next navigation are mapped | Current local contract: all six viewport/palette cases pass the exact two-heading and one-entry TOC order, zero table/code inventory, four exact next-step destinations, edit and Fine-tuning footer targets, required fail-closed training markers, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate page copy by keyboard, verify exact clipboard text and visible success state, retain focus and zero overflow, and rerun Axe |
| Optimization | `/docs/transformers/main/en/optimization_overview` | `/voicehub/guides/optimization-overview/` | The current Overview hierarchy, universal lifecycle, six public passes, evidence boundaries, active navigation, right TOC, page actions, and preserved detailed optimization guides are mapped | Current local contract: all six viewport/palette cases pass the exact eight-heading and seven-entry TOC order, one six-row technique table, one code block and copy action, all six public pass names, five exact next-step targets, edit and previous/next footer destinations, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate page copy by keyboard, verify exact clipboard text and visible success state, retain focus and zero overflow, and rerun Axe |
| Contribution | `/docs/transformers/main/en/modular_transformers` | `/voicehub/project/adding-a-model/` | The current Contribute navigation, modular contribution boundary, explicit eight-step speech integration path, per-step file inventory, page actions, and responsive process overview are mapped; the legacy `add_new_model` route is retained only as a secondary boundary | Current local contract: all six viewport/palette cases pass the exact 10-heading and nine-entry TOC order, eight process labels, three tables with 8/3/7 rows, 13 code blocks and copy actions, the complete scaffold/registry/training/optimization/package boundary markers, two final guide targets, edit and previous/next footer destinations, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate page copy by keyboard, verify exact clipboard text and visible success state, retain focus and zero overflow, and rerun Axe |
| API reference | `/docs/transformers/main/en/main_classes/model` | `/voicehub/reference/models/` | The current Main Classes navigation, Models title, pretrained speech bases, task-specific lifecycle, normalized outputs, source links, artifact boundary, page actions, and preserved comprehensive API are mapped | Current local contract: all six viewport/palette cases pass the exact five-heading and four-entry TOC order, four tables with 6/3/3/5 rows, two code blocks and copy actions, four exact facade-source links, three exact internal lifecycle links, the complete pretrained/output/save boundary markers, edit and previous/next footer destinations, geometry, Axe, reviewed screenshot signatures, and complete focus traversal. Six additional cases activate page copy by keyboard, verify exact clipboard text and visible success state, retain focus and zero overflow, and rerun Axe |

## API main-class route inventory

The current Transformers API `Main Classes` group is mapped below. A VoiceHub
route is provided when the public speech contract exists. A non-applicable
entry states the missing or out-of-domain contract explicitly instead of
inventing parity.

| Transformers entry | VoiceHub route or disposition | Status and speech-domain boundary |
| --- | --- | --- |
| Auto Classes | [Auto Classes](../models/providers/index.md) | Mapped to registry-derived configuration, processor, and task-model factories |
| Backbones | Not applicable | VoiceHub has no registry-wide public backbone base; model graphs remain explicit in their integrations |
| Callbacks | [Callbacks](../reference/api.md#callbacks) | Mapped to the shared Trainer callback contract |
| Configuration | [Configuration and processor factories](../reference/api.md#configuration-and-processor-factories) | Mapped to `VoiceHubConfig` and `AutoConfig` |
| Continuous batching | Not applicable | No registry-wide public continuous-batching scheduler is exposed; serving backends own that policy |
| Data Collator | [Data collators](../reference/api.md#data-collators) | Mapped to task-neutral and speech-specific collators |
| Logging | Not applicable | VoiceHub does not expose a public logging facade |
| Models | [Models](../reference/models.md) | Mapped to the representative pretrained speech-model page |
| Text Generation | [Generation](../reference/api.md#generation) | Adapted to speech generation and normalized audio output |
| Optimization | [Explicit optimization passes](../reference/api.md#explicit-optimization-passes) | Mapped to the universal optimization registry |
| Model outputs | [Model outputs](../reference/models.md#model-outputs) | Mapped to normalized TTS, ASR, VAD, and training outputs |
| PEFT | Not applicable | VoiceHub has model-specific training adapters but no public registry-wide PEFT contract |
| Pipelines | [Pipeline](../reference/api.md#pipeline) | Mapped to TTS, ASR, and VAD task pipelines |
| Processors | [Configuration and processor factories](../reference/api.md#configuration-and-processor-factories) | Mapped to `AutoProcessor`, text processing, and `AudioProcessor` |
| Exporters | [Save, load, and resume boundaries](../reference/api.md#save-load-and-resume-boundaries) | Mapped to portable and provider-native artifact boundaries |
| Quantization | Not applicable | VoiceHub exposes no registry-wide public quantization pass |
| Tokenizer | Not applicable | Speech tokenizers remain processor or model-local components; no public `AutoTokenizer` exists |
| Trainer | [Trainer](../reference/api.md#trainer) | Mapped to the shared speech Trainer loop |
| DeepSpeed | Not applicable | No public DeepSpeed integration is supported |
| ExecuTorch | Not applicable | No public ExecuTorch exporter or runtime is supported |
| Feature Extractor | [Configuration and processor factories](../reference/api.md#configuration-and-processor-factories) | Adapted to the shared audio processor contract |
| Image Processor | Not applicable | Image processing is outside the speech-model product scope |
| Video Processor | Not applicable | Video processing is outside the speech-model product scope |
| Kernels | [Explicit optimization passes](../reference/api.md#explicit-optimization-passes) | Mapped to capability-validated shared kernel passes |

The speech-specific [public exports inventory](../reference/public-api.md)
complements these task-oriented mappings. It is generated from the package-root
surface and records every export's canonical module, source line, kind,
signature or constant marker, summary, and lazy-loading state. This mirrors the
discoverability role of Transformers' `Main Classes` group without inventing
NLP-only abstractions.

## Shared component and interaction inventory

| Component or state | Current evidence | Remaining gate |
| --- | --- | --- |
| Global header and documentation controls | The 65-pixel brand row, 270-pixel rail, 128-pixel control block, product/search/utility coordinates, and exact compact-control geometry match at desktop and tablet widths; both palettes, LTR/RTL mirroring, zero overflow, and pointer/keyboard behavior pass. VoiceHub intentionally uses its own brand rather than Hugging Face-specific corporate links. All ten representative routes pass complete native focus traversal at desktop, tablet, and mobile widths in both palettes. A 60-case search matrix opens the desktop/tablet command dialog by keyboard and the mobile dialog by its visible pointer trigger, validates focus and ARIA/inert/tab-order state, reruns Axe, closes with Escape, and restores focus to the visible breakpoint-specific control; native Tab away from the desktop input cancels delayed restoration so traversal continues. A separate 60-case version matrix opens the exact three-destination menu by keyboard or pointer, verifies its current item, focus, geometry, ARIA state, overflow, and Axe result, then closes it with Escape and restores focus to the summary. A 40-case desktop/tablet language matrix verifies all 11 exact locale destinations, uses keyboard navigation to Turkish and pointer/semantic selection to Arabic, preserves the active palette, validates LTR/RTL direction and localized selection, reruns Axe after navigation, and retains zero overflow. A 40-case desktop/tablet theme matrix switches every representative route to the opposite palette by keyboard or pointer, preserves route, locale, geometry, focus, and overflow, and reruns Axe after every switch. A 40-case desktop/tablet source matrix focuses the exact repository link, validates its name, target, geometry, outline, overflow, and Axe result, and performs deterministic browser navigation by Enter or pointer | Non-representative routes and future shared controls remain outside the closed representative matrix |
| Left navigation | Desktop and tablet use a persistent 270-pixel rail with four root sections. Model list activates `Models → Model list`; every generated model page activates `Models → task → model`. The interaction matrix activates and restores all four roots plus the visible TTS, ASR, and VAD model branches in both palettes while preserving route, focus, ARIA, sticky geometry, overflow, and accessibility | Non-representative routes and future navigation structures remain outside the closed Pipeline and SpeechT5 matrices |
| Right table of contents | The representative routes use a 270-pixel desktop rail at x = 1,170; both rails remain fixed in the viewport while the article scrolls, and both collapse at 1,024 and 390 pixels with zero overflow. All ten routes now pass pointer and Enter activation in both palettes. Each of the 40 cases requires the exact hash and CSS target, one matching active link after smooth scrolling settles, visible target alignment beneath the header, Enter focus retention, zero overflow, and a post-action Axe pass. Complete native traversal covers every representative desktop TOC | Non-representative routes and future page templates remain outside the closed representative matrix |
| Main content | The ten representative pages retain the reference shell widths and zero overflow. Installation protects two platform-tab sets, 12 code-copy actions, source installation, cache, and offline boundaries. Quickstart protects platform and task tabs, 11 code-copy actions, the training boundary, and concise model/train/opt destinations. The model index protects all 68 generated provider links plus its compact searchable comparison grid; every model page protects paper, GitHub, source, checkpoint, training, and optimization evidence | Remaining page-specific states outside the ten representative inventories |
| Footer and page actions | Edit, copy-page, previous/next, back-to-top, and footer regions render on the representative pages. Installation links Overview to Quickstart; Quickstart links Installation to Pipeline. Shared copy actions activate once on Enter or Space, restore focus, preserve their focus treatment, and return success states to idle | Representative page-action behavior closed; non-representative routes, future controls, and intercepted external GitHub destinations remain outside the local matrix |
| Theme | VoiceHub and reference light and dark renders completed at desktop, tablet, and mobile. All ten representative VoiceHub routes now switch palette at both visible widths through 20 Enter and 20 pointer cases with exact focus, geometry, route, locale, overflow, and post-switch Axe checks; all 20 mobile base cases retain the intentionally hidden state | Non-representative routes and future page templates remain outside the closed representative matrix |

## Registry, public-contract, and contribution inventory

- The live registry contains 68 integrations: 34 TTS, 23 ASR, and 11 VAD.
  All 68 generated model pages and their `Base classes → Models` navigation entries are
  current; Auto Classes is generated once under `API → Main Classes`. Every
  `ModelSpec` derives an uppercase-first presentation label from its public
  class name while retaining its canonical lowercase registry key. The model
  index, page title, and navigation label consume that one display contract.
  Each page now follows the same nine-section model-detail contract and links
  its paper, upstream GitHub repository, resolved configuration, and model facade
  source without importing the implementation.
- The public optimization registry exposes `codec-kernels`, `compile`,
  `custom-kernels`, `diffusion-cache`, `diffusion-sampling`, and
  `flash-attention-4`. The registry inventory therefore contains 408
  optimization/model pairs. A registry-derived CPU-safe contract executes all
  408 pairs through public application and validation, manifest reporting,
  strict JSON serialization, normalized TTS/ASR/VAD output and state
  preservation, deterministic restoration, and post-restore cleanup. It
  rejects silent skips: the synthetic CPU inventory reports 68 reasoned eager
  fallbacks for `compile` and 340 reasoned not-applicable universal fallbacks
  for architecture-specific passes. Separate pass-specific suites exercise
  actual configured or compiled paths, failure rollback, and restoration.
  Five opt-in Triton/CUDA-extension checks remain hardware-limited and are not
  counted as passes.
- `pipeline()`, its three task adapters, `AutoConfig`, task-specific auto
  models, `AutoProcessor`, pretrained model bases, normalized typed outputs,
  `Trainer`, `TrainingArguments`, generation and inference configurations,
  registry APIs, and save/load methods provide the current Transformers-style
  mental-model surface. The pipeline dispatch and Torch-free public import are
  covered locally. `AutoProcessor.from_pretrained()` now separates
  configuration loader or override values in `config_kwargs` from processor
  construction and artifact-restoration values, including explicit-model-type
  and local-artifact paths. It delegates local directories, direct processor
  configuration files, and Hub identifiers to the selected processor class,
  reuses supported Hub options for both configuration and processor artifact
  resolution, and consumes loader-only values before processor state is built.
  A missing optional base processor artifact falls back explicitly to the
  supplied construction values. `VoiceHubProcessor` and `AudioProcessor`
  reject top-level or nested secrets during construction and untrusted artifact
  loading, recheck mutable state, and validate a subclass's final mapping before
  creating `processor_config.json` or its artifact directory. Hub tokens remain
  loader-only and safe construction fields still round-trip. Matching current
  Transformers configuration
  loading, `AutoConfig.from_pretrained()` now applies `subfolder` during both
  automatic model-type discovery and the concrete configuration load; task
  factories preserve that path through `config_kwargs` without eager model
  loading. `ModelSpec` also records each task-default or explicitly registered
  processor as a lazy import target. Registry-wide
  processor selection therefore constructs and offline-loads all 68 processors
  without importing a model wrapper or heavy backend. All 68 registered
  configurations likewise construct and serialize through `AutoConfig` in a
  fresh process without importing PyTorch or a named optional speech backend.
  All 68 configurations now share the same base credential boundary:
  construction rejects nested or top-level secrets, runtime Hub tokens are
  omitted, every current config rejects a secret added after construction, and
  final diff, JSON, representation, and checkpoint serialization fail before
  exposing an unsafe subclass payload. Embedded generation defaults now enter
  that constructor boundary for all 68 configurations. Standalone
  `TTSGenerationConfig` construction, untrusted checkpoint loading,
  dictionary conversion, representation, and checkpoint output enforce the
  same policy while retaining legitimate token-ID fields and a runtime-only
  Hub loader token. `VoiceHubManifest.metadata` now shares the serialization
  boundary: constructor and untrusted-load validation reject nested secrets,
  mutable post-construction state is rechecked, and save validates the final
  subclass payload before creating the artifact directory or a temporary
  manifest. Safe descriptive metadata continues to round-trip.
  `ASRInferenceConfig` and `VADInferenceConfig` now enforce the same final
  serialization boundary for representation and checkpoint output, in
  addition to their existing constructor, untrusted-load, and mutable-state
  checks. A malicious subclass fails before a task configuration file or its
  artifact directory is created; runtime Hub tokens remain excluded and safe
  fields such as `max_new_tokens` still round-trip.
  `TrainerState` construction, untrusted state loading, and final checkpoint
  serialization now enforce the same boundary across nested `log_history`
  values. Mutable or subclass-added credentials fail before an artifact path
  is created, and `Trainer.log()` rejects them before state mutation or
  callback dispatch. Ordinary metric fields, including `token_count`, still
  round-trip. Exact-resume dataset, collator, stateful-callback, optimizer, and
  scheduler fingerprints now reject credential-shaped fields after
  normalization. The complete checkpoint manifest is revalidated immediately
  before its atomic write, including subclass output, and untrusted manifests
  fail before model or runtime restoration. Safe identity and metric fields
  remain serializable. Exact-resume optimizer, scheduler, random-generator,
  gradient-scaler, callback, sampler, and strategy mappings now pass through
  the same final binary-state boundary. Unsafe writes fail atomically, and
  loaded state is checked before model or runtime restoration. This check is a
  credential boundary, not a claim that Python pickle accepts untrusted input;
  only trusted checkpoints with intact manifest integrity are supported.
  Model wrappers retain the same config class object when their runtime is
  eventually imported. Ambiguous or invalid mappings fail before loading. An
  object-by-object parity audit remains open and this inventory does not treat
  naming alone as behavioral parity.
- The documented contribution path contains eight steps: create the package,
  record provenance and license, define the config, implement the task wrapper,
  register once, declare training and optimization support, test the contract,
  and generate the model page. Its scaffold covers the model package,
  configuration, runtime, registration, manifest, pinned source and license,
  focused test, optional architecture package, and generated page. Exact
  comparison with the current Modular Transformers contribution path remains
  open.

## Home shell baseline

At a 1440 x 900 desktop viewport, the reference page rendered a persistent
left documentation navigation and a right table of contents. VoiceHub
previously set `hide: [navigation, toc]` on every localized home page,
producing one wide content column. The home sources now retain both shell
regions for all eleven built locales. Localized home hero images use
parent-relative asset URLs so they resolve through the shared site asset
directory instead of locale-local 404 routes. The rebuilt English home
rendered a 242-pixel left navigation, 736-pixel content column, and 242-pixel
right table of contents inside a 1,220-pixel main region with no horizontal
overflow.

The responsive comparison used the same rendered routes at 1024 x 768 and
390 x 844. VoiceHub had no horizontal overflow at either size. At tablet
width, both sites now retain a 270-pixel left navigation and remove the right
table of contents; VoiceHub's 1,009-pixel main region leaves a 739-pixel content
region beside the navigation after accounting for the viewport scrollbar.
VoiceHub hides the redundant documentation-drawer button in this state. At
mobile width, both sites collapse their documentation navigation and table of
contents, and VoiceHub retains the working drawer button. VoiceHub rendered its
default light palette and `slate` dark palette at all three viewports without
horizontal overflow. The reference also rendered in light and dark modes at
all three viewports; its mobile shell does not expose the theme selector, so
the light and dark mobile captures were reached from the desktop selector
state. The VoiceHub drawer opened from its unique header control. Its LTR
backdrop starts after the 242-pixel drawer and occupies the remaining 133 x 844
rendered pixels. The Arabic RTL render mirrors those regions, placing the
drawer at x = 133 and the backdrop at x = 0. Pointer dismissal closed both
drawers. Escape also closed the English drawer through the loaded keyboard
handler. The English paths returned the sidebar to x = -242 pixels and the RTL
path returned it to x = 375 after the transition; horizontal overflow remained
zero throughout. The desktop active link rendered with a highlighted
background and 700 font weight, while keyboard focus rendered a two-pixel
indigo outline with a two-pixel offset.

## Installation navigation-state evidence

The mapped Transformers Installation page renders its current left-navigation
item as a rounded filled control. At 1440 x 900, the reference item measured
about 218 x 31 pixels and VoiceHub measured 219 x 34 pixels. VoiceHub rendered
one visible active item, kept the `Get started` branch checked, and had no
horizontal overflow in its default light and `slate` dark themes. Keyboard
traversal rendered a two-pixel theme-colored outline with a two-pixel offset.

At 1024 x 768, the reference retained its 218 x 31-pixel active item. VoiceHub
retained a persistent 270-pixel sidebar, rendered one 212 x 34-pixel active
item in both themes, hid the mobile drawer button, kept `Get started` checked,
and had no horizontal overflow. At 390 x 844, the VoiceHub drawer opened to
242 pixels and its backdrop occupied the remaining 133 pixels. The current
page became one visible 251 x 48-pixel active row, `Get started` remained
checked, and a keyboard-focused drawer link rendered the same two-pixel focus
treatment in both themes. The mapped Transformers drawer rendered one filled
323 x 31-pixel Installation item with a visible keyboard focus outline. Both
mobile pages had zero horizontal overflow.

## Quickstart representative-page evidence

The mapped reference introduces setup, pretrained-model, inference, trainer,
and next-step concepts through a compact title hierarchy. VoiceHub now maps
that mental model to its own speech contracts: setup, lazy registry discovery,
pretrained task auto models, normalized TTS/ASR/VAD inference, training-support
inspection before `Trainer`, and task-oriented next steps. The reference's
agent-skill section, Hugging Face community panel, and Colab/Studio launch
badges have no current VoiceHub product contract, so VoiceHub does not fabricate
equivalents for them.

At 1440 x 900, both articles begin at x = 318 and measure 804 pixels. The title
uses 24/32-pixel type, 600 weight, and a 27.2-pixel bottom margin; section
headings use 20/28-pixel type with 40-pixel top and 23.2-pixel bottom margins;
and body copy uses 16.8/29.4-pixel type. VoiceHub's one-line code block measures
804 x 50 pixels with 14/24-pixel code type, matching the reference code
contract. VoiceHub keeps its 270/900/270 shell and the reference keeps its
Hugging Face-specific panels above the article.

At 1024 x 768, both articles remain at x = 318 and measure 658 pixels while the
right table of contents is hidden. At 390 x 844, both articles begin at x = 24
and measure 342 pixels. VoiceHub's mobile code block uses the reference's
12-pixel edge inset, 366-pixel width, and 13.16/22.56-pixel code type. Fresh
VoiceHub light and dark renders at all three viewports preserved those bounds
and had zero horizontal overflow; the reference default palette was measured
at the same three viewports.

The English navigation label and document title now both say `Quickstart`, and
the right table of contents exposes Set up, Discover models, Pretrained models,
Inference, Trainer, and Next steps. The page-level copy button aligns with the
title at 100 x 28 pixels. Pointer activation replaced a clipboard sentinel with
the 5,252-character article and left `aria-busy="false"`. Enter and Space both
left the named button focused and rendered its compatibility focus class with a
solid two-pixel outline and two-pixel offset. The in-app driver did not dispatch
the native button `click` after either key, so its clipboard sentinel remained
unchanged. Chrome could not navigate to the localhost preview in two bounded
attempts. Native keyboard activation is therefore inaccessible and is not
counted as passed; sequential focus remains pending separately.

The following gaps remain explicit and are not passed by this slice:

- Header controls now match the reference desktop and tablet documentation-rail
  structure and geometry. The corporate-row decision is resolved as a content
  difference: VoiceHub keeps its own brand and does not fabricate Hugging Face
  ecosystem links. Complete sequential focus order is still pending.
- The right table-of-contents sticky and scroll-tracking states are verified on
  Pipeline. The remaining representative-route focus and expanded-state matrix
  is not yet verified.
- All ten representative page types now have partial evidence. Their stated
  interaction gates remain open; no representative type is unexamined.

## Pipeline representative-page evidence

The mapped Transformers tutorial presents one `pipeline(task=..., model=...)`
entry point, then organizes the workflow by tasks, parameters, device, batch
inference, task-specific parameters, chunking, large datasets, and large
models. VoiceHub previously exposed only a TTS-specific guide and had no public
pipeline entry point. The bounded slice adds one dependency-light `pipeline()`
contract that selects the TTS, ASR, or VAD auto factory, preserves normalized
`TTSOutput`, `ASROutput`, or `VADOutput`, and wraps an existing model without
changing its device or runtime state. It does not fabricate vectorized batch or
universal chunking support; those paths remain explicit model or serving
contracts.

The focused source regression first failed because the public pipeline classes
and function did not exist. After the implementation, seven pipeline tests and
eight subtests covered aliases, all three inference methods, unchanged outputs,
task mismatch, incomplete models, loader-option separation, lifecycle
delegation, reserved options, and a Torch-free root import. The mapped guide
regression then passed with the Pipeline navigation label, task/parameter
hierarchy, all three normalized outputs, the existing measured-duration TTS
workflow, and the explicit non-batching boundary.

At 1440 x 900, the reference and VoiceHub articles both begin at x = 318 and
measure 804 pixels; their Pipeline titles use 24/32-pixel type. At 1024 x 768,
both articles measure 658 pixels from x = 318. At 390 x 844, both begin at
x = 24 and measure 342 pixels. The reference and VoiceHub had zero document
overflow at every viewport. VoiceHub light and dark renders retained the same
geometry at all three sizes, the complete right TOC appeared on desktop, and
the mobile drawer opened to x = 0 without overflow.

The Tasks anchor positioned its heading below the 65-pixel shell. Pointer copy
wrote the complete 6,742-character Pipeline article and reported `Copied`;
Command+K focused the mobile search field and Escape dismissed it. A synthetic
Enter probe restored named-button focus and the compatibility focus class but
did not receive clipboard permission, while pointer activation in the same
harness did. Native keyboard clipboard activation is therefore inaccessible
and not counted as passed; complete sequential focus remains pending.

## Auto Classes representative-page evidence

The current official reference is the Auto Classes page at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its 301-line source has SHA-256
`557f5836c0722fef6a484c46805dfab0eb69a387b028a914b132350edf09f167`,
and the pinned toctree has SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The reference presents extension, configuration, processing, generic and
task-specific auto-model entry points, then the generated class inventory.
VoiceHub maps that mental model to speech with `AutoConfig`, `AutoProcessor`,
the TTS, ASR, and VAD auto factories, lazy `available_models()` discovery, and
68 generated model guides. The upstream main-version banner, Hugging Face
community panel, and non-speech class families are product-specific and are not
fabricated locally.

The first focused regression failed because the local index still began with
`Model guides`. The generated contract now starts with `Auto Classes`, parses
all three Python examples, retains the nine required sections on every provider
page, and supplies one registry-derived display label to each index link, page
heading, and navigation entry without changing the canonical `model_type`.
The focused index and generated-guide slice reported two passes and 136
subtests; the separate all-task display-name regression reported one pass and
68 subtests. All 68 pages and all 59 model notebooks remain generator-current.

At 1,440 x 900, the local article begins at x = 318 and measures 804 pixels;
the 270-pixel primary and secondary rails are visible and `Auto Classes` is the
only active navigation item. At 1,024 x 768, the article remains at x = 318 and
measures 658 pixels while the right rail collapses. At 390 x 844, the article
begins at x = 24 and measures 342 pixels; all four tables scroll within their
own wrappers. Light and dark renders preserve the same geometry and have zero
document overflow at every viewport. The focused desktop navigation item
renders a solid two-pixel outline plus the existing inset focus cue. The mobile
drawer exposes Base classes, Models, and Auto Classes as the active path in both
palettes. Pointer page copy replaced a sentinel with the complete
6,200-character article. Registered models and Voice activity detection pointer
transitions retained their hashes, positioned their targets at y = 64, and left
exactly one matching TOC link active in the light and dark palettes.

The official page rendered at the same 1,440 x 900, 1,024 x 768, and
390 x 844 viewports in both light and dark. Its article begins at x = 318 and
measures 804 pixels on desktop, begins at x = 318 and measures 658 pixels on
tablet, and begins at x = 24 and measures 342 pixels on mobile. Desktop retains
the active Auto Classes route and its generated API TOC; tablet retains the
active left navigation; mobile uses the compact Auto Classes header; and every
size has zero document overflow. A pointer TOC transition preserved the
`#transformers.AutoConfig` hash and positioned its target at y = 36. The
reference's community panel and 324,041-pixel generated API article make its
page height intentionally different from VoiceHub's compact registry catalog.

The official copy button did not replace a clipboard sentinel in this harness,
so that reference action is not counted as passed. Native keyboard activation
and complete sequential focus remain inaccessible and are also pending rather
than passed.

## SpeechT5 model-detail representative-page evidence

The current reference is the official Transformers SpeechT5 page at commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its 87-line source has SHA-256
`71bba8a2921cf637383fb8d6f2fd66df9cd95deb59118b9f49e1362485c27eb5`,
and the pinned toctree has SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The page leads with an overview and then exposes generated configuration,
processing, base-model, task-model, and vocoder API cards. It includes
`SpeechT5Config`, tokenizer, feature extractor, processor, base model,
speech-to-text, text-to-speech, speech-to-speech, and HiFi-GAN headings,
expandable parameter controls, and copy actions. The main-version banner,
community panel, paper metadata, and provider-specific class inventory are
product content and are not fabricated locally.

VoiceHub maps that mental model through one generated contract shared by all 68
registered integrations. Every page now begins with a copyable Usage example,
then documents overview, serializable configuration, processing, inference,
training and universal optimization, checkpoint/provenance/license/limitations,
and public API. The public API names the stable configuration and task-model
facades, links directly to their resolved repository source, and keeps the
implementation lazy during generation. Checkpoint defaults remain explicitly
separate from real-checkpoint execution evidence. The first focused SpeechT5
regression failed on the old page hierarchy; after generation, the
representative and all-page contract passed for every configuration and model
source path, at least four parsed Python examples per page, explicit optional
dependency and hardware statements, limitations, and current navigation.

At 1,440 x 900, the VoiceHub article begins at x = 318 and measures 804 pixels;
both 270-pixel rails are present, SpeechT5 is active, the right table of
contents exposes all eight sections and nested public facades, and all eight
tables fit their wrappers. At 1,024 x 768, the article measures 658 pixels and
the right rail collapses. At 390 x 844, the article measures 342 pixels, the
drawer exposes the active Base classes, Models, Text to speech, and SpeechT5
path, and each table scrolls inside its wrapper while document overflow remains
zero. Light and dark renders retain that geometry and behavior at all three
widths. The right rail has zero width at tablet and mobile sizes. Pointer clicks
on Public API and SpeechT5ForTextToSpeech preserve their hashes, place the
target at y = 64, and leave exactly the matching TOC link active in the light
and dark palettes. The edit action targets the SpeechT5 source page, the source
control targets the repository, and both facade links resolve to the declared
local source paths.

The official page now also has exact 1,440 x 900, 1,024 x 768, and 390 x 844
renders in both its light and dark palettes. Its article begins at x = 318 and
measures 804 pixels on desktop, begins at x = 318 and measures 658 pixels on
tablet, and begins at x = 24 and measures 342 pixels on mobile. Desktop exposes
both rails and the SpeechT5 API TOC; tablet retains its active left navigation;
mobile retains the compact SpeechT5 header; and all sizes expose the page-copy
action and generated signature controls. Activating the first 60-parameter
control removes its collapsed overlay and exposes the full signature. The
reference theme menu successfully produced verified light and dark renders at
all three sizes.

At 390 pixels, the official generated page has a 594-pixel document scroll
width and a 570-pixel article scroll width. Long generated code comments and
API type names extend beyond the article instead of being contained. This is
an observed upstream defect, not a VoiceHub target: the local page retains a
390-pixel document scroll width and internal scrolling for wide content. Native
keyboard activation and complete sequential focus remain pending and are not
counted as passed.

## Trainer representative-page evidence

The mapped reference is the official Trainer overview at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its source has SHA-256
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`
across 30 lines, and the pinned toctree has SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
It introduces `Trainer` and `TrainingArguments` in two short paragraphs, then
provides one `Next steps` section linking fine-tuning, customization, data
collators, and callbacks. The rendered reference exposes one active `Trainer
overview` navigation item, one right-TOC link, copy and source actions, and a
Fine-tuning next link. Its main-version banner, community panel, and Trainer
video are product content that VoiceHub does not fabricate.

VoiceHub previously mapped this route directly to its 250-line fine-tuning
workflow, producing ten sections, twelve code blocks, and an unfamiliar
active `Get started` label. The bounded correction preserves that workflow as
`Fine-tuning` and adds a separate concise `Trainer overview` before it in the
nested Training navigation. The overview follows the reference's title and
single-section hierarchy while explaining the necessary speech-domain
difference: a model-owned objective and training profile must validate before
the shared loop loads. Its Next steps link the fine-tuning workflow, Trainer
architecture, model/checkpoint support matrix, and licensed data-preparation
path. No generic fallback loss or unsupported runtime claim was added.

At 1,440 x 900, the local article starts at x = 318 and measures 804 pixels;
both rails are present, exactly one visible `Trainer overview` item is active,
and the right TOC exposes `Next steps`. At 1,024 x 768, the article remains at
x = 318 and measures 658 pixels while the right rail collapses. At 390 x 844,
the article begins at x = 24 and measures 342 pixels; the opened drawer shows
one visible active overview item. Light and dark renders retained the same
geometry at all three widths with zero document overflow. Pointer copy
reported `Copied`, the edit action targeted `docs/guides/trainer.md`, the next
footer link targeted `Fine-tuning`, and focused navigation rendered a solid
two-pixel indigo outline with a two-pixel offset.

The exact reference sweep now closes the prior controller limitation. The
official and local pages rendered at 1,440 x 900, 1,024 x 768, and 390 x 844
with matching 804-, 658-, and 342-pixel articles. Light and dark were verified
on both sites at all three sizes. Each site retained zero document overflow,
one active Trainer overview route when its navigation was visible, a collapsed
right rail below desktop width, a mobile navigation path, page actions, and a
Fine-tuning destination. VoiceHub's mobile drawer kept its active Trainer
overview item in both palettes; the official mobile documentation menu did the
same.

VoiceHub page copy replaced a sentinel with the complete 1,257-character
article. The official copy action did not replace its 25-character sentinel in
this harness, so that interaction is a recorded failure rather than a pass.
The official `#next-steps` pointer transition retained its hash in both themes;
the short document bottom-clamped the target at y = 417 and did not mark a TOC
item active. Before the bounded fix, VoiceHub similarly scrolled its short page
but Material's tracking observer cleared the hash and left no active item. The
shared header-control script now waits for the pointer scroll to settle, then
preserves the requested hash and sole active TOC link without preventing the
native navigation. The corrected local transition retains `#next-steps`, marks
exactly one matching active link in both themes, and bottom-clamps the target at
y = 329. Direct anchored loading remains valid.

The first focused pytest invocation used the wrong test-class name and
collected no tests; the corrected Trainer contract passed. Native keyboard
activation and complete sequential focus remain inaccessible in the current
driver and are not counted as passed.

After the bounded repair, the focused Trainer and TOC contracts reported two
passes. The complete documentation contract reported 45 passes and 1,379
subtests, while the registry contract reported 17 passes and 193 subtests. All
68 model pages and 59 model notebooks remained generator-current, release
alignment found five benchmark files and 68 documented providers, and the
strict eleven-language documentation build passed. The selected code,
credential, whitespace, and file-integrity pre-commit hooks passed; the
Markdown hook had no matching files and is not counted as executed evidence.
Public optimization suites, distribution probes, and the complete Python suite
were not rerun for this documentation-shell-only change. Exact-commit remote
CI, protected publisher configuration, and publication approval remain open.
The untracked `uv.lock` stayed unchanged.

## Optimization representative-page evidence

The current reference is the official Transformers Optimization overview at
commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its source has SHA-256
`19622667a7299f258f5c9a72940c9f26492619636f35d9bd592701c02745b620`,
and the pinned toctree has SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The previous `perf_infer_gpu_one` mapping is absent from that toctree. The
current route is `optimization_overview`, whose hierarchy is Overview,
Compilation, Attention backends, Kernels, Quantization, Caching, Parallelism,
and Continuous batching.

VoiceHub now places a concise `Optimization overview` before the preserved TTS,
codec, diffusion, and VITS-family workflows. The overview derives its inventory
from the six public passes: `compile`, `flash-attention-4`, `custom-kernels`,
`codec-kernels`, `diffusion-cache`, and `diffusion-sampling`. One copyable helper
shows discovery, application, manifest reporting, and restoration through the
universal contract. The page does not infer compatibility from discovery: it
states that runtime validation happens before mutation and that failures roll
back reversible work. It also records exact public boundaries instead of
inventing parity: VoiceHub has no registry-wide public quantization pass,
parallelism remains a training or serving topology, and continuous batching
belongs to the serving scheduler.

At 1,440 x 900 the local article starts at x = 318 and measures 804 pixels,
with one active Overview item, the expected seven-link right TOC, copy and edit
actions, and Model support/TTS optimization previous-next navigation. At
1,024 x 768 the article measures 658 pixels and the right rail collapses. At
390 x 844 it starts at x = 24 and measures 342 pixels; the wide technique table
is contained by an internal 374-pixel wrapper with no document overflow, and
the opened drawer exposes Overview followed by all four detailed guides. Light
and dark renders retained the same geometry and zero document overflow at all
three widths.

The exact comparison now closes the previous reference-theme gap. The official
and local pages rendered at 1,440 x 900, 1,024 x 768, and 390 x 844 with
matching 804-, 658-, and 342-pixel articles. Light and dark were verified on
both sites at all three sizes. Both retained zero document overflow, an active
Overview route when navigation was visible, a desktop right TOC, collapsed
responsive rails, page actions, and mobile documentation navigation. The local
mobile technique table remained contained by a 374-pixel wrapper with a
752-pixel internal scroll width. The main-version banner and community panel
are upstream product content that VoiceHub does not fabricate.

VoiceHub page copy replaced a sentinel with the complete 3,730-character
article. Its Compilation and Diffusion sampling pointer transitions retained
their requested hashes, landed between y = 64.1 and y = 64.2, and left exactly
one matching TOC link active in both palettes. The official Compilation and
Caching pointer transitions retained their hashes at y = 35.9 and y = 36.2.
The official page-copy control did not replace its 21-character sentinel in
this harness, so that interaction remains a recorded failure rather than a
pass. Native keyboard activation and complete sequential focus remain pending
and are not counted as passed.

The original focused regression failed because the fifth navigation group
still pointed to the detailed TTS workflow; that development failure is not a
pass. For this evidence closure, the focused Optimization contract reported one
pass and six subtests. The complete documentation contract reported 45 passes
and 1,379 subtests. The registry and all documented public optimization suites
reported 178 passes and 944 subtests; one platform-specific PyTorch
decomposition warning and three weight-normalization deprecation warnings are
recorded warnings, not passes or failures. All 68 model pages and 59 model
notebooks remained generator-current, `scripts/check_release.py` found five
benchmark records and 68 documented providers, and the strict eleven-language
documentation build passed. The selected file-integrity, whitespace,
credential, and case-conflict pre-commit hooks passed; code and Markdown hooks
had no matching files and are not counted as executed evidence. Distribution
probes and the complete Python suite were not rerun for this evidence-only
change. Exact-commit remote CI, native keyboard activation, complete sequential
focus, protected publisher configuration, and publication remain open. The
untracked `uv.lock` stayed unchanged.

## Contribution representative-page evidence

The current reference is the official Transformers modular contribution guide
at commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its source has SHA-256
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`,
the legacy contribution source has SHA-256
`4d7e7066deeefde340c3e0460eae540f343cdfe642d690a600cba0a90441cb03`,
and the pinned toctree has SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The toctree places `modular_transformers` under Base classes, Models,
Contribute and labels `add_new_model` as Legacy model contribution. VoiceHub
now mirrors that information architecture with Add a model, Add an ASR or VAD
provider, and Add an optimization under the same nested contribution group.

VoiceHub maps the reference's reduced-boilerplate, standalone-file outcome to
speech-specific composition instead of generating a provider runtime through
inheritance. The guide states that boundary directly: checkpoint conversion,
audio processing, codecs, and streaming state remain locally explicit. Its
eight observable steps are Create, Audit, Configure, Wrap, Register, Support,
Test, and Document. A matching table names the package, optional architecture,
provenance and license records, configuration, runtime wrapper, manifest,
training and optimization factories, CPU-safe tests, generated provider page,
and navigation artifact produced or checked by each step.

At 1,440 x 900 the local article starts at x = 318 and measures 804 pixels. Its
eight-card process renders as two 392-pixel columns, the Contribute branch is
expanded, Add a model is active, the eight step headings appear in the right
TOC, all three tables are contained, and the copy/edit actions and adjacent
contribution navigation render. At 1,024 x 768 the cards remain a two-column
319-pixel grid in a 658-pixel article and the right rail collapses. At 390 x
844 the 342-pixel article contains one 342-pixel card column, the drawer exposes
Contribute and Add a model, all three tables scroll inside their 374-pixel
wrappers, and the document has zero horizontal overflow. Light and dark
VoiceHub renders preserve those states. Pointer copy replaced a sentinel with
the complete 18,197-character article. The Register once pointer transition
retained `#5-register-once`, placed the heading at y = 64, and left exactly one
matching right-TOC link active.

The official page rendered 804-, 658-, and 342-pixel articles at the same three
viewports in light and dark with its current title, active
modular-contribution item, desktop right TOC, main-version banner, community
panel, copy action, table, and mobile documentation menu. It retained zero
document overflow at every size. Pointer copy replaced a sentinel with the
complete 24,443-character Markdown source. The Generate the modeling files
transition retained `#generate-the-modeling-files`, focused the requested link,
and placed the target at y = 36; the official TOC does not expose a separate
active-link state for that transition. The mobile menu retained its filled
active modular-contribution item. Upstream product banners and the
inheritance-based implementation technique are intentional differences;
VoiceHub preserves the shared contribution outcome and explicit generated-file
contract. Native keyboard activation and complete sequential focus remain
pending and are not counted as passed.

The initial focused contract failed because the contribution route still used
the legacy mapping and seven process cards. A later stylesheet regression
failed before the selector was corrected, and the first mobile render exposed
that the strengthened desktop selector also needed a matching mobile
specificity. None of those failures is counted as passed. Two exploratory
inventory commands also failed after assuming nonexistent optimization and
model-spec attributes; the corrected live inventory used the public
optimization exports and `MODEL_ALIASES`.

After correction, the focused contribution/navigation/stylesheet regression
reported three passes, 38 subtests, and 41 deselections. The complete
documentation contract reported 44 passes and 1,310 subtests; the model
scaffold contract reported 20 passes and 35 subtests; and the registry plus
public optimization slice reported 76 passes and 786 subtests. All 68 model
pages and 59 model notebooks remained generator-current,
`scripts/check_release.py` found five benchmark files and 68 documented
providers, and the strict eleven-language build passed. The full suite and
distribution probes were not rerun for this navigation, prose, test, and CSS
slice; their preceding evidence remains applicable. A later evidence-only pass
confirmed the exact light/dark geometry and pointer states above without a
product change. Exact-commit remote CI, native keyboard activation, complete
sequential focus, protected publisher configuration, and publication remain
open.

## Models API representative-page evidence

The current reference is the official Transformers Models API page at commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its 44-line source has SHA-256
`a4899c758b5d621075b2e2f39f0aa79671010c88b08d1508f7af5731375c2871`,
and the pinned toctree has SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The current API Main Classes group contains 24 entries. The source inventory
above maps each entry to a VoiceHub route or records an explicit unsupported or
out-of-domain boundary.

VoiceHub now exposes `API > Main Classes > Models` before the preserved Full
API reference. The representative page follows the reference's Models title
and pretrained-model emphasis while adapting the class hierarchy to speech:
`PreTrainedSpeechModel` is the common marker, and the TTS, audio, ASR, and VAD
bases own their explicit input/output semantics. The page documents the shared
lazy `from_pretrained()`, `load()`, `load_for_training()`, support-validation,
and `save_pretrained()` lifecycle; normalized TTS, ASR, VAD, and training
outputs; direct source links; and the portable/native artifact boundary. It
states that VoiceHub has no public `push_to_hub()` contract instead of implying
unsupported registry-wide sharing.

At 1,440 x 900 the local article starts at x = 318 and measures 804 pixels.
Both rails render, Models is active under Main Classes, the right TOC exposes
all four sections, two code blocks fit their containers, all four tables are
contained, and the previous/next links target Model audit and Full API
reference. At 1,024 x 768 the article remains at x = 318 and measures 658
pixels while the right rail collapses. At 390 x 844 it starts at x = 24 and
measures 342 pixels; the drawer exposes API, Main Classes, Models, and Full API
reference, and all four tables scroll inside 374-pixel wrappers. Light and dark
VoiceHub renders retain the same geometry and zero document overflow.

Pointer page copy replaced a sentinel with the 5,106-character page and left
`aria-busy="false"`. The three declared source targets resolve to the public
modeling, audio-modeling, and output modules. The initial Model outputs pointer
probe positioned the target at y = 68 while Material's observer remained on the
preceding section; that failed state is not a pass. The shared target offset now
lands all four Models headings between y = 63.8 and y = 64.1. In both VoiceHub
palettes, each pointer click preserves its requested hash and leaves exactly its
matching link active with zero overflow. Enter left the copy button focused
with a solid two-pixel outline and two-pixel offset but did not replace the
clipboard sentinel. Native keyboard copy activation and complete sequential
focus remain pending and are not counted as passed.

The official page rendered 804-, 658-, and 342-pixel articles at the same three
viewports in light and dark with its Models, PreTrainedModel, ModuleUtilsMixin,
and Pushing to the Hub hierarchy, active Models item, desktop right TOC,
generated signature expand and copy controls, responsive navigation, and
internally scrollable generated API content. Every checked viewport had zero
document overflow. Pointer copy replaced a sentinel with the complete
49,539-character Markdown source. The PreTrainedModel transition retained
`#transformers.PreTrainedModel`, focused the requested link, and placed the
target at y = 36. Expanding the first collapsed control exposed all 14 hidden
parameters, and the light and dark mobile menus preserved their active Models
item. The main-version banner, community panel, token-embedding utility, and
Hub-specific mixin are upstream product/content differences that VoiceHub does
not fabricate.

The initial focused regression failed because API was still a flat navigation
entry and the representative page did not exist. After correction it reported
one pass and 50 subtests. The complete documentation contract reported 45
passes and 1,379 subtests; the base API, speech-core, and registry slice
reported 49 passes and 216 subtests; and the public optimization slice reported
59 passes and 593 subtests. All 68 model pages and 59 model notebooks remained
generator-current, `scripts/check_release.py` found five benchmark files and
68 documented providers, and the strict eleven-language build passed. An
exploratory optimization inventory incorrectly treated its registry as
iterable and failed; the corrected public `list()` query reported six passes
and 408 model/pass pairs. The failed query is not evidence.

The full suite and distribution probes were not rerun for this navigation,
documentation, and test slice; their preceding evidence remains applicable.
Exact-commit remote CI, native keyboard copy activation, complete sequential
focus, protected publisher configuration, and publication remain open.

## Shared page-copy keyboard evidence

The bounded shared-action correction makes the page-copy button own its
Enter and Space paths instead of depending on a browser-generated click. The
handler prevents the second native activation, invokes the existing copy
routine once, preserves the `aria-busy` lifecycle and live `Copied` state, and
restores focus with the established two-pixel treatment. Pointer activation
continues through the existing click path.

Before correction, a rendered Enter probe left its seeded clipboard unchanged.
After correction, Enter and Space each replaced a fresh sentinel with the full
5,252-character Quickstart article at 1280 x 720. Both activations left the
button focused, retained its focus-visible state, returned `aria-busy` to
`false`, and reported `Copied`. The same Enter path passed in the dark palette,
and pointer activation still copied the article while removing the
keyboard-only focus class. The light and dark pages retained zero horizontal
overflow. The official Quicktour exposed its corresponding action as a button,
but its Enter probe left the seeded clipboard unchanged, so that upstream probe
is recorded as failed rather than passed.

The in-app browser surface did not expose viewport resizing in this iteration.
The shared JavaScript correction changes no markup or CSS, so the immediately
preceding exact 1440 x 900, 1024 x 768, and 390 x 844 geometry remains the
applicable responsive evidence; no new exact-viewport keyboard claim is made.
At that point, native Enter activation of focused right-TOC links and complete
sequential focus remained separate pending gates.

## Right table-of-contents interaction evidence

The current official Pipeline page and the mapped VoiceHub Pipeline guide were
remeasured at the pinned Transformers revision. At 1,440 x 900, both right
rails begin at x = 1,170 and measure 270 pixels wide. The reference rail stays
at the viewport top and its Tasks and Large models links remain at y = 128 and
y = 352 while the matching headings move through the article. VoiceHub's rail
stays below the 65-pixel global header; its scroll region remains at y = 89 and
its links do not move while the article scrolls.

Material's tracking contract marks exactly one VoiceHub link after its heading
crosses the fixed-header threshold. Tasks, Parameters, and Large models each
became the sole active link in separate scroll states. The earlier shared
filled navigation pill made this secondary state heavier than the reference.
The bounded correction retains the active-heading cue but removes its fill and
inset border, leaving only the permitted VoiceHub palette color and weight.
Light and dark renders both produced a transparent background, no box shadow,
and zero horizontal overflow.

The Models API route exposed a four-pixel mismatch between Material's default
hash target position and that same observer: the target stopped at y = 68, while
the active link changed only after the heading reached approximately y = 64.
The shared target margin now resolves to 64 pixels, aligning navigation and
tracking without adding a second JavaScript observer. All four Models links and
the Tasks, Parameters, and Large models Pipeline links retained their requested
hash and became the sole active link in light and dark renders. Manual Pipeline
scroll states at the same threshold preserved the existing one-link tracking.

At 1,024 x 768 and 390 x 844, both official and VoiceHub right rails are hidden
and both documents retain zero horizontal overflow. Pointer activation places
the target heading below the fixed header, and the focused VoiceHub link shows
a solid two-pixel outline with a two-pixel offset. Locator-driven Enter and a
native browser keypress both left the link focused but did not navigate to its
target in this harness. Keyboard anchor activation therefore remains an exact
pending gate and is not counted as passed.

A later bounded correction gives the shared secondary rail one unmodified
Enter path. It prevents the second native activation, delegates to the existing
click path so hash-settling behavior remains single-sourced, and restores focus
without scrolling. The strengthened source contract failed before the handler
existed and passed after correction.

At the available 1280 x 720 desktop viewport, Enter on Parameters in the dark
palette preserved `#parameters`, settled the heading at y = 63.8, left exactly
that TOC link active and focused, and retained the solid two-pixel outline with
a two-pixel offset. Enter on Large models repeated those results in the light
palette at y = 63.9. Pointer activation of Tasks still settled at y = 64.0 with
one matching active link. Modified Ctrl+Enter remained unintercepted and did
not navigate in the harness. Every checked state retained zero horizontal
overflow. The official Pipeline link also remained focused without navigating
under the same Enter probe, so that upstream result is failed evidence rather
than a pass.

The in-app surface again exposed no viewport resizing. The handler changes no
markup or CSS, the right rail is already verified as hidden at 1024 and 390
pixels, and the immediately preceding exact responsive geometry remains
applicable; no fresh 1440-pixel Enter claim is made. Complete sequential focus
is now the remaining shared keyboard gate.

## Left documentation rail scroll-offset evidence

The current official Pipeline page lets its 65-pixel global row leave the
viewport while the left product controls and navigation settle at the top. At
1,440 x 900, its product label moves from y = 81 to y = 16 and its navigation
scroll region moves from y = 192.5 to y = 127.5 while retaining a 772.5-pixel
height. VoiceHub previously kept the global row at y = 0 and both left-rail
regions below y = 65 for the entire article scroll.

VoiceHub now tracks at most the first 65 pixels of article scrolling without
changing the mobile shell. At the desktop page start, its product controls are
at y = 65, the product label is at y = 77, the 270-pixel primary rail is at
y = 65, and the independent navigation region is at y = 193 with 707 pixels
available. After the threshold, the visual global row is translated out of the
viewport, the product controls and primary rail settle at y = 0, the product
label settles at y = 12, and the navigation region settles at y = 128 with
772 pixels available. The article and navigation can scroll independently,
and the document retains zero horizontal overflow.

At 1,024 x 768, the same final state keeps the 270-pixel product and primary
rail at y = 0, reserves the first 128 pixels for its controls, provides a
640-pixel navigation region, and keeps the secondary rail hidden. The geometry
is identical in the light and dark palettes. At 390 x 844, the existing
64-pixel mobile header remains at y = 0 while the article scrolls, and the
242-pixel drawer still opens without document overflow. A locator-driven
keyboard probe focused the desktop search input and rendered a solid two-pixel
indigo outline with a two-pixel offset around its form.

The first focused regression failed because no scroll-offset contract existed.
An initial transform on the header changed the containing block for fixed
descendants, and a later attempt to move the header itself clipped or obscured
the product controls; neither render is counted. Material also wrote inline
sidebar top and height values and retained a tablet z-index, so the final
scoped rules explicitly override those framework-owned values and preserve the
tablet control stack. The exact reference and local positions above are the
post-correction evidence.

## Left navigation focus-order evidence

The current reference remains the official Transformers Pipeline page at
commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. Its pinned navigation
SHA-256 remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the rendered Pipeline endpoint returned HTTP 200. At the available 1280 x
720 desktop viewport, the official left rail exposes the current inference
branch rather than every descendant of every collapsed branch.

VoiceHub previously left focusable descendants in every visually collapsed
desktop branch. The unchecked Base classes branch alone exposed 88 focusable
descendants, including all 68 model pages, before the current Pipeline article.
The first correction moved the existing grid, opacity, and visibility collapse
to desktop. Its source contract passed, but the rendered inventory still found
the same inactive descendants because descendant visibility rules overrode the
parent. That intermediate render is failed evidence and is not counted as a
pass.

Unchecked desktop branches now use `display: none`, while the checked or
indeterminate branch uses `display: block`. In both VoiceHub palettes, the
rendered Pipeline page exposed 63 visible focusable elements, including the
version summary, and zero focusable descendants inside inactive branches. All
eight root controls remained visible. The checked Inference and Pipeline API
branches retained the active Pipeline link and its Speech recognition and VAD
siblings, while unchecked Serving descendants remained hidden. The article
started at x = 318 and measured 644 pixels, matching the official page at this
viewport, and every checked state retained zero horizontal overflow.

Desktop section labels are now pointer-operable controls with synchronized
`aria-expanded` and `aria-controls` state. Pointer activation changed Base
classes from collapsed to expanded, displayed exactly its Models,
Preprocessors, and Architecture child controls, and left the nested Models
branch collapsed. The branch returned to the same inactive-descendant-free
state when collapsed.

Native Tab presses from the in-app browser, DOM keyboard surface, and locator
surface did not advance focus, so complete sequential focus remains unavailable
and unpassed. Locator-driven Enter and Space on the branch label also failed
because the browser reported that its focused input target no longer matched
the resolved locator. The shared handler and ARIA synchronization are protected
by the source regression, but rendered Enter and Space activation remain a
precise pending gate. The in-app surface exposed no viewport resizing, so no
fresh 1440 x 900, 1024 x 768, or 390 x 844 claim is made; the immediately
preceding responsive shell evidence remains applicable outside this desktop-
only minimum-width rule.

## Representative-route navigation regression evidence

The current official reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its navigation SHA-256 remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official Installation endpoint returned HTTP 200. The bounded
VoiceHub matrix covers Home, Installation, Quickstart, Pipeline, Auto Classes,
SpeechT5, Trainer, Optimization overview, Add a model, and Models API.

At the available 1280 x 720 viewport, all ten routes rendered exactly one
visible active primary-navigation link, the expected checked ancestor branches,
all eight root controls, zero focusable descendants in inactive branches, and
zero horizontal overflow. The complete matrix passed first in the dark palette
and then in the light palette. The initial audit script failed because its
browser evaluation used an unavailable `HTMLElement` constructor; the corrected
node-type check produced the recorded results. That tooling failure is not a
pass.

The route matrix had previously depended on manual evidence. A new dependency-
free post-build validator now parses the generated English HTML for all ten
pages. It requires the expected H1, the exact eight-root order, one active
primary-navigation anchor, the exact checked ancestor sequence, a focusable
label for every `__nav_` branch, and `aria-expanded` state consistent with each
toggle. Documentation CI and the tagged release workflow execute the validator
immediately after the strict build.

The focused source contract failed before the validator and workflow steps
existed. Its first rendered-site execution then failed because the checker
mistook Material's active-page `__toc` toggle for a navigation branch. Scoping
the contract to `__nav_` toggles retained every product branch and excluded
that separate page control; the corrected validator reported ten routes and
the eight expected roots. The first selected hook run also failed after YAPF
modified the new script and docformatter exited nonzero; the formatted rerun
passed every applicable hook. None of those failed or intermediate runs is
counted as passing evidence.

The complete documentation contract now reports 46 passes and 1,379 subtests.
The strict eleven-language build and its post-build route validator pass; the
release-workflow contract reports nine passes. All 68 model pages remain
generator-current, release alignment finds five benchmark records and all 68
documented providers, and the live inventories remain 34 TTS, 23 ASR, 11 VAD,
102 aliases, six public optimization passes, 408 model/pass pairs, and eight
contribution steps. Runtime, optimization-lifecycle, package, and complete-suite
checks were not rerun for this documentation validation and workflow slice;
their preceding evidence remains applicable.

Native Tab still did not move focus away from `BODY`, so complete sequential
focus remains unavailable and unpassed. The in-app browser exposed no viewport
resize control; the fresh computed-state matrix is therefore limited to 1280 x
720, while the existing exact desktop, tablet, and mobile geometry evidence
remains applicable. The post-build validator protects rendered structure and
ARIA state, not pixel geometry or native keyboard behavior, and is not reported
as either kind of evidence.

## Representative-route responsive visual regression evidence

The current official reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its navigation SHA-256 remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official Pipeline endpoint returned HTTP 200. The same ten mapped page
types used by the structural validator now have one executable responsive and
palette contract instead of relying only on manual viewport records.

The checker uses pinned Playwright 1.62.0 and its Chromium 151.0.7922.34
revision 1234. It renders every route in the default and slate palettes at
1,440 x 900, 1,024 x 768, and 390 x 844, producing 60 cases. Desktop requires
an article at x = 318 with an 804-pixel width, a 65-pixel header, a 270-pixel
left rail, and a 270-pixel right rail at x = 1,170. Tablet retains the same
left edge and header, uses a 658-pixel article, and hides the right rail.
Mobile uses a 342-pixel article at x = 24, a 64-pixel header, a hidden right
rail, and a closed 242-pixel drawer positioned off canvas at x = -242.

Every case additionally requires the expected H1, one exact active primary
link, the exact checked ancestors, all eight top-level roots, a closed drawer,
and zero document overflow. Desktop and tablet also require zero visible
focusable descendants inside inactive branches. The default palette resolves
the body to `rgb(255, 255, 255)` with `rgb(41, 43, 50)` text; slate resolves to
`rgb(30, 33, 41)` with `rgb(232, 233, 239)` text. Documentation and tagged-
release CI install the pinned browser and run this checker after the strict
build and structural DOM validation.

The pre-implementation source regression failed because the checker did not
exist. The first post-implementation regression then exposed an over-specific
selector fragment in the test, and the first selected hook run exited nonzero
after YAPF formatted the checker. A scratch forced checkbox action also failed
because the input was outside the viewport; the final checker uses the same
change-event path as the rendered theme control. None of those failed or
intermediate results is counted as a pass.

After correction, the focused contract passed, every selected hook passed, the
strict eleven-language build completed, the DOM validator reported ten routes
and eight roots, and the visual validator passed all 60 cases. The complete
documentation contract reported 47 passes and 1,379 subtests. The broader
documentation, release, distribution, and packaging slice reported 70 passes
and 1,455 subtests. The complete Python 3.12.12 suite reported 2,479 passes,
15 skipped, 3,797 subtests, and 35 warnings in 112.83 seconds; skipped paths
remain unverified. The distribution probe passed for wheel, source
distribution, and editable installs with a 57,192,274-byte wheel and a
55,444,958-byte source distribution.

The refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, 68 complete model pages with no missing, orphaned, or lowercase-first
display names, six public optimization passes, 408 model/pass pairs, five
benchmark records, eight contribution steps, and ten representative page
pairs. All 68 model pages and 59 notebooks remain generator-current, and
release alignment finds all five benchmark files and all 68 documented
providers.

This gate protects computed geometry, palette state, navigation state, and
responsive visibility. At that point it did not compare screenshot pixels or
prove native sequential Tab or rendered Enter/Space branch activation; the
following keyboard slice closes the two executable interaction gaps. Remote CI
has not executed the uncommitted worktree, and publisher configuration, tags,
and publication still require maintainer action. The untracked `uv.lock`
remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Native keyboard focus regression evidence

The current official reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its navigation SHA-256 remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official Pipeline endpoint returned HTTP 200. At 1,440 x 900, the first
25 native Tab stops on that page were all visible and reached the Pipeline
route after the product controls. VoiceHub keeps its speech-specific shell and
modern palette while following the same visible, ordered keyboard model.

The first pinned-Chromium VoiceHub probe exposed actual product defects. Tab
entered a zero-height search scroll region at step four and an invisible
palette radio at step seven. All eight visible root branch labels carried an
empty `tabindex` and were absent from the sequence. Programmatically focused
Base classes retained focus, but Enter left it collapsed while Space opened it,
showing that the earlier click delegation double-handled one native key path.
None of those states is counted as passing evidence.

The shared correction removes the search result viewport and native palette
radios from sequential focus, normalizes every primary-navigation branch label
to `tabindex="0"`, and gives branch focus a two-pixel solid outline with a
two-pixel offset. Enter and Space now run in the capture phase, prevent the
competing handler, toggle the associated input exactly once, and dispatch the
same bubbling change event used by pointer activation. The active branch label
keeps focus while its `aria-expanded` state and panel visibility synchronize.

The Playwright gate now completes four full native Tab cycles on the Pipeline
page: default and slate at both desktop and mobile widths. The stable sequence
contains 228 focus stops. A repeat contained 230 because Material's
scroll-dependent Back to top control became focusable in two cycles; both
conditional stops were visible and passed the same checks. Desktop begins with
skip, logo, search, version, language, theme, and source controls; then visits
all eight root branches in document order, only the expanded Inference
descendants, the complete right table of contents, article controls, previous
and next links, and footer links.
Mobile begins with skip, drawer, search, and version controls. The closed
drawer is inert, its off-canvas descendants are absent, and the closed search
input is not a stop. Every stop has rendered width and height, no stop belongs
to an inactive panel, every branch has the expected visible outline, each cycle
reaches `BODY`, and the next Tab returns to the skip link.

Two additional native activation cases reach Base classes through nine Tab
presses rather than programmatic focus. Enter in the default palette and Space
in slate each open and close the branch. Both preserve focus, the two-pixel
outline, the sole active Pipeline link, zero horizontal overflow, exact checked
state, `aria-expanded`, and `block`/`none` panel display. The inspected slate
screenshot shows the focused Base classes outline without changing the
established 1,440-pixel shell geometry.

The responsive continuation makes the mobile drawer label a native button with
an accessible name, `aria-controls`, `aria-expanded`, and a visible outline.
The off-canvas navigation stays inert until the drawer opens. Enter in the
default palette and Space in slate each open the drawer, wait for its transition
to finish, move focus to the fully visible navigation control, and leave zero
overflow. Escape closes the drawer, restores its inert boundary, synchronizes
`aria-expanded`, and returns focus to the trigger. The inspected 390 x 844
screenshot shows the opened Pipeline drawer and focus outline without exposing
the obscured page to sequential navigation.

The pre-implementation source regression failed against the old focus
contract. A first corrected regression then over-counted a Jinja loop's single
source declaration, and the first selected hook run exited nonzero after
pyupgrade made the JavaScript-evaluation string raw. The first two rendered
checker runs also began after the theme control because the palette setup
intentionally restores focus there; reloading the persisted palette resets the
test to a real document-start Tab sequence. None of those failed or
intermediate runs is counted as a pass.

A follow-up mobile probe then exposed the closed drawer, its source link, and
its full navigation tree in sequential focus while the visible menu label was
not operable; it also reached the closed off-canvas search input. The expanded
focused regression failed before the drawer trigger, inert lifecycle, and
responsive search tabindex existed. Initial rendered checks applied viewport
position to long article stops that Playwright did not scroll and sampled the
drawer during its transition; those harness failures are not passes. The final
gate enforces full viewport containment on the fixed shell prefix and waits for
the drawer control to be fully on canvas.

The corrected focused regressions reported two passes. The complete
documentation and release-contract slice reported 57 passes and 1,379
subtests. The strict eleven-language build, ten-route DOM checker, 60-case
responsive matrix, eight keyboard cases, and 228 stable native focus stops
passed; a repeat also validated two conditional Back to top stops.
Every applicable selected hook passed after the formatter correction. The
complete Python 3.12.12 suite then reported 2,480 passes, 15 skipped, 3,797
subtests, and 35 warnings in 113.79 seconds; skipped paths remain unverified.

The refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, 68 model pages without missing, orphaned, or lowercase-first display
names, six public optimization passes, 408 model/pass pairs, five benchmark
records, eight contribution steps, and ten representative page pairs. Package
code and metadata did not change, so the distribution probe was not rerun; the
immediately preceding 57,192,274-byte wheel and 55,444,958-byte source-
distribution evidence remains applicable but is not new evidence for this
slice.

Desktop branch navigation plus closed and opened mobile-drawer keyboard
behavior are now executable CI gates rather than pending manual checks.
At that point screenshot pixel comparison, exact-commit remote CI for the
uncommitted worktree, protected publisher configuration, tags, and publication
remain open. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Screenshot-derived visual regression evidence

The current official reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; its navigation SHA-256 remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official Pipeline endpoint returned HTTP 200. The local visual matrix
previously proved computed geometry and interaction state but did not compare
the rendered raster against a reviewed baseline.

The same ten representative routes now produce in-memory PNG screenshots in
both palettes at 1,440 x 900, 1,024 x 768, and 390 x 844. After fonts settle and
animations are disabled, the checker verifies the exact raster dimensions and
uses Pillow 12.3.0 to apply a two-pixel Gaussian blur. It then records an
aspect-preserving 64-pixel-wide difference hash and the screenshot's mean RGB
values. The schema-one manifest contains all 60 cases and records Playwright
1.62.0 with Chromium 151.0.7922.34. A default check fails for a missing,
orphaned, malformed, or dimension-mismatched signature, a perceptual hash more
than 8% from its reviewed baseline, or a mean channel delta above 6.0.

Baseline generation is deliberately separate from validation. The explicit
`--update-screenshot-baselines` mode prints replacement JSON for review; the
documentation and tagged-release workflows only run comparison mode. The docs
extra and the standalone documentation workflow pin Pillow 12.3.0, so the
pixel algorithm does not depend on an undeclared transitive package.

The pre-implementation regression failed at the absent manifest. The first
selected hook run exited nonzero after YAPF formatted the implementation, and
the first manifest application attempt failed before writing because its patch
lacked the final newline. Those failed tooling runs are not passes. The
corrected focused test passed, and a fresh strict eleven-language build passed
all 60 responsive cases, all 60 screenshot comparisons, eight keyboard cases,
and four complete focus cycles.

Two intentional negative controls demonstrate observable failure. Replacing
the first reference hash with zeroes changed 566 of its bits, or 22.109%, and
failed above the 8% threshold. Replacing its mean RGB with zeroes produced
channel deltas of 229.142, 231.456, and 239.484 and failed above 6.0. Neither
deliberate corruption is reported as a product pass.

Fresh official Pipeline screenshot requests returned HTTP 200 for light
desktop, tablet, and mobile and for dark desktop and tablet. The first dark-
mobile request returned HTTP 429 and is not a pass; a later isolated retry
returned HTTP 200 and its screenshot was inspected. The local desktop and
mobile screenshots were inspected in light and dark. Upstream retains Hugging
Face's global product header, version banner,
community panel, and NLP-specific content; VoiceHub retains original speech
content, its own brand, and its approved modern palette. The new signatures
protect the local rendered raster but do not claim raw-pixel identity across
those intentional differences.

The complete documentation and release-contract slice reported 58 passes and
1,379 subtests. The full Python 3.12.12 suite reported 2,481 passes, 15 skipped,
3,797 subtests, and 35 warnings in 110.41 seconds. The fresh package probe
passed wheel, source-distribution, and editable installs with a 57,192,282-byte
wheel and 55,446,461-byte source distribution. All 68 model pages and 59
notebooks remain generator-current, release alignment still finds five
benchmark records and all 68 documented providers, and the registry-derived
inventories remain 34 TTS, 23 ASR, 11 VAD, 102 aliases, six public optimization
passes, 408 model/pass pairs, eight contribution steps, and ten representative
routes.

Local screenshot regression is now an executable CI gate. Exact-commit remote
CI for the uncommitted worktree, protected publisher configuration, tags,
publication, five native-
kernel hardware gates, and seven inaccessible asset/checkpoint/oracle paths
remain open. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Complete representative-route focus evidence

The local interaction matrix now covers native sequential focus for all ten
representative routes at 1,440 x 900, 1,024 x 768, and 390 x 844 in both
VoiceHub palettes. Its official structural reference remains Transformers
commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the pinned navigation
fingerprint remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
This gate validates the local interaction contract and does not claim that
upstream's different content has an identical focus sequence.

Before editing, a native Chromium probe confirmed the desktop, tablet, and
mobile shell prefixes and measured up to 239 stops on the model index. The
checker therefore replaced its fixed 200-step limit with a DOM-derived bound
and moved complete traversal into the existing 60-case render matrix. Desktop
requires skip, logo, search, version, language, theme, and source. Tablet uses
the same compact header without the hidden logo. Mobile requires skip, drawer,
search trigger, and version. Route-specific left navigation, TOC, content,
actions, previous and next links, conditional Back to top, and footer controls
then follow their native order without a hard-coded page sequence.

The first matrix run exposed three concrete gaps rather than producing a false
pass. Palette changes left Chromium's internal focus cursor at the theme
control; a document-start reset now restores the true first stop. Installation
focused a zero-sized native content-tab radio while its rendered label had no
indicator; the label now receives a two-pixel focus proxy that the checker
validates. The long Models API tablet route reached body focus and then resumed
near the scrolled article; an unmodified forward Tab from the document
boundary now returns deterministically to Skip to content. Each failed run and
the initial zero-test invocation are recorded failures, not passing evidence.

The final rendered gate passed all 60 geometry cases, all 60 screenshot
comparisons, all 60 focus cycles, and four independent Enter/Space branch and
drawer activation cases. It visited 4,075 rendered focus stops, rejected every
inactive-branch target, checked visible branch, drawer, and content-tab
outlines, reached `BODY` in each cycle, and returned to the skip link. The
observed stop total may vary when Material's scroll-dependent Back to top
control becomes available, so the checker protects behavior rather than a
fixed count.

The focused source slice passed two tests. The complete documentation and
release slice passed 59 tests and 1,379 subtests; the registry and public-
optimization slice passed 231 tests and 1,203 subtests with four warnings; and
the full Python 3.12.12 suite passed 2,482 tests and 3,797 subtests with 15
skips and 35 warnings in 110.60 seconds. The strict eleven-language build,
ten-route DOM validator, selected hooks, 68 generated pages, 59 generated
notebooks, five benchmark records, and fresh wheel, source-distribution, and
editable probes remain current for this worktree.

Local complete focus traversal is now an executable CI gate. Exact-commit
remote CI, remaining route-specific activation behavior, protected publisher
configuration, tags, publication, five native-kernel hardware gates, and seven
inaccessible asset, checkpoint, or oracle paths remain open. The untracked
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Rendered semantic-accessibility evidence

The representative-route checker now pins `axe-playwright-python` 0.1.8 and
reports Axe Core 4.12.1 across the same ten routes, three viewports, and two
VoiceHub palettes used for geometry, screenshots, and complete focus cycles.
This protects automated detectable accessibility rules for the local product
shell; it does not claim a complete WCAG or manual audit, nor does it assert
that upstream's different content produces identical Axe results.

The pre-edit matrix exposed shared semantic defects instead of page-specific
exceptions: pseudo-buttons implemented as labels, low-contrast syntax and
footer tokens, unnamed repeated code-action landmarks, paragraph links without
a non-color cue, and a closed search-results scroller left in the accessibility
tree. The shell now uses native buttons for drawer, search, theme, and runtime
navigation controls, keeps closed results inert and `aria-hidden`, assigns
unique names to code-action landmarks, underlines paragraph links, and uses
accessible token colors. A subsequent failed run found the mobile Quickstart
table lacked keyboard scrolling; shared table wrappers now expose a named
keyboard stop and a two-pixel focus indicator.

The final current-code matrix passed 60 Axe cases with zero reported
violations, 60 geometry cases, 60 screenshot comparisons, 60 complete focus
cycles with 4,213 observed focus stops, and four Enter/Space activation cases.
An injected unnamed button failed with `button-name`, proving the new audit is
not a vacuous pass. The documentation source suite reported 51 passes and
1,379 subtests; the complete local suite reported 2,483 passes, 15 skips,
3,797 subtests, and 35 warnings in 113.36 seconds. Exact-commit remote CI and
the already documented external, hardware, and checkpoint gates remain open;
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Expanded shared-shell accessibility evidence

The rendered parity checker now audits shared-shell interaction states after
their visual transitions and asynchronous content updates have settled. The
official structural reference remains Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; the current navigation fingerprint
remains
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.

Thirty new Axe Core 4.12.1 cases cover open search, populated results, empty
results, the version disclosure, desktop/tablet branch expansion, and the
mobile drawer across both VoiceHub palettes and all applicable viewports. The
pre-edit audit failed 12 cases on dynamically generated scrollable code,
version list/menu role conflicts, repeated nested navigation landmark names,
and opened-drawer title contrast. Shared runtime normalization now provides
named keyboard stops for result code, the version selector uses native
disclosure and list semantics, expanded navigation landmarks have unique
labels, and mobile nested titles retain readable contrast.

The final exact-code matrix passed all 30 expanded states with zero reported
Axe violations while preserving the existing 60 base accessibility, geometry,
screenshot, and complete-focus cases. It traversed 4,229 visible focus stops
and retained four independent Enter/Space branch and drawer activation checks.
The documentation source suite reported 52 passes and 1,379 subtests, the full
Python 3.12.12 suite reported 2,484 passes, 15 skips, 3,797 subtests, and 35
warnings in 110.14 seconds, and fresh distribution probes passed. This is
automated evidence for the enumerated local states, not a complete WCAG/manual
audit or raw-pixel identity claim against upstream's different content and
branding.

The current inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, 68 current provider pages, 59 current notebooks, six public
optimization passes, 408 model/pass pairs, five benchmark records, eight
top-level navigation roots, ten representative routes, and eight contribution
steps. Exact-current-worktree remote CI, remaining route-specific interactions,
protected publication configuration, tags, publication approval, five
native-kernel hardware gates, and seven inaccessible asset, checkpoint, or
oracle paths remain open. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Installation page evidence

The current official Installation reference was refreshed at Transformers
commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the page returned HTTP 200. Its rendered hierarchy is `Installation`,
`Virtual environment`, `Python`, `Source install`, `Editable install`,
`conda`, `Set up`, `Cache directory`, and `Offline mode`. The current page has
no content tabs. At the matched viewports its article measured x = 318 and 804
pixels wide on desktop, x = 318 and 658 pixels wide on tablet, and x = 24 and
342 pixels wide on mobile, with zero overflow.

VoiceHub previously exposed numbered install sections and an OS tab set from
an older route structure. A focused source contract failed on that mismatch
before editing. The corrected route uses the same current hierarchy and right
TOC while keeping VoiceHub prose and speech semantics original. Its workflows
cover uv environments, published and training installs, source and editable
installs, a conda-owned environment, the shared Hub cache precedence, and
offline configuration. The source contract rejects a fabricated conda-forge
package and parses every Python example. It also requires a rendered route
validator so source headings alone cannot satisfy the parity claim.

The first rendered audit failed rather than hiding a new contrast regression:
the cache examples introduced Pygments variable-name tokens at 4.486:1 in the
light theme. The shared variable token now uses `#686a72` against `#f4f5f8`,
measuring 4.948:1. The final Playwright 1.62.0 and Chromium 151.0.7922.34
matrix passed all six Installation viewport/palette cases for exact headings,
TOC, tab absence, required content, geometry, Axe Core 4.12.1, reviewed
screenshot signatures, and complete native focus. These cases are part of the
same 60-route matrix, which passed 60 base Axe checks, 60 screenshot checks,
60 full focus cycles with 4,250 observed stops, 30 expanded-state Axe checks,
and four independent Enter/Space activations.

The strict eleven-language build, ten-route DOM validator, selected hooks, 53
documentation tests with 1,380 subtests, 231 registry/optimization tests with
1,203 subtests, and the complete 2,485-test Python 3.12.12 suite passed.
VoiceHub's current Installation structure and local rendered contract are now
protected. This does not claim raw-pixel identity with upstream's different
brand, prose, or package ecosystem, and it does not close the remaining
route-specific interaction, exact-commit CI, publication, hardware, or
checkpoint gates. `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Quickstart page evidence

The current official Quickstart reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the rendered page returned HTTP 200. Its current hierarchy is `Quickstart`,
`Set up`, `Agent skills`, `Pretrained models`, `Pipeline`, `Trainer`, and
`Next steps`. At the matched viewports its article measured x = 318 and 804
pixels wide on desktop, x = 318 and 658 pixels wide on tablet, and x = 24 and
342 pixels wide on mobile, with zero document overflow.

VoiceHub previously exposed the stale `Discover models` and `Inference`
hierarchy plus three task subheadings, without current content tabs or
callouts. A focused source contract failed on that mismatch before editing.
The corrected page preserves the upstream information hierarchy while keeping
VoiceHub content original: uv and pip setup, repository-local agent-skill
guidance, shared configuration and pretrained-model contracts, TTS/ASR/VAD
pipeline examples, and an explicit Trainer support boundary. Seven tabs in
three sets, two tips, one model table, and at least twelve code blocks are
protected in source and rendered checks. Repository-local skills are described
as source-checkout guidance, not wheel features or authorization.

The rendered iteration kept every failure explicit. Early runs found an
off-canvas tab target, persisted Material tab state between viewports, an
unnamed keyboard-inaccessible code scroller, eight pixels of mobile overflow,
an unnamed horizontally scrollable tab-label strip, and transient Back to top
contrast below Axe's threshold. Shared runtime normalization now names and
focuses overflowing code and option strips, the mobile tab strip stays inside
the document, the test resets tab state deterministically, and Back to top no
longer fades through low-contrast intermediate opacity. Those failed runs are
not passing evidence.

The final Playwright 1.62.0 and Chromium 151.0.7922.34 matrix passed all six
Quickstart viewport/palette cases for exact headings, TOC, initial tab state,
component inventory, required content, geometry, Axe Core 4.12.1, reviewed
screenshot signatures, and complete native focus. Six additional interaction
cases focused and activated the last option in every tab set with Space,
verified the selected panel and zero overflow, and reran Axe after activation.
They are part of a 60-route matrix that passed 60 base Axe checks, 60
screenshot checks, 60 focus cycles with 4,297 observed stops, 30 expanded-state
Axe checks, four independent branch/drawer activations, and six Quickstart tab
interaction checks.

The focused source slice passed two tests; the complete documentation suite
passed 53 tests and 1,380 subtests; the registry and public-optimization slice
passed 231 tests and 1,203 subtests with four warnings; and the complete Python
3.12.12 suite passed 2,485 tests and 3,798 subtests with 15 skips and 35
warnings in 107.12 seconds. The strict eleven-language build, ten-route DOM
validator, 68 generated pages, 59 generated notebooks, and five-record release
alignment also passed. The refreshed inventories remain 68 models (34 TTS, 23
ASR, and 11 VAD), 102 aliases, six public optimization passes, 408 model/pass
pairs, eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Quickstart structure and tab-interaction gate; it
does not claim raw-pixel identity with upstream's different brand, prose,
model ecosystem, or approved VoiceHub palette. Exact-current-worktree remote
CI, protected publisher configuration, tags, publication approval, five
native-kernel hardware gates, and seven inaccessible asset, checkpoint, or
oracle paths remain open. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Pipeline page evidence

The official task-guide reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The mapped pair remains `/docs/transformers/main/en/pipeline_tutorial` and
`/voicehub/guides/inference/`. VoiceHub keeps speech-specific examples and
normalized TTS, ASR, and VAD outputs while matching the task-guide hierarchy,
task and parameter grouping, persistent navigation, right TOC, responsive
article geometry, tables, code presentation, page actions, and keyboard
interaction boundary.

The source contract now binds the rendered validator to the exact fourteen-
heading order and thirteen-entry right TOC. Each of the six viewport/palette
cases requires one parameter table, six code blocks, six accessible code-copy
buttons, and the rendered task, batching, large-model, and discovery markers.
This supplements the existing geometry, active-navigation, zero-overflow,
screenshot-signature, complete-focus, and Axe coverage rather than replacing
it.

The first exact render failed because all six code blocks exposed zero copy
buttons even though `content.code.copy` was configured. That run is not
passing evidence. The shared page-action runtime now creates one native button
per rendered code block, uses the existing Clipboard API plus selection
fallback, exposes busy and success states, and retains keyboard focus with a
visible two-pixel outline. The first implementation then copied highlighted
code with extra visual line breaks; its failed clipboard comparison is also
excluded. Copying `textContent` preserves the exact source text.

The final Playwright 1.62.0 and Chromium 151.0.7922.34 matrix passed all six
Pipeline viewport/palette cases. Six additional cases used Enter in the
default palette and Space in slate at desktop, tablet, and mobile widths,
verified exact clipboard contents and the visible success state, retained
focus and zero overflow, and reran Axe Core 4.12.1 after activation. The
complete ten-route matrix passed 60 base Axe checks, 60 reviewed screenshot
checks, 60 native focus cycles with 4,572 observed stops, 30 expanded-state
Axe checks, four independent branch/drawer activations, six Quickstart tab
interaction cases, and six Pipeline code-copy interaction cases.

The focused Pipeline source contract and the complete documentation suite
passed; the latter reported 53 tests and 1,380 subtests. The strict eleven-
language build and ten-route DOM validator also passed. This closes the
current local Pipeline content and code-copy interaction inventory. It does
not claim raw-pixel identity with upstream's different brand, prose, model
ecosystem, or approved VoiceHub palette. Exact-current-worktree remote CI,
protected publisher configuration, tags, publication approval, five native-
kernel hardware gates, and two inaccessible WeNet asset paths remain open.
The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Installation interaction evidence

The official Installation reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. The 164-line
`docs/source/en/installation.md` source has SHA-256
`d050e3e0e1c89d543c71c25367a455a9b49ea89c92f8c2376bb58294a4a4cf3b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The toctree still begins with Transformers, Installation, and Quickstart, and
the official Installation route returned HTTP 200. The rendered reference
retains the Installation, Virtual environment, Python, Source install,
Editable install, conda, Set up, Cache directory, and Offline mode hierarchy,
code-copy controls, edit action, and Transformers/Quickstart footer sequence.
The mapped VoiceHub route remains `/voicehub/getting-started/installation/`.

This bounded iteration defined the observable route contract before editing:
all six viewport/palette cases must expose the exact nine headings and eight
TOC entries, zero stale tab sets, 15 code blocks and copy buttons, one page-
copy action, four exact external package-manager destinations, three exact
internal workflow destinations, the edit and previous/next targets, and the
installation, cache, offline, and evidence-safety markers. Six Enter/Space
cases must copy the first command, and six more must copy the readable page.
Both actions must preserve exact clipboard text, visible success and idle
states, focus, zero overflow, and a post-action zero-violation Axe result.

The first focused source test intentionally failed because none of the new
rendered-action constants, helpers, or counters existed. The first rendered
run then failed on an overly narrow phrase assertion that did not span the
existing `without downloading a checkpoint or importing PyTorch` sentence.
The corrected exact phrase and final rendered contract pass; neither failed
run is counted as passing evidence. No Installation prose, package command, or
link needed to change.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all
60 representative route/viewport/palette cases, including six exact
Installation structure/destination cases, six Installation first-code copy
cases, and six Installation page-copy cases. The complete matrix recorded 60
base Axe checks, 60 reviewed screenshot signatures, 60 complete focus cycles
with 4,613 observed stops, 30 expanded-state Axe checks, four branch/drawer
activation cases, and 130 keyboard cases. Installation's two action families
used Enter in the default palette and Space in slate at desktop, tablet, and
mobile widths and returned from their success state to idle without losing
focus or introducing overflow.

The complete documentation suite reported 54 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes the current local Installation structure, destination, code-copy,
and page-copy inventory. The complete Python suite and physical package builds
were not rerun for this documentation-checker-only slice; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Auto Classes page evidence

The official Auto Classes reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
and the official route returned HTTP 200. The pinned 301-line
`docs/source/en/model_doc/auto.md` source has SHA-256
`557f5836c0722fef6a484c46805dfab0eb69a387b028a914b132350edf09f167`.
The mapped pair remains `/docs/transformers/main/en/model_doc/auto` and
`/voicehub/models/providers/`.

The focused source contract first failed because the rendered validator had
no model-index-specific contract. That intentional failure reported one
failure and 68 passing subtests and is not passing evidence. The new contract
requires the exact nine-heading and eight-entry TOC order, four tables with
3, 34, 23, and 11 body rows, 68 unique provider links whose labels start with
uppercase characters, three code blocks and three shared copy actions, and
the expected AutoConfig, AutoProcessor, task AutoModel, `available_models`,
`lazy_load`, and nine-section markers.

The first selected-file pre-commit run then failed when docformatter
misclassified a multiline JavaScript predicate and rewrote it. That run is
also excluded. The predicate now uses formatter-safe concatenated strings,
and the complete selected-file pre-commit sequence passes.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Auto Classes cases. Six
additional cases used Enter in the default palette and Space in slate to
activate page copy at desktop, tablet, and mobile widths. They verified exact
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
452 subtests. All 68 generated model pages and 59 generated notebooks remain
current, and the five-record release alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Auto Classes structure and page-copy interaction
inventory. The complete Python suite and physical package builds were not
rerun for this documentation-only slice; their exact preceding current-
worktree evidence remains separate. Exact-current-worktree remote CI, exact
tagged artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current SpeechT5 model-detail page evidence

The official SpeechT5 reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`,
its `model_doc/speecht5` entry retained the `SpeechT5` title, and the official
route returned HTTP 200. The pinned 87-line source has SHA-256
`71bba8a2921cf637383fb8d6f2fd66df9cd95deb59118b9f49e1362485c27eb5`.
The mapped pair remains `/docs/transformers/main/en/model_doc/speecht5` and
`/voicehub/models/providers/speecht5/`.

The first focused command used the wrong unittest class name, collected no
tests, and exited 4; it is unexecuted evidence, not a pass. The corrected
focused source contract then failed because the rendered validator had no
SpeechT5-specific contract. That expected failure is also excluded. The new
contract protects the exact 16-heading and 15-entry TOC order, eight tables
with 7, 3, 4, 2, 6, 1, 9, and 8 body rows, seven code blocks and seven shared
copy actions, verified paper and upstream GitHub references, local
configuration/model facade-source links, and the required auto-model,
processor, normalized-output, checkpoint, real-evidence, and fail-closed
optimization markers.

The first full render expected ten rows in the checkpoint table, observed the
source-correct nine rows, and failed. That run is not passing evidence. The
corrected table inventory then passed without changing generated page content.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact SpeechT5 cases. Six
additional cases used Enter in the default palette and Space in slate to
activate page copy at desktop, tablet, and mobile widths. They verified exact
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
452 subtests. All 68 generated model pages and 59 generated notebooks remain
current, and the five-record release alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local SpeechT5 model-detail structure and page-copy
interaction inventory. The complete Python suite and physical package builds
were not rerun for this documentation-only slice; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Trainer overview evidence

The official Trainer reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml` retained
SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and still places `Trainer overview` before `Fine-tuning`. The official route
returned HTTP 200, and its pinned 30-line source retained SHA-256
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`.
The mapped pair remains `/docs/transformers/main/en/trainer` and
`/voicehub/guides/trainer/`.

The focused source contract first failed because the rendered validator had
no Trainer-specific contract. That expected failure is not passing evidence.
The new contract protects the exact `Trainer` and `Next steps` heading order,
the sole right-TOC entry, zero tables and code blocks, one page-copy action,
the four fine-tuning, architecture, support-matrix, and data-preparation
destinations, the edit target, the Fine-tuning footer destination, and the
required Trainer, TrainingArguments, model-owned objective, exact-resume, and
fail-closed speech-training markers.

The final strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Trainer cases. Six
additional cases used Enter in the default palette and Space in slate to
activate page copy at desktop, tablet, and mobile widths. They verified exact
clipboard text, visible success and idle state, focus retention, zero
overflow, and a second zero-violation Axe result. The complete matrix recorded
60 base Axe checks, 60 reviewed screenshot signatures, 60 complete focus
cycles with 4,771 stops, 30 expanded-state Axe checks, four branch/drawer
activation cases, six Quickstart tab cases, six Pipeline copy cases, six Auto
Classes page-copy cases, six SpeechT5 page-copy cases, and six Trainer
page-copy cases, for 94 keyboard cases total. The ten-route DOM validator also
passed.

The complete documentation suite reported 53 passes and 1,380 subtests. A
risk-proportional registry, speech-task, universal-optimization,
distribution-compliance, and release-readiness slice reported 97 passes and
452 subtests. All 68 generated model pages and 59 generated notebooks remain
current, and the five-record release alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Trainer structure and page-copy interaction
inventory. The complete Python suite and physical package builds were not
rerun for this documentation-only slice; their exact preceding current-
worktree evidence remains separate. Exact-current-worktree remote CI, exact
tagged artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Optimization overview evidence

The official Optimization overview reference was refreshed at Transformers
commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml`
retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and still maps `optimization_overview` to `Overview`. The official route
returned HTTP 200. Its pinned 178-line source retained SHA-256
`19622667a7299f258f5c9a72940c9f26492619636f35d9bd592701c02745b620`.
The mapped pair remains `/docs/transformers/main/en/optimization_overview`
and `/voicehub/guides/optimization-overview/`.

The focused source contract intentionally failed before implementation with
one failure and six passing subtests because the rendered validator had no
Optimization-specific contract. That run is not passing evidence. The new
contract protects the exact eight-heading and seven-entry TOC inventory, one
six-row technique table, one code block and shared copy action, one page-copy
action, all six registered pass names, five exact next-step destinations, the
edit target, the Model support and TTS optimization footer destinations, and
the application, validation, manifest, restoration, unsupported-quantization,
parallelism, and continuous-batching boundary markers.

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
452 subtests. All 68 generated model pages and 59 generated notebooks remain
current, and the five-record release alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Optimization overview structure and page-copy
interaction inventory. The complete Python suite and physical package builds
were not rerun for this documentation-only slice; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Contribution page evidence

The official modular-contribution reference was refreshed at Transformers
commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml`
retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and still places `Add a model with modular transformers` before the legacy
model-contribution route. The official route returned HTTP 200. Its pinned
500-line source retained SHA-256
`a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.
The mapped pair remains `/docs/transformers/main/en/modular_transformers` and
`/voicehub/project/adding-a-model/`.

The focused source contract intentionally failed before implementation with
one failure and 18 passing subtests because the rendered validator had no
Contribution-specific contract. That run is not passing evidence. Two
read-only pre-edit HTML inventory probes also failed because Beautiful Soup is
not installed and the first built-in parser selector assumed the content root
was a `div`; neither is verification evidence. The corrected dependency-free
parser established the exact local inventory before the contract was added.

The new contract protects the exact 10-heading and nine-entry TOC order, the
Create/Audit/Configure/Wrap/Register/Support/Test/Document process, three
tables with 8, 3, and 7 body rows, 13 code blocks and shared copy actions, one
page-copy action, the two final contribution-guide destinations, the edit and
adjacent footer targets, and the scaffold, registry, task bases, training,
optimization, generation, package, and unverified-evidence boundary markers.

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
452 subtests. All 68 generated model pages and 59 generated notebooks remain
current, and the five-record release alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Contribution structure and page-copy interaction
inventory. The complete Python suite and physical package builds were not
rerun for this documentation-only slice; their exact preceding current-
worktree evidence remains separate. Exact-current-worktree remote CI, exact
tagged artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Models API page evidence

The official Models API reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`; `_toctree.yml`
retained SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and still places Models between Logging and Text Generation in Main Classes.
The official route returned HTTP 200. Its pinned 44-line source retained
SHA-256
`a4899c758b5d621075b2e2f39f0aa79671010c88b08d1508f7af5731375c2871`.
The mapped pair remains `/docs/transformers/main/en/main_classes/model` and
`/voicehub/reference/models/`.

The focused source contract intentionally failed before implementation with
one failure and 26 passing subtests because the rendered validator had no
Models-API-specific contract. That run is not passing evidence. The first full
render then failed because code-line number anchors were included in the
internal lifecycle-link inventory. That run is also excluded. The corrected
selector rejects links inside `pre` while retaining the exact user-facing
content links.

The final contract protects the exact five-heading and four-entry TOC order,
four tables with 6, 3, 3, and 5 body rows, two code blocks and shared copy
actions, one page-copy action, four exact facade-source links, three exact
internal lifecycle links, the edit target, the Model audit and Full API footer
destinations, and the configuration, factory, lazy-loading, task-base,
normalized-output, training-output, portable-state, and unsupported-sharing
boundary markers.

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
452 subtests. All 68 generated model pages and 59 generated notebooks remain
current, and the five-record release alignment check passed. Refreshed
inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero
invalid display names, six public optimization passes, 408 model/pass pairs,
eight top-level navigation roots, ten representative routes, and eight
contribution steps.

This closes the current local Models API structure and page-copy interaction
inventory. The complete Python suite and physical package builds were not
rerun for this documentation-only slice; their exact preceding current-
worktree evidence remains separate. Exact-current-worktree remote CI, exact
tagged artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Home page evidence

The official Home reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. The 65-line `index.md`
source has SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The toctree still begins with Transformers, Installation, and Quickstart, and
the official route returned HTTP 200. The mapped pair remains
`/docs/transformers/main/en/index` and `/voicehub/`.

The focused source contract intentionally failed before implementation
because the local Home still exposed only `What is VoiceHub?` instead of the
current Features, Design, and Learn hierarchy. An intermediate test also
failed because its raw-source assertion did not normalize an existing wrapped
bold registry label. Neither run is passing evidence. The corrected source
contract normalizes Markdown before checking content without weakening the
rendered assertions. The first rendered run passed the ten-route DOM validator
but failed the old desktop-light Home screenshot baseline by 289 bits
(11.289%); that expected stale-baseline run is also excluded.

The reviewed replacement candidate contained material differences above the
configured threshold only for the six Home viewport/palette signatures. Only
those six entries were accepted; no non-Home baseline was replaced. The
rendered review also exposed a wrapping mismatch in the long local title. The
final H1 is now the product name `VoiceHub`, matching the reference's compact
product-title geometry, while the original one-lifecycle message remains in
the VoiceHub-specific tagline.

The final contract protects the exact VoiceHub, Features, Design, and Learn
heading order; the three-entry right TOC; Pipeline, Trainer, and speech
generation targets; one design tip; two design principles; all 13 existing
resource cards; four exact status-badge targets; six images with two
decorative and four non-empty badge alternatives; zero tables and code blocks;
one page-copy action; the edit target; no previous destination; Installation
as next; the 68/34/23/11 registry counts; and the lazy-checkpoint, training-
extra, and third-party-license boundaries.

The strict eleven-language build passed. Playwright 1.62.0 with Chromium
151.0.7922.34 and Axe Core 4.12.1 passed all 60 representative
route/viewport/palette cases, including six exact Home structure and content
cases. Six additional cases used Enter in the default palette and Space in
slate at desktop, tablet, and mobile widths to activate Home page copy. They
verified exact clipboard text, visible success and idle state, focus retention,
zero overflow, and a second zero-violation Axe result. The complete matrix
recorded 60 base Axe checks, 60 reviewed screenshot signatures, 60 complete
focus cycles with 4,713 stops, 30 expanded-state Axe checks, four branch/drawer
activation cases, six Home page-copy cases, six Quickstart tab cases, six
Pipeline copy cases, and six page-copy cases for each of Auto Classes,
SpeechT5, Trainer, Optimization, Contribution, and Models API, for 118 keyboard
cases total. The ten-route DOM validator also passed.

The current official rendered page confirmed the exact Transformers, Features,
Design, and Learn hierarchy, an 804-pixel desktop article, and zero overflow
in a successful fresh probe. Subsequent attempts to repeat all six remote
states returned an unhydrated shell and were interrupted; those retries are
not counted as passed. The preceding six-state official Home shell evidence
remains tied to the same upstream commit, while current source, route, and
local six-state evidence support this slice.

The complete documentation suite reported 54 passes and 1,408 subtests. A
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated model pages and 59 generated
notebooks remain current, and the five-record release alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes the current local Home structure and page-copy interaction
inventory. The complete Python suite and physical package builds were not
rerun for this documentation-only slice; their exact preceding current-
worktree evidence remains separate. Exact-current-worktree remote CI, exact
tagged artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Quickstart interaction evidence

The official Quickstart reference was refreshed at Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. The 312-line
`docs/source/en/quicktour.md` source has SHA-256
`ecfb99781204bcaea1ca63bcb4ad9ef70c99812e5f965a49b29de23cedd25bd7`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The toctree still begins with Transformers, Installation, and Quickstart, and
the official route returned HTTP 200. The rendered reference retains the
Quickstart, Set up, Agent skills, Pretrained models, Pipeline, Trainer, and
Next steps hierarchy, content tabs, code-copy controls, page actions, and
previous/next navigation. The mapped VoiceHub route remains
`/voicehub/getting-started/quickstart/`. All three declared GitHub skill
destinations also returned HTTP 200 in the refreshed link probe.

This bounded iteration defined the remaining observable Quickstart contract
before editing. All six viewport/palette cases must expose the exact seven
headings and six TOC entries, three tab sets with seven options, two tips, one
three-row model table, 12 code blocks and copy buttons, one page-copy action,
three exact external skill destinations, nine exact internal workflow-link
occurrences, and the edit and previous/next targets. The existing six tab
cases must still activate the last option in each set. Six Enter/Space page-
copy cases must then copy the readable active-tab state with exact clipboard
text, visible success and idle states, focus retention, zero overflow, and a
post-action zero-violation Axe result.

The focused source test intentionally failed before implementation because
the rendered checker had no Quickstart destination constants, page-copy
helper, or result counter. The first rendered run also failed because
Material's seven generated tab-control hashes entered the initial content-link
inventory. The corrected inventory excludes tab controls, which remain
covered by their dedicated activation checks, and requires the nine actual
workflow-link occurrences. Neither failed run is counted as passing evidence.
No Quickstart prose, code example, destination, or screenshot baseline needed
to change.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all
60 representative route/viewport/palette cases, including six exact
Quickstart structure/destination cases, six Quickstart tab cases, and six
Quickstart page-copy cases. The complete matrix recorded 60 base Axe checks,
60 reviewed screenshot signatures, 60 complete focus cycles with 4,644
observed stops, 30 expanded-state Axe checks, four branch/drawer activation
cases, and 136 keyboard cases. Page copy used Enter in the default palette and
Space in slate at desktop, tablet, and mobile widths and returned from its
success state to idle without losing focus or introducing overflow.

The complete documentation suite reported 54 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes the current local Quickstart structure, destination, tab, and
page-copy inventory. The complete Python suite and physical package builds
were not rerun for this documentation-checker-only slice; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current representative table-of-contents activation evidence

The shared table-of-contents contract was refreshed against Transformers
commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. The 65-line
`docs/source/en/index.md` source has SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. Existing official rendered-shell
evidence remains separate; this iteration does not claim a new remote
interaction run.

The observable local contract was fixed before implementation. Each of the
ten representative routes must activate its final table-of-contents target by
pointer and by unmodified Enter in both palettes at the 1,440-pixel desktop
viewport. All 40 cases must expose a current hash that identifies the CSS
`:target`, exactly one matching active TOC link after smooth scrolling settles,
a visible heading aligned beneath the header, zero overflow, and a post-action
zero-violation Axe result. The 20 keyboard cases must also retain focus on the
activated link.

The focused source contract intentionally failed before implementation because
the rendered checker had no representative-route activation matrix, helper,
or counters. The first full rendered run then failed because the Home target
settled at 63.703 pixels beneath the intentional 65-pixel header boundary; the
two-pixel observer tolerance now matches the existing CSS contract. An initial
focused rendered probe also failed because it inspected Installation before
smooth scrolling and Material's observer had settled. The corrected helper
waits for the exact hash and active link, then requires another 600 milliseconds
of stable state. None of those failed runs is counted as passing evidence.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,607 observed stops, 30 expanded-state Axe checks,
and four branch/drawer activation cases. The new matrix added 20 pointer and 20
Enter TOC activation cases, each with a second Axe pass; the complete keyboard
inventory is now 156 cases.

The complete documentation suite reported 55 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. Selected pre-commit hooks passed. All 68 generated
provider pages and 59 generated model notebooks remain current, and the
five-record release-alignment check passed. Refreshed inventories remain 68
models (34 TTS, 23 ASR, and 11 VAD), 102 aliases, zero invalid display names,
six public optimization passes, 408 model/pass pairs, eight top-level
navigation roots, ten representative routes, and eight contribution steps.

This closes route-specific TOC activation for the representative local matrix.
Non-representative routes and future templates remain outside that matrix. The
complete Python suite and physical package builds were not rerun for this
documentation-checker-only slice; their exact preceding current-worktree
evidence remains separate. Exact-current-worktree remote CI, exact tagged
artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current representative search activation evidence

The shared header search contract was refreshed against Transformers commit
`b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`. The 65-line
`docs/source/en/index.md` source has SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. Existing official rendered-shell
evidence remains separate; no new remote search interaction was executed.

The observable local contract was fixed before implementation. All ten
representative routes must activate and close search at desktop, tablet, and
mobile widths in both palettes. The 40 desktop/tablet cases use the documented
Ctrl+K shortcut; the 20 mobile cases use the visible search trigger. Every
opened state must expose the checked toggle, expanded trigger, open body state,
focused and visible input, active input and result tab stops, non-hidden and
non-inert result output, zero overflow, and a zero-violation Axe result. Escape
must close the dialog, restore hidden/inert/tab-order state, keep zero overflow,
and return focus to the visible inline input on desktop/tablet or the trigger
on mobile.

The focused source contract intentionally failed before implementation because
the checker had no representative-route search matrix, helper, counters, or
breakpoint-aware close-focus contract. The implementation replaced the hidden
desktop trigger as Escape's universal focus destination with the visible
inline input outside the mobile breakpoint. A six-case Home breakpoint probe
then passed. The first complete documentation-suite run after the rendered
matrix failed one legacy static assertion that still required
`trigger.focus()`; the assertion now enforces the breakpoint-aware focus
helper. Neither failed run is counted as passing evidence.

The final strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,654 observed stops, 30 expanded-state Axe checks,
four branch/drawer activation cases, 60 search activation/closure cases, and 40
TOC activation cases. Each opened search state received another Axe pass. The
complete keyboard inventory is now 196 cases.

The complete documentation suite reported 56 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes search activation and Escape restoration across the representative
local route/viewport/palette matrix. Version, language, theme, and source
route-specific activation remain outside this slice. The complete Python suite
and physical package builds were not rerun; their exact preceding
current-worktree evidence remains separate. Exact-current-worktree remote CI,
exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current representative version-control activation evidence

The shared header version contract was refreshed against Transformers commit
`af0993dda925a8cac0a590f6e43a239933cc6d5b`. The 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. Existing official rendered-shell
evidence remains separate; no new remote version-menu interaction was
executed.

The observable local contract was fixed before implementation. All ten
representative routes must activate and close the version control at desktop,
tablet, and mobile widths in both palettes. The 30 default-palette cases use
unmodified Enter and the 30 slate cases use the pointer. Every opened state
must expose exactly three destinations with the expected labels and targets,
one current item, the open and expanded state, summary focus, a visible menu
inside the viewport, zero overflow, and a zero-violation Axe result. Escape
must close and hide the menu, restore the collapsed ARIA state, retain zero
overflow, and return focus to the version summary.

The focused source contract intentionally failed before implementation because
the checker had no version activation matrix, helper, or counters. The first
rendered probe then failed because the desktop and tablet version menu was
right-aligned to a 64-pixel left-rail control and extended outside the viewport.
The minimal responsive CSS correction left-aligns the LTR menu and right-aligns
the RTL menu inside that rail while preserving mobile placement. A fresh
six-case Home breakpoint/palette probe passed. Neither failed run is counted as
passing evidence.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,704 observed stops, 30 expanded-state Axe cases,
four branch/drawer activation cases, 60 version activation/closure cases, 60
search activation/closure cases, and 40 TOC activation cases. Each opened
version state received another Axe pass. The complete keyboard inventory is
now 226 cases.

The complete documentation suite reported 57 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
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

## Current representative language-control activation evidence

The shared header language contract was refreshed against Transformers commit
`af0993dda925a8cac0a590f6e43a239933cc6d5b`. The 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. Existing official rendered-shell
evidence remains separate; no new remote language-switch interaction was
executed.

The observable local contract was fixed before implementation. Because the
reference-sized language control is intentionally hidden at the shared mobile
breakpoint, all ten representative routes must switch locale at desktop and
tablet widths in both palettes. Each of the 40 cases must expose one visible,
focusable select with the exact ordered EN, TR, ES, FR, DE, PT, ZH, JA, KO, RU,
and AR inventory and production-base destination for the current route. The 20
default-palette cases use unmodified ArrowDown to select Turkish; the 20 slate
cases use pointer activation and semantic native-option selection to select
Arabic. The target must retain the palette, route, locale selection, LTR or RTL
direction, zero overflow, and a zero-violation Axe result.

The focused source contract intentionally failed before implementation because
the checker had no language activation matrix, helper, or counters. Rendered
probes then exposed three boundaries: the local server did not mount the
configured `/voicehub/` production base, headless Chromium did not surface
native picker key events, and a locale navigation reset slate to default. The
test server now serves the production base, the native select explicitly
commits unmodified ArrowUp and ArrowDown changes, and a one-navigation
`sessionStorage` transfer restores the chosen palette before removing its
temporary key. Follow-up assertion failures corrected native option selection,
the body-owned direction contract, and nested-route mounting. The first full
matrix still failed on that nested mount before the corrected four-case
Installation probe passed. None of the failed or timed-out runs is counted as
passing evidence.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed all 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,752 observed stops, 30 expanded-state Axe cases,
four branch/drawer activation cases, 40 language switches, 60 version
activation/closure cases, 60 search activation/closure cases, and 40 TOC
activation cases. Each localized destination received another Axe pass. The
complete keyboard inventory is now 246 cases.

The complete documentation suite reported 58 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
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

## Current representative theme-control activation evidence

The shared theme contract was refreshed against Transformers commit
`b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`. The 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200. Later repeated rendered probes were
rate-limited with HTTP 429 and are not counted as comparison evidence; the
successful current revision, source fingerprints, and route response remain
the upstream basis for this bounded local interaction slice.

The observable contract was fixed before implementation. The theme control is
intentionally hidden at the shared mobile breakpoint, so all ten
representative routes must switch palette at desktop and tablet widths from
both starting palettes. The 20 default-palette cases use Enter to select
`slate`; the 20 slate cases use pointer activation to select `default`. Every
case requires the exact two next-action labels and target inputs, a visible
native tab stop at the reference-sized 34 by 24-pixel geometry, route and
English-locale stability, focus transfer to the newly visible toggle, exact
target colors, zero overflow, and a zero-violation Axe result after the switch.

The focused source contract intentionally failed before implementation because
the checker had no theme activation matrix, helper, or counters. The first
four-case rendered probe rejected an incorrect y-coordinate expectation by one
pixel; the corrected Home probe passed all four cases. The first full matrix
then exposed an asynchronous setup race: a programmatic palette change could
complete the theme control's two-frame focus transfer during the later native
Tab cycle. The palette helper now waits for those exact animation frames. The
previously failing Models tablet/slate cycle then passed all 239 focus stops,
and the complete matrix passed from a fresh process. Neither failed run is
counted as passing evidence. No production theme template, JavaScript, or CSS
change was required.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,709 observed stops, 30 expanded-state Axe cases,
four branch/drawer activation cases, 40 theme switches, 40 language switches,
60 version activation/closure cases, 60 search activation/closure cases, and
40 TOC activation cases. Every switched theme received another Axe pass. The
complete keyboard inventory increased from 246 to 266 cases.

The complete documentation suite reported 59 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
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

## Current representative source-link activation evidence

The shared source-control contract was refreshed against Transformers commit
`b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`. The 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official Home route returned HTTP 200.

The observable contract was fixed before implementation. The source link is
intentionally hidden at the shared mobile breakpoint, so all ten
representative routes must activate it at desktop and tablet widths from both
palettes. Twenty default-palette cases use Enter and 20 slate cases use a
pointer. Every case requires exactly one native tab stop named `Open VoiceHub
source repository`, the exact `https://github.com/kadirnar/voicehub` target,
55 by 16-pixel geometry at x = 198 and y = 155, a two-pixel focus outline,
English route and palette stability before activation, zero overflow, and a
zero-violation Axe result. The browser must then perform an exact navigation;
the checker intercepts that declared external target with a deterministic
fixture so the matrix neither depends on GitHub availability nor claims to
audit GitHub's page.

The focused contract first failed because no source activation matrix, helper,
or counters existed. A later focused assertion also rejected a quote-sensitive
checker fragment and was corrected before the contract passed. The four-case
Home render then passed both palettes and visible viewports. The first two full
matrix attempts exposed a real sequential-focus race: closing Material search
after native Tab could asynchronously restore focus to the search input and
skip the source link. Those runs are excluded. The production header control
now cancels only that delayed restoration when Tab leaves the desktop input;
Escape closure still restores focus. A 20-cycle focused probe and a fresh full
matrix passed after rebuilding the site.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,621 observed stops, 30 expanded-state Axe cases,
four branch/drawer activation cases, 40 source navigations, 40 theme switches,
40 language switches, 60 version activation/closure cases, 60 search
activation/closure cases, and 40 TOC activation cases. Every focused source
link received its own local-page Axe pass. The complete keyboard inventory is
now 286 cases.

The complete documentation suite reported 60 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps.

This closes source-link activation across the visible representative local
desktop/tablet matrix. The mobile-hidden state remains covered by all 20 mobile
base cases. Non-representative routes and future shared controls remain outside
the matrix. The complete Python suite and physical package builds were not
rerun; their exact preceding current-worktree evidence remains separate.
Exact-current-worktree remote CI, exact tagged artifact hashes, protected
publisher configuration, tags, publication approval, five native-kernel
hardware gates, and two inaccessible WeNet asset paths remain open. The
untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current root left-navigation activation evidence

The next bounded shell iteration refreshed the navigation reference at
Transformers commit `b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`. Its 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official rendered Home route returned HTTP 200. Its current rail presents
the same eight ordered top-level sections as clickable disclosures. VoiceHub's
native buttons, `aria-expanded`, and labeled controlled panels are an
accessibility-preserving implementation of that interaction on the MkDocs
platform.

The observable contract was fixed before implementation. On the Pipeline
route, every one of the eight root branches must activate and return to its
initial state at desktop and tablet widths in both palettes. Sixteen
default-palette cases use Enter and 16 slate cases use a pointer. Each case
requires the exact root order, one exact target, correct initial and restored
checked state, `aria-expanded`, `aria-controls`, panel visibility and label,
the active Pipeline link and route, palette stability, zero horizontal
overflow, in-viewport rail geometry, and a zero-violation Axe result whenever
the branch is expanded. Keyboard cases additionally require retained focus and
the two-pixel visible outline; pointer cases require retained DOM focus without
incorrectly displaying the keyboard-only ring. The mobile drawer remains a
separate responsive contract.

The focused contract first failed because the checker did not yet define the
root matrix, helper, or counters. The first rendered probe rejected a desktop
coordinate incorrectly applied to tablet. The second exposed the legitimate
eight-pixel tablet rail scroll needed to keep the final API button and outline
visible. The third rejected a pointer activation for lacking a keyboard-only
focus ring. Those expectations were corrected without changing production
markup or styling, and none of the failed probes is counted. The complete
32-case rerun passed with Axe Core 4.12.1.

The strict eleven-language build and ten-route DOM validator passed.
Playwright 1.62.0 with Chromium 151.0.7922.34 and Axe Core 4.12.1 passed 60
base route/viewport/palette cases, 60 reviewed screenshot signatures, 60
complete focus cycles with 4,737 observed stops, 30 expanded-state Axe cases,
32 root-branch activation/restoration cases, two mobile drawer activation
cases, 40 source navigations, 40 theme switches, 40 language switches, 60
version activation/closure cases, 60 search activation/closure cases, and 40
TOC activation cases. The complete keyboard inventory is now 300 cases.

The complete documentation suite reported 61 passes and 1,408 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 generated provider pages and 59 generated model
notebooks remain current, and the five-record release-alignment check passed.
Refreshed inventories remain 68 models (34 TTS, 23 ASR, and 11 VAD), 102
aliases, zero invalid display names, six public optimization passes, 408
model/pass pairs, eight top-level navigation roots, ten representative routes,
and eight contribution steps. The selected pre-commit sequence exited zero for
every applicable hook; its Markdown hook reported no matching files and is not
counted as a pass.

This closes root-branch activation on the representative Pipeline route across
the visible desktop/tablet matrix. Nested branch activation and sticky behavior
outside that recorded route remain open. The complete Python suite and
physical package builds were not rerun; their preceding exact-current-worktree
evidence remains separate. Exact-current-worktree remote CI, exact tagged
artifact hashes, protected publisher configuration, tags, publication
approval, five native-kernel hardware gates, and two inaccessible WeNet asset
paths remain open. The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current API model-navigation hierarchy evidence

The next bounded hierarchy iteration refreshed Transformers `main` at commit
`b317ff31cd2491c2d4fc05d25fa06f35c527bcf6`. Its 65-line
`docs/source/en/index.md` source retains SHA-256
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b`,
and the 1,576-line `_toctree.yml` retains SHA-256
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The source places Auto Classes under `API → Main Classes` and individual model
guides, including SpeechT5, under `API → Models`. VoiceHub keeps Auto Classes
under `API → Main Classes` but groups model guides under `Base classes → Models`
beside catalogs and support. The official Home and Auto Classes routes returned
HTTP 200 and rendered the current API rail. The
official SpeechT5 shell also loaded its selected model link, but its content
pane displayed a rate-limit 429. That content capture is unavailable and is
not counted as current model-page comparison evidence.

The shared generator owns the canonical placement: Auto Classes appears exactly
once under `API → Main Classes`, and its generated block writes all 68 unique
model paths under `Base classes → Models`, grouped into 34 TTS, 23 ASR, and 11
VAD guides. Base classes retains lifecycle and support material, contribution,
preprocessors, and architecture. Auto Classes expands `API → Main Classes`;
SpeechT5 expands `Base classes → Models → Text to speech`. The generator remains
idempotent after the navigation update.

The first complete visual run after the hierarchy change rejected the stale
Quickstart footer expectation; it is not counted as passing evidence. After
the validator was aligned with the rendered document order, the strict
eleven-language build and ten-route DOM validator passed. The reviewed local
light and dark comparisons covered Auto Classes and SpeechT5 at desktop,
tablet, and mobile widths. The complete Playwright 1.62.0, Chromium
151.0.7922.34, and Axe Core 4.12.1 matrix then passed 60 base cases, 60 existing
screenshot signatures, 60 complete focus cycles with 4,575 stops, 30 expanded-
state Axe cases, 32 root-branch activation/restoration cases, two mobile drawer
activation cases, 40 source navigations, 40 theme switches, 40 language
switches, 60 version cases, 60 search cases, and 40 TOC cases. A fresh final
12-case model-route slice rechecked exact ancestry, screenshots, Axe, and 1,970
focus stops after formatting and the final strict build. The complete keyboard
inventory remains 300 cases.

The complete documentation suite reported 62 passes and 1,476 subtests. The
risk-proportional registry, speech-task, universal-optimization, packaging-
metadata, distribution-compliance, and release-readiness slice reported 97
passes and 522 subtests. All 68 model pages and 59 model notebooks remain
current, and the five-record release-alignment check passed. Refreshed
inventories remain 68 models, 102 aliases, zero invalid display names, six
public optimization passes, 408 model/pass pairs, eight top-level navigation
roots, ten representative routes, and eight contribution steps. The selected
pre-commit sequence passed every applicable hook on its second invocation; the
first invocation reformatted files and is not counted, and the Markdown hook
reported no matching files on both invocations.

This closes the upstream API placement gap for Auto Classes and every generated
model guide. The current official SpeechT5 content capture remains rate-limited
and unpassed; nested branch activation and sticky behavior outside the existing
representative matrix also remain open. The complete Python suite and physical
package builds were not rerun for this navigation-only slice; their preceding
exact-current-worktree evidence remains separate. Exact-current-worktree remote
CI, exact tagged artifact hashes, protected publisher configuration, tags,
publication approval, five native-kernel hardware gates, and two inaccessible
WeNet asset paths remain open. The untracked `uv.lock` stayed unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current nested navigation and sticky-rail evidence

This representative shell slice uses Transformers `main` commit
`339c18c08bf0c143b8307c255004506e358984f2`. Its 65-line
`docs/source/en/index.md` and 1,576-line `_toctree.yml` retain SHA-256 values
`e24e0eddadccef8b15e3f25c55acb0df0f65de21a4b5bf709d119d22d0eb555b` and
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`.
The official SpeechT5 route returned HTTP 200 and rendered current content at
desktop, tablet, and mobile widths without the earlier rate-limit page.

VoiceHub's SpeechT5 route is the deepest representative navigation path. Its
nine visible nested controls cover Models, catalogs, the three speech-task
groups, Contribute, Preprocessors, Architecture, and the active SpeechT5 page.
A 36-case matrix activates and restores each control at desktop and tablet
widths in both palettes: 18 default-palette cases use Enter and 18 slate cases
use a pointer.
Every case requires exact ancestry, initial and restored checked state,
`aria-expanded`, `aria-controls`, visible controlled panel, active SpeechT5
link, unchanged route and palette, retained focus, keyboard-only outline
semantics, and zero overflow. The document then scrolls to 320 pixels; the
270-pixel rail must remain sticky while its top changes from 65 to 0 pixels,
its height expands to the viewport, and the recorded shell offset becomes 65
pixels. Axe reruns in every sticky target state.

The source contract intentionally failed before the matrix existed. The first
rendered case then failed because the checker mixed the embedded active
`Usage` TOC link into the primary-navigation active-link inventory. After that
probe was corrected, the first complete matrix failed on four duplicate
subsection landmarks exposed by opening the active SpeechT5 page panel. Neither
run is counted as a pass. The shared navigation initializer now assigns unique
SpeechT5-scoped accessible names to all nested page-section landmarks. The
corrected 24-case focused matrix passed with Axe Core 4.12.1.

The strict eleven-language build, ten-route DOM validator, 63 documentation
tests with 1,476 subtests, 97-test release-risk slice with 522 subtests, 68-page
and 59-notebook generators, and five-record release alignment check passed.
The complete Playwright 1.62.0 and Chromium 151.0.7922.34 matrix passed 60
base cases, 60 screenshot signatures, 60 focus cycles with 4,628 stops, 30
expanded states, 32 root disclosures, 24 nested sticky disclosures, two mobile
drawer activations, and the existing source, theme, language, version, search,
and TOC matrices. The keyboard inventory is now 312 cases.

This closes nested disclosure activation and sticky behavior on the deepest
representative model route. Non-representative navigation structures and
future navigation controls remain outside the closed Pipeline and SpeechT5
matrices. The complete Python suite and physical package builds were not rerun
for this documentation-only slice; their preceding exact-current-worktree
evidence remains separate. Exact-current-worktree remote CI, exact tagged
artifacts, publisher configuration, tags, publication approval, five native-
kernel hardware gates, and two inaccessible WeNet asset paths remain open.
The untracked `uv.lock` stayed unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current representative page-action evidence

This slice uses Transformers `main` commit
`179a24360e55f1daff1bc20f0d11f4e8f47a6f44`; its current SpeechT5 source
retains the shared Copy page control and `DocFooterNav` previous/next region.
VoiceHub now validates the corresponding local edit, previous/next, and Back
to top behaviors on all ten representative routes at desktop, tablet, and
mobile widths in both palettes.

The 60-case matrix uses Enter in 30 default-palette cases and pointer input in
30 slate cases. It activates 60 exact edit targets through deterministic
external interception, 114 exact local footer destinations, and 60 Back to top
buttons. Every case checks native semantics, exact labels and targets, visible
focus, route and palette preservation where applicable, zero overflow, and an
Axe audit while Back to top is visible and focused. A missing Back to top focus
outline and a transient transition-state contrast failure were found and are
not counted as passes; the shared focus style and settled-state audit now pass.

The complete matrix passed with Playwright 1.62.0, Chromium 151.0.7922.34, and
Axe Core 4.12.1. The repository also passed 64 documentation tests with 1,476
subtests, the strict eleven-language build, ten-route DOM validation, the
97-test release-risk slice with 522 subtests, all 68 model-page and 59 notebook
freshness checks, the five-record release-alignment check, and fresh wheel,
sdist, and editable-install validation.

This closes representative local page-action behavior. Non-representative
routes, future controls, and accessibility or availability of the intercepted
external GitHub destination are not claimed. Exact-current-worktree remote CI,
tagged artifacts, protected publisher configuration, tags, publication
approval, hardware-only optimization checks, and inaccessible checkpoints
remain separate pending gates.

## Current TrainingArguments persistence-safety evidence

This public-lifecycle slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`. The current 2,906-line upstream
`src/transformers/training_args.py` has SHA-256
`cb55368c4d4b80633a7d289a790c5e4ab9f7ad03b527a969be7b0b0c2c432313`;
its dictionary and JSON methods remain the structural reference for
VoiceHub's speech-training arguments. The shared 1,576-line navigation and
500-line Modular Transformers sources retain SHA-256 values
`f7d0504e36cd7c312968b549af4fe02b6ee7b3c23d8023986e6a5824680c8f3a`
and `a52af1ee1754e6e658535f071a727fac31a057a3a848c28c18873a5ff371dd96`.

VoiceHub now applies its runtime-only credential rule across the complete
portable `TrainingArguments` lifecycle. Normal construction rejects nested
credential-shaped subclass fields. Untrusted JSON is checked before subclass
construction. Inherited dictionary, JSON-string, and file writers recheck the
complete current dataclass payload, and `Trainer.save_model()` performs that
preflight before creating the artifact destination. Safe metadata such as
`dataset_id` and `token_count` still round-trips.

The focused pre-implementation command is excluded: three tests failed while
only the safe case passed. After the shared fix, four focused tests and two
subtests passed. The proportional training/checkpoint/optimization slice passed
180 tests and 212 subtests, and the registry/public-API/release slice passed
151 tests and 621 subtests. The complete Python 3.12.12 suite passed 2,537
tests and 4,045 subtests with 35 warnings; its 15 explicit skips remain
unpassed.

Documentation passed 64 tests and 1,476 subtests, the strict eleven-language
build, and the ten-route DOM validator. All 68 generated model pages and 59
notebooks remain current, and five benchmark records align with version 0.3.0.
Fresh wheel, sdist, and editable-install validation passed with 68 models, 81
provenance manifests, 193 compliance files, zero runtime-dependency
violations, and no eager PyTorch import. The artifacts measured 57,193,213 and
55,456,837 bytes. The unchanged rendered shell retains its preceding visual
evidence; no new visual-parity claim is made.

This closes the argument-persistence credential gap without adding a task,
provider, reporting-backend, or subclass allowlist. The broader object-by-
object public API audit, exact-current-worktree remote platform and publication
gates, five native-kernel gates, and seven opt-in or inaccessible asset paths
remain open. No protected action was taken. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current optimization-manifest persistence-safety evidence

This public-lifecycle slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the current Home, navigation,
Modular Transformers, Trainer, and TrainingArguments source fingerprints match
the release-readiness ledger. VoiceHub's optimization extension contract now
applies the same runtime-only credential rule to every canonical strict-JSON
tree.

Pass configuration is checked before application, result metadata is checked
before the transformed graph is published, and evolving runtime status is
rechecked whenever a manifest is requested. An unsafe metadata result rolls
back the reversible pass. Error messages expose only the owning artifact and
field path. Safe fields such as `token_count` remain serializable. Because all
generic and TTS optimization declarations, snapshots, metadata, and manifests
share this boundary, no provider, pass, or architecture-specific list was
introduced.

The direct pre-fix probe accepted both configuration and runtime credentials;
the focused regression then failed one test with 29 deselected. The corrected
focused contract passed three tests. Two proportional optimization and
training slices passed 394 tests and 902 subtests. The 408 registry-wide
model/pass inventory, all 68 provider pages, all 59 generated notebooks, and
the five benchmark records remain current. The complete Python 3.12.12 suite
passed 2,538 tests and 4,045 subtests with 35 warnings; its 15 explicit skips
remain unpassed.

Documentation passed 64 tests and 1,476 subtests, the strict eleven-language
build, and the ten-route DOM validator. Fresh wheel, sdist, and editable
validation passed with 68 models, 81 provenance manifests, 193 compliance
files, all required package data, zero runtime-dependency violations, and no
eager PyTorch import. The unchanged representative shell retains its preceding
visual evidence; this slice makes no new visual-parity claim.

This closes the optimization-manifest credential gap. The broader object-by-
object public API audit, exact-current-worktree remote platform and publication
gates, five native-kernel gates, and seven opt-in or inaccessible asset paths
remain open. No protected action was taken. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current training-recipe persistence-safety evidence

This public-lifecycle slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the current Home, navigation,
Modular Transformers, Trainer, and TrainingArguments source fingerprints match
the release-readiness ledger. VoiceHub now applies its runtime-only credential
rule to the training-recipe manifest produced by every adapter extension.

`Trainer.save_model()` validates the exact adapter mapping before backend
import, model-state access, destination creation, or native export. It retains
that validated payload and rechecks the final mapping immediately before
writing `training_recipe.json`. Errors expose only the owning manifest and key
path. Safe metadata such as `dataset_id` and `token_count` remains serializable.
Because this is the shared Trainer boundary, no task, provider, adapter, or
recipe-specific list was introduced.

The direct pre-fix probe persisted a nested credential and created the output
directory; the focused regression then failed one test with 39 deselected. The
corrected focused contract passed one test. The proportional training,
checkpoint, optimization, registry, documentation, distribution-policy, and
release slices passed 285 tests and 2,633 subtests. The complete Python 3.12.12
suite passed 2,539 tests and 4,045 subtests with 35 warnings; its 15 explicit
skips remain unpassed.

Documentation passed 64 tests and 1,476 subtests, the strict eleven-language
build, and the ten-route DOM validator. All 68 generated model pages and 59
notebooks remain current, and five benchmark records align with version 0.3.0.
Fresh wheel, sdist, and editable validation passed with 68 models, 81
provenance manifests, 193 compliance files, all required package data, zero
runtime-dependency violations, and no eager PyTorch import. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the adapter-supplied training-recipe credential gap. The broader
object-by-object public API audit, exact-current-worktree remote platform and
publication gates, five native-kernel gates, and seven opt-in or inaccessible
asset paths remain open. No protected action was taken. The untracked
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current model-artifact state persistence-safety evidence

This public-lifecycle slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the current Home, navigation,
Modular Transformers, Trainer, and TrainingArguments source fingerprints match
the release-readiness ledger. VoiceHub now applies its runtime-only credential
rule to the binary model state written by the shared Trainer lifecycle.

`Trainer.save_model()` checks the exact state returned by the model or portable
optimization result before artifact mutation and again immediately before
writing `model_state.pt`. The second boundary catches a mapping changed by an
intervening model-owned save. Errors expose only the owning artifact and field
path. Safe tensor state and descriptive `dataset_id` and `token_count` metadata
still round-trip. Because this is the shared Trainer boundary, no task,
provider, model, or optimization-pass allowlist was introduced.

The direct pre-fix probe persisted a nested credential and created the output
directory; the focused regression then failed one test with 40 deselected. The
corrected focused contract passed one test. The complete training-runtime file
passed 41 tests and seven subtests. Proportional training, checkpoint,
optimization, registry, public-API, documentation, packaging, and release
slices passed 505 tests and 2,762 subtests. The complete Python 3.12.12 suite
passed 2,540 tests and 4,045 subtests with 35 warnings; its 15 explicit skips
remain unpassed.

Documentation passed 64 tests and 1,476 subtests, the strict eleven-language
build, and the ten-route DOM validator. All 68 generated model pages and 59
notebooks remain current, and five benchmark records align with version 0.3.0.
Fresh wheel, sdist, and editable validation passed with 68 models, 81
provenance manifests, 193 compliance files, all required package data, zero
runtime-dependency violations, and no eager PyTorch import. The artifacts
measured 57,193,419 and 55,458,392 bytes. The unchanged representative shell
retains its preceding visual evidence; this slice makes no new visual-parity
claim.

This closes the shared Trainer model-state credential gap. The broader object-
by-object public API audit, exact-current-worktree remote platform and
publication gates, five native-kernel gates, and seven opt-in or inaccessible
asset paths remain open. No protected action was taken. The untracked
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current model-state load safety evidence

This public-lifecycle slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the current Home, navigation,
Modular Transformers, Trainer, and TrainingArguments source fingerprints match
the release-readiness ledger. VoiceHub now applies its runtime-only credential
rule to Trainer-owned model state on both sides of the exact-resume boundary.

`Trainer.save_model()` already checked the exact state before artifact mutation
and at the final write boundary. Exact checkpoint resume now checks the
deserialized state before calling the model's `load_state_dict()`. Errors expose
only the owning artifact and field path, and ordinary model state still
restores exactly once. This remains a credential boundary rather than a claim
that Python pickle accepts untrusted input. No task, provider, model, or
optimization-pass allowlist was introduced.

The focused pre-fix regression reached the tracking model's load boundary; it
failed with a PyTorch key error and is excluded. The corrected
focused contract passed, and the complete training-runtime file passed 42
tests and seven subtests. Proportional training, checkpoint, optimization,
registry, public-API, speech-task, documentation, packaging-policy, and release
slices passed 561 tests and 2,959 subtests. The complete Python 3.12.12 suite
passed 2,541 tests and 4,045 subtests with 35 warnings; its 15 explicit skips
remain unpassed.

Documentation retained the strict eleven-language build and ten-route DOM
contract. All 68 generated model pages, 59 generated model notebooks, six
public optimization passes, 408 model/pass pairs, and five benchmark records
remain current. Fresh isolated wheel, sdist, and editable validation passed
with 68 models, 81 provenance manifests, 193 compliance files, required
package data, zero runtime-dependency violations, and no eager PyTorch import.
The unchanged representative shell retains its preceding visual evidence; this
slice makes no new visual-parity claim.

This closes the Trainer model-state read boundary. The broader object-by-object
public API audit, exact-current-worktree remote platform and publication gates,
five native-kernel gates, and seven opt-in or inaccessible asset paths remain
open. No protected action was taken. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current public portable model-state load safety evidence

This public-lifecycle slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the current Home, navigation,
Modular Transformers, Trainer, and TrainingArguments source fingerprints match
the release-readiness ledger. VoiceHub now applies its runtime-only credential
rule to portable model state across the shared public TTS, ASR, and VAD
pretrained bases.

One task-neutral validator checks deserialized `model_state.pt` before the
training-adapter and ordinary-runtime restore branches. Credential-shaped
fields therefore fail before adapter `setup()` or runtime `load_state_dict()`
can mutate a fresh model. Errors expose only the owning artifact and field
path, rejected state remains retryable, and safe state still restores exactly
once in inference mode. This remains a credential boundary rather than a
safe-unpickling claim. No task, provider, model, adapter, or optimization-pass
allowlist was introduced.

The two-test pre-fix regression reached both mutation paths without a policy
error and is excluded. The corrected focused contract passed two tests; the
complete inference-lifecycle and speech-core files passed 76 tests and 41
subtests. Proportional training, checkpoint, optimization, registry,
public-API, automatic-model, pipeline, speech-task, documentation, packaging,
distribution-policy, release, and scaffold slices passed 608 tests and 3,051
subtests. The complete Python 3.12.12 suite passed 2,543 executed tests and
4,045 subtests with 35 warnings; its 15 explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Documentation retained the strict eleven-language build and
ten-route DOM contract. Fresh isolated wheel, sdist, and editable validation
passed with 68 models, 81 provenance manifests, 193 compliance files,
required package data, zero runtime-dependency violations, and no eager
PyTorch import. The unchanged representative shell retains its preceding
visual evidence; this slice makes no new visual-parity claim.

This closes the shared public portable model-state read boundary. The broader
object-by-object public API audit, exact-current-worktree remote platform and
publication gates, five native-kernel gates, and seven opt-in or inaccessible
asset paths remain open. No protected action was taken. The untracked
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current speech dataset manifest persistence-safety evidence

This data-contract slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the current Home, navigation,
Modular Transformers, and TrainingArguments fingerprints match the release
ledger. Revision-pinned revalidation shows that the Trainer source is 30 lines
with SHA-256
`e7c5368c1223c2b195321468fcd1ac64c5cda52a1e44d4adf854a6473b6c9ee0`.
The earlier 54-line response was not reproducible against that revision and is
excluded rather than transferred.

TTS and ASR portable manifest readers now reject credential-shaped fields
after parsing and before dataset construction. Both `to_jsonl()` paths reject
the same fields after portable path normalization and before filesystem
mutation. TTS export additionally precomputes every line before opening the
destination, so a credential or ordinary serialization failure cannot create
a parent directory or truncate existing data. Errors expose only the task,
record index, and field path; `token_count` remains portable. No task-provider
or architecture allowlist was introduced.

The four-test pre-fix regression accepted every unsafe record and is excluded.
The corrected focused contract passed five tests; the complete TTS and ASR
dataset files passed 79 tests and 116 subtests. Proportional training,
registry, optimization, public-API, pipeline, inference, speech-task,
documentation, packaging, distribution-policy, release, and scaffold slices
passed 578 tests and 3,021 subtests. The complete Python 3.12.12 suite passed
2,548 executed tests and 4,045 subtests with 35 warnings; its 15 explicit skips
remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Documentation retained the strict eleven-language build and
ten-route DOM contract. Fresh isolated wheel, sdist, and editable validation
passed with 68 models, 81 provenance manifests, 193 compliance files,
required package data, zero runtime-dependency violations, and no eager
PyTorch import. The unchanged representative shell retains its preceding
visual evidence; this slice makes no new visual-parity claim.

This closes the public TTS/ASR dataset-manifest credential and preflight gap.
The broader object-by-object public API audit, exact-current-worktree remote
platform and publication gates, five native-kernel gates, and seven opt-in or
inaccessible asset paths remain open. No protected action was taken. The
untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current shared JSON artifact atomicity evidence

This data-safety slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. VoiceHub's shared model-artifact JSON writer now
encodes finite deterministic JSON before filesystem mutation, flushes a
temporary sibling, and atomically replaces the destination. Trainer JSON output
uses that same path. Serialization and replacement failures therefore preserve
existing documents, avoid rejected parent-directory creation, and remove
temporary files; safe `token_count` metadata retains the existing stable format.
Multi-file native exports continue to rely on their separate staging and
manifest contracts.

The pre-fix command exited one with nine failing subcases and is excluded. The
corrected focused contract passed three tests and 12 subtests. Proportional
public-API, configuration, processing, pipeline, training, checkpoint,
inference, registry, optimization, documentation-policy, packaging-policy,
distribution-policy, and release slices passed 400 tests and 2,692 subtests.
The complete Python 3.12.12 suite passed 2,551 executed tests and 4,057 subtests
with 35 warnings; its 15 explicit skips remain unpassed. Corrected focused
probes also passed on Python 3.10.19 and 3.11.15; their first shared harness run
is excluded because it asserted the wrong newline representation.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Documentation passed 87 tests and 1,552 subtests, the strict
eleven-language build, and the ten-route DOM contract. Fresh isolated wheel,
sdist, and editable validation passed with 68 models, 81 provenance manifests,
193 compliance files, required package data, zero runtime-dependency
violations, and no eager PyTorch import. The unchanged representative shell
retains its preceding visual evidence; this slice makes no new visual-parity
claim.

This closes the shared JSON document preflight and atomic-replacement gap. The
broader object-by-object public API audit, exact-current-worktree remote
platform, complete Python 3.10/3.11, default-runtime, and publication gates,
five native-kernel gates, and seven opt-in or inaccessible asset paths remain
open. No protected action was taken. The untracked `uv.lock` remained unchanged
at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current shared JSON artifact read-safety evidence

This data-safety slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official Modular Transformers route returned
HTTP 429 and is unavailable rather than passed; its pinned raw source still
matched the recorded fingerprint.

VoiceHub's shared JSON loader now rejects duplicate object keys at every
nesting level and rejects `NaN`, positive or negative `Infinity`, and exponent
overflow. Configuration and automatic-model callers therefore fail before
construction or provider dispatch can interpret an ambiguous artifact.
Diagnostics identify the source and duplicate key or numeric path without
printing discarded values. Ordinary syntax errors retain their parser type,
and finite metadata such as `token_count` remains compatible. No task,
provider, model, or optimization-pass allowlist was introduced.

The pre-fix command exited one with seven failures and is excluded. The first
post-fix command is also excluded because three diagnostic-case assertions
failed. The corrected focused contract passed seven tests and 18 subtests.
Dependent API, auto-configuration, inference, pipeline, speech-core, and
checkpoint contracts passed 137 tests and 53 subtests. Registry and
optimization contracts passed 234 tests and 1,339 subtests. The complete
Python 3.12.12 suite passed 2,555 tests and 4,063 subtests with 35 warnings;
its 15 explicit skips remain unpassed. Focused probes passed on Python 3.10.19
and 3.11.15, while complete exact-current-worktree runs remain pending there.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The unchanged
representative shell retains its preceding visual evidence. The focused reader
plus release, distribution, and documentation-policy selection passed 84 tests
and 1,494 subtests, followed by the strict eleven-language build and ten-route
DOM validator. This slice makes no new visual-parity claim.

This closes the shared artifact read-ambiguity and non-finite-number gap. The
broader object-by-object public API audit, exact-current-worktree remote
platform, complete Python 3.10/3.11, default-runtime, and publication gates,
five native-kernel gates, and seven opt-in or inaccessible asset paths remain
open. No protected action was taken. The untracked `uv.lock` remained unchanged
at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Trainer JSON read-safety evidence

This exact-resume slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The corresponding official documentation and source
routes returned current content.

VoiceHub now routes `TrainingArguments`, `TrainerState`, checkpoint discovery,
checkpoint manifests, optimization manifests, and exact-resume Trainer state
through one strict JSON reader. Duplicate keys and non-finite numbers therefore
fail before object construction or model/runtime restoration. Discovery skips
an invalid candidate. Diagnostics name the source and duplicate key or numeric
path without printing discarded values. Valid saved objects and exact resume
retain their existing public behavior. No task, provider, model, or
optimization-pass branch was introduced.

The pre-fix command exited one with 11 failures and is excluded. The corrected
focused contract passed three tests and 11 subtests after a formatting-only
YAPF rerun. Complete Trainer/runtime coverage passed 73 tests and 34 subtests;
proportional training/checkpoint/optimization coverage passed 285 tests and
692 subtests; and registry/lifecycle coverage passed 139 tests and 499
subtests. The complete Python 3.12.12 suite passed 2,558 tests and 4,074
subtests with 35 warnings; its 15 explicit skips remain unpassed. Focused
probes passed on Python 3.10.19 and 3.11.15, while complete exact-current-
worktree runs remain pending there.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The unchanged
representative shell retains its preceding visual evidence. The focused
Trainer readers plus release, distribution, and documentation-policy selection
passed 80 tests and 1,487 subtests, followed by the strict eleven-language
build and ten-route DOM validator. This data-safety slice makes no new visual-
parity claim.

This closes the Trainer-owned JSON read-ambiguity and non-finite-number gap.
The broader object-by-object public API audit, exact-current-worktree remote
platform, complete Python 3.10/3.11, default-runtime, and publication gates,
five native-kernel gates, and seven opt-in or inaccessible asset paths remain
open. No protected action was taken. The untracked `uv.lock` remained unchanged
at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current native checkpoint JSON read-safety evidence

This data-safety slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

VoiceHub now routes `VoiceHubManifest`, bounded Safetensors headers, and
sharded Safetensors indexes through the shared strict JSON decoder. Duplicate
keys and non-finite numbers therefore fail before manifest construction,
tensor materialization, or shard lookup. Diagnostics name the source and
duplicate key or numeric path without printing discarded values. Valid
manifests, deterministic Safetensors files, and descriptive `token_count`
metadata retain their formats. No provider, model, task, or optimization-pass
branch was introduced.

The pre-fix command exited one with eight failing subcases and is excluded;
its three top-level passes are not counted. The corrected focused contract
passed three tests and eight subtests. Native-checkpoint and shared-reader
coverage passed 26 tests and 26 subtests; proportional public lifecycle and
training coverage passed 165 tests and 89 subtests; and registry, task,
inference, and optimization coverage passed 124 tests and 969 subtests.
Dependency-light probes passed on Python 3.10.19 and 3.11.15. The complete
Python 3.12.12 suite passed 2,561 tests and 4,082 subtests with 35 warnings;
its 15 explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the native checkpoint JSON ambiguity and non-finite-number gap.
The broader object-by-object public API audit, exact-current-worktree remote
platform, complete Python 3.10/3.11, default-runtime, and publication gates,
five native-kernel gates, and seven opt-in or inaccessible asset paths remain
open. No protected action was taken. The untracked `uv.lock` remained unchanged
at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current portable-state config read-safety evidence

This public-lifecycle slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

VoiceHub's TTS and audio-input pretrained base loaders now route a portable
state artifact's sibling `config.json` through the shared strict reader even
when an explicit configuration was supplied. Duplicate keys and non-finite
numbers fail before wrapper construction across TTS, ASR, and VAD. Diagnostics
name the source and duplicate key or numeric path without printing discarded
values. Valid saved base-checkpoint identity and lazy restoration retain their
existing behavior. No provider, model, task, or optimization-pass branch was
introduced.

The pre-fix command exited one with six failing subcases and is excluded; its
two top-level passes are not counted. A first expanded post-fix command is also
excluded because two valid-artifact assertions did not normalize macOS's
`/var` symlink. The corrected focused contract passed two tests and six
subtests. Public lifecycle and training coverage passed 221 tests and 181
subtests; registry, task, and optimization coverage passed 76 tests and 953
subtests. Dependency-light probes passed on Python 3.10.19 and 3.11.15 after
direct pytest attempts reported that pytest was not installed. The complete
Python 3.12.12 suite passed 2,563 tests and 4,088 subtests with 35 warnings;
its 15 explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the explicit-config portable-state JSON bypass. The broader
object-by-object public API audit, exact-current-worktree remote platform,
complete Python 3.10/3.11, default-runtime, and publication gates, five native-
kernel gates, and seven opt-in or inaccessible asset paths remain open. No
protected action was taken. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current Hub JSON trust-boundary evidence

This public-artifact slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

VoiceHub now routes remote Hub API payloads and native file/snapshot cache
metadata through the same strict, dependency-light JSON decoder as local
artifacts. Duplicate keys and non-finite numbers fail before remote commit or
tree interpretation. Ambiguous cache objects become offline cache misses.
Diagnostics retain source context without printing discarded values or Hub
tokens. Valid list-valued repository trees, online/offline resolution,
redirects, and PyTorch-free import behavior are preserved. No provider, model,
task, or optimization-pass branch was introduced.

The pre-fix command exited one with five failures and one reported pass and is
excluded. The corrected focused contract passed three tests and three
subtests. Hub, strict-artifact, and native-checkpoint coverage passed 48 tests
and 34 subtests; public lifecycle coverage passed 123 tests and 59 subtests;
and registry, task, and optimization coverage passed 76 tests and 953
subtests. Dependency-light exact probes passed on Python 3.10.19 and 3.11.15.
The complete Python 3.12.12 suite passed 2,566 tests and 4,091 subtests with 35
warnings; its 15 explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the Hub API and native cache JSON ambiguity gap. The broader
object-by-object public API audit, exact-current-worktree remote platform,
complete Python 3.10/3.11, default-runtime, and publication gates, five native-
kernel gates, and seven opt-in or inaccessible asset paths remain open. No
protected action was taken. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current tokenizer JSON trust-boundary evidence

This public-artifact slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

VoiceHub's shared Hugging Face Byte-BPE and SentencePiece-BPE loaders now route
their bounded tokenizer documents through the common strict JSON decoder before
interpreting model graphs, vocabularies, merges, or added tokens. Duplicate
keys and non-finite numbers fail with source-aware diagnostics that omit
discarded values. Existing graph and resource limits, valid load/save behavior,
registered consumers, and PyTorch-free imports are preserved. No provider,
model, task, or optimization-pass branch was introduced.

The pre-fix command exited one with six failing subcases and two reported
top-level passes and is excluded. The corrected focused contract passed two
tests and six subtests. Core tokenizer and closely related native coverage
passed 59 tests and 27 subtests; eleven additional native consumer suites
passed 166 tests and 27 subtests; and registry, task, optimization,
documentation, and native dependency-policy coverage passed 159 tests and
2,429 subtests. Corrected dependency-light probes passed on Python 3.10.19 and
3.11.15 after an initial helper-based probe failed before the contract because
PyTorch was unavailable. The complete Python 3.12.12 suite passed 2,568 tests
and 4,097 subtests with 35 warnings; its 15 explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the two common tokenizer loaders' JSON ambiguity gap. The broader
object-by-object public API audit, exact-current-worktree remote platform,
complete Python 3.10/3.11, default-runtime, and publication gates, five native-
kernel gates, and seven opt-in or inaccessible asset paths remain open. No
protected action was taken. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current speech manifest JSON trust-boundary evidence

This public-dataset slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

VoiceHub's TTS and ASR dataset loaders now route JSON manifests, JSON Lines
records, and JSON-shaped CSV/TSV fields through the shared strict decoder.
Duplicate keys and non-finite numbers fail before dataset construction with
source-aware diagnostics that omit discarded values. Existing malformed
tabular-string coercion, ASR syntax-only NeMo JSON Lines fallback, credential
checks, portable round trips, and PyTorch-free imports are preserved. No
provider, model, task, or optimization-pass branch was introduced.

The pre-fix command exited one with 18 failures and two reported passes and is
excluded. The corrected focused contract passed two tests and 18 subtests.
Dataset coverage passed 81 tests and 134 subtests; proportional training
coverage passed 168 tests and 212 subtests; and registry, task, optimization,
documentation, scaffold, and native dependency-policy coverage passed 179
tests and 2,464 subtests. Dependency-light exact probes passed on Python
3.10.19 and 3.11.15. The complete Python 3.12.12 suite passed 2,570 tests and
4,115 subtests with 35 warnings; its 15 explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the speech dataset manifest JSON ambiguity gap. The broader
object-by-object public API audit, exact-current-worktree remote platform,
complete Python 3.10/3.11, default-runtime, and publication gates, five native-
kernel gates, and seven opt-in or inaccessible asset paths remain open. No
protected action was taken. The untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current external LLM JSON response trust-boundary evidence

This public-serving slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

VoiceHub's bounded vLLM and SGLang response path now uses the shared strict
JSON decoder before interpreting token IDs, usage metadata, or other protocol
fields. Duplicate keys and non-finite numbers fail through the existing
sanitized backend error with route and key or numeric-path context, without
echoing discarded values. Valid responses, transport limits, redirect
rejection, credential redaction, capability dispatch, and PyTorch-free imports
are preserved. No provider, model, task, or optimization-pass branch was
introduced.

An initial pre-fix command is excluded because its test control flow produced
a secondary error. The corrected red command failed all three ambiguity
subcases and is also excluded. The post-fix focused contract passed one test
and three subtests. LLM-serving coverage passed 53 tests and 57 subtests;
proportional serving, lifecycle, registry, task, and optimization coverage
passed 237 tests and 1,185 subtests; release, distribution, and documentation
policy coverage passed 77 tests and 1,476 subtests. Dependency-light exact
probes passed on Python 3.10.19 and 3.11.15. The complete Python 3.12.12 suite
passed 2,571 tests and 4,118 subtests with 35 warnings; its 15 explicit skips
remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the external LLM JSON response ambiguity gap. The broader object-
by-object public API audit, exact-current-worktree remote platform, complete
Python 3.10/3.11, default-runtime, and publication gates, five native-kernel
gates, and seven opt-in or inaccessible asset paths remain open. No protected
action was taken. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current model-integration JSON trust-boundary evidence

This contribution-path slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

Activated package-local `model-integration.json` files and their required
`source/SOURCE.json` records now pass through strict JSON decoding before
registry or training-profile construction. Duplicate keys and non-finite
numbers fail with file-, key-, or numeric-path-aware diagnostics that omit
discarded values. The standalone scaffold checker and catalog renderer enforce
the same boundary without importing VoiceHub or PyTorch. Inactive
work-in-progress manifests remain undiscovered, valid zero-central-edit TTS,
ASR, and VAD integrations retain their behavior, and no provider branch was
introduced.

The pre-fix command exited one with six failures, two reported passes, and
three reported subtests and is excluded. The corrected focused contract passed
two tests and nine subtests; the complete scaffold suite passed 22 tests and 44
subtests; and proportional registry, task, training, documentation, release,
optimization, and distribution coverage passed 241 tests and 2,659 subtests.
Dependency-light exact probes passed on Python 3.10.19 and 3.11.15. The
complete Python 3.12.12 suite passed 2,573 tests and 4,127 subtests with 35
warnings; its 15 explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the activated model-integration metadata ambiguity gap. The
broader object-by-object public API audit, exact-current-worktree remote
platform, complete Python 3.10/3.11, default-runtime, and publication gates,
five native-kernel gates, and seven opt-in or inaccessible asset paths remain
open. No protected action was taken. The untracked `uv.lock` remained
unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current shared JSON artifact byte-bound evidence

This public-artifact slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

VoiceHub's shared JSON object reader now validates a positive integer byte
ceiling and defaults to 64 MiB. It rejects a file already over the ceiling and
reads at most one byte beyond the limit so a document that grows during the
operation also fails before decoding. Diagnostics identify the source and size
or limit without exposing document content. Exact-limit finite JSON, strict
duplicate and non-finite rejection, caller-selected smaller ceilings, and lazy
dependency boundaries remain intact. The configuration, auto-loading,
processing, training, checkpoint, and native-artifact surfaces inherit one
capability-driven implementation.

The pre-fix command exited one with six failures and one reported pass and is
excluded. The corrected focused contract passed four tests and four subtests;
the complete shared JSON artifact suite passed 11 tests and 22 subtests; and
proportional public lifecycle, registry, optimization, documentation, release,
and distribution coverage passed 409 tests and 2,663 subtests. Dependency-
light exact probes passed on Python 3.10.19 and 3.11.15. The complete Python
3.12.12 suite passed 2,577 tests and 4,131 subtests with 35 warnings; its 15
explicit skips remain unpassed.

All 68 generated model pages, 59 generated model notebooks, six public
optimization passes, 408 model/pass pairs, and five benchmark records remain
current. Fresh isolated wheel, sdist, and editable validation passed with 68
models, 81 provenance manifests, 193 compliance files, required package data,
zero runtime-dependency violations, and no eager PyTorch import. The strict
eleven-language build and ten-route DOM validator passed. The unchanged
representative shell retains its preceding visual evidence; this slice makes
no new visual-parity claim.

This closes the unbounded shared JSON artifact-read gap. The broader object-by-
object public API audit, exact-current-worktree remote platform, complete
Python 3.10/3.11, default-runtime, and publication gates, five native-kernel
gates, and seven opt-in or inaccessible asset paths remain open. No protected
action was taken. The untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current supported-Python macOS evidence

This platform slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official reference documentation and source
routes returned current content.

Fresh temporary CPython 3.10.19 and 3.11.15 environments installed the current
checkout with its declared `test` extra through uv 0.11.21's CPU PyTorch
resolver without a lockfile operation. Both imported VoiceHub from the working
tree and used macOS 26.5.2 arm64, PyTorch 2.8.0, Transformers 5.14.1, and
pytest 9.1.1.

The complete Python 3.10.19 suite passed 2,577 tests and 4,131 subtests with 35
warnings in 173.72 seconds. The complete Python 3.11.15 suite passed the same
counts in 163.92 seconds. Together with the existing Python 3.12.12 result of
the same counts in 114.47 seconds, the exact current macOS runtime passes the
complete CPU-safe suite on all three declared Python versions. Each run's 15
explicit skips remain unpassed.

After this evidence-only update, release, distribution, and documentation-
policy coverage passed 77 tests and 1,476 subtests independently on Python
3.10.19, 3.11.15, and 3.12.12. The strict eleven-language build, ten-route DOM
validator, and isolated wheel, sdist, and editable checks also passed.

This closes the locally executable complete Python 3.10/3.11 gap. Exact-
current-worktree Linux, Windows, default-runtime, publication, five native-
kernel, and seven opt-in or inaccessible asset gates remain open, as does the
broader object-by-object public API audit. No protected action was taken. The
untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current full-dependency default-runtime evidence

This release slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned Home,
navigation, Modular Transformers, Trainer, and TrainingArguments fingerprints
match the release ledger. The official navigation, documentation specification,
Modular Transformers, and Trainer references returned current content. The
documentation Home endpoint returned HTTP 429 and is not counted as a pass.

A pristine Python 3.12.12 direct install of
`.[test,training,docs]` first failed because librosa's broad numba range allowed
uv to select Python-3.12-incompatible numba 0.53.1. VoiceHub's test extra now
declares `numba>=0.59` without adding numba to the default inference runtime.
The corrected direct install resolved 135 compatible packages without a
lockfile operation, and focused packaging plus activated default-runtime
coverage passed 16 tests and 214 subtests.

The complete activated suite passed 2,581 tests and 4,269 subtests with 35
warnings in 176.53 seconds. Twelve CUDA/Triton, opt-in asset/oracle, and
inaccessible WeNet checks remain explicitly unpassed. Proportional inventory
coverage passed 205 tests and 2,556 subtests; the five benchmark records, 68
model pages, 59 notebooks, six optimizations over 408 model/pass pairs, and
eight navigation roots remain current. The strict eleven-language build,
ten-route DOM validator, and isolated wheel, source-distribution, and editable
checks passed.

This closes the locally executable full-dependency/default-runtime gate
without changing a parity mapping or making a new visual claim. Exact-current-
worktree Linux, Windows, remote default-runtime, publication, five native-
kernel, seven asset, and broader public-API audit gates remain open. No
protected action was taken, and the untracked `uv.lock` remained unchanged at
SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current generated public API inventory

This public-contract slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned source
fingerprints match the release ledger. The Transformers Models main-class page
is the rendered structural reference, while VoiceHub keeps speech-specific
grouping and names.

VoiceHub now generates an object-by-object reference for all 261 unique root
exports. Every row records canonical repository source and line, kind,
signature or explicit constant/type-alias marker, summary, and lazy state.
Static root metadata validation and generic facade re-export resolution make
duplicate, unresolved, undocumented, source-less, and stale entries fail.
The inventory contains 131 classes, 73 callables, 39 enums, 11 exceptions,
four constants, and three type aliases across seven speech-domain groups.

The initial red test failed because no generator existed. Intermediate source-
resolution failures and three rendered-comparison setup failures are excluded.
A fresh Python 3.10 run then found that `EnumMeta` introspection produced
version-dependent output; its 66 passes, two failures, and 1,741 subtests are
not a pass. Enum entries now use the stable public `(value)` contract. The
corrected public-API/documentation selection passed 68 tests and 2,002
subtests on Python 3.10.19, 3.11.15, and 3.12.12. The final Python 3.12 complete
suite passed 2,582 tests and 4,657 subtests with 15 explicit skips and 35
warnings.

The strict eleven-language build and ten-route DOM validator passed. The
complete representative Playwright/Axe matrix passed 60 base and screenshot
cases, 342 keyboard cases, and 4,613 focus steps after its first run correctly
failed a stale Models-next-page expectation. A focused six-case render covered
Public exports in both palettes at desktop, tablet, and mobile widths with 261
source links, seven tables, active navigation, and zero overflow. The official
Transformers Models main-class page rendered at the same three viewports.
This establishes structural comparison, not raw-pixel equivalence across
different products and content.

All 68 generated provider pages, 59 notebooks, six public optimizations over
408 model/pass pairs, and five benchmark records remain current. This closes
the broader package-root public API audit. Exact-current complete Python
3.10/3.11, Linux, Windows, remote default-runtime, hardware/asset, tagged-
workflow, publisher, and publication gates remain open. No protected action
was taken, and the untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current final supported-Python evidence

This evidence-only platform slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; all five revision-pinned source
fingerprints match the release ledger, and the four official reference routes
returned HTTP 200.

Fresh CPython 3.10.19 and 3.11.15 environments installed the exact checkout
directly with its declared test extra through uv 0.11.21's CPU PyTorch resolver
without a lockfile synchronization. Both passed dependency checks, resolved
VoiceHub to this worktree, and imported all 261 root exports without importing
PyTorch. They used macOS 26.5.2 arm64, PyTorch 2.8.0, Transformers 5.14.1, and
pytest 9.1.1.

The complete Python 3.10 suite passed 2,582 tests and 4,657 subtests with 15
skips and 35 warnings in 229.03 seconds. Python 3.11 passed the same counts in
214.94 seconds. Together with Python 3.12's exact-current result of the same
counts in 123.52 seconds, the complete CPU-safe suite now passes on every
supported interpreter for the exact current macOS worktree. The 15 skips remain
unpassed: three default-runtime checks, three Triton checks, two CUDA-extension
checks, five opt-in asset/oracle checks, and two inaccessible WeNet paths.

The public-API, documentation, release, packaging, and distribution-policy
selection then passed 92 tests and 2,078 subtests independently on Python 3.10,
3.11, and 3.12. All registry, model-page, notebook, optimization, navigation,
contribution, public-export, and benchmark inventories remain current. No
runtime, test, workflow, or generated artifact source changed in this slice.

The strict eleven-language build and ten-route DOM validator passed. Fresh
isolated wheel, source-distribution, and editable validation also passed with
68 models, 81 provenance manifests, 193 compliance files, required package
data, zero runtime-dependency violations, and no eager PyTorch import. The
unchanged representative shell retains its preceding visual evidence; this
platform-only slice makes no new visual-parity claim.

This closes the locally executable exact-current supported-Python gap. Linux,
Windows, remote default-runtime, hardware/asset, tagged-workflow, publisher,
and publication gates remain open. No protected action was taken, and the
untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current pinned release assets

This checkpoint slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the revision-pinned reference
fingerprints remain unchanged. It changes no parity mapping or model contract.

The pinned ESPNet configuration, SenseVoice tokenizer, and SpeechBrain
tokenizer reproduced their immutable Hugging Face revisions, declared sizes,
SHA-256 values, and published token or semantic vectors. Their combined
exact-current command passed three tests with 49 deselections. The official
315,449-byte TEN-VAD ONNX graph at revision
`22a3bcd4509d0faaa8eef4881e8af5f39c178950` reproduced digest
`e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4`,
converted to the native Safetensors boundary, and matched ONNX Runtime 1.22.1
across 25 recurrent steps. The official 70,993,538-byte NVIDIA QuartzNet15x5
1.0.0rc1 archive reproduced digest
`1b9b7b87a9277e6fef164d8f99d1226f0511af154423bbf919b920421ac9602f`
and converted tensor fingerprint
`47c098414f58e8380868692db82cf0e4cde3b2777be1cdfd557cb7c5865ef37e`.

The TEN-VAD and QuartzNet focused tests passed independently, and one combined
command passed all five asset/oracle tests with 64 deselections. Broader native
architecture, registry, task, optimization, release, packaging, and
distribution-compliance coverage passed 169 tests and 1,029 subtests with all
five paths active. No runtime, test, workflow, generated page, or generated
notebook source changed in this evidence-only slice.

The release-policy selection, strict eleven-language build, ten-route DOM
validator, and isolated wheel, source-distribution, and editable installs also
passed. The unchanged rendered shell retains its preceding visual evidence;
this checkpoint-only slice makes no new visual-parity claim.

This closes the five pinned opt-in release-asset gates for the exact current
macOS worktree. The two WeNet paths remain inaccessible and unpassed; three
Triton and two CUDA-extension paths remain hardware-limited. Exact-current
Linux, Windows, remote default-runtime, tagged-workflow, publisher, and
publication gates remain open. No protected action was taken, and the untracked
`uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current WeNet release assets

This release slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; the five revision-pinned
reference fingerprints remain unchanged. It changes no Transformers parity
mapping or shell claim.

The original WeNet UCloud routes remain unavailable, and the current download
portal also fails its documented path. The public `openspeech/wenet-models`
mirror at immutable revision
`90acd57d17169a15d5ceab462c6e7db3bd003921` supplied the exact
503,845,602-byte GigaSpeech U2++ archive with the already-audited SHA-256
`061ccfa51d64ebe7ea091a5a13ae31e37d9c36f4eface5c7bafc80bd4a06b26e`.
Independent verification proved byte identity before either test ran.

The restricted checkpoint conversion and tokenizer behavior passed two focused
tests. Package CI and the tagged build now pin and hash the mirror before
running both isolated gates. Provenance retains the original WeNet source,
failed upstream endpoints, immutable mirror identity, undeclared checkpoint
license, and explicit pickle trust boundary; the generated page continues to
lead with a converted local artifact instead of presenting the mirror as a
native model repository.

The broader activated release selection passed 224 tests and 2,621 subtests,
and the selected pre-commit sequence passed. The post-evidence documentation,
release, packaging, generator, workflow-YAML, strict multilingual site, DOM,
wheel, source-distribution, and editable-install checks also passed. The
unchanged rendered shell retains its preceding visual evidence; this asset
slice makes no new parity claim.

This closes both WeNet asset gates on the exact current macOS worktree. Three
Triton and two CUDA-extension gates remain hardware-limited; exact-current
Linux, Windows, remote default-runtime, tagged-workflow, publisher, and
publication gates remain open. No protected action was taken, and the
untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current full-dependency runtime refresh

This release-only slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; all five pinned reference
fingerprints remain unchanged. It changes no parity mapping, page structure,
interaction contract, or visual claim.

A fresh macOS 26.5.2 arm64 CPython 3.12.12 environment installed all test,
training, and documentation extras directly from the current worktree. Its 135
packages passed dependency validation. With the complete runtime activated,
the focused import contract passed five tests and 138 subtests, and the full
suite passed 2,585 tests and 4,795 subtests with 12 explicit skips and 35
warnings in 181.72 seconds.

The post-evidence documentation/release selection, generators, release
alignment, selected lint, strict multilingual site, ten-route DOM validator,
and isolated wheel, source-distribution, and editable installs also passed. The
unchanged rendered shell retains its preceding visual evidence; this runtime
slice makes no new parity claim.

The seven opt-in asset skips retain separate exact-current passing evidence;
the three Triton and two CUDA-extension paths remain hardware-limited and
unpassed. This refresh proves only local Python 3.12. Python 3.10/3.11 must be
rerun after the WeNet source/test slice, and exact-current Linux, Windows,
remote default-runtime, tagged-workflow, publisher, and publication gates
remain open. No protected action was taken, and the untracked `uv.lock`
remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current final supported-Python refresh

This release-only slice uses Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd`; all five reference fingerprints
remain unchanged. It changes no parity mapping, component, interaction, or
visual claim.

Fresh macOS 26.5.2 arm64 Python 3.10.19 and 3.11.15 environments installed the
exact current worktree through the CPU PyTorch resolver, passed dependency
validation, resolved all 261 root exports without importing PyTorch, and passed
15 focused changed-contract tests plus 76 subtests each. Their complete suites
each passed 2,582 tests and 4,657 subtests with 15 explicit skips and 35
warnings: Python 3.10 in 226.26 seconds and Python 3.11 in 212.37 seconds.
The final public-API, documentation, release, and packaging selection also
passed 92 tests and 2,078 subtests on both interpreters, and every generated
inventory remained current.
Selected lint, the strict eleven-language build, the ten-route DOM validator,
and isolated wheel, source-distribution, and editable installs passed after the
evidence update. The rendered shell is unchanged, so this slice makes no new
visual-parity claim.

The three default-runtime and seven asset skips retain separate exact-current
passing evidence; three Triton and two CUDA-extension paths remain hardware-
limited and unpassed. Together with the exact-current Python 3.12 result, this
closes the local supported-Python gate after the WeNet slice. Exact-current
remote Linux, macOS, Windows, tagged-workflow, publisher, and publication gates
remain open. No protected action was taken, and the untracked `uv.lock`
remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Exact-current repository-wide lint refresh

This release-only slice retains Transformers `main` commit
`d09f53a801f45ad73ec3510e17972024234bc0fd` and changes no parity mapping,
component, interaction, or visual claim. Every configured pre-commit hook
passed in one invocation over all 4,675 tracked and candidate untracked files
except the protected `uv.lock`, and before/after SHA-256 manifests were
identical. The registry, public-API, documentation, optimization, scaffold,
release, and packaging selection then passed 157 tests and 2,797 subtests, and
all generated inventories remained current.
The strict eleven-language build, ten-route DOM validator, and isolated wheel,
source-distribution, and editable-install probes also passed. The rendered
shell is unchanged, so this release-only slice makes no new visual-parity
claim.

This closes the exact-current local repository-wide formatting-and-lint gate.
Exact-current remote lint, Linux, macOS, Windows, tagged-workflow, publisher,
publication, and five hardware gates remain open. No protected action was
taken, and the untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Pull-request release-candidate parity evidence

The final comparison refresh retrieved Transformers `main` commit
`ff2421c67f35cc83a0fbabbc2633c96734685918` on 2026-08-04. VoiceHub
implementation commit `aead0611b9eafa0e20d32900568c063073976741` is the
reviewed head of [pull request 73](https://github.com/kadirnar/voicehub/pull/73).
It changes no approved parity mapping and preserves the modern VoiceHub color
palette as the only default visual deviation.

Local documentation evidence passed 64 source tests and 1,480 subtests plus a
complete Playwright 1.62.0/Axe 4.12.1 matrix of 60 rendered and screenshot
cases, 60 accessibility cases, 342 keyboard cases, and 4,591 focus steps. The
content-tab regression now validates native radio-group ArrowRight activation
from a clean route state instead of programmatically focusing an unchecked
radio while Material's selected-tab animation is still active.

The exact-head [documentation run](https://github.com/kadirnar/voicehub/actions/runs/30875378385)
passed the strict eleven-language build, ten-route DOM inventory, all eight
ordered navigation roots, the reviewed Linux screenshot signatures, 60
accessibility cases, 342 keyboard cases, and 4,587 focus steps. Exact-head
[package](https://github.com/kadirnar/voicehub/actions/runs/30875378415) and
[cross-platform CI](https://github.com/kadirnar/voicehub/actions/runs/30875378456)
also passed wheel/source-distribution/editable validation, Linux/macOS/Windows,
Python 3.10/3.11/3.12, full-runtime, training, lint, and runtime-smoke gates.
The generated inventory remains 68 model pages, 59 model notebooks, 261 public
exports, six optimizations over 408 model/pass pairs, ten representative page
pairs, and five benchmark records.

Three Triton and two compiled CUDA-extension paths remain hardware-limited and
unpassed. Pull-request Pages deployment was skipped and is not counted. No tag,
merge, release, publisher change, or PyPI publication was performed. The
protected untracked `uv.lock` remained unchanged at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## Current neutral documentation-theme evidence

This documentation-shell slice refreshed the official Transformers reference
on 2026-08-05 at `main` commit
`838763bf4372a5d0e5643fbd76f88294fb66277f`. The retrieved `_toctree.yml`
digest was
`9081408c4dc6b97cdcfb940f49868cddd51bd565aa56400f88d12a111dd485ea`.
The rendered Home reference used a 65-pixel global row, 270-pixel left and
right rails, an 804-pixel desktop article at x = 318, white and near-black
foundations, thin neutral dividers, compact group labels, and a high-contrast
active navigation pill.

VoiceHub now uses the same restrained visual grammar while retaining its own
logo and focused rose, violet, green, and teal semantic accents. The previous
rainbow global header, page-wide radial glows, gradient tabs, gradient titles,
gradient primary buttons, elevated tables, and elevated process cards were
removed. Light mode now uses `#ffffff` with `#111827` text; dark mode uses the
reference's `#0b0f19` foundation with `#f3f4f6` text. Header, documentation
rails, search, selectors, footer, tables, cards, and process steps use solid
surfaces and thin borders. Primary navigation uses a white-on-neutral active
pill, while every right-table-of-contents instance retains transparent,
accent-only tracking. Desktop and tablet root groups use compact small-cap
typography without changing their accessible text or existing geometry.

The first screenshot update exposed insufficient light-footer contrast and is
not counted. Later full interaction probes exposed a tablet active-label
inheritance error, a compact mobile header contrast error, long-label wrapping,
and a model-page sticky-TOC contrast error; those failed runs are also excluded.
The corrected local macOS matrix passed 60 rendered and Axe cases, 60 reviewed
screenshot cases, 342 keyboard cases, 32 root-branch activations, 24 nested-
branch activations, 40 TOC activations, and 4,604 native focus steps across all
ten representative routes, both palettes, and desktop, tablet, and mobile
viewports. The strict multilingual build and rendered DOM inventory passed
after this evidence edit. The refreshed Linux screenshot signature candidate
remains subject to exact pull-request CI validation and is not counted as a
Linux pass here.

This slice changes no documentation route, public Python API, model registry,
checkpoint, optimization, packaging contract, or publication state. VoiceHub's
brand mark remains multicolor, but decorative brand gradients no longer define
the documentation shell. The protected untracked `uv.lock` remained unchanged
at SHA-256
`48f7d98d6eab756580348b081e8fc891d3a5dd2847433e41766b3b45854a70b1`.

## 2026-08-09 model-guide reference refresh

The model documentation comparison was refreshed against Transformers `main`
commit `e8ea728a3eeeb903e77c7d1bd29267c80a1be71f` on 2026-08-09. The retrieved
SHA-256 digests are:

| Upstream source | SHA-256 |
| --- | --- |
| `docs/source/en/_toctree.yml` | `9081408c4dc6b97cdcfb940f49868cddd51bd565aa56400f88d12a111dd485ea` |
| `docs/source/en/model_doc/auto.md` | `557f5836c0722fef6a484c46805dfab0eb69a387b028a914b132350edf09f167` |
| `docs/source/en/model_doc/speecht5.md` | `71bba8a2921cf637383fb8d6f2fd66df9cd95deb59118b9f49e1362485c27eb5` |

VoiceHub keeps one generated page per registry model under `Base classes → Models`.
The pages retain the current Auto-class and model-detail structure while using
short speech-specific contracts, runnable code, source links, and
checkpoint-evidence boundaries. Installation examples now use the repository
source instead of assuming a package-index release.

## 2026-08-13 compact model-explorer evidence

The Auto Classes comparison was refreshed against Transformers `main` commit
`918dbf131d0df5b46e3f6e1d96174d62aa4d16d6`. The retrieved
`docs/source/en/model_doc/auto.md` source has SHA-256
`c3a46d6a8271f239ab5ec1cf4df4368cc1924df88fb8b814fe3139eb5492e84b`,
and `_toctree.yml` has SHA-256
`79da32ab73d6445f12e5e14ec5d76ed8e67c4b0724b2856095bbe86aa5eb0af9`.
The official route returned HTTP 200. VoiceHub intentionally retains a
speech-specific comparator because its catalog spans TTS, ASR, and VAD rather
than upstream Auto-class mappings alone.

The generated explorer covers all 68 registered models with search, seven
select facets, 11 capability chips, two resource chips, removable active
filters, reset and empty states, and URL-restorable state. Its parameter facet
covers seven exact bands. Sorting covers model type, parameter count in both
directions, supported-language count in both directions, name, and training
support; unavailable parameter totals remain last in either parameter order.
The compact responsive grid keeps at least three cards in the initial desktop
viewport, two on tablet, and one on mobile, with no horizontal overflow.

The exact-current source suite passed 74 tests and 1,901 subtests. Generation
checks passed for all 68 model pages, the strict build passed for all eleven
locales, and the rendered DOM inventory passed 11 routes. The final-layout
macOS matrix passed all 60 viewport/palette screenshots and Axe checks before
the last metadata-only corrections. The exact-current strengthened mobile
default-palette shard then passed all ten representative routes on both macOS
and a genuine Ubuntu browser run, including the model explorer's complete
filter/sort exercise, screenshots, Axe checks, and complete focus traversal.
No publication or deployment action was performed.
