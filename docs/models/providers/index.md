---
description: Search and filter every registered VoiceHub TTS, ASR, and VAD model.
---

# Model list

<div class="vh-model-explorer" data-vh-model-explorer data-model-count="68">
  <section class="vh-model-explorer__hero" aria-labelledby="vh-model-explorer-title">
    <div class="vh-model-explorer__hero-copy">
      <p class="vh-model-explorer__eyebrow">Model discovery</p>
      <h2 id="vh-model-explorer-title">Find the right speech model</h2>
      <p>Search the complete VoiceHub catalog by language, task, training path,
      architecture, checkpoint source, license, and production capability.</p>
    </div>
    <dl class="vh-model-explorer__stats" aria-label="Catalog summary">
      <div><dt>68</dt><dd>models</dd></div>
      <div><dt>779</dt><dd>indexed codes</dd></div>
      <div><dt>34</dt><dd>TTS</dd></div>
      <div><dt>23</dt><dd>ASR</dd></div>
    </dl>
  </section>

  <form class="vh-model-filters" data-vh-model-filters>
    <div class="vh-model-filters__search">
      <label for="vh-model-query">Search models</label>
      <div class="vh-model-search-field">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21 21-4.35-4.35m2.35-5.4A7.75 7.75 0 1 1 3.5 11.25a7.75 7.75 0 0 1 15.5 0Z"/></svg>
        <input id="vh-model-query" name="query" type="search" autocomplete="off"
          placeholder="Search model, architecture, feature, or Hub ID…"
          aria-describedby="vh-model-search-hint" data-vh-model-query>
        <kbd>/</kbd>
      </div>
      <span id="vh-model-search-hint" class="vh-model-filters__hint">Try “Turkish voice cloning” or “Whisper timestamps”.</span>
    </div>

    <div class="vh-model-filters__controls">
      <div class="vh-model-filters__quick" aria-label="Quick filters">
        <div class="vh-model-filter-field">
        <label for="vh-model-language">Language</label>
        <select id="vh-model-language" name="language" data-vh-model-select>
          <option value="">Any language</option>
<option value="not-text-conditioned">Language-neutral (VAD)</option>
        </select>
      </div>
        <div class="vh-model-filter-field">
        <label for="vh-model-task">Task</label>
        <select id="vh-model-task" name="task" data-vh-model-select>
          <option value="">Any task</option>
<option value="text-to-speech">Text to speech (34)</option>
<option value="automatic-speech-recognition">Automatic speech recognition (23)</option>
<option value="voice-activity-detection">Voice activity detection (11)</option>
        </select>
      </div>
        <div class="vh-model-filter-field">
        <label for="vh-model-training">Training</label>
        <select id="vh-model-training" name="training" data-vh-model-select>
          <option value="">Any training path</option>
<option value="native">Native training (46)</option>
<option value="preprocessed">Prepared-data training (17)</option>
<option value="custom">Custom training (3)</option>
<option value="inference-only">Inference only (2)</option>
        </select>
      </div>
      </div>

      <details class="vh-model-filters__advanced">
        <summary>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M7 12h10m-7 6h4"/></svg>
          <span>More filters</span>
          <span class="vh-model-filters__advanced-count"
            data-vh-model-advanced-count hidden>0</span>
        </summary>
        <div class="vh-model-filters__advanced-body">
          <div class="vh-model-filters__secondary">
            <div class="vh-model-filter-field">
        <label for="vh-model-checkpoint">Checkpoint</label>
        <select id="vh-model-checkpoint" name="checkpoint" data-vh-model-select>
          <option value="">Any checkpoint source</option>
<option value="huggingface">Hugging Face (59)</option>
<option value="external-archive">External archive (1)</option>
<option value="local">Local or caller-provided (8)</option>
        </select>
      </div>
            <div class="vh-model-filter-field">
        <label for="vh-model-license">License</label>
        <select id="vh-model-license" name="license" data-vh-model-select>
          <option value="">Any license status</option>
<option value="commercial">Commercial use declared (3)</option>
<option value="noncommercial">Non-commercial (5)</option>
<option value="review">Review required (6)</option>
<option value="checkpoint-specific">Checkpoint-specific (54)</option>
        </select>
      </div>
            <div class="vh-model-filter-field">
        <label for="vh-model-architecture">Architecture</label>
        <select id="vh-model-architecture" name="architecture" data-vh-model-select>
          <option value="">Any architecture</option>
<option value="bark">bark (1)</option>
<option value="causal-lm">causal-lm (1)</option>
<option value="chatterbox">chatterbox (1)</option>
<option value="cohere-asr">cohere-asr (1)</option>
<option value="conversationtts">conversationtts (1)</option>
<option value="cosyvoice-native">cosyvoice-native (1)</option>
<option value="csm">csm (1)</option>
<option value="dia">dia (1)</option>
<option value="echo-dit">echo-dit (1)</option>
<option value="energy-vad">energy-vad (1)</option>
<option value="espnet-librispeech-transformer-e18">espnet-librispeech-transformer-e18 (1)</option>
<option value="f5tts">f5tts (1)</option>
<option value="fish-s2">fish-s2 (1)</option>
<option value="fsmn-vad">fsmn-vad (1)</option>
<option value="gptsovits">gptsovits (1)</option>
<option value="granite-speech">granite-speech (1)</option>
<option value="higgs_audio_v2">higgs_audio_v2 (1)</option>
<option value="hubert">hubert (1)</option>
<option value="inflecttts">inflecttts (1)</option>
<option value="irodoritts-rf-dit">irodoritts-rf-dit (1)</option>
<option value="kokoro">kokoro (1)</option>
<option value="lasr-ctc">lasr-ctc (1)</option>
<option value="llasa">llasa (1)</option>
<option value="marblenet-vad">marblenet-vad (1)</option>
<option value="melotts">melotts (1)</option>
<option value="moonshine">moonshine (1)</option>
<option value="moss-tts">moss-tts (1)</option>
<option value="native-asr-dispatch">native-asr-dispatch (1)</option>
<option value="native-vad-dispatch">native-vad-dispatch (1)</option>
<option value="nemo-asr">nemo-asr (1)</option>
<option value="nemotron-3.5-rnnt">nemotron-3.5-rnnt (1)</option>
<option value="neutts">neutts (1)</option>
<option value="omnivoice">omnivoice (1)</option>
<option value="openvoice-v2-converter">openvoice-v2-converter (1)</option>
<option value="outetts">outetts (1)</option>
<option value="parakeet-tdt">parakeet-tdt (1)</option>
<option value="parlertts">parlertts (1)</option>
<option value="pyannet">pyannet (3)</option>
<option value="qwen3-asr">qwen3-asr (1)</option>
<option value="qwen3-tts">qwen3-tts (1)</option>
<option value="seamless-m4t-v2-s2t">seamless-m4t-v2-s2t (1)</option>
<option value="sensevoice-small">sensevoice-small (1)</option>
<option value="silero-vad">silero-vad (1)</option>
<option value="speechbrain-crdnn-asr">speechbrain-crdnn-asr (1)</option>
<option value="speechbrain-crdnn-vad">speechbrain-crdnn-vad (1)</option>
<option value="speecht5">speecht5 (1)</option>
<option value="styletts2">styletts2 (1)</option>
<option value="supertonic">supertonic (1)</option>
<option value="vibevoice-asr">vibevoice-asr (1)</option>
<option value="vibevoice-tts">vibevoice-tts (1)</option>
<option value="vits">vits (1)</option>
<option value="voxcpm2">voxcpm2 (1)</option>
<option value="vui">vui (1)</option>
<option value="wav2vec2">wav2vec2 (2)</option>
<option value="wavlm">wavlm (1)</option>
<option value="webrtc-vad">webrtc-vad (1)</option>
<option value="wenet-asr">wenet-asr (1)</option>
<option value="whisper">whisper (5)</option>
<option value="xtts2">xtts2 (1)</option>
<option value="zonos">zonos (1)</option>
<option value="zonos2">zonos2 (1)</option>
        </select>
      </div>
          </div>
          <fieldset>
            <legend>Capabilities <span>Models must include every selected capability</span></legend>
            <div class="vh-model-filter-chips"><label class="vh-model-filter-chip" for="vh-model-feature-voice-cloning">
            <input id="vh-model-feature-voice-cloning" name="feature"
              type="checkbox" value="voice-cloning"
              data-vh-model-checkbox data-filter-label="Voice cloning">
            <span>Voice cloning <small>22</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-streaming">
            <input id="vh-model-feature-streaming" name="feature"
              type="checkbox" value="streaming"
              data-vh-model-checkbox data-filter-label="Streaming">
            <span>Streaming <small>4</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-timestamps">
            <input id="vh-model-feature-timestamps" name="feature"
              type="checkbox" value="timestamps"
              data-vh-model-checkbox data-filter-label="Timestamps">
            <span>Timestamps <small>14</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-translation">
            <input id="vh-model-feature-translation" name="feature"
              type="checkbox" value="translation"
              data-vh-model-checkbox data-filter-label="Translation">
            <span>Translation <small>4</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-voice-design">
            <input id="vh-model-feature-voice-design" name="feature"
              type="checkbox" value="voice-design"
              data-vh-model-checkbox data-filter-label="Voice design">
            <span>Voice design <small>4</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-expressive-speech">
            <input id="vh-model-feature-expressive-speech" name="feature"
              type="checkbox" value="expressive-speech"
              data-vh-model-checkbox data-filter-label="Expressive speech">
            <span>Expressive speech <small>3</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-long-form">
            <input id="vh-model-feature-long-form" name="feature"
              type="checkbox" value="long-form"
              data-vh-model-checkbox data-filter-label="Long-form audio">
            <span>Long-form audio <small>4</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-speaker-attribution">
            <input id="vh-model-feature-speaker-attribution" name="feature"
              type="checkbox" value="speaker-attribution"
              data-vh-model-checkbox data-filter-label="Speaker attribution">
            <span>Speaker attribution <small>2</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-language-identification">
            <input id="vh-model-feature-language-identification" name="feature"
              type="checkbox" value="language-identification"
              data-vh-model-checkbox data-filter-label="Language identification">
            <span>Language identification <small>3</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-hotwords">
            <input id="vh-model-feature-hotwords" name="feature"
              type="checkbox" value="hotwords"
              data-vh-model-checkbox data-filter-label="Hotwords">
            <span>Hotwords <small>3</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-feature-frame-scores">
            <input id="vh-model-feature-frame-scores" name="feature"
              type="checkbox" value="frame-scores"
              data-vh-model-checkbox data-filter-label="Frame scores">
            <span>Frame scores <small>8</small></span>
          </label></div>
          </fieldset>
          <fieldset>
            <legend>Resources</legend>
            <div class="vh-model-filter-chips"><label class="vh-model-filter-chip" for="vh-model-resource-notebook">
            <input id="vh-model-resource-notebook" name="resource"
              type="checkbox" value="notebook"
              data-vh-model-checkbox data-filter-label="Colab notebook">
            <span>Colab notebook <small>59</small></span>
          </label>
<label class="vh-model-filter-chip" for="vh-model-resource-huggingface">
            <input id="vh-model-resource-huggingface" name="resource"
              type="checkbox" value="huggingface"
              data-vh-model-checkbox data-filter-label="Hugging Face page">
            <span>Hugging Face page <small>63</small></span>
          </label></div>
          </fieldset>
        </div>
      </details>
    </div>
  </form>

  <div class="vh-model-results__toolbar">
    <div>
      <p class="vh-model-results__count" role="status" aria-live="polite">
        <strong data-vh-model-result-count>68</strong>
        <span data-vh-model-result-label>models</span>
      </p>
      <div class="vh-model-active-filters" data-vh-model-active-filters hidden></div>
    </div>
    <div class="vh-model-results__actions">
      <button type="button" class="vh-model-clear" data-vh-model-clear hidden>Clear filters</button>
      <label for="vh-model-sort">Sort</label>
      <select id="vh-model-sort" name="sort" data-vh-model-sort>
        <option value="name">Name A–Z</option>
        <option value="languages">Language coverage</option>
        <option value="task">Task</option>
        <option value="training">Training readiness</option>
      </select>
    </div>
  </div>

  <div class="vh-model-results" data-vh-model-results><article class="vh-model-card"
      data-vh-model-card
      data-name="auditokvad"
      data-model-type="vad_auditok"
      data-task="voice-activity-detection"
      data-training="inference-only"
      data-training-rank="3"
      data-checkpoint="local"
      data-license="checkpoint-specific"
      data-architecture="energy-vad"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection energy-based adaptive-threshold algorithmic voicehub-native"
      data-resources=""
      data-search="AuditokVAD vad_auditok Voice activity detection VAD energy-vad auditok-energy-vad  Inference only Local or caller-provided Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Inference only</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_auditok/">AuditokVAD</a></h2>
        <code>vad_auditok</code>
      </div>
      <p class="vh-model-card__summary">Runs Auditok&#x27;s weightless energy detector with conservative speech/silence durations.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>energy-vad</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_auditok/">View model <span aria-hidden="true">→</span></a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="bark"
      data-model-type="bark"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="bark"
      data-language-kind="enumerated"
      data-language-count="13"
      data-languages="de en es fr hi it ja ko pl pt ru tr zh"
      data-capabilities="text-to-speech expressive-speech voice-prompt safetensors fine-tuning voicehub-native native-runtime preencoded-stage-training restricted-pickle-conversion"
      data-resources="notebook huggingface"
      data-search="Bark bark Text to speech TTS bark suno/bark-small suno/bark-small Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="bark/">Bark</a></h2>
        <code>bark</code>
      </div>
      <p class="vh-model-card__summary">Selects a Bark history prompt and bounds semantic token sampling.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>bark</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>de</code><code>en</code><code>es</code><code>fr</code><code>hi</code><span class="vh-model-card__more-languages">+8</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Expressive speech</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="bark/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/suno/bark-small">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/bark.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="chatterbox"
      data-model-type="chatterbox"
      data-task="text-to-speech"
      data-training="custom"
      data-training-rank="2"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="chatterbox"
      data-language-kind="enumerated"
      data-language-count="23"
      data-languages="ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh"
      data-capabilities="text-to-speech voice-cloning fine-tuning safetensors voicehub-native native-runtime raw-audio-fine-tuning"
      data-resources="notebook huggingface"
      data-search="Chatterbox chatterbox Text to speech TTS chatterbox ResembleAI/chatterbox ResembleAI/chatterbox Custom training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Custom training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="chatterbox/">Chatterbox</a></h2>
        <code>chatterbox</code>
      </div>
      <p class="vh-model-card__summary">Demonstrates Chatterbox voice prompting through VoiceHub&#x27;s normalized reference-audio field.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>chatterbox</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>ar</code><code>da</code><code>de</code><code>el</code><code>en</code><span class="vh-model-card__more-languages">+18</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="chatterbox/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/ResembleAI/chatterbox">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/chatterbox.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="cohere"
      data-model-type="asr_cohere"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="cohere-asr"
      data-language-kind="enumerated"
      data-language-count="14"
      data-languages="ar de el en es fr it ja ko nl pl pt vi zh"
      data-capabilities="automatic-speech-recognition multilingual long-form punctuation gated-checkpoint safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Cohere asr_cohere Automatic speech recognition ASR cohere-asr CohereLabs/cohere-transcribe-03-2026 CohereLabs/cohere-transcribe-03-2026 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_cohere/">Cohere</a></h2>
        <code>asr_cohere</code>
      </div>
      <p class="vh-model-card__summary">Requests Cohere Transcribe punctuation with an explicit language and bounded decoding.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>cohere-asr</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>ar</code><code>de</code><code>el</code><code>en</code><code>es</code><span class="vh-model-card__more-languages">+9</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Long-form audio</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_cohere/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/CohereLabs/cohere-transcribe-03-2026">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_cohere.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="conversationtts"
      data-model-type="conversationtts"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="noncommercial"
      data-architecture="conversationtts"
      data-language-kind="enumerated"
      data-language-count="3"
      data-languages="en zh yue"
      data-capabilities="text-to-speech voice-cloning conversation multilingual fine-tuning safetensors voicehub-native native-runtime raw-audio-fine-tuning preencoded-code-fine-tuning noncommercial"
      data-resources="notebook huggingface"
      data-search="ConversationTTS conversationtts Text to speech TTS conversationtts AudioFoundation/SpeechFoundation AudioFoundation/SpeechFoundation Native training Hugging Face Non-commercial">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="conversationtts/">ConversationTTS</a></h2>
        <code>conversationtts</code>
      </div>
      <p class="vh-model-card__summary">Assigns an explicit conversation speaker and caps the generated audio duration.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>conversationtts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>yue</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Non-commercial</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="conversationtts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/AudioFoundation/SpeechFoundation">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/conversationtts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="cosyvoice"
      data-model-type="cosyvoice"
      data-task="text-to-speech"
      data-training="custom"
      data-training-rank="2"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="cosyvoice-native"
      data-language-kind="enumerated"
      data-language-count="9"
      data-languages="zh en ja ko de es fr it ru"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning flow-matching adversarial-vocoder-training safetensors voicehub-native native-runtime precomputed-speaker-embedding preencoded-speech-token-fine-tuning"
      data-resources="notebook huggingface"
      data-search="CosyVoice cosyvoice Text to speech TTS cosyvoice-native FunAudioLLM/Fun-CosyVoice3-0.5B-2512 FunAudioLLM/Fun-CosyVoice3-0.5B-2512 Custom training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Custom training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="cosyvoice/">CosyVoice</a></h2>
        <code>cosyvoice</code>
      </div>
      <p class="vh-model-card__summary">Loads the required 192-value speaker embedding from a reviewable JSON file.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>cosyvoice-native</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>en</code><code>ja</code><code>ko</code><code>de</code><span class="vh-model-card__more-languages">+4</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="cosyvoice/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/cosyvoice.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="csm"
      data-model-type="csm"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="csm"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech voice-cloning conversation safetensors fine-tuning raw-audio-training preencoded-code-training voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="CSM csm Text to speech TTS csm sesame/csm-1b sesame/csm-1b Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="csm/">CSM</a></h2>
        <code>csm</code>
      </div>
      <p class="vh-model-card__summary">Builds CSM speaker context from a stable speaker index and paired reference recording.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>csm</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="csm/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/sesame/csm-1b">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/csm.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="dia"
      data-model-type="dia"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="dia"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech dialogue safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Dia dia Text to speech TTS dia nari-labs/Dia-1.6B-0626 nari-labs/Dia-1.6B-0626 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="dia/">Dia</a></h2>
        <code>dia</code>
      </div>
      <p class="vh-model-card__summary">Uses Dia speaker tags in the text instead of an unrelated generic single-speaker prompt.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>dia</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="dia/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/nari-labs/Dia-1.6B-0626">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/dia.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="echotts"
      data-model-type="echo"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="echo-dit"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech voice-cloning fine-tuning flow-matching safetensors voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="EchoTTS echo Text to speech TTS echo-dit jordand/echo-tts-base jordand/echo-tts-base Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="echo/">EchoTTS</a></h2>
        <code>echo</code>
      </div>
      <p class="vh-model-card__summary">Exposes Echo&#x27;s flow-matching step count and separate text/speaker guidance scales.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>echo-dit</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="echo/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/jordand/echo-tts-base">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/echo.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="espnetasr"
      data-model-type="asr_espnet"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="espnet-librispeech-transformer-e18"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition english safetensors fine-tuning voicehub-native native-runtime raw-audio-fine-tuning hybrid-ctc-attention"
      data-resources="notebook huggingface"
      data-search="ESPnetASR asr_espnet Automatic speech recognition ASR espnet-librispeech-transformer-e18 espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_espnet/">ESPnetASR</a></h2>
        <code>asr_espnet</code>
      </div>
      <p class="vh-model-card__summary">Uses the audited ESPnet LibriSpeech transformer with an explicit beam size.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>espnet-librispeech-transformer-e18</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_espnet/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_espnet.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="f5tts"
      data-model-type="f5tts"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="local"
      data-license="checkpoint-specific"
      data-architecture="f5tts"
      data-language-kind="enumerated"
      data-language-count="2"
      data-languages="en zh"
      data-capabilities="text-to-speech voice-cloning fine-tuning flow-matching safetensors voicehub-native native-runtime"
      data-resources="huggingface"
      data-search="F5TTS f5tts Text to speech TTS f5tts F5TTS_v1_Base SWivid/F5-TTS Prepared-data training Local or caller-provided Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="f5tts/">F5TTS</a></h2>
        <code>f5tts</code>
      </div>
      <p class="vh-model-card__summary">Supplies F5-TTS with the mandatory reference waveform and matching transcript.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>f5tts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="f5tts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/SWivid/F5-TTS">Hugging Face</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="fasterwhisper"
      data-model-type="asr_faster_whisper"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="whisper"
      data-language-kind="enumerated"
      data-language-count="99"
      data-languages="en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su"
      data-capabilities="automatic-speech-recognition multilingual translation timestamps safetensors fine-tuning voicehub-native"
      data-resources="notebook huggingface"
      data-search="FasterWhisper asr_faster_whisper Automatic speech recognition ASR whisper openai/whisper-small openai/whisper-small Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_faster_whisper/">FasterWhisper</a></h2>
        <code>asr_faster_whisper</code>
      </div>
      <p class="vh-model-card__summary">Uses the faster-whisper backend with language selection, word timestamps, and a bounded beam.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>whisper</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>de</code><code>es</code><code>ru</code><span class="vh-model-card__more-languages">+94</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Translation</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_faster_whisper/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/openai/whisper-small">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_faster_whisper.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="fishtts"
      data-model-type="fishtts"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="noncommercial"
      data-architecture="fish-s2"
      data-language-kind="enumerated"
      data-language-count="83"
      data-languages="zh en ja ko es pt ar ru fr de sv it tr no nl cy eu ca da gl ta hu fi pl et hi la ur th vi jw bn yo sl cs sw nn he ms uk id kk bg lv my tl sk ne fa af el bo hr ro sn mi yi am be km is az sd br sq ps mn ht ml sr sa te ka bs pa lt kn si hy mr as gu fo"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime preprocessed-training noncommercial"
      data-resources="notebook huggingface"
      data-search="FishTTS fishtts Text to speech TTS fish-s2 fishaudio/s2-pro fishaudio/s2-pro Prepared-data training Hugging Face Non-commercial">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="fishtts/">FishTTS</a></h2>
        <code>fishtts</code>
      </div>
      <p class="vh-model-card__summary">Pairs Fish S2 reference audio and text while keeping semantic sampling bounded.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>fish-s2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>en</code><code>ja</code><code>ko</code><code>es</code><span class="vh-model-card__more-languages">+78</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Non-commercial</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="fishtts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/fishaudio/s2-pro">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/fishtts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="funasr"
      data-model-type="asr_funasr"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="sensevoice-small"
      data-language-kind="enumerated"
      data-language-count="5"
      data-languages="zh en ja ko yue"
      data-capabilities="automatic-speech-recognition multilingual timestamps language-identification emotion-recognition audio-events fine-tuning safetensors voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="FunASR asr_funasr Automatic speech recognition ASR sensevoice-small FunAudioLLM/SenseVoiceSmall FunAudioLLM/SenseVoiceSmall Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_funasr/">FunASR</a></h2>
        <code>asr_funasr</code>
      </div>
      <p class="vh-model-card__summary">Runs SenseVoiceSmall&#x27;s native SANM-CTC graph with language detection and word timestamps.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>sensevoice-small</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>en</code><code>ja</code><code>ko</code><code>yue</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Language identification</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_funasr/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_funasr.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="funasrvad"
      data-model-type="vad_funasr"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="fsmn-vad"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection voicehub-native safetensors trusted-checkpoint-conversion frame-scores streaming fine-tuning modelscope-compatible"
      data-resources="notebook huggingface"
      data-search="FunASRVAD vad_funasr Voice activity detection VAD fsmn-vad funasr/fsmn-vad funasr/fsmn-vad Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_funasr/">FunASRVAD</a></h2>
        <code>vad_funasr</code>
      </div>
      <p class="vh-model-card__summary">Uses the FSMN endpoint model with speech padding and maximum-segment limits.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>fsmn-vad</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Streaming</span><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_funasr/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/funasr/fsmn-vad">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_funasr.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="gptsovits"
      data-model-type="gptsovits"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="gptsovits"
      data-language-kind="enumerated"
      data-language-count="5"
      data-languages="zh en ja ko yue"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime preprocessed-training gpt-sovits-v1 gpt-sovits-v2 gpt-sovits-v2-pro gpt-sovits-v2-pro-plus prepared-pro-speaker-conditioning variant-aware-safetensors-export"
      data-resources="notebook huggingface"
      data-search="GPTSoVITS gptsovits Text to speech TTS gptsovits lj1995/GPT-SoVITS lj1995/GPT-SoVITS Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="gptsovits/">GPTSoVITS</a></h2>
        <code>gptsovits</code>
      </div>
      <p class="vh-model-card__summary">Defines both target and prompt languages for GPT-SoVITS zero-shot voice prompting.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>gptsovits</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>en</code><code>ja</code><code>ko</code><code>yue</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="gptsovits/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/lj1995/GPT-SoVITS">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/gptsovits.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="granitespeech"
      data-model-type="asr_granite_speech"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="granite-speech"
      data-language-kind="enumerated"
      data-language-count="6"
      data-languages="en fr de es pt ja"
      data-capabilities="automatic-speech-recognition multilingual hotwords translation safetensors fine-tuning lora voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="GraniteSpeech asr_granite_speech Automatic speech recognition ASR granite-speech ibm-granite/granite-speech-4.1-2b ibm-granite/granite-speech-4.1-2b Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_granite_speech/">GraniteSpeech</a></h2>
        <code>asr_granite_speech</code>
      </div>
      <p class="vh-model-card__summary">Uses Granite Speech&#x27;s instruction prompt with deterministic generation.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>granite-speech</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>fr</code><code>de</code><code>es</code><code>pt</code><span class="vh-model-card__more-languages">+1</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Translation</span><span>Hotwords</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_granite_speech/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/ibm-granite/granite-speech-4.1-2b">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_granite_speech.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="higgstts"
      data-model-type="higgstts"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="higgs_audio_v2"
      data-language-kind="enumerated"
      data-language-count="4"
      data-languages="en zh de ko"
      data-capabilities="text-to-speech voice-cloning expressive-speech multilingual fine-tuning safetensors voicehub-native native-runtime raw-audio-fine-tuning preencoded-code-fine-tuning"
      data-resources="notebook huggingface"
      data-search="HiggsTTS higgstts Text to speech TTS higgs_audio_v2 bosonai/higgs-tts-2-3b-base bosonai/higgs-tts-2-3b-base Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="higgstts/">HiggsTTS</a></h2>
        <code>higgstts</code>
      </div>
      <p class="vh-model-card__summary">Provides Higgs Audio with paired reference context and a bounded semantic-token budget.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>higgs_audio_v2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>de</code><code>ko</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span><span>Expressive speech</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="higgstts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/bosonai/higgs-tts-2-3b-base">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/higgstts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="hubert"
      data-model-type="asr_hubert"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="hubert"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition timestamps safetensors fine-tuning voicehub-native"
      data-resources="notebook huggingface"
      data-search="Hubert asr_hubert Automatic speech recognition ASR hubert facebook/hubert-large-ls960-ft facebook/hubert-large-ls960-ft Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_hubert/">Hubert</a></h2>
        <code>asr_hubert</code>
      </div>
      <p class="vh-model-card__summary">Uses the HuBERT CTC fine-tuned head with an explicit English transcription task.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>hubert</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_hubert/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/facebook/hubert-large-ls960-ft">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_hubert.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="inflecttts"
      data-model-type="inflecttts"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="inflecttts"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en-US"
      data-capabilities="text-to-speech fine-tuning safetensors voicehub-native native-runtime preprocessed-training vits-warm-start explicit-phonemes"
      data-resources="notebook huggingface"
      data-search="InflectTTS inflecttts Text to speech TTS inflecttts owensong/Inflect-Micro-v2 owensong/Inflect-Micro-v2 Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="inflecttts/">InflectTTS</a></h2>
        <code>inflecttts</code>
      </div>
      <p class="vh-model-card__summary">Uses Inflect&#x27;s normalized-text frontend with explicit speed and variation controls.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>inflecttts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en-US</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="inflecttts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/owensong/Inflect-Micro-v2">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/inflecttts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="irodoritts"
      data-model-type="irodoritts"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="irodoritts-rf-dit"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="ja"
      data-capabilities="text-to-speech voice-cloning voice-design multilingual fine-tuning flow-matching safetensors voicehub-native native-runtime raw-audio-fine-tuning preencoded-latent-fine-tuning duration-prediction"
      data-resources="notebook huggingface"
      data-search="IrodoriTTS irodoritts Text to speech TTS irodoritts-rf-dit Aratako/Irodori-TTS-500M-v3 Aratako/Irodori-TTS-500M-v3 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="irodoritts/">IrodoriTTS</a></h2>
        <code>irodoritts</code>
      </div>
      <p class="vh-model-card__summary">Exercises Irodori-TTS&#x27;s explicit no-reference path and flow sampler controls.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>irodoritts-rf-dit</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>ja</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span><span>Voice design</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="irodoritts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/Aratako/Irodori-TTS-500M-v3">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/irodoritts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="kokoro"
      data-model-type="kokoro"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="kokoro"
      data-language-kind="enumerated"
      data-language-count="9"
      data-languages="en-US en-GB es fr hi it pt-BR ja zh"
      data-capabilities="text-to-speech multilingual fine-tuning safetensors voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Kokoro kokoro Text to speech TTS kokoro hexgrad/Kokoro-82M hexgrad/Kokoro-82M Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="kokoro/">Kokoro</a></h2>
        <code>kokoro</code>
      </div>
      <p class="vh-model-card__summary">Selects a Kokoro voice ID and explicit speaking speed.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>kokoro</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en-US</code><code>en-GB</code><code>es</code><code>fr</code><code>hi</code><span class="vh-model-card__more-languages">+4</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="kokoro/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/hexgrad/Kokoro-82M">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/kokoro.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="llasa"
      data-model-type="llasa"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="noncommercial"
      data-architecture="llasa"
      data-language-kind="enumerated"
      data-language-count="11"
      data-languages="zh en de fr ja ko nl es it pt pl"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime raw-audio-fine-tuning preencoded-code-fine-tuning"
      data-resources="notebook huggingface"
      data-search="Llasa llasa Text to speech TTS llasa HKUSTAudio/Llasa-1B-Multilingual HKUSTAudio/Llasa-1B-Multilingual Native training Hugging Face Non-commercial">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="llasa/">Llasa</a></h2>
        <code>llasa</code>
      </div>
      <p class="vh-model-card__summary">Pairs LLaSA reference audio with its exact transcript for voice cloning.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>llasa</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>en</code><code>de</code><code>fr</code><code>ja</code><span class="vh-model-card__more-languages">+6</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Non-commercial</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="llasa/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/HKUSTAudio/Llasa-1B-Multilingual">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/llasa.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="medasr"
      data-model-type="asr_medasr"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="review"
      data-architecture="lasr-ctc"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition medical gated-checkpoint safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="MedASR asr_medasr Automatic speech recognition ASR lasr-ctc google/medasr google/medasr Native training Hugging Face Review required">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_medasr/">MedASR</a></h2>
        <code>asr_medasr</code>
      </div>
      <p class="vh-model-card__summary">Selects the audited English MedASR decoding path without pretending it is a clinical decision system.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>lasr-ctc</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Review required</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_medasr/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/google/medasr">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_medasr.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="melotts"
      data-model-type="melotts"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="local"
      data-license="checkpoint-specific"
      data-architecture="melotts"
      data-language-kind="enumerated"
      data-language-count="6"
      data-languages="en fr ja es zh ko"
      data-capabilities="text-to-speech multilingual fine-tuning safetensors voicehub-native native-runtime preprocessed-training explicit-linguistic-features"
      data-resources="huggingface"
      data-search="MeloTTS melotts Text to speech TTS melotts EN myshell-ai/MeloTTS-English Prepared-data training Local or caller-provided Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="melotts/">MeloTTS</a></h2>
        <code>melotts</code>
      </div>
      <p class="vh-model-card__summary">Opts into the pinned legacy MeloTTS release explicitly and selects its English speaker table.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>melotts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>fr</code><code>ja</code><code>es</code><code>zh</code><span class="vh-model-card__more-languages">+1</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="melotts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/myshell-ai/MeloTTS-English">Hugging Face</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="moonshine"
      data-model-type="asr_moonshine"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="moonshine"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition safetensors fine-tuning compact voicehub-native"
      data-resources="notebook huggingface"
      data-search="Moonshine asr_moonshine Automatic speech recognition ASR moonshine UsefulSensors/moonshine-tiny UsefulSensors/moonshine-tiny Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_moonshine/">Moonshine</a></h2>
        <code>asr_moonshine</code>
      </div>
      <p class="vh-model-card__summary">Uses Moonshine&#x27;s short-form speech path with deterministic decoding.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>moonshine</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_moonshine/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/UsefulSensors/moonshine-tiny">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_moonshine.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="mosstts"
      data-model-type="mosstts"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="moss-tts"
      data-language-kind="enumerated"
      data-language-count="31"
      data-languages="zh yue en ar cs da de nl es fr fi el he hi hu ja it ko mk ms ru fa pl pt sv ro sw tl th tr vi"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime delay-variant local-variant local-v1.5-variant realtime-variant raw-audio-fine-tuning preencoded-rvq-fine-tuning native-codec-v1 native-codec-v2 buffered-generation"
      data-resources="notebook huggingface"
      data-search="MossTTS mosstts Text to speech TTS moss-tts OpenMOSS-Team/MOSS-TTS-v1.5 OpenMOSS-Team/MOSS-TTS-v1.5 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="mosstts/">MossTTS</a></h2>
        <code>mosstts</code>
      </div>
      <p class="vh-model-card__summary">Combines MOSS-TTS language, instruction, and quality controls without importing upstream demo code.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>moss-tts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>yue</code><code>en</code><code>ar</code><code>cs</code><span class="vh-model-card__more-languages">+26</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="mosstts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/mosstts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="nemoasr"
      data-model-type="asr_nemo"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="local"
      data-license="review"
      data-architecture="nemo-asr"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition english timestamps safetensors fine-tuning voicehub-native ctc"
      data-resources=""
      data-search="NeMoASR asr_nemo Automatic speech recognition ASR nemo-asr nvidia/nemo/stt_en_quartznet15x5  Native training Local or caller-provided Review required">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_nemo/">NeMoASR</a></h2>
        <code>asr_nemo</code>
      </div>
      <p class="vh-model-card__summary">Runs VoiceHub&#x27;s native QuartzNet15x5 graph from the pinned NeMo/NGC source.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>nemo-asr</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Review required</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_nemo/">View model <span aria-hidden="true">→</span></a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="nemotron"
      data-model-type="asr_nemotron"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="commercial"
      data-architecture="nemotron-3.5-rnnt"
      data-language-kind="enumerated"
      data-language-count="40"
      data-languages="en-US en-GB es-US es-ES fr-FR fr-CA it-IT pt-BR pt-PT nl-NL de-DE tr-TR ru-RU ar-AR hi-IN ja-JP ko-KR vi-VN uk-UA pl-PL sv-SE cs-CZ nb-NO da-DK bg-BG fi-FI hr-HR sk-SK zh-CN hu-HU ro-RO et-EE el-GR lt-LT lv-LV mt-MT sl-SI he-IL th-TH nn-NO"
      data-capabilities="automatic-speech-recognition multilingual language-identification timestamps streaming-architecture safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Nemotron asr_nemotron Automatic speech recognition ASR nemotron-3.5-rnnt nvidia/nemotron-3.5-asr-streaming-0.6b nvidia/nemotron-3.5-asr-streaming-0.6b Native training Hugging Face Commercial use declared">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_nemotron/">Nemotron</a></h2>
        <code>asr_nemotron</code>
      </div>
      <p class="vh-model-card__summary">Uses Nemotron&#x27;s cache-aware native decoder and requests word timestamps.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>nemotron-3.5-rnnt</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en-US</code><code>en-GB</code><code>es-US</code><code>es-ES</code><code>fr-FR</code><span class="vh-model-card__more-languages">+35</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Commercial use declared</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Language identification</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_nemotron/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_nemotron.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="nemovad"
      data-model-type="vad_nemo"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="marblenet-vad"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection voicehub-native safetensors trusted-checkpoint-conversion frame-scores fine-tuning"
      data-resources="notebook huggingface"
      data-search="NeMoVAD vad_nemo Voice activity detection VAD marblenet-vad nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_nemo/">NeMoVAD</a></h2>
        <code>vad_nemo</code>
      </div>
      <p class="vh-model-card__summary">Runs multilingual MarbleNet frame VAD and retains frame scores for inspection.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>marblenet-vad</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_nemo/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_nemo.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="neutts"
      data-model-type="neutts"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="review"
      data-architecture="neutts"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech voice-cloning multilingual emotion safetensors fine-tuning default-checkpoint-inference-only raw-audio-training preencoded-code-training voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="NeuTTS neutts Text to speech TTS neutts neuphonic/neutts-2e neuphonic/neutts-2e Native training Hugging Face Review required">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="neutts/">NeuTTS</a></h2>
        <code>neutts</code>
      </div>
      <p class="vh-model-card__summary">Uses NeuTTS&#x27;s required one-of speaker source with the matching reference transcript.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>neutts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Review required</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="neutts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/neuphonic/neutts-2e">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/neutts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="omnivoice"
      data-model-type="omnivoice"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="omnivoice"
      data-language-kind="enumerated"
      data-language-count="646"
      data-languages="aae aal aao ab abb abn abr abs abv acm acw acx adf adx ady aeb aec af afb afo ahl ahs ajg aju ala aln alo am amu an anc ank anp anw aom apc apd arb arq ars ary arz as ast avl awo ayl ayp az ba bag bas bax bba bbj bbl bbu bce bci bcs bcy bda bde bdm be beb bew bfd bft bg bgp bhb bhh bho bhp bhr bjj bjk bjn bjt bkh bkm bky bmm bmq bn bnm bnn bns bo bou bqg br bra brh bri brx bs bsh bsj bsk btm btv bug bum buo bux bwr bxf byc bys byv byx bzc bzw ca ccg ceb cen cfa cgg chq cjk ckb ckl ckr cky cnh cpy cs cte ctl cut cux cv cy da dag dar dav dbd dcc de deg dgh dgo dje dmk dml dru dty dua dv dyu dzg ebr ebu ego eiv eko ekr el elm en eo es esu et eto ets etu eu ewo ext eyo fa fan fat ff ffm fi fia fil fip fkk fmp fr fub fuc fue fuf fuh fui fuq fuv fy ga gbm gbr gby gcc gdf gej ges ggg gid gig giz gjk gju gl glw gn gol gom gsl gu gui gur guz gv gwc gwe gwt gya gyz ha hah hao haw haz hbb he hem hi hia hkk hla hno hoj hr hsb ht hu hue hul hux hwo hy hz ia ibb id ida idu ig ijc ijn ik ikw is ish iso it its itw itz ja jal jax jgo jmx jns jqr juk juo jv ka kab kai kaj kam kbd kbl kbt kcq kdh kea keu kfe kfk kfp khg khw kj kjc kjk kk kln kls km kmr kmy kn kna knn ko kol koo kpo kqo ks ksd ksf kto kuh kvx kw kwm kxp ky kyx lag lb lcm ldb lg lij lir lkb lla ln lnu lo loa lrk lss lt ltg lto lua luo lus lv lwg mab maf mai mau max mbo mcf mcn mcx mdd mde mdf mek mer meu mfm mfn mfo mfv mgg mgi mhk mhr mi mig miu mk mkf mki ml mlq mn mne mni mqy mr mrj mrr mrt ms mse msh msw mt mtr mtu mtx mua mug mui mve mvy mxs mxu mxy my myv mzl nal nan nap nb nbh ncf nco ncx ndi ng ngi nhg nhi nhn nhq nja nl nla nlv nmg nmz nn nnh no noe npi nso ny nyu oc odk odu ogo om orc oru ory os pa pbs pbt pbu pcm pex phl phr pip piy pko pl plk plt pmq pms pmy pnb poc poe pow prq ps pst pt pua pwn qug qum qup qur qus quv qux quy qva qvi qvj qvl qwa qws qxa qxp qxt qxu qxw rag rm ro rob rof roo rth ru rup rw sa sah sat sau say sbn sc scl scn sd sei shu si sip siw sjr sk skg skr sl sn snc snk so sol sps sq sr src sro ssi ste sua sv sva sw szy ta tan tar tay tbf tcf tcy tdn tdx te tg tgc th the thq thr thv ti tig tio tk tkg tkt tli tlp tn tok tpl tpz tqp tr trp trq trv trw tt ttj ttr ttu tui tul tuq tuv tuy tvo tvu tw twu txs txy udl ug uk uki umb ur ush uz uzn vai var ver vi vmc vmj vmm vmp vmz vot vro wbl wci weo wes wja wji wo wof xh xhe xka xmf xmv xmw xpe xti xtu yaq yav yay ydd ydg yer yes yi yo yue zga zgh zh zoc zoh zor zpv zpy ztg ztn ztp zts ztu zu zza"
      data-capabilities="text-to-speech voice-cloning voice-design multilingual fine-tuning safetensors voicehub-native native-runtime raw-audio-fine-tuning preencoded-code-fine-tuning"
      data-resources="notebook huggingface"
      data-search="OmniVoice omnivoice Text to speech TTS omnivoice k2-fsa/OmniVoice k2-fsa/OmniVoice Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="omnivoice/">OmniVoice</a></h2>
        <code>omnivoice</code>
      </div>
      <p class="vh-model-card__summary">Pairs OmniVoice speaker audio with its transcript and selects the native iterative decoder controls.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>omnivoice</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>aae</code><code>aal</code><code>aao</code><code>ab</code><code>abb</code><span class="vh-model-card__more-languages">+641</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span><span>Voice design</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="omnivoice/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/k2-fsa/OmniVoice">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/omnivoice.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="openaiwhisper"
      data-model-type="asr_openai_whisper"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="whisper"
      data-language-kind="enumerated"
      data-language-count="99"
      data-languages="en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su"
      data-capabilities="automatic-speech-recognition multilingual translation timestamps safetensors fine-tuning voicehub-native"
      data-resources="notebook huggingface"
      data-search="OpenAIWhisper asr_openai_whisper Automatic speech recognition ASR whisper openai/whisper-small openai/whisper-small Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_openai_whisper/">OpenAIWhisper</a></h2>
        <code>asr_openai_whisper</code>
      </div>
      <p class="vh-model-card__summary">Runs the original OpenAI Whisper backend with deterministic beam decoding.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>whisper</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>de</code><code>es</code><code>ru</code><span class="vh-model-card__more-languages">+94</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Translation</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_openai_whisper/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/openai/whisper-small">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_openai_whisper.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="openvoice"
      data-model-type="openvoice"
      data-task="text-to-speech"
      data-training="custom"
      data-training-rank="2"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="openvoice-v2-converter"
      data-language-kind="enumerated"
      data-language-count="6"
      data-languages="en es fr zh ja ko"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime paired-waveform-training explicit-base-waveform"
      data-resources="notebook huggingface"
      data-search="OpenVoice openvoice Text to speech TTS openvoice-v2-converter myshell-ai/OpenVoiceV2 myshell-ai/OpenVoiceV2 Custom training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Custom training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="openvoice/">OpenVoice</a></h2>
        <code>openvoice</code>
      </div>
      <p class="vh-model-card__summary">Runs OpenVoice tone-color transfer from a source utterance to an authorized target-speaker recording.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>openvoice-v2-converter</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>es</code><code>fr</code><code>zh</code><code>ja</code><span class="vh-model-card__more-languages">+1</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="openvoice/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/myshell-ai/OpenVoiceV2">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/openvoice.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="orpheustts"
      data-model-type="orpheustts"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="causal-lm"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech expressive-speech safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="OrpheusTTS orpheustts Text to speech TTS causal-lm canopylabs/orpheus-3b-0.1-ft canopylabs/orpheus-3b-0.1-ft Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="orpheustts/">OrpheusTTS</a></h2>
        <code>orpheustts</code>
      </div>
      <p class="vh-model-card__summary">Selects a released Orpheus voice and bounds autoregressive audio-token generation.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>causal-lm</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Expressive speech</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="orpheustts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/canopylabs/orpheus-3b-0.1-ft">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/orpheustts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="outetts"
      data-model-type="outetts"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="noncommercial"
      data-architecture="outetts"
      data-language-kind="enumerated"
      data-language-count="23"
      data-languages="en ar zh nl fr de it ja ko lt ru es pt be bn ka hu lv fa pl sw ta uk"
      data-capabilities="text-to-speech voice-cloning fine-tuning safetensors voicehub-native native-runtime preprocessed-training speaker-profile-training"
      data-resources="notebook huggingface"
      data-search="OuteTTS outetts Text to speech TTS outetts OuteAI/Llama-OuteTTS-1.0-1B OuteAI/Llama-OuteTTS-1.0-1B Prepared-data training Hugging Face Non-commercial">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="outetts/">OuteTTS</a></h2>
        <code>outetts</code>
      </div>
      <p class="vh-model-card__summary">Uses the audited OuteTTS V3 regular generation path with an explicit token limit.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>outetts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>ar</code><code>zh</code><code>nl</code><code>fr</code><span class="vh-model-card__more-languages">+18</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Non-commercial</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="outetts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/outetts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="parakeettdt"
      data-model-type="asr_parakeet_tdt"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="commercial"
      data-architecture="parakeet-tdt"
      data-language-kind="enumerated"
      data-language-count="25"
      data-languages="en es fr de bg hr cs da nl et fi el hu it lv lt mt pl pt ro sk sl sv ru uk"
      data-capabilities="automatic-speech-recognition multilingual timestamps long-form safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="ParakeetTDT asr_parakeet_tdt Automatic speech recognition ASR parakeet-tdt nvidia/parakeet-tdt-0.6b-v3 nvidia/parakeet-tdt-0.6b-v3 Native training Hugging Face Commercial use declared">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_parakeet_tdt/">ParakeetTDT</a></h2>
        <code>asr_parakeet_tdt</code>
      </div>
      <p class="vh-model-card__summary">Runs the native Parakeet TDT decoder and returns its calibrated timestamp segments.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>parakeet-tdt</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>es</code><code>fr</code><code>de</code><code>bg</code><span class="vh-model-card__more-languages">+20</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Commercial use declared</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Long-form audio</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_parakeet_tdt/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_parakeet_tdt.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="parlertts"
      data-model-type="parlertts"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="parlertts"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech prompted-style fine-tuning safetensors voicehub-native native-runtime raw-audio-fine-tuning"
      data-resources="notebook huggingface"
      data-search="ParlerTTS parlertts Text to speech TTS parlertts parler-tts/parler-tts-mini-v1 parler-tts/parler-tts-mini-v1 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="parlertts/">ParlerTTS</a></h2>
        <code>parlertts</code>
      </div>
      <p class="vh-model-card__summary">Separates the spoken text from Parler-TTS&#x27;s acoustic style description.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>parlertts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="parlertts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/parler-tts/parler-tts-mini-v1">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/parlertts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="pyannotebrouhahavad"
      data-model-type="vad_pyannote_brouhaha"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="pyannet"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection gated-checkpoint voicehub-native trusted-checkpoint-conversion safetensors frame-scores snr c50 fine-tuning"
      data-resources="notebook huggingface"
      data-search="PyannoteBrouhahaVAD vad_pyannote_brouhaha Voice activity detection VAD pyannet pyannote/brouhaha pyannote/brouhaha Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_pyannote_brouhaha/">PyannoteBrouhahaVAD</a></h2>
        <code>vad_pyannote_brouhaha</code>
      </div>
      <p class="vh-model-card__summary">Uses Brouhaha speech scores while returning frames for downstream SNR or quality review.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>pyannet</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_pyannote_brouhaha/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/pyannote/brouhaha">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_pyannote_brouhaha.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="pyannotesegmentationvad"
      data-model-type="vad_pyannote_segmentation"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="pyannet"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection voicehub-native gated-checkpoint trusted-checkpoint-conversion safetensors powerset frame-scores fine-tuning"
      data-resources="notebook huggingface"
      data-search="PyannoteSegmentationVAD vad_pyannote_segmentation Voice activity detection VAD pyannet pyannote/segmentation-3.0 pyannote/segmentation-3.0 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_pyannote_segmentation/">PyannoteSegmentationVAD</a></h2>
        <code>vad_pyannote_segmentation</code>
      </div>
      <p class="vh-model-card__summary">Runs pyannote segmentation 3.0 as VAD with hysteresis and frame evidence enabled.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>pyannet</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_pyannote_segmentation/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/pyannote/segmentation-3.0">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_pyannote_segmentation.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="pyannotevad"
      data-model-type="vad_pyannote"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="pyannet"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection voicehub-native gated-checkpoint trusted-checkpoint-conversion safetensors frame-scores fine-tuning"
      data-resources="notebook huggingface"
      data-search="PyannoteVAD vad_pyannote Voice activity detection VAD pyannet pyannote/voice-activity-detection pyannote/voice-activity-detection Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_pyannote/">PyannoteVAD</a></h2>
        <code>vad_pyannote</code>
      </div>
      <p class="vh-model-card__summary">Applies pyannote VAD with separate onset and offset thresholds.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>pyannet</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_pyannote/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/pyannote/voice-activity-detection">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_pyannote.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="qwen3asr"
      data-model-type="asr_qwen3"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="qwen3-asr"
      data-language-kind="enumerated"
      data-language-count="30"
      data-languages="ar yue zh cs da nl en fil fi fr de el hi hu id it ja ko mk ms fa pl pt ro ru es sv th tr vi"
      data-capabilities="automatic-speech-recognition multilingual language-identification hotwords long-form safetensors fine-tuning lora voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Qwen3ASR asr_qwen3 Automatic speech recognition ASR qwen3-asr Qwen/Qwen3-ASR-0.6B Qwen/Qwen3-ASR-0.6B Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_qwen3/">Qwen3ASR</a></h2>
        <code>asr_qwen3</code>
      </div>
      <p class="vh-model-card__summary">Provides Qwen3-ASR with a domain prompt and deterministic decoding controls.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>qwen3-asr</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>ar</code><code>yue</code><code>zh</code><code>cs</code><code>da</code><span class="vh-model-card__more-languages">+25</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Long-form audio</span><span>Language identification</span><span>Hotwords</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_qwen3/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/Qwen/Qwen3-ASR-0.6B">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_qwen3.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="qwen3tts"
      data-model-type="qwen3tts"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="qwen3-tts"
      data-language-kind="enumerated"
      data-language-count="10"
      data-languages="zh en ja ko de fr ru pt es it"
      data-capabilities="text-to-speech voice-cloning voice-design multilingual fine-tuning lora-fine-tuning default-checkpoint-inference-only safetensors voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Qwen3TTS qwen3tts Text to speech TTS qwen3-tts Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="qwen3tts/">Qwen3TTS</a></h2>
        <code>qwen3tts</code>
      </div>
      <p class="vh-model-card__summary">Uses the registered Qwen3-TTS CustomVoice role with an explicit language and speaker.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>qwen3-tts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>en</code><code>ja</code><code>ko</code><code>de</code><span class="vh-model-card__more-languages">+5</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span><span>Voice design</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="qwen3tts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/qwen3tts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="seamlessm4tv2"
      data-model-type="asr_seamless_m4t_v2"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="noncommercial"
      data-architecture="seamless-m4t-v2-s2t"
      data-language-kind="enumerated"
      data-language-count="98"
      data-languages="afr amh arb ary arz asm azj bel ben bos bul cat ceb ces ckb cmn cmn_Hant cym dan deu ell eng est eus fin fra fuv gaz gle glg guj heb hin hrv hun hye ibo ind isl ita jav jpn kan kat kaz khk khm kir kor lao lit lug luo lvs mai mal mar mkd mlt mni mya nld nno nob npi nya ory pan pbt pes pol por ron rus sat slk slv sna snd som spa srp swe swh tam tel tgk tgl tha tur ukr urd uzn vie yor yue zlm zul"
      data-capabilities="automatic-speech-recognition multilingual safetensors fine-tuning voicehub-native native-runtime greedy-decoding full-model-training"
      data-resources="notebook huggingface"
      data-search="SeamlessM4Tv2 asr_seamless_m4t_v2 Automatic speech recognition ASR seamless-m4t-v2-s2t facebook/seamless-m4t-v2-large facebook/seamless-m4t-v2-large Native training Hugging Face Non-commercial">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_seamless_m4t_v2/">SeamlessM4Tv2</a></h2>
        <code>asr_seamless_m4t_v2</code>
      </div>
      <p class="vh-model-card__summary">Selects SeamlessM4T v2 transcription rather than speech translation.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>seamless-m4t-v2-s2t</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>afr</code><code>amh</code><code>arb</code><code>ary</code><code>arz</code><span class="vh-model-card__more-languages">+93</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Non-commercial</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_seamless_m4t_v2/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/facebook/seamless-m4t-v2-large">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_seamless_m4t_v2.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="sherpaonnxvad"
      data-model-type="vad_sherpa_onnx"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="review"
      data-architecture="native-vad-dispatch"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection voicehub-native safetensors explicit-onnx-weight-conversion fine-tuning streaming sherpa-compatible-segmentation silero ten-vad"
      data-resources="notebook huggingface"
      data-search="SherpaONNXVAD vad_sherpa_onnx Voice activity detection VAD native-vad-dispatch safestack/silero-vad safestack/silero-vad Native training Hugging Face Review required">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_sherpa_onnx/">SherpaONNXVAD</a></h2>
        <code>vad_sherpa_onnx</code>
      </div>
      <p class="vh-model-card__summary">Uses sherpa-onnx streaming Silero state with an explicit threshold and segment padding.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>native-vad-dispatch</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Review required</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Streaming</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_sherpa_onnx/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/safestack/silero-vad">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_sherpa_onnx.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="silerovad"
      data-model-type="vad_silero"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="silero-vad"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection voicehub-native safetensors jit-weight-import frame-scores streaming fine-tuning"
      data-resources="notebook huggingface"
      data-search="SileroVAD vad_silero Voice activity detection VAD silero-vad safestack/silero-vad safestack/silero-vad Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_silero/">SileroVAD</a></h2>
        <code>vad_silero</code>
      </div>
      <p class="vh-model-card__summary">Uses Silero&#x27;s probability threshold plus minimum speech/silence duration smoothing.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>silero-vad</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Streaming</span><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_silero/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/safestack/silero-vad">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_silero.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="speechbrainasr"
      data-model-type="asr_speechbrain"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="commercial"
      data-architecture="speechbrain-crdnn-asr"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition english beam-search safetensors fine-tuning voicehub-native crdnn ctc-seq2seq rnnlm-shallow-fusion"
      data-resources="notebook huggingface"
      data-search="SpeechBrainASR asr_speechbrain Automatic speech recognition ASR speechbrain-crdnn-asr speechbrain/asr-crdnn-rnnlm-librispeech speechbrain/asr-crdnn-rnnlm-librispeech Native training Hugging Face Commercial use declared">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_speechbrain/">SpeechBrainASR</a></h2>
        <code>asr_speechbrain</code>
      </div>
      <p class="vh-model-card__summary">Uses the audited SpeechBrain CRDNN/RNNLM decoder with an explicit beam size.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>speechbrain-crdnn-asr</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Commercial use declared</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_speechbrain/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_speechbrain.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="speechbrainvad"
      data-model-type="vad_speechbrain"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="speechbrain-crdnn-vad"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection voicehub-native safetensors trusted-checkpoint-conversion frame-scores fine-tuning offline-bidirectional"
      data-resources="notebook huggingface"
      data-search="SpeechBrainVAD vad_speechbrain Voice activity detection VAD speechbrain-crdnn-vad speechbrain/vad-crdnn-libriparty speechbrain/vad-crdnn-libriparty Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_speechbrain/">SpeechBrainVAD</a></h2>
        <code>vad_speechbrain</code>
      </div>
      <p class="vh-model-card__summary">Uses SpeechBrain CRDNN probabilities with explicit hysteresis and silence merging.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>speechbrain-crdnn-vad</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_speechbrain/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/speechbrain/vad-crdnn-libriparty">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_speechbrain.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="speecht5"
      data-model-type="speecht5"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="speecht5"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech speaker-embedding safetensors fine-tuning voicehub-native native-runtime raw-audio-fine-tuning inference-reloadable-training-export"
      data-resources="notebook huggingface"
      data-search="SpeechT5 speecht5 Text to speech TTS speecht5 microsoft/speecht5_tts microsoft/speecht5_tts Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="speecht5/">SpeechT5</a></h2>
        <code>speecht5</code>
      </div>
      <p class="vh-model-card__summary">Passes a reviewed speaker-embedding file through SpeechT5&#x27;s safe public loader.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>speecht5</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="speecht5/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/microsoft/speecht5_tts">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/speecht5.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="styletts2"
      data-model-type="styletts2"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="local"
      data-license="checkpoint-specific"
      data-architecture="styletts2"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en-US"
      data-capabilities="text-to-speech voice-cloning fine-tuning safetensors voicehub-native native-runtime preprocessed-training explicit-phonemes"
      data-resources="huggingface"
      data-search="StyleTTS2 styletts2 Text to speech TTS styletts2  yl4579/StyleTTS2-LibriTTS Prepared-data training Local or caller-provided Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="styletts2/">StyleTTS2</a></h2>
        <code>styletts2</code>
      </div>
      <p class="vh-model-card__summary">Uses an explicit local VoiceHub artifact and the native phoneme boundary required by StyleTTS 2.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>styletts2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en-US</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="styletts2/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/yl4579/StyleTTS2-LibriTTS">Hugging Face</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="supertonic"
      data-model-type="supertonic"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="supertonic"
      data-language-kind="enumerated"
      data-language-count="31"
      data-languages="en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi"
      data-capabilities="text-to-speech multilingual fine-tuning safetensors voicehub-native native-runtime preprocessed-training"
      data-resources="notebook huggingface"
      data-search="Supertonic supertonic Text to speech TTS supertonic Supertone/supertonic-3 Supertone/supertonic-3 Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="supertonic/">Supertonic</a></h2>
        <code>supertonic</code>
      </div>
      <p class="vh-model-card__summary">Selects a Supertonic style ID, language, diffusion-step count, and speaking speed.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>supertonic</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>ko</code><code>ja</code><code>ar</code><code>bg</code><span class="vh-model-card__more-languages">+26</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="supertonic/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/Supertone/supertonic-3">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/supertonic.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="tiron"
      data-model-type="asr_tiron"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="whisper"
      data-language-kind="enumerated"
      data-language-count="2"
      data-languages="en zh"
      data-capabilities="automatic-speech-recognition multilingual speaker-attribution timestamps safetensors fine-tuning constrained-decoding voicehub-native"
      data-resources="notebook huggingface"
      data-search="Tiron asr_tiron Automatic speech recognition ASR whisper Trelis/tiron Trelis/tiron Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_tiron/">Tiron</a></h2>
        <code>asr_tiron</code>
      </div>
      <p class="vh-model-card__summary">Enables Tiron constrained decoding and caps diarized speakers for a meeting recording.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>whisper</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Speaker attribution</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_tiron/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/Trelis/tiron">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_tiron.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="transformersasr"
      data-model-type="asr_transformers"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="native-asr-dispatch"
      data-language-kind="enumerated"
      data-language-count="99"
      data-languages="en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su"
      data-capabilities="automatic-speech-recognition multilingual timestamps safetensors fine-tuning ctc speech-seq2seq voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="TransformersASR asr_transformers Automatic speech recognition ASR native-asr-dispatch openai/whisper-small openai/whisper-small Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_transformers/">TransformersASR</a></h2>
        <code>asr_transformers</code>
      </div>
      <p class="vh-model-card__summary">Runs the generic Transformers ASR adapter with Whisper language/task controls and timestamps.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>native-asr-dispatch</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>de</code><code>es</code><code>ru</code><span class="vh-model-card__more-languages">+94</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_transformers/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/openai/whisper-small">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_transformers.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="transformersvad"
      data-model-type="vad_transformers"
      data-task="voice-activity-detection"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="local"
      data-license="checkpoint-specific"
      data-architecture="wav2vec2"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection frame-scores safetensors fine-tuning voicehub-native native-runtime"
      data-resources=""
      data-search="TransformersVAD vad_transformers Voice activity detection VAD wav2vec2   Native training Local or caller-provided Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_transformers/">TransformersVAD</a></h2>
        <code>vad_transformers</code>
      </div>
      <p class="vh-model-card__summary">Runs a caller-selected Transformers frame classifier with explicit hysteresis and frame output.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>wav2vec2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Frame scores</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_transformers/">View model <span aria-hidden="true">→</span></a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="vibevoice"
      data-model-type="asr_vibevoice"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="vibevoice-asr"
      data-language-kind="enumerated"
      data-language-count="51"
      data-languages="en zh es pt de ja ko fr ru id sv it he nl pl no tr th ar hu ca cs da fa af hi fi et aa el ro vi bg is sl sk lt sw uk kl lv hr ne sr tl yi ms ur mn hy jv"
      data-capabilities="automatic-speech-recognition multilingual speaker-attribution timestamps hotwords long-form safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="VibeVoice asr_vibevoice Automatic speech recognition ASR vibevoice-asr microsoft/VibeVoice-ASR-HF microsoft/VibeVoice-ASR-HF Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_vibevoice/">VibeVoice</a></h2>
        <code>asr_vibevoice</code>
      </div>
      <p class="vh-model-card__summary">Requests VibeVoice-ASR timestamps with a concise transcription prompt.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>vibevoice-asr</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>es</code><code>pt</code><code>de</code><span class="vh-model-card__more-languages">+46</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Long-form audio</span><span>Speaker attribution</span><span class="vh-model-card__more-features">+1 more</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_vibevoice/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/microsoft/VibeVoice-ASR-HF">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_vibevoice.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="vibevoice"
      data-model-type="vibevoice"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="vibevoice-tts"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech voice-prompt fine-tuning default-checkpoint-inference-only safetensors voicehub-native native-runtime preprocessed-training verified-low-level-realtime-stages high-level-generation-fails-closed"
      data-resources="notebook huggingface"
      data-search="VibeVoice vibevoice Text to speech TTS vibevoice-tts microsoft/VibeVoice-Realtime-0.5B microsoft/VibeVoice-Realtime-0.5B Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vibevoice/">VibeVoice</a></h2>
        <code>vibevoice</code>
      </div>
      <p class="vh-model-card__summary">Loads the audited VibeVoice realtime stages without claiming an unverified text-to-waveform loop.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>vibevoice-tts</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vibevoice/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vibevoice.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="vits"
      data-model-type="vits"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="vits"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech multilingual mms-tts safetensors fine-tuning voicehub-native native-runtime raw-audio-training preprocessed-training adversarial-training generator-warm-start explicit-acoustic-training-config"
      data-resources="notebook huggingface"
      data-search="Vits vits Text to speech TTS vits facebook/mms-tts-eng facebook/mms-tts-eng Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vits/">Vits</a></h2>
        <code>vits</code>
      </div>
      <p class="vh-model-card__summary">Controls MMS-VITS speaking rate, stochastic duration, and output-frame guardrails.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>vits</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vits/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/facebook/mms-tts-eng">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vits.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="voxcpm"
      data-model-type="voxcpm"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="voxcpm2"
      data-language-kind="enumerated"
      data-language-count="30"
      data-languages="zh en ar my da nl fi fr de el he hi id it ja km ko lo ms no pl pt ru es sw sv tl th tr vi"
      data-capabilities="text-to-speech voice-cloning voice-design audio-continuation multilingual fine-tuning safetensors voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="VoxCPM voxcpm Text to speech TTS voxcpm2 openbmb/VoxCPM2 openbmb/VoxCPM2 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="voxcpm/">VoxCPM</a></h2>
        <code>voxcpm</code>
      </div>
      <p class="vh-model-card__summary">Conditions VoxCPM2 on a reference timbre and exposes its diffusion guidance and step count.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>voxcpm2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>zh</code><code>en</code><code>ar</code><code>my</code><code>da</code><span class="vh-model-card__more-languages">+25</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span><span>Voice design</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="voxcpm/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/openbmb/VoxCPM2">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/voxcpm.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="vui"
      data-model-type="vui"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="local"
      data-license="checkpoint-specific"
      data-architecture="vui"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="text-to-speech fine-tuning safetensors standalone-safetensors-export voicehub-native native-runtime preprocessed-training"
      data-resources="huggingface"
      data-search="Vui vui Text to speech TTS vui vui-abraham-100m.pt fluxions/vui Prepared-data training Local or caller-provided Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vui/">Vui</a></h2>
        <code>vui</code>
      </div>
      <p class="vh-model-card__summary">Uses Vui&#x27;s bounded chunk retry and duration controls with the registered pinned artifacts.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>vui</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vui/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/fluxions/vui">Hugging Face</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="wav2vec2"
      data-model-type="asr_wav2vec2"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="wav2vec2"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition timestamps safetensors fine-tuning voicehub-native"
      data-resources="notebook huggingface"
      data-search="Wav2Vec2 asr_wav2vec2 Automatic speech recognition ASR wav2vec2 facebook/wav2vec2-base-960h facebook/wav2vec2-base-960h Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_wav2vec2/">Wav2Vec2</a></h2>
        <code>asr_wav2vec2</code>
      </div>
      <p class="vh-model-card__summary">Runs the native Wav2Vec2 CTC path and requests word-level alignment where supported.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>wav2vec2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_wav2vec2/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/facebook/wav2vec2-base-960h">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_wav2vec2.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="wavlm"
      data-model-type="asr_wavlm"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="wavlm"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition timestamps safetensors fine-tuning voicehub-native"
      data-resources="notebook huggingface"
      data-search="WavLM asr_wavlm Automatic speech recognition ASR wavlm patrickvonplaten/wavlm-libri-clean-100h-base-plus patrickvonplaten/wavlm-libri-clean-100h-base-plus Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_wavlm/">WavLM</a></h2>
        <code>asr_wavlm</code>
      </div>
      <p class="vh-model-card__summary">Runs the registered WavLM CTC checkpoint with greedy single-beam decoding.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>wavlm</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_wavlm/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_wavlm.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="webrtcvad"
      data-model-type="vad_webrtc"
      data-task="voice-activity-detection"
      data-training="inference-only"
      data-training-rank="3"
      data-checkpoint="local"
      data-license="checkpoint-specific"
      data-architecture="webrtc-vad"
      data-language-kind="not-text-conditioned"
      data-language-count="0"
      data-languages=""
      data-capabilities="voice-activity-detection fixed-point voicehub-native native-runtime streaming"
      data-resources=""
      data-search="WebRTCVAD vad_webrtc Voice activity detection VAD webrtc-vad webrtc-vad  Inference only Local or caller-provided Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--vad">VAD</span>
        <span class="vh-model-card__training">Inference only</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="vad_webrtc/">WebRTCVAD</a></h2>
        <code>vad_webrtc</code>
      </div>
      <p class="vh-model-card__summary">Runs weightless WebRTC VAD with frame-compatible duration controls.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>webrtc-vad</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><span class="vh-model-card__neutral-language">Language-neutral</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Local or caller-provided</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Streaming</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="vad_webrtc/">View model <span aria-hidden="true">→</span></a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="wenetasr"
      data-model-type="asr_wenet"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="external-archive"
      data-license="review"
      data-architecture="wenet-asr"
      data-language-kind="enumerated"
      data-language-count="1"
      data-languages="en"
      data-capabilities="automatic-speech-recognition english timestamps safetensors fine-tuning voicehub-native ctc attention-rescoring"
      data-resources=""
      data-search="WeNetASR asr_wenet Automatic speech recognition ASR wenet-asr wenet/gigaspeech-u2pp-conformer  Native training External archive Review required">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_wenet/">WeNetASR</a></h2>
        <code>asr_wenet</code>
      </div>
      <p class="vh-model-card__summary">Loads a reviewed VoiceHub conversion of WeNet GigaSpeech U2++ and requests word timestamps.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>wenet-asr</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code></dd></div>
        <div><dt>Checkpoint</dt><dd>External archive</dd></div>
        <div><dt>License</dt><dd>Review required</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_wenet/">View model <span aria-hidden="true">→</span></a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="whisper"
      data-model-type="asr_whisper"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="whisper"
      data-language-kind="enumerated"
      data-language-count="99"
      data-languages="en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su"
      data-capabilities="automatic-speech-recognition multilingual translation timestamps safetensors fine-tuning voicehub-native"
      data-resources="notebook huggingface"
      data-search="Whisper asr_whisper Automatic speech recognition ASR whisper openai/whisper-large-v3-turbo openai/whisper-large-v3-turbo Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_whisper/">Whisper</a></h2>
        <code>asr_whisper</code>
      </div>
      <p class="vh-model-card__summary">Uses VoiceHub&#x27;s native Whisper graph with explicit transcription language and word timestamps.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>whisper</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>de</code><code>es</code><code>ru</code><span class="vh-model-card__more-languages">+94</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Timestamps</span><span>Translation</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_whisper/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/openai/whisper-large-v3-turbo">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_whisper.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="whisperx"
      data-model-type="asr_whisperx"
      data-task="automatic-speech-recognition"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="whisper"
      data-language-kind="enumerated"
      data-language-count="99"
      data-languages="en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su"
      data-capabilities="automatic-speech-recognition multilingual word-timestamps alignment safetensors fine-tuning voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="WhisperX asr_whisperx Automatic speech recognition ASR whisper openai/whisper-small openai/whisper-small Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--asr">ASR</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="asr_whisperx/">WhisperX</a></h2>
        <code>asr_whisperx</code>
      </div>
      <p class="vh-model-card__summary">Requests WhisperX alignment timestamps through VoiceHub&#x27;s normalized ASR output.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>whisper</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>de</code><code>es</code><code>ru</code><span class="vh-model-card__more-languages">+94</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Core speech inference</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="asr_whisperx/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/openai/whisper-small">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_whisperx.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="xtts"
      data-model-type="xtts"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="review"
      data-architecture="xtts2"
      data-language-kind="enumerated"
      data-language-count="17"
      data-languages="en es fr de it pt pl tr ru nl cs ar zh-CN hu ko ja hi"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime preencoded-code-fine-tuning gpt-fine-tuning restricted-pickle-conversion"
      data-resources="notebook huggingface"
      data-search="XTTS xtts Text to speech TTS xtts2 coqui/XTTS-v2 coqui/XTTS-v2 Prepared-data training Hugging Face Review required">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="xtts/">XTTS</a></h2>
        <code>xtts</code>
      </div>
      <p class="vh-model-card__summary">Supplies the mandatory XTTS v2 speaker reference and a supported language code.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>xtts2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>es</code><code>fr</code><code>de</code><code>it</code><span class="vh-model-card__more-languages">+12</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Review required</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="xtts/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/coqui/XTTS-v2">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/xtts.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="zonos"
      data-model-type="zonos"
      data-task="text-to-speech"
      data-training="preprocessed"
      data-training-rank="1"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="zonos"
      data-language-kind="enumerated"
      data-language-count="5"
      data-languages="en ja zh fr de"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Zonos zonos Text to speech TTS zonos Zyphra/Zonos-v0.1-transformer Zyphra/Zonos-v0.1-transformer Prepared-data training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Prepared-data training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="zonos/">Zonos</a></h2>
        <code>zonos</code>
      </div>
      <p class="vh-model-card__summary">Conditions Zonos on an eSpeak language code and an authorized speaker reference.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>zonos</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>ja</code><code>zh</code><code>fr</code><code>de</code></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="zonos/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/Zyphra/Zonos-v0.1-transformer">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/zonos.ipynb">Colab</a></footer>
    </article>
<article class="vh-model-card"
      data-vh-model-card
      data-name="zonos2"
      data-model-type="zonos2"
      data-task="text-to-speech"
      data-training="native"
      data-training-rank="0"
      data-checkpoint="huggingface"
      data-license="checkpoint-specific"
      data-architecture="zonos2"
      data-language-kind="enumerated"
      data-language-count="34"
      data-languages="en zh ja ko ru it pt fr es vi de he nl sv hi ta te th no bn tl ar da id pl uk ro fi hu lt et sk hr lv"
      data-capabilities="text-to-speech voice-cloning multilingual fine-tuning safetensors voicehub-native native-runtime"
      data-resources="notebook huggingface"
      data-search="Zonos2 zonos2 Text to speech TTS zonos2 Zyphra/ZONOS2 Zyphra/ZONOS2 Native training Hugging Face Checkpoint-specific">
      <div class="vh-model-card__topline">
        <span class="vh-model-badge vh-model-badge--tts">TTS</span>
        <span class="vh-model-card__training">Native training</span>
      </div>
      <div class="vh-model-card__heading">
        <h2><a href="zonos2/">Zonos2</a></h2>
        <code>zonos2</code>
      </div>
      <p class="vh-model-card__summary">Uses ZONOS2&#x27;s language, speed, accurate-mode, and speaker-conditioning controls.</p>
      <dl class="vh-model-card__metadata">
        <div><dt>Architecture</dt><dd><code>zonos2</code></dd></div>
        <div><dt>Languages</dt><dd class="vh-model-card__languages"><code>en</code><code>zh</code><code>ja</code><code>ko</code><code>ru</code><span class="vh-model-card__more-languages">+29</span></dd></div>
        <div><dt>Checkpoint</dt><dd>Hugging Face</dd></div>
        <div><dt>License</dt><dd>Checkpoint-specific</dd></div>
      </dl>
      <div class="vh-model-card__features" aria-label="Capabilities"><span>Voice cloning</span></div>
      <footer class="vh-model-card__actions"><a class="vh-model-card__primary-action" href="zonos2/">View model <span aria-hidden="true">→</span></a> <a href="https://huggingface.co/Zyphra/ZONOS2">Hugging Face</a> <a href="https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/zonos2.ipynb">Colab</a></footer>
    </article></div>
  <div class="vh-model-empty" data-vh-model-empty hidden>
    <span aria-hidden="true">⌕</span>
    <h2>No models match these filters</h2>
    <p>Try a broader language, remove a capability, or clear the search.</p>
    <button type="button" data-vh-model-clear>Clear all filters</button>
  </div>
  <noscript><p class="vh-model-explorer__noscript">Enable JavaScript to filter this catalog. All model cards remain available below.</p></noscript>
</div>

## Search the registry in Python

Registry discovery stays lazy and imports no model runtime.

```python
from voicehub import list_model_specs

for model in list_model_specs():
    print(model.task.value, model.display_name, model.model_type)
```

Use the [training matrix](../training-support.md) and
[optimization catalog](../../optimizations/index.md) for deeper comparisons.

Generated by `scripts/generate_model_pages.py` from lazy registry metadata.
