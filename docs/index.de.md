---
description: VoiceHub-Dokumentation für einheitliche TTS-Inferenz, Datenaufbereitung und architekturspezifisches Fine-Tuning.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: Text-zu-Sprache-Inferenz und Training

<p class="vh-doc-tagline">
  Eine Python-Bibliothek mit integriertem Quellcode für Inferenz,
  Datenaufbereitung und modellspezifisches Fine-Tuning moderner TTS-Familien.
</p>

<div class="vh-doc-teaser" role="img" aria-label="Text durchläuft einen VoiceHub-Modelladapter und wird zu einer Audiowellenform">
  <div class="vh-doc-teaser__label">
    <strong>TEXT</strong>
    <span>„Eine klare, natürliche Stimme.“</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>MODELLADAPTER</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">AUDIO</span>
</div>

<section class="vh-home-models" aria-labelledby="vh-home-models-title">
  <p class="vh-home-models__eyebrow">Modellkatalog</p>
  <h2 id="vh-home-models-title">Finden Sie ein Modell für Ihre Sprache und Aufgabe</h2>
  <p class="vh-home-models__description">Durchsuchen Sie alle 68 TTS-, ASR- und VAD-Integrationen nach Sprache, Funktion, Trainingspfad, Lizenz, Architektur und Checkpoint-Quelle.</p>
  <p class="vh-home-models__actions">
    <a class="vh-home-models__primary" href="models/providers/">Alle Modelle entdecken <span aria-hidden="true">→</span></a>
    <a class="vh-home-models__secondary" href="models/training-support/">Trainingsunterstützung vergleichen</a>
  </p>
  <ul class="vh-home-models__stats" aria-label="Zusammenfassung des Modellregisters">
    <li><strong>68</strong><span>Modelle</span></li>
    <li><strong>34</strong><span>TTS</span></li>
    <li><strong>23</strong><span>ASR</span></li>
    <li><strong>11</strong><span>VAD</span></li>
  </ul>
</section>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="Status der kontinuierlichen Integration von VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="Status des VoiceHub-Dokumentations-Builds">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub unterstützt Python 3.10 und neuere Versionen">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub wird unter der Apache-2.0-Lizenz bereitgestellt">
  </a>
</p>

## Was ist VoiceHub?

VoiceHub stellt TTS-, Spracherkennungs- und Sprachaktivitätsmodelle über
gemeinsame APIs für Konfiguration, Verarbeitung, Modelle, Ergebnisse und
Training bereit. Die Implementierungen berücksichtigen weiterhin ihre jeweilige
Architektur: Codec-Sprachmodelle, Sequence-to-Sequence-Systeme, Flow-Matching-
und Diffusionsmodelle, akustische Modelle, adversarielle Systeme im VITS-Stil
und zusammengesetzte Pipelines behalten ihre eigene Konditionierung, ihre
Zielgrößen, die Zuständigkeit für Parameter und ihre Exportregeln bei.

Die Registry enthält **34 TTS-Integrationen**, **23 ASR-Anbieter** und **11
VAD-Anbieter**. Die Fine-Tuning-Unterstützung hängt vom Checkpoint und von der
Laufzeitumgebung ab; eine Inferenzintegration bedeutet nicht, dass ihr
aktuelles VoiceHub-Artefakt differenzierbar ist. Wählen Sie eine Integration
mithilfe des [TTS-Modellkatalogs](models/index.md), der
[TTS-Trainingsmatrix](models/training-support.md) und der
[ASR/VAD-Unterstützungsmatrix](models/asr-vad-support.md) aus.

Der Modellquellcode und alle integrierten TTS-, ASR- und
VAD-Inferenzlaufzeiten werden standardmäßig mit VoiceHub installiert.
Checkpoint-Gewichte werden weiterhin bei Bedarf heruntergeladen oder über
lokale Pfade bereitgestellt. Fügen Sie nur `voicehub[training]` für
Fine-Tuning und Reporting hinzu. Die Apache-2.0-Lizenz gilt nur für VoiceHub
selbst; integrierter Quellcode, Checkpoints, Codecs, Datensätze und erzeugte
Audiodateien können anderen Bedingungen unterliegen.

<div class="grid cards" markdown>

-   **Erste Schritte**

    ---

    Installieren Sie VoiceHub aus dem aktuellen Quellbaum und führen Sie die
    erste Generierungsanfrage über die gemeinsame Model Factory aus.

    [Schnellstart](getting-started/quickstart.md)

-   **Inferenz**

    ---

    Entdecken Sie Integrationen, laden Sie Checkpoints aus Hub oder von lokalen
    Pfaden, konfigurieren Sie reproduzierbare Generierung und verarbeiten Sie
    normalisierte Audiodaten.

    [Inferenzleitfaden](guides/inference.md)

-   **Datenaufbereitung**

    ---

    Erstellen Sie überprüfbare Manifeste, validieren Sie Audiodaten, verhindern
    Sie Datenlecks zwischen Sprechern oder Sitzungen und erzeugen Sie
    modellspezifische Trainingseingaben.

    [Leitfaden zur Datenaufbereitung](guides/data-preparation.md)

-   **Training**

    ---

    Validieren Sie Checkpoint-Grenzen, führen Sie native Zielfunktionen aus,
    evaluieren Sie Modelle, setzen Sie vollständige Checkpoints fort und
    speichern Sie portable Artefakte.

    [Trainingsleitfaden](guides/training.md)

-   **Modelle**

    ---

    Vergleichen Sie TTS-Registry-Einträge, Standard-Checkpoints, Funktionen,
    Herkunft des Quellcodes und Einschränkungen.

    [Modellkatalog](models/index.md)

-   **Trainingsunterstützung**

    ---

    Prüfen Sie für jede Integration genau, ob Fine-Tuning mit Rohdaten oder
    vorverarbeiteten Daten möglich ist, eine spezialisierte Verarbeitung
    erfordert oder nicht verfügbar ist.

    [Trainingsmatrix](models/training-support.md)

-   **Notebooks**

    ---

    Nutzen Sie vier Notebooks: fokussierte Beispiele für Inferenz,
    Datenaufbereitung und Training sowie den vollständigen Dia-Workflow bis
    zum Export und erneuten Laden in einer frischen Laufzeitumgebung.

    [Notebook-Galerie öffnen](guides/notebook.md)

-   **API-Referenz**

    ---

    Schlagen Sie Factories, Ergebnisse, Trainer-Argumente, Callbacks,
    Collators, Strategien, Artefakte und Erweiterungs-Registries nach.

    [API durchsuchen](reference/api.md)

-   **Architektur**

    ---

    Lernen Sie Registry, Model Wrapper, Adapter, Laufzeitstrategien, Checkpoints
    und die Grenzen portabler Artefakte kennen.

    [Bibliotheksarchitektur](concepts/architecture.md)

-   **Modell hinzufügen**

    ---

    Implementieren und testen Sie einen lazy geladenen Wrapper, eine
    Trainingsspezifikation, bei Bedarf einen spezialisierten Adapter sowie
    einen Exportvertrag.

    [Leitfaden zur Modellintegration](project/adding-a-model.md)

</div>

</div>
