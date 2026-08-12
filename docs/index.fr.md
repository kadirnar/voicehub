---
description: Documentation de VoiceHub pour l'inférence TTS unifiée, la préparation des données et l'ajustement adapté à chaque architecture.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub : inférence et entraînement de synthèse vocale

<p class="vh-doc-tagline">
  Une bibliothèque Python intégrée au code source pour l'inférence, la
  préparation des données et l'ajustement propre à chaque modèle dans les familles TTS modernes.
</p>

<div class="vh-doc-teaser" role="img" aria-label="Le texte traverse un adaptateur de modèle VoiceHub et devient une forme d'onde audio">
  <div class="vh-doc-teaser__label">
    <strong>TEXTE</strong>
    <span>« Une voix claire et naturelle. »</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>ADAPTATEUR DE MODÈLE</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">AUDIO</span>
</div>

<section class="vh-home-models" aria-labelledby="vh-home-models-title">
  <p class="vh-home-models__eyebrow">Catalogue de modèles</p>
  <h2 id="vh-home-models-title">Trouvez un modèle pour votre langue et votre tâche</h2>
  <p class="vh-home-models__description">Explorez les 68 intégrations TTS, ASR et VAD par langue, capacité, parcours d’entraînement, licence, architecture et source de checkpoint.</p>
  <p class="vh-home-models__actions">
    <a class="vh-home-models__primary" href="models/providers/">Explorer tous les modèles <span aria-hidden="true">→</span></a>
    <a class="vh-home-models__secondary" href="models/training-support/">Comparer la prise en charge de l’entraînement</a>
  </p>
  <ul class="vh-home-models__stats" aria-label="Résumé du registre de modèles">
    <li><strong>68</strong><span>Modèles</span></li>
    <li><strong>34</strong><span>TTS</span></li>
    <li><strong>23</strong><span>ASR</span></li>
    <li><strong>11</strong><span>VAD</span></li>
  </ul>
</section>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="État de l'intégration continue de VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="État de la compilation de la documentation VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub prend en charge Python 3.10 et les versions ultérieures">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub est distribué sous licence Apache 2.0">
  </a>
</p>

## Qu'est-ce que VoiceHub ?

VoiceHub présente les modèles TTS, de reconnaissance automatique de la parole
et de détection d'activité vocale au moyen d'API communes pour la
configuration, le traitement, les modèles, les résultats et l'entraînement.
Les implémentations restent adaptées à leur
architecture : les modèles de langage à codec, les systèmes séquence à
séquence, les modèles de flow matching et de diffusion, les modèles
acoustiques, les systèmes antagonistes de type VITS et les pipelines composites
conservent leurs propres conditionnements, objectifs, règles de propriété des
paramètres et règles d'exportation.

Le registry contient **34 intégrations TTS**, **23 fournisseurs ASR** et **11
fournisseurs VAD**. La prise en charge de l'ajustement dépend du checkpoint et
de l'environnement d'exécution ; une intégration d'inférence ne garantit pas
que son artefact VoiceHub actuel soit différentiable. Consultez le
[catalogue TTS](models/index.md), la
[matrice d'entraînement TTS](models/training-support.md) et la
[matrice de prise en charge ASR/VAD](models/asr-vad-support.md) pour choisir
une intégration.

Le code source des modèles et tous les environnements d'inférence TTS, ASR et
VAD intégrés sont installés par défaut avec VoiceHub. Les poids des checkpoints
sont téléchargés à la demande ou fournis sous forme de chemins locaux. Ajoutez
uniquement `voicehub[training]` pour l'ajustement et le reporting. La licence
Apache-2.0 couvre uniquement VoiceHub ; le code intégré, les checkpoints, les
codecs, les jeux de données et les fichiers audio générés peuvent être soumis
à des conditions distinctes.

<div class="grid cards" markdown>

-   **Bien démarrer**

    ---

    Installez VoiceHub depuis l'arborescence source actuelle et exécutez votre
    première requête de génération avec le model factory partagé.

    [Démarrage rapide](getting-started/quickstart.md)

-   **Inférence**

    ---

    Découvrez les intégrations, chargez des checkpoints depuis Hub ou des
    chemins locaux, configurez une génération reproductible et exploitez un
    signal audio normalisé.

    [Guide d'inférence](guides/inference.md)

-   **Préparation des données**

    ---

    Créez des manifests auditables, validez le signal audio, empêchez les fuites
    entre locuteurs ou sessions et générez des entrées d'entraînement propres
    à chaque modèle.

    [Guide de préparation des données](guides/data-preparation.md)

-   **Entraînement**

    ---

    Validez les limites des checkpoints, exécutez les objectifs natifs, évaluez,
    reprenez à partir de checkpoints complets et enregistrez des artefacts
    portables.

    [Guide d'entraînement](guides/training.md)

-   **Modèles**

    ---

    Comparez les entrées TTS du registry, leurs checkpoints par défaut, leurs
    capacités, la provenance du code source et leurs contraintes.

    [Catalogue des modèles](models/index.md)

-   **Prise en charge de l'entraînement**

    ---

    Vérifiez précisément, pour chaque intégration, si l'ajustement accepte des
    données brutes ou prétraitées, requiert un traitement spécialisé ou n'est
    pas disponible.

    [Matrice d'entraînement](models/training-support.md)

-   **Notebooks**

    ---

    Utilisez quatre notebooks : des exemples ciblés d'inférence, de préparation
    des données et d'entraînement, ainsi que le workflow Dia complet jusqu'à
    l'exportation et au rechargement dans un nouvel environnement d'exécution.

    [Ouvrir la galerie de notebooks](guides/notebook.md)

-   **Référence de l'API**

    ---

    Consultez les factories, les résultats, les arguments du trainer, les
    callbacks, les collators, les stratégies, les artefacts et les registries
    d'extensions.

    [Parcourir l'API](reference/api.md)

-   **Architecture**

    ---

    Comprenez le registry, les model wrappers, les adaptateurs, les stratégies
    d'exécution, les checkpoints et les limites des artefacts portables.

    [Architecture de la bibliothèque](concepts/architecture.md)

-   **Ajouter un modèle**

    ---

    Implémentez et testez un wrapper lazy, une spécification d'entraînement, un
    adaptateur spécialisé si nécessaire et un contrat d'exportation.

    [Guide d'intégration d'un modèle](project/adding-a-model.md)

</div>

</div>
