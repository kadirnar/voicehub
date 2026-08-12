---
description: Documentación de VoiceHub para inferencia TTS unificada, preparación de datos y ajuste fino adaptado a cada arquitectura.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: inferencia y entrenamiento de texto a voz

<p class="vh-doc-tagline">
  Una biblioteca de Python integrada con el código fuente para inferencia,
  preparación de datos y ajuste fino específico de cada modelo en familias TTS modernas.
</p>

<div class="vh-doc-teaser" role="img" aria-label="El texto pasa por un adaptador de modelo de VoiceHub y se convierte en una onda de audio">
  <div class="vh-doc-teaser__label">
    <strong>TEXTO</strong>
    <span>“Una voz clara y natural.”</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>ADAPTADOR DE MODELO</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">AUDIO</span>
</div>

<section class="vh-home-models" aria-labelledby="vh-home-models-title">
  <p class="vh-home-models__eyebrow">Catálogo de modelos</p>
  <h2 id="vh-home-models-title">Encuentre un modelo para su idioma y tarea</h2>
  <p class="vh-home-models__description">Busque las 68 integraciones de TTS, ASR y VAD por idioma, capacidad, ruta de entrenamiento, licencia, arquitectura y origen del checkpoint.</p>
  <p class="vh-home-models__actions">
    <a class="vh-home-models__primary" href="models/providers/">Explorar todos los modelos <span aria-hidden="true">→</span></a>
    <a class="vh-home-models__secondary" href="models/training-support/">Comparar el soporte de entrenamiento</a>
  </p>
  <ul class="vh-home-models__stats" aria-label="Resumen del registro de modelos">
    <li><strong>68</strong><span>Modelos</span></li>
    <li><strong>34</strong><span>TTS</span></li>
    <li><strong>23</strong><span>ASR</span></li>
    <li><strong>11</strong><span>VAD</span></li>
  </ul>
</section>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="Estado de la integración continua de VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="Estado de la compilación de la documentación de VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub es compatible con Python 3.10 y versiones posteriores">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub se distribuye bajo la licencia Apache 2.0">
  </a>
</p>

## ¿Qué es VoiceHub?

VoiceHub presenta modelos de TTS, reconocimiento automático de voz y detección
de actividad de voz mediante API compartidas de configuración, procesamiento,
modelo, resultados y entrenamiento. Las implementaciones conservan las particularidades de cada
arquitectura: los modelos de lenguaje con codec, los sistemas de secuencia a
secuencia, los modelos de flow matching y difusión, los modelos acústicos, los
sistemas adversariales de tipo VITS y los pipelines compuestos mantienen sus
propios condicionamientos, objetivos, propiedad de parámetros y reglas de
exportación.

El registry contiene **34 integraciones TTS**, **23 proveedores ASR** y **11
proveedores VAD**. La compatibilidad con el ajuste fino depende del checkpoint
y del entorno de ejecución; disponer de una integración de inferencia no
implica que su artefacto actual de VoiceHub sea diferenciable. Consulte el
[catálogo TTS](models/index.md), la
[matriz de entrenamiento TTS](models/training-support.md) y la
[matriz de compatibilidad ASR/VAD](models/asr-vad-support.md) para seleccionar
una integración.

El código fuente de los modelos y todos los entornos de inferencia TTS, ASR y
VAD integrados se instalan de forma predeterminada con VoiceHub. Los pesos de
los checkpoints se descargan bajo demanda o se proporcionan mediante rutas
locales. Añada únicamente `voicehub[training]` para el ajuste fino y los
informes. La licencia Apache-2.0 cubre únicamente VoiceHub; el código
integrado, los checkpoints, los codecs, los conjuntos de datos y el audio
generado pueden estar sujetos a condiciones distintas.

<div class="grid cards" markdown>

-   **Primeros pasos**

    ---

    Instale VoiceHub desde el árbol de código fuente actual y ejecute la primera
    solicitud de generación mediante el model factory compartido.

    [Inicio rápido](getting-started/quickstart.md)

-   **Inferencia**

    ---

    Descubra integraciones, cargue checkpoints desde Hub o rutas locales,
    configure una generación reproducible y utilice audio normalizado.

    [Guía de inferencia](guides/inference.md)

-   **Preparación de datos**

    ---

    Cree manifests auditables, valide el audio, evite fugas entre hablantes o
    sesiones y genere entradas de entrenamiento específicas para cada modelo.

    [Guía de preparación de datos](guides/data-preparation.md)

-   **Entrenamiento**

    ---

    Valide los límites de los checkpoints, ejecute objetivos nativos, evalúe,
    reanude checkpoints completos y guarde artefactos portátiles.

    [Guía de entrenamiento](guides/training.md)

-   **Modelos**

    ---

    Compare las entradas TTS del registry, los checkpoints predeterminados,
    las capacidades, la procedencia del código fuente y las restricciones.

    [Catálogo de modelos](models/index.md)

-   **Compatibilidad de entrenamiento**

    ---

    Consulte el límite exacto del ajuste fino con datos sin procesar,
    preprocesados, especializados o no disponibles para cada integración.

    [Matriz de entrenamiento](models/training-support.md)

-   **Notebooks**

    ---

    Use cuatro notebooks: ejemplos centrados en inferencia, preparación de
    datos y entrenamiento, además del flujo de trabajo completo de Dia hasta
    la exportación y la recarga en un entorno nuevo.

    [Abrir la galería de notebooks](guides/notebook.md)

-   **Referencia de la API**

    ---

    Consulte factories, resultados, argumentos del trainer, callbacks,
    collators, estrategias, artefactos y registries de extensiones.

    [Explorar la API](reference/api.md)

-   **Arquitectura**

    ---

    Comprenda el registry, los model wrappers, los adaptadores, las estrategias
    de ejecución, los checkpoints y los límites de los artefactos portátiles.

    [Arquitectura de la biblioteca](concepts/architecture.md)

-   **Añadir un modelo**

    ---

    Implemente y pruebe un wrapper lazy, una especificación de entrenamiento,
    un adaptador especializado cuando sea necesario y un contrato de
    exportación.

    [Guía de integración de modelos](project/adding-a-model.md)

</div>

</div>
