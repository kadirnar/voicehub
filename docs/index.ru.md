---
description: Документация VoiceHub по унифицированному TTS-инференсу, подготовке данных и дообучению с учетом архитектуры модели.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: инференс и обучение моделей синтеза речи

<p class="vh-doc-tagline">
  Python-библиотека с интегрированным исходным кодом моделей для инференса,
  подготовки данных и специализированного дообучения современных семейств TTS.
</p>

<div class="vh-doc-teaser" role="img" aria-label="Текст проходит через адаптер модели VoiceHub и преобразуется в звуковую волну">
  <div class="vh-doc-teaser__label">
    <strong>ТЕКСТ</strong>
    <span>«Чистый, естественный голос».</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>АДАПТЕР МОДЕЛИ</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">АУДИО</span>
</div>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="Статус непрерывной интеграции VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="Статус сборки документации VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub поддерживает Python 3.10 и более поздние версии">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub распространяется по лицензии Apache 2.0">
  </a>
</p>

## Что такое VoiceHub?

VoiceHub предоставляет модели TTS, автоматического распознавания речи и
детекции речевой активности через унифицированные API конфигурации, процессора,
модели, результата и тренера. Реализации
сохраняют особенности каждой архитектуры: языковые модели с кодеками, системы
sequence-to-sequence, модели flow-matching и диффузионные модели, акустические
модели, состязательные системы в стиле VITS и составные пайплайны используют
собственные способы кондиционирования, целевые функции, правила управления
параметрами и экспорта.

Реестр содержит **34 интеграции TTS**, **23 провайдеров ASR** и **11 провайдеров
VAD**. Возможность дообучения зависит от конкретных checkpoint и среды
выполнения; поддержка инференса не означает, что текущий артефакт VoiceHub
поддерживает дифференцируемое обучение. Для выбора интеграции используйте
[матрицу обучения TTS](models/training-support.md) и [матрицу поддержки
ASR/VAD](models/asr-vad-support.md).

Исходный код моделей и все встроенные среды инференса TTS, ASR и VAD входят в
стандартную установку VoiceHub. Веса checkpoint загружаются по мере
необходимости или указываются через локальные пути. Добавляйте
`voicehub[training]` только для дообучения и отчётности. Лицензия Apache-2.0
распространяется на сам VoiceHub; для интегрированного исходного кода,
checkpoint, кодеков, наборов данных и сгенерированного аудио могут действовать
отдельные условия.

<div class="grid cards" markdown>

-   **Начало работы**

    ---

    Установите VoiceHub из текущего дерева исходного кода и выполните первый
    запрос генерации через общую фабрику моделей.

    [Краткое руководство](getting-started/quickstart.md)

-   **Инференс**

    ---

    Находите интеграции, загружайте checkpoint из Hub или локального хранилища,
    настраивайте воспроизводимую генерацию и получайте нормализованное аудио.

    [Руководство по инференсу](guides/inference.md)

-   **Подготовка данных**

    ---

    Создавайте пригодные для аудита манифесты, проверяйте аудио, предотвращайте
    утечку данных между дикторами или сессиями и формируйте входные данные для
    обучения конкретной модели.

    [Руководство по подготовке данных](guides/data-preparation.md)

-   **Обучение**

    ---

    Проверяйте состав и границы checkpoint, выполняйте нативные целевые функции,
    оценивайте модель, возобновляйте работу с полных checkpoint и сохраняйте
    переносимые артефакты.

    [Руководство по обучению](guides/training.md)

-   **Поддержка обучения**

    ---

    Проверяйте точные границы дообучения для каждой интеграции: на
    необработанных или предобработанных данных, по специализированной схеме
    либо без поддержки.

    [Матрица обучения](models/training-support.md)

-   **Jupyter-ноутбуки**

    ---

    Используйте четыре Jupyter-ноутбука: тематические примеры по инференсу,
    подготовке данных и обучению, а также полный рабочий процесс Dia вплоть до
    экспорта и повторной загрузки в новой среде выполнения.

    [Открыть галерею ноутбуков](guides/notebook.md)

-   **Справочник API**

    ---

    Изучите фабрики, выходные данные, аргументы тренера, callbacks, collators,
    стратегии, артефакты и реестры расширений.

    [Открыть справочник API](reference/api.md)

-   **Архитектура**

    ---

    Разберитесь в устройстве реестра, оберток моделей, адаптеров, стратегий
    выполнения, checkpoint и границ переносимых артефактов.

    [Архитектура библиотеки](concepts/architecture.md)

-   **Добавление модели**

    ---

    Реализуйте и протестируйте обертку с отложенной загрузкой, спецификацию
    обучения, специализированный адаптер при необходимости и контракт экспорта.

    [Руководство по интеграции модели](project/adding-a-model.md)

</div>

</div>
