---
description: Documentação do VoiceHub para inferência TTS unificada, preparação de dados e ajuste fino orientado à arquitetura.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: inferência e treinamento de síntese de voz

<p class="vh-doc-tagline">
  Uma biblioteca Python integrada ao código-fonte para inferência, preparação
  de dados e ajuste fino específico para diferentes famílias modernas de TTS.
</p>

<div class="vh-doc-teaser" role="img" aria-label="O texto passa por um adaptador de modelo do VoiceHub e se transforma em uma forma de onda de áudio">
  <div class="vh-doc-teaser__label">
    <strong>TEXTO</strong>
    <span>“Uma voz clara e natural.”</span>
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
  <span class="vh-doc-teaser__audio">ÁUDIO</span>
</div>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="Status da integração contínua do VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="Status da compilação da documentação do VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="O VoiceHub é compatível com Python 3.10 ou posterior">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="O VoiceHub é licenciado sob a Apache 2.0">
  </a>
</p>

## O que é o VoiceHub?

O VoiceHub disponibiliza modelos de TTS, reconhecimento automático de fala e
detecção de atividade de voz por meio de APIs compartilhadas de configuração,
processador, modelo, saída e treinador. As implementações permanecem cientes da arquitetura: modelos de
linguagem com codec, sistemas de sequência para sequência, modelos de
flow-matching e difusão, modelos acústicos, sistemas adversariais no estilo
VITS e pipelines compostos mantêm seus próprios condicionamentos, objetivos,
regras de propriedade de parâmetros e de exportação.

O registro contém **34 integrações TTS**, **23 provedores ASR** e **11 provedores
VAD**. O suporte a ajuste fino depende do checkpoint e do ambiente de execução;
uma integração de inferência não implica que seu artefato atual do VoiceHub
seja diferenciável. Consulte a [matriz de treinamento TTS](models/training-support.md) e a
[matriz de suporte ASR/VAD](models/asr-vad-support.md) para selecionar uma
integração.

O código-fonte dos modelos e todos os ambientes de inferência TTS, ASR e VAD
integrados são instalados por padrão com o VoiceHub. Os pesos dos checkpoints
são baixados sob demanda ou fornecidos por caminhos locais. Adicione apenas
`voicehub[training]` para ajuste fino e relatórios. A licença Apache-2.0
abrange o próprio VoiceHub; código integrado, checkpoints, codecs, conjuntos
de dados e áudio gerado podem estar sujeitos a termos distintos.

<div class="grid cards" markdown>

-   **Primeiros passos**

    ---

    Instale o VoiceHub a partir da árvore de código-fonte atual e execute a
    primeira solicitação de geração usando a fábrica de modelos compartilhada.

    [Início rápido](getting-started/quickstart.md)

-   **Inferência**

    ---

    Encontre integrações, carregue checkpoints do Hub ou locais, configure uma
    geração reproduzível e consuma áudio normalizado.

    [Guia de inferência](guides/inference.md)

-   **Preparação de dados**

    ---

    Crie manifestos auditáveis, valide o áudio, evite vazamento entre locutores
    ou sessões e produza entradas de treinamento específicas para cada modelo.

    [Guia de preparação de dados](guides/data-preparation.md)

-   **Treinamento**

    ---

    Valide os limites dos checkpoints, execute objetivos nativos, avalie,
    retome checkpoints completos e salve artefatos portáteis.

    [Guia de treinamento](guides/training.md)

-   **Suporte a treinamento**

    ---

    Verifique, para cada integração, o limite exato de ajuste fino com dados
    brutos, dados pré-processados, fluxo especializado ou sem suporte.

    [Matriz de treinamento](models/training-support.md)

-   **Notebooks**

    ---

    Use quatro notebooks: exemplos focados em inferência, preparação de dados
    e treinamento, além do fluxo de trabalho completo do Dia até a exportação
    e o recarregamento em um novo ambiente de execução.

    [Abrir a galeria de notebooks](guides/notebook.md)

-   **Referência da API**

    ---

    Consulte fábricas, saídas, argumentos do treinador, callbacks, collators,
    estratégias, artefatos e registros de extensão.

    [Explorar a API](reference/api.md)

-   **Arquitetura**

    ---

    Entenda o registro, os wrappers de modelo, os adaptadores, as estratégias de
    execução, os checkpoints e os limites dos artefatos portáteis.

    [Arquitetura da biblioteca](concepts/architecture.md)

-   **Adicionar um modelo**

    ---

    Implemente e teste um wrapper com carregamento tardio, uma especificação de
    treinamento, um adaptador especializado quando necessário e um contrato de
    exportação.

    [Guia de integração de modelos](project/adding-a-model.md)

</div>

</div>
