---
description: 통합 TTS 추론, 데이터 준비, 아키텍처 인식 미세 조정을 위한 VoiceHub 문서입니다.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: 텍스트 음성 변환 추론 및 학습

<p class="vh-doc-tagline">
  최신 TTS 모델 계열을 위한 추론, 데이터 준비, 모델별 미세 조정을 제공하는
  소스 코드 통합형 Python 라이브러리입니다.
</p>

<div class="vh-doc-teaser" role="img" aria-label="텍스트가 VoiceHub 모델 어댑터를 거쳐 오디오 파형으로 변환됩니다">
  <div class="vh-doc-teaser__label">
    <strong>텍스트</strong>
    <span>“선명하고 자연스러운 목소리.”</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>모델 어댑터</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">오디오</span>
</div>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="VoiceHub 지속적 통합 상태">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="VoiceHub 문서 빌드 상태">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub는 Python 3.10 이상을 지원합니다">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub는 Apache 2.0 라이선스로 제공됩니다">
  </a>
</p>

## VoiceHub란?

VoiceHub는 공통 설정, 프로세서, 모델, 출력, 트레이너 API를 통해 TTS, 자동 음성
인식, 음성 활동 감지 모델을 제공합니다. 모델 구현은 각 아키텍처의 특성을 그대로
유지합니다. 코덱 언어 모델, 시퀀스-투-시퀀스 시스템, 플로 매칭 및 확산 모델,
음향 모델, VITS 스타일 적대적 시스템, 복합 파이프라인은 각자의 조건화 방식,
목적 함수, 파라미터 소유권, 내보내기 규칙을 유지합니다.

레지스트리에는 **34개의 TTS 통합**, **23개의 ASR 제공자**, **11개의 VAD
제공자**가 있습니다. 미세 조정 지원 여부는 checkpoint와 런타임에 따라
달라집니다. 추론을 지원한다고 해서 현재 VoiceHub 아티팩트가 미분 가능한 것은
아닙니다. 통합 기능을 선택할 때는 [TTS 학습 매트릭스](models/training-support.md)와
[ASR/VAD 지원 매트릭스](models/asr-vad-support.md)를 확인하세요.

모델 소스 코드와 내장 TTS, ASR, VAD 추론 런타임은 모두 VoiceHub 기본 설치에
포함됩니다. checkpoint 가중치는 필요할 때 다운로드하거나 로컬 경로로 제공할 수
있습니다. 미세 조정과 리포팅이 필요할 때만 `voicehub[training]`을 추가하세요.
Apache-2.0 라이선스는 VoiceHub 자체에 적용됩니다. 통합된 소스 코드,
checkpoint, 코덱, 데이터 세트, 생성된 오디오에는 별도의 조건이 적용될 수
있습니다.

<div class="grid cards" markdown>

-   **시작하기**

    ---

    현재 소스 트리에서 VoiceHub를 설치하고 공통 모델 팩토리를 통해 첫 번째 생성
    요청을 실행합니다.

    [빠른 시작](getting-started/quickstart.md)

-   **추론**

    ---

    통합 기능을 찾고, Hub 또는 로컬 checkpoint를 불러오고, 재현 가능한 생성을
    설정하여 정규화된 오디오를 가져옵니다.

    [추론 가이드](guides/inference.md)

-   **데이터 준비**

    ---

    감사 가능한 매니페스트를 만들고, 오디오를 검증하고, 화자 또는 세션 간 누출을
    방지하며, 모델별 학습 입력을 생성합니다.

    [데이터 준비 가이드](guides/data-preparation.md)

-   **학습**

    ---

    checkpoint 경계를 검증하고, 네이티브 목적 함수를 실행하고, 평가하고, 완전한
    checkpoint에서 재개하며, 이식 가능한 아티팩트를 저장합니다.

    [학습 가이드](guides/training.md)

-   **학습 지원**

    ---

    각 통합 기능의 정확한 미세 조정 경계를 원시 데이터, 전처리된 데이터, 전용
    워크플로, 지원되지 않음으로 구분해 확인합니다.

    [학습 매트릭스](models/training-support.md)

-   **Notebook 모음**

    ---

    네 개의 Notebook에서 추론, 데이터 준비, 학습에 초점을 맞춘 예제를
    실행하거나 내보내기와 새로운 런타임에서의 다시 불러오기까지 포함한 전체
    Dia 워크플로를 따라갑니다.

    [Notebook 갤러리 열기](guides/notebook.md)

-   **API 레퍼런스**

    ---

    팩토리, 출력, 트레이너 인자, 콜백, collator, 전략, 아티팩트, 확장
    레지스트리를 찾아봅니다.

    [API 살펴보기](reference/api.md)

-   **아키텍처**

    ---

    레지스트리, 모델 wrapper, 어댑터, 런타임 전략, checkpoint, 이식 가능한
    아티팩트 경계를 이해합니다.

    [라이브러리 아키텍처](concepts/architecture.md)

-   **모델 추가**

    ---

    지연 로딩 wrapper, 학습 사양, 필요한 경우 전용 어댑터, 내보내기 계약을
    구현하고 테스트합니다.

    [모델 통합 가이드](project/adding-a-model.md)

</div>

</div>
