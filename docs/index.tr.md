---
description: Birleşik TTS çıkarımı, veri hazırlama ve mimariye duyarlı ince ayar için VoiceHub belgeleri.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: Metinden Sese Çıkarım ve Eğitim

<p class="vh-doc-tagline">
  Modern TTS ailelerinde çıkarım, veri hazırlama ve modele özgü ince ayar için
  kaynak koduyla birlikte sunulan bir Python kütüphanesi.
</p>

<div class="vh-doc-teaser" role="img" aria-label="Metin, bir VoiceHub model adaptöründen geçerek ses dalga biçimine dönüşür">
  <div class="vh-doc-teaser__label">
    <strong>METİN</strong>
    <span>“Net ve doğal bir ses.”</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>MODEL ADAPTÖRÜ</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">SES</span>
</div>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="VoiceHub sürekli entegrasyon durumu">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="VoiceHub belge derleme durumu">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub, Python 3.10 ve sonraki sürümleri destekler">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub, Apache 2.0 lisansı altında sunulur">
  </a>
</p>

## VoiceHub nedir?

VoiceHub; TTS, otomatik konuşma tanıma ve ses etkinliği algılama modellerini
ortak yapılandırma, işlemci, model, çıktı ve trainer API'leri üzerinden sunar. Model uygulamaları
mimarilerinin gereksinimlerini korur: codec dil modelleri, diziden diziye
sistemler, flow-matching ve difüzyon modelleri, akustik modeller, VITS tarzı
çekişmeli sistemler ve bileşik pipeline'lar kendi koşullandırma yöntemlerini,
amaç fonksiyonlarını, parametre sahipliğini ve dışa aktarma kurallarını korur.

Registry **34 TTS entegrasyonu**, **23 ASR sağlayıcısı** ve **11 VAD
sağlayıcısı** içerir. İnce ayar desteği kontrol noktasına ve çalışma ortamına
özeldir; bir çıkarım entegrasyonunun bulunması, mevcut VoiceHub model
çıktısının türevlenebilir olduğu anlamına gelmez. Uygun entegrasyonu seçmek
için [TTS eğitim matrisini](models/training-support.md) ve
[ASR/VAD destek matrisini](models/asr-vad-support.md) kullanın.

Model kaynak kodu ile yerleşik TTS, ASR ve VAD çıkarım çalışma ortamlarının
tamamı varsayılan VoiceHub kurulumuna dahildir. Kontrol noktası ağırlıkları
gerektiğinde indirilir veya yerel yollar üzerinden sağlanır. İnce ayar ve
raporlama için yalnızca `voicehub[training]` ekleyin. Apache-2.0 lisansı
yalnızca VoiceHub'ı kapsar; entegre kaynak kodu, kontrol noktaları, codec'ler,
veri kümeleri ve üretilen sesler farklı koşullara tabi olabilir.

<div class="grid cards" markdown>

-   **Başlarken**

    ---

    VoiceHub'ı mevcut kaynak ağacından kurun ve paylaşılan model factory
    üzerinden ilk üretim isteğinizi çalıştırın.

    [Hızlı başlangıç](getting-started/quickstart.md)

-   **Çıkarım**

    ---

    Entegrasyonları keşfedin, Hub veya yerel kontrol noktalarını yükleyin,
    yeniden üretilebilir üretimi yapılandırın ve normalleştirilmiş sesi kullanın.

    [Çıkarım rehberi](guides/inference.md)

-   **Veri hazırlama**

    ---

    Denetlenebilir manifest'ler oluşturun, sesi doğrulayın, konuşmacı veya
    oturum sızıntısını önleyin ve modele özgü eğitim girdileri hazırlayın.

    [Veri hazırlama rehberi](guides/data-preparation.md)

-   **Eğitim**

    ---

    Kontrol noktası sınırlarını doğrulayın, özgün amaç fonksiyonlarını
    çalıştırın, değerlendirin, eksiksiz kontrol noktalarından devam edin ve
    taşınabilir model çıktıları kaydedin.

    [Eğitim rehberi](guides/training.md)

-   **Eğitim desteği**

    ---

    Her entegrasyon için ham veri, önceden işlenmiş veri, özelleştirilmiş veya
    kullanılamayan ince ayar sınırını tam olarak inceleyin.

    [Eğitim matrisi](models/training-support.md)

-   **Notebook galerisi**

    ---

    Dört notebook kullanın: çıkarım, veri hazırlama ve eğitime odaklanan
    örneklerin yanı sıra dışa aktarma ve yeni bir çalışma ortamında yeniden
    yüklemeye kadar eksiksiz Dia iş akışını izleyin.

    [Notebook galerisini açın](guides/notebook.md)

-   **API referansı**

    ---

    Factory'leri, çıktıları, trainer argümanlarını, callback'leri,
    collator'ları, stratejileri, model çıktılarını ve genişletme registry'lerini
    inceleyin.

    [API'ye göz atın](reference/api.md)

-   **Mimari**

    ---

    Registry'yi, model wrapper'larını, adaptörleri, çalışma zamanı
    stratejilerini, kontrol noktalarını ve taşınabilir model çıktısı sınırlarını
    anlayın.

    [Kütüphane mimarisi](concepts/architecture.md)

-   **Model ekleme**

    ---

    Lazy yüklenen bir wrapper, eğitim tanımı, gerektiğinde özelleştirilmiş
    adaptör ve dışa aktarma sözleşmesi geliştirin ve test edin.

    [Model entegrasyonu rehberi](project/adding-a-model.md)

</div>

</div>
