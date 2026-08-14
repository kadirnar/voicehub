---
description: وثائق VoiceHub للاستدلال الموحّد لأنظمة TTS، وإعداد البيانات، والضبط الدقيق المراعي للبنية المعمارية.
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub: الاستدلال والتدريب لتحويل النص إلى كلام

<p class="vh-doc-tagline">
  مكتبة Python تدمج الشيفرة المصدرية لتوفير الاستدلال وإعداد البيانات
  والضبط الدقيق الخاص بكل نموذج عبر عائلات TTS الحديثة.
</p>

<div class="vh-doc-teaser" role="img" aria-label="يمر النص عبر مهايئ نموذج VoiceHub ويتحول إلى شكل موجي صوتي">
  <div class="vh-doc-teaser__label">
    <strong>نص</strong>
    <span>«صوت واضح وطبيعي.»</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>مهايئ النموذج</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">صوت</span>
</div>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="حالة التكامل المستمر لـ VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="حالة بناء وثائق VoiceHub">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="يدعم VoiceHub الإصدار Python 3.10 والإصدارات الأحدث">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="يُرخَّص VoiceHub بموجب Apache 2.0">
  </a>
</p>

## ما هو VoiceHub؟

يقدّم VoiceHub نماذج TTS والتعرّف التلقائي على الكلام واكتشاف النشاط الصوتي من
خلال واجهات API مشتركة للتكوين والمعالج والنموذج والمخرجات والمدرّب. وتظل تطبيقات النماذج
مراعية لبنيتها المعمارية: إذ تحتفظ نماذج اللغة المعتمدة على codec، وأنظمة
sequence-to-sequence، ونماذج flow-matching والانتشار، والنماذج الصوتية،
والأنظمة التنافسية بأسلوب VITS، وخطوط المعالجة المركبة بأساليب التكييف
والأهداف وملكية المعلمات وقواعد التصدير الخاصة بها.

يضم registry **34 تكامل TTS** و**23 مزوّدي ASR** و**11 مزوّدي VAD**. يعتمد دعم
الضبط الدقيق على نقطة التحقق وبيئة التشغيل تحديدًا؛ فوجود تكامل للاستدلال لا
يعني أن artifact الحالي الخاص به في VoiceHub قابل للاشتقاق. استخدم
[مصفوفة تدريب TTS](models/training-support.md) و
[مصفوفة دعم ASR/VAD](models/asr-vad-support.md) لاختيار التكامل المناسب.

تُثبّت الشيفرة المصدرية للنماذج وجميع بيئات استدلال TTS وASR وVAD المدمجة
مع VoiceHub افتراضيًا، بينما تُنزّل أوزان نقاط التحقق عند الحاجة أو تُوفّر
عبر مسارات محلية. أضف `voicehub[training]` فقط للضبط الدقيق وإعداد التقارير.
لا يغطي ترخيص Apache-2.0 سوى VoiceHub نفسه؛ وقد تخضع الشيفرة المصدرية
المدمجة ونقاط التحقق وبرامج codec ومجموعات البيانات والصوت المُنشأ لشروط
منفصلة.

<div class="grid cards" markdown>

-   **البدء**

    ---

    ثبّت VoiceHub من شجرة المصدر الحالية، ونفّذ أول طلب توليد عبر
    model factory المشترك.

    [البدء السريع](getting-started/quickstart.md)

-   **الاستدلال**

    ---

    اكتشف التكاملات، وحمّل نقاط تحقق من Hub أو من مسارات محلية، واضبط
    عملية توليد قابلة لإعادة الإنتاج، واستخدم الصوت المُطبّع.

    [دليل الاستدلال](guides/inference.md)

-   **إعداد البيانات**

    ---

    أنشئ ملفات manifest قابلة للتدقيق، وتحقق من صحة الصوت، وامنع تسرّب
    البيانات بين المتحدثين أو الجلسات، وأنشئ مدخلات تدريب خاصة بكل نموذج.

    [دليل إعداد البيانات](guides/data-preparation.md)

-   **التدريب**

    ---

    تحقّق من حدود نقاط التحقق، ونفّذ الأهداف الأصلية، وقيّم النتائج،
    واستأنف العمل من نقاط تحقق كاملة، واحفظ artifacts قابلة للنقل.

    [دليل التدريب](guides/training.md)

-   **دعم التدريب**

    ---

    تحقّق بدقة من نطاق الضبط الدقيق لكل تكامل: هل يدعم البيانات الأولية
    أو المعالجة مسبقًا، أو يتطلب مسارًا متخصصًا، أو أنه غير متاح.

    [مصفوفة التدريب](models/training-support.md)

-   **دفاتر Notebook**

    ---

    استخدم أربعة دفاتر Notebook: أمثلة مركّزة للاستدلال وإعداد البيانات
    والتدريب، أو اتبع سير عمل Dia الكامل حتى التصدير وإعادة التحميل في بيئة
    تشغيل جديدة.

    [افتح معرض دفاتر Notebook](guides/notebook.md)

-   **مرجع API**

    ---

    ابحث في factories والمخرجات ووسائط trainer وcallbacks وcollators
    والاستراتيجيات وartifacts وسجلات الامتدادات.

    [تصفّح API](reference/api.md)

-   **البنية المعمارية**

    ---

    تعرّف إلى registry وmodel wrappers والمهايئات واستراتيجيات وقت التشغيل
    ونقاط التحقق وحدود artifacts القابلة للنقل.

    [بنية المكتبة](concepts/architecture.md)

-   **إضافة نموذج**

    ---

    نفّذ واختبر lazy wrapper ومواصفات التدريب، وأضف مهايئًا متخصصًا عند
    الحاجة، وحدّد عقد التصدير.

    [دليل تكامل النماذج](project/adding-a-model.md)

</div>

</div>
