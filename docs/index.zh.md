---
description: VoiceHub 文档：统一的 TTS 推理、数据准备和架构感知微调。
---

<div class="vh-doc-home" markdown>

<p class="vh-doc-logo">
  <img src="../assets/voicehub-mark.svg" alt="">
</p>

# VoiceHub：文本转语音推理与训练

<p class="vh-doc-tagline">
  一款集成模型源码的 Python 库，面向现代 TTS 模型家族提供推理、数据准备和
  模型专用微调能力。
</p>

<div class="vh-doc-teaser" role="img" aria-label="文本经过 VoiceHub 模型适配器后转换为音频波形">
  <div class="vh-doc-teaser__label">
    <strong>文本</strong>
    <span>“清晰、自然的声音。”</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-teaser__model">
    <img src="../assets/voicehub-mark.svg" alt="">
    <strong>VoiceHub</strong>
    <span>模型适配器</span>
  </div>
  <span class="vh-doc-teaser__arrow" aria-hidden="true">→</span>
  <div class="vh-doc-waveform" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i>
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
  <span class="vh-doc-teaser__audio">音频</span>
</div>

<section class="vh-home-models" aria-labelledby="vh-home-models-title">
  <p class="vh-home-models__eyebrow">模型目录</p>
  <h2 id="vh-home-models-title">按语言和任务查找合适的模型</h2>
  <p class="vh-home-models__description">可按语言、能力、训练方式、许可证、架构和 checkpoint 来源搜索全部 68 个 TTS、ASR 和 VAD 集成。</p>
  <p class="vh-home-models__actions">
    <a class="vh-home-models__primary" href="models/providers/">浏览所有模型 <span aria-hidden="true">→</span></a>
    <a class="vh-home-models__secondary" href="models/training-support/">比较训练支持</a>
  </p>
  <ul class="vh-home-models__stats" aria-label="模型注册表摘要">
    <li><strong>68</strong><span>模型</span></li>
    <li><strong>34</strong><span>TTS</span></li>
    <li><strong>23</strong><span>ASR</span></li>
    <li><strong>11</strong><span>VAD</span></li>
  </ul>
</section>

<p class="vh-badges">
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/ci.yml/badge.svg?branch=main" alt="VoiceHub 持续集成状态">
  </a>
  <a href="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml">
    <img src="https://github.com/kadirnar/voicehub/actions/workflows/docs.yml/badge.svg?branch=main" alt="VoiceHub 文档构建状态">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="VoiceHub 支持 Python 3.10 及更高版本">
  </a>
  <a href="https://github.com/kadirnar/voicehub/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/VoiceHub%20license-Apache--2.0-4051b5" alt="VoiceHub 采用 Apache 2.0 许可证">
  </a>
</p>

## VoiceHub 是什么？

VoiceHub 通过统一的配置、处理器、模型、输出和训练器 API 提供 TTS、自动语音识别
和语音活动检测模型。模型实现始终遵循各自的架构特性：编解码器语言模型、序列到序列系统、
流匹配和扩散模型、声学模型、VITS 风格的对抗系统以及复合流水线，均保留各自的
条件机制、训练目标、参数归属和导出规则。

模型注册表包含 **34 个 TTS 集成**、**23 个 ASR 提供程序**和 **11 个 VAD
提供程序**。微调支持取决于具体的 checkpoint 和运行时；支持推理并不意味着当前的
VoiceHub 工件支持可微分训练。请参考 [TTS 模型目录](models/index.md)、
[TTS 训练矩阵](models/training-support.md)和
[ASR/VAD 支持矩阵](models/asr-vad-support.md)来选择合适的集成。

模型源码以及所有内置 TTS、ASR 和 VAD 推理运行时都会随 VoiceHub 默认安装。
checkpoint 权重仍按需下载，也可以通过本地路径提供。只有在微调和记录实验时才
需要添加 `voicehub[training]`。Apache-2.0 许可证仅适用于 VoiceHub 本身；
集成的源码、checkpoint、编解码器、数据集和生成的音频可能适用其他条款。

<div class="grid cards" markdown>

-   **快速入门**

    ---

    从当前源码树安装 VoiceHub，并通过统一的模型工厂完成第一次生成请求。

    [快速入门](getting-started/quickstart.md)

-   **推理**

    ---

    查找集成、加载 Hub 或本地 checkpoint、配置可复现的生成流程，并获取
    标准化音频。

    [推理指南](guides/inference.md)

-   **数据准备**

    ---

    构建可审计的清单、验证音频、防止说话人或会话信息泄漏，并生成模型专用的
    训练输入。

    [数据准备指南](guides/data-preparation.md)

-   **训练**

    ---

    验证 checkpoint 边界、运行原生训练目标、执行评估、从完整 checkpoint
    恢复训练，并保存可移植工件。

    [训练指南](guides/training.md)

-   **模型**

    ---

    对比 TTS 注册项的默认 checkpoint、功能、源码来源和使用限制。

    [模型目录](models/index.md)

-   **训练支持**

    ---

    查看每个集成准确的微调边界：原始数据、预处理数据、专用流程或暂不支持。

    [训练矩阵](models/training-support.md)

-   **Notebook 示例集**

    ---

    使用四个 Notebook：运行聚焦推理、数据准备和训练的示例，或按照完整的
    Dia 工作流完成导出，并在全新运行时中重新加载。

    [打开 Notebook 示例库](guides/notebook.md)

-   **API 参考**

    ---

    查阅工厂、输出、训练器参数、回调、collator、策略、工件和扩展注册表。

    [浏览 API](reference/api.md)

-   **架构**

    ---

    了解注册表、模型 wrapper、适配器、运行时策略、checkpoint 和可移植工件
    边界。

    [库架构](concepts/architecture.md)

-   **添加模型**

    ---

    实现并测试延迟加载 wrapper、训练规范、必要时使用的专用适配器以及导出
    契约。

    [模型集成指南](project/adding-a-model.md)

</div>

</div>
