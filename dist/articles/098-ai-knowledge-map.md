---
title: 2026 AI全链路知识图谱：从入门到精通的完整工具索引
category: AI
excerpt: 系统性梳理AI领域全链路知识工具与产品，涵盖大模型、开发工具、知识管理、创作工具等，附知识图谱与思维导图，助你快速构建AI知识体系。
tags: AI, 知识图谱, 工具索引, 大模型, 开发工具, AI应用, 思维导图
date: 2026-02-20
readTime: 40
---
## 一、AI知识全景图谱

### 1.1 全链路架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI 全链路知识生态系统                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  基础设施层   │───▶│   模型层     │───▶│   应用层     │───▶│  终端层   │  │
│  │              │    │              │    │              │    │          │  │
│  │ • 云计算平台  │    │ • 大语言模型  │    │ • 对话应用   │    │ • Web    │  │
│  │ • 算力服务   │    │ • 多模态模型  │    │ • 编程工具   │    │ • App    │  │
│  │ • 开发框架   │    │ • 垂直模型   │    │ • 创作工具   │    │ • 桌面端  │  │
│  │ • 数据服务   │    │ • 开源模型   │    │ • 知识管理   │    │ • API    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│         │                   │                   │                 │        │
│         ▼                   ▼                   ▼                 ▼        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         支撑体系层                                    │   │
│  │  • 学习资源  • 社区论坛  • 开源生态  • 行业报告  • 政策法规  • 伦理安全  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 知识图谱节点关系

```
                    ┌─────────────┐
                    │   AI核心    │
                    │  知识体系   │
                    └──────┬──────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  技术基础    │    │  应用场景    │    │  工具产品    │
│             │    │             │    │             │
│ • 机器学习   │    │ • 内容创作   │    │ • ChatGPT   │
│ • 深度学习   │    │ • 代码开发   │    │ • Claude    │
│ • NLP       │    │ • 数据分析   │    │ • Cursor    │
│ • CV        │    │ • 智能客服   │    │ • Midjourney│
│ • RL        │    │ • 自动驾驶   │    │ • Runway    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  实践落地    │
                    │  与商业模式  │
                    └─────────────┘
```

## 二、基础设施层：AI的底座

### 2.1 云计算与算力平台

| 平台 | 官网 | 定位 | 特点 |
|------|------|------|------|
| **NVIDIA DGX Cloud** | [nvidia.com/dgx-cloud](https://www.nvidia.com/dgx-cloud/) | 企业级AI算力 | H100/H200集群，完整软件栈 |
| **AWS SageMaker** | [aws.amazon.com/sagemaker](https://aws.amazon.com/sagemaker/) | 云原生ML平台 | 与AWS生态深度集成 |
| **Google Cloud Vertex AI** | [cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai) | 全托管ML平台 | Gemini原生支持 |
| **Azure OpenAI Service** | [azure.microsoft.com/openai](https://azure.microsoft.com/products/ai-services/openai-service/) | 企业OpenAI API | 合规、安全、SLA保障 |
| **阿里云PAI** | [pai.console.aliyun.com](https://pai.console.aliyun.com/) | 国产AI平台 | 通义系列模型原生支持 |
| **火山引擎方舟** | [console.volcengine.com/ark](https://console.volcengine.com/ark/) | 字节跳动AI平台 | 豆包大模型、高性价比 |
| **SiliconCloud** | [siliconflow.cn](https://siliconflow.cn/) | 国产大模型云 | 聚合多家模型，价格优势 |
| **Together AI** | [together.ai](https://www.together.ai/) | 开源模型推理 | 开源模型高性能推理 |

### 2.2 开发框架与工具链

| 框架 | 官网 | 用途 | 推荐指数 |
|------|------|------|----------|
| **PyTorch** | [pytorch.org](https://pytorch.org/) | 深度学习框架 | ⭐⭐⭐⭐⭐ |
| **TensorFlow** | [tensorflow.org](https://www.tensorflow.org/) | 工业级ML框架 | ⭐⭐⭐⭐ |
| **JAX** | [jax.readthedocs.io](https://jax.readthedocs.io/) | 高性能数值计算 | ⭐⭐⭐⭐ |
| **Hugging Face Transformers** | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers/) | 预训练模型库 | ⭐⭐⭐⭐⭐ |
| **LangChain** | [langchain.com](https://www.langchain.com/) | LLM应用开发 | ⭐⭐⭐⭐ |
| **LlamaIndex** | [llamaindex.ai](https://www.llamaindex.ai/) | RAG框架 | ⭐⭐⭐⭐ |
| **Ollama** | [ollama.com](https://ollama.com/) | 本地模型运行 | ⭐⭐⭐⭐⭐ |
| **vLLM** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | 高性能推理引擎 | ⭐⭐⭐⭐ |
| **Text Generation Inference** | [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) | HF推理服务 | ⭐⭐⭐⭐ |

### 2.3 数据与标注平台

| 平台 | 官网 | 功能 | 适用场景 |
|------|------|------|----------|
| **Hugging Face Datasets** | [huggingface.co/datasets](https://huggingface.co/datasets) | 开源数据集 | 模型训练、研究 |
| **Kaggle Datasets** | [kaggle.com/datasets](https://www.kaggle.com/datasets) | 数据科学数据集 | 竞赛、学习 |
| **Scale AI** | [scale.com](https://scale.com/) | 数据标注服务 | 企业级标注需求 |
| **Label Studio** | [labelstud.io](https://labelstud.io/) | 开源标注工具 | 自建标注流程 |
| **Snorkel** | [snorkel.ai](https://snorkel.ai/) | 弱监督学习 | 大规模标注 |
| **Weights & Biases** | [wandb.ai](https://wandb.ai/) | 实验跟踪 | MLOps、实验管理 |
| **MLflow** | [mlflow.org](https://mlflow.org/) | 开源ML生命周期 | 模型管理、部署 |

## 三、模型层：AI的大脑

### 3.1 国际大语言模型

```
┌─────────────────────────────────────────────────────────────────┐
│                    国际大语言模型生态                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  第一梯队（闭源商用）                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   GPT-4o     │  │ Claude 3.5   │  │   Gemini 2   │          │
│  │   OpenAI     │  │  Anthropic   │  │    Google    │          │
│  │  $20/月      │  │  $20/月      │  │  $20/月      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  API: openai.com    API: anthropic.com   API: makersuite...   │
│                                                                 │
│  第二梯队（特色模型）                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Grok 2     │  │  Mistral     │  │   Cohere     │          │
│  │    xAI       │  │   Large      │  │  Command     │          │
│  │  实时信息    │  │  欧洲开源    │  │  企业RAG     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  开源模型（可本地部署）                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Llama 3    │  │   Qwen 2.5   │  │  DeepSeek    │          │
│  │   Meta       │  │    阿里      │  │   深度求索   │          │
│  │  405B最强    │  │  72B开源    │  │  V3 671B    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| 模型 | 官网/API | 特点 | 价格 |
|------|----------|------|------|
| **GPT-4o** | [platform.openai.com](https://platform.openai.com/) | 多模态、速度快 | $2.5/M tokens |
| **GPT-4o-mini** | [platform.openai.com](https://platform.openai.com/) | 性价比高 | $0.15/M tokens |
| **o1/o3** | [platform.openai.com](https://platform.openai.com/) | 推理能力强 | $15/M tokens |
| **Claude 3.5 Sonnet** | [console.anthropic.com](https://console.anthropic.com/) | 代码能力顶级 | $3/M tokens |
| **Claude 3.5 Haiku** | [console.anthropic.com](https://console.anthropic.com/) | 快速响应 | $0.25/M tokens |
| **Gemini 2.0 Flash** | [ai.google.dev](https://ai.google.dev/) | 多模态、免费额度高 | 免费/$0.35/M |
| **Gemini 2.0 Pro** | [ai.google.dev](https://ai.google.dev/) | 长上下文 | $3.5/M tokens |
| **Grok 2** | [x.ai](https://x.ai/) | X平台实时信息 | $5/M tokens |
| **Mistral Large** | [mistral.ai](https://mistral.ai/) | 欧洲最强 | $3/M tokens |
| **Cohere Command R+** | [cohere.com](https://cohere.com/) | 企业RAG优化 | $3/M tokens |

### 3.2 国产大语言模型

| 模型 | 官网 | 特点 | 应用场景 |
|------|------|------|----------|
| **通义千问 Qwen2.5** | [tongyi.aliyun.com](https://tongyi.aliyun.com/) | 72B开源最强 | 通用对话、代码 |
| **文心一言 4.0** | [yiyan.baidu.com](https://yiyan.baidu.com/) | 百度生态集成 | 搜索、办公 |
| **智谱清言 GLM-4** | [chatglm.cn](https://chatglm.cn/) | 清华出品 | 学术研究、长文本 |
| **讯飞星火 4.0** | [xinghuo.xfyun.cn](https://xinghuo.xfyun.cn/) | 语音能力突出 | 语音交互 |
| **Kimi K1.5** | [kimi.moonshot.cn](https://kimi.moonshot.cn/) | 200K长上下文 | 文档分析 |
| **豆包 Pro** | [doubao.com](https://www.doubao.com/) | 字节跳动 | 内容创作 |
| **DeepSeek V3** | [deepseek.com](https://www.deepseek.com/) | 671B MoE | 代码、推理 |
| **MiniMax abab6.5** | [minimaxi.com](https://www.minimaxi.com/) | 多模态 | 角色扮演 |
| **百川智能 Baichuan4** | [baichuan-ai.com](https://www.baichuan-ai.com/) | 医疗法律 | 垂直领域 |
| **零一万物 Yi-Large** | [lingyiwanwu.com](https://www.lingyiwanwu.com/) | 李开复团队 | 中英文 |

### 3.3 多模态模型

| 模型 | 官网 | 能力 | 特色 |
|------|------|------|------|
| **GPT-4o Vision** | [openai.com](https://openai.com/) | 图文理解、生成 | 端到端多模态 |
| **Claude 3.5 Vision** | [anthropic.com](https://www.anthropic.com/) | 图像分析 | 精准描述 |
| **Gemini 2.0** | [deepmind.google/gemini](https://deepmind.google/gemini/) | 原生多模态 | 视频理解 |
| **Qwen-VL** | [huggingface.co/Qwen](https://huggingface.co/Qwen/) | 视觉语言 | 中文优化 |
| **InternVL** | [internvl.opengvlab.com](https://internvl.opengvlab.com/) | 开源视觉模型 | 上海AI Lab |
| **LLaVA** | [llava-vl.github.io](https://llava-vl.github.io/) | 开源多模态 | 社区活跃 |

### 3.4 图像生成模型

| 模型/工具 | 官网 | 特点 | 价格 |
|-----------|------|------|------|
| **Midjourney V6.1** | [midjourney.com](https://www.midjourney.com/) | 艺术风格顶级 | $10/月起 |
| **DALL-E 3** | [openai.com/dall-e-3](https://openai.com/dall-e-3) | ChatGPT集成 | API按量 |
| **Stable Diffusion 3** | [stability.ai](https://stability.ai/) | 开源可商用 | 免费/企业版 |
| **Flux** | [blackforestlabs.ai](https://blackforestlabs.ai/) | 开源新标杆 | 免费 |
| **Ideogram 2.0** | [ideogram.ai](https://ideogram.ai/) | 文字渲染强 | 免费/付费 |
| **Adobe Firefly** | [adobe.com/firefly](https://www.adobe.com/products/firefly.html) | 商业安全 | 订阅制 |
| **通义万相** | [tongyi.aliyun.com/wanxiang](https://tongyi.aliyun.com/wanxiang/) | 国产中文 | 免费额度 |
| **文心一格** | [yige.baidu.com](https://yige.baidu.com/) | 百度生态 | 免费/付费 |

### 3.5 视频生成模型

| 模型/工具 | 官网 | 特点 | 时长/分辨率 |
|-----------|------|------|-------------|
| **Sora** | [openai.com/sora](https://openai.com/sora) | 物理模拟强 | 60s/1080p |
| **Runway Gen-3** | [runwayml.com](https://runwayml.com/) | 电影级质量 | 10s/1080p |
| **Pika 2.0** | [pika.art](https://pika.art/) | 风格多样 | 3s/720p |
| **Kling 1.6** | [klingai.com](https://klingai.com/) | 国产最强 | 2min/1080p |
| **Luma Dream Machine** | [lumalabs.ai](https://lumalabs.ai/) | 快速生成 | 5s/1080p |
| **Stable Video Diffusion** | [stability.ai](https://stability.ai/) | 开源 | 4s/576p |
| **可灵 AI** | [klingai.com](https://klingai.com/) | 快手出品 | 2min/1080p |
| **即梦 AI** | [jimeng.jianying.com](https://jimeng.jianying.com/) | 字节跳动 | 免费额度 |

### 3.6 音频与音乐模型

| 模型/工具 | 官网 | 能力 | 特色 |
|-----------|------|------|------|
| **Suno V3** | [suno.ai](https://www.suno.ai/) | 音乐生成 | 人声、编曲完整 |
| **Udio** | [udio.com](https://www.udio.com/) | 音乐生成 | 高保真音质 |
| **ElevenLabs** | [elevenlabs.io](https://elevenlabs.io/) | TTS/语音克隆 | 最自然人声 |
| **Whisper V3** | [github.com/openai/whisper](https://github.com/openai/whisper) | 语音识别 | 开源多语言 |
| **Bark** | [github.com/suno-ai/bark](https://github.com/suno-ai/bark) | TTS开源 | 情感丰富 |
| **Fish Speech** | [github.com/fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) | 中文TTS | 开源中文优化 |

## 四、应用层：AI的落地场景

### 4.1 AI编程工具

```
┌─────────────────────────────────────────────────────────────────┐
│                      AI编程工具全景                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IDE/编辑器                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Cursor    │  │   GitHub     │  │   Windsurf   │          │
│  │   $20/月     │  │   Copilot    │  │  Codeium出品 │          │
│  │  AI原生IDE   │  │   $10/月     │  │   $15/月     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  国产编程助手                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   通义灵码    │  │   CodeGeeX   │  │   文心快码    │          │
│  │   阿里巴巴    │  │   智谱AI     │  │   百度       │          │
│  │   免费       │  │   免费       │  │   免费       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  代码审查/测试                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  CodeRabbit  │  │  CodiumAI    │  │  Snyk        │          │
│  │  AI代码审查  │  │  测试生成    │  │  安全扫描    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| 工具 | 官网 | 类型 | 价格 | 推荐指数 |
|------|------|------|------|----------|
| **Cursor** | [cursor.com](https://cursor.com/) | AI原生IDE | $20/月 | ⭐⭐⭐⭐⭐ |
| **GitHub Copilot** | [github.com/copilot](https://github.com/features/copilot/) | IDE插件 | $10/月 | ⭐⭐⭐⭐⭐ |
| **Windsurf** | [codeium.com/windsurf](https://codeium.com/windsurf/) | AI IDE | $15/月 | ⭐⭐⭐⭐ |
| **通义灵码** | [tongyi.aliyun.com/lingma](https://tongyi.aliyun.com/lingma/) | IDE插件 | 免费 | ⭐⭐⭐⭐⭐ |
| **CodeGeeX** | [codegeex.cn](https://codegeex.cn/) | IDE插件 | 免费 | ⭐⭐⭐⭐ |
| **Codeium** | [codeium.com](https://codeium.com/) | 免费Copilot | 免费 | ⭐⭐⭐⭐ |
| **Tabnine** | [tabnine.com](https://www.tabnine.com/) | 企业AI助手 | $12/月 | ⭐⭐⭐⭐ |
| **Amazon CodeWhisperer** | [aws.amazon.com/codewhisperer](https://aws.amazon.com/codewhisperer/) | AWS生态 | 免费/$19 | ⭐⭐⭐ |

### 4.2 AI对话与助手

| 产品 | 官网 | 模型 | 特色 | 价格 |
|------|------|------|------|------|
| **ChatGPT** | [chatgpt.com](https://chatgpt.com/) | GPT-4o | 功能最全 | 免费/$20 |
| **Claude** | [claude.ai](https://claude.ai/) | Claude 3.5 | 代码、长文 | 免费/$20 |
| **Gemini** | [gemini.google.com](https://gemini.google.com/) | Gemini 2.0 | 多模态、免费 | 免费/$20 |
| **Perplexity** | [perplexity.ai](https://www.perplexity.ai/) | 多模型 | 搜索增强 | 免费/$20 |
| **Poe** | [poe.com](https://poe.com/) | 多模型聚合 | 机器人市场 | 免费/$20 |
| **You.com** | [you.com](https://you.com/) | 多模型 | 隐私优先 | 免费/$15 |
| **文心一言** | [yiyan.baidu.com](https://yiyan.baidu.com/) | 文心4.0 | 中文、搜索 | 免费/付费 |
| **通义千问** | [tongyi.aliyun.com](https://tongyi.aliyun.com/) | Qwen2.5 | 中文、代码 | 免费/付费 |
| **Kimi** | [kimi.moonshot.cn](https://kimi.moonshot.cn/) | Moonshot | 长文本 | 免费 |
| **豆包** | [doubao.com](https://www.doubao.com/) | 豆包 | 多模态 | 免费 |

### 4.3 AI内容创作工具

#### 写作助手

| 工具 | 官网 | 功能 | 价格 |
|------|------|------|------|
| **Jasper** | [jasper.ai](https://www.jasper.ai/) | 营销文案 | $49/月起 |
| **Copy.ai** | [copy.ai](https://www.copy.ai/) | 营销内容 | 免费/$36 |
| **Writesonic** | [writesonic.com](https://writesonic.com/) | SEO文章 | 免费/$16 |
| **Notion AI** | [notion.so/product/ai](https://www.notion.so/product/ai) | 笔记助手 | $10/月 |
| **Grammarly** | [grammarly.com](https://www.grammarly.com/) | 语法检查 | 免费/$12 |
| **秘塔写作猫** | [xiezuocat.com](https://www.xiezuocat.com/) | 中文写作 | 免费/付费 |
| **讯飞听见** | [tingxie.iflyrec.com](https://tingxie.iflyrec.com/) | 语音转写 | 免费/付费 |

#### 图像创作

| 工具 | 官网 | 功能 | 价格 |
|------|------|------|------|
| **Midjourney** | [midjourney.com](https://www.midjourney.com/) | AI绘画 | $10/月起 |
| **Stable Diffusion WebUI** | [github.com/AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) | 开源绘画 | 免费 |
| **ComfyUI** | [github.com/comfyanonymous](https://github.com/comfyanonymous/ComfyUI) | 工作流绘画 | 免费 |
| **Canva AI** | [canva.com/ai-image-generator](https://www.canva.com/ai-image-generator/) | 设计+AI | 免费/付费 |
| **Remove.bg** | [remove.bg](https://www.remove.bg/) | 抠图 | 免费/付费 |
| **Upscayl** | [upscayl.org](https://upscayl.org/) | 图片放大 | 免费 |

#### 视频创作

| 工具 | 官网 | 功能 | 价格 |
|------|------|------|------|
| **Runway** | [runwayml.com](https://runwayml.com/) | 视频生成/编辑 | $15/月起 |
| **Pika** | [pika.art](https://pika.art/) | 视频生成 | 免费/付费 |
| **HeyGen** | [heygen.com](https://www.heygen.com/) | 数字人视频 | $24/月起 |
| **Synthesia** | [synthesia.io](https://www.synthesia.io/) | 企业数字人 | $22/月起 |
| **CapCut** | [capcut.com](https://www.capcut.com/) | AI剪辑 | 免费/付费 |
| **剪映** | [jianying.com](https://www.jianying.com/) | 国产剪辑 | 免费/付费 |

### 4.4 AI知识管理工具

| 工具 | 官网 | 功能 | 特色 |
|------|------|------|------|
| **Notion** | [notion.so](https://www.notion.so/) | 笔记+数据库+AI | 全能工作空间 |
| **Obsidian** | [obsidian.md](https://obsidian.md/) | 本地知识库 | 插件生态丰富 |
| **Logseq** | [logseq.com](https://logseq.com/) | 大纲笔记 | 开源免费 |
| **Mem.ai** | [mem.ai](https://mem.ai/) | AI知识库 | 自动关联 |
| **Readwise** | [readwise.io](https://readwise.io/) | 阅读高亮管理 | 学习闭环 |
| **Zotero** | [zotero.org](https://www.zotero.org/) | 文献管理 | 学术必备 |
| **飞书文档** | [feishu.cn](https://www.feishu.cn/) | 协作文档 | 国产办公 |
| **语雀** | [yuque.com](https://www.yuque.com/) | 知识库 | 阿里出品 |

### 4.5 AI搜索与信息获取

| 工具 | 官网 | 特点 | 适用场景 |
|------|------|------|----------|
| **Perplexity** | [perplexity.ai](https://www.perplexity.ai/) | 引用来源、实时搜索 | 研究、学习 |
| **Devv** | [devv.ai](https://devv.ai/) | 开发者搜索 | 编程问题 |
| **Phind** | [phind.com](https://www.phind.com/) | 开发者AI搜索 | 技术问题 |
| **You.com** | [you.com](https://you.com/) | 隐私搜索 | 日常搜索 |
| **Kimi探索版** | [kimi.moonshot.cn](https://kimi.moonshot.cn/) | 深度搜索 | 复杂问题 |
| **秘塔AI搜索** | [metaso.cn](https://metaso.cn/) | 国产搜索 | 中文内容 |
| **天工AI** | [tiangong.cn](https://www.tiangong.cn/) | 昆仑万维 | 多模态搜索 |
| **360AI搜索** | [ai.so.com](https://ai.so.com/) | 360出品 | 安全搜索 |

## 五、终端层：触达用户

### 5.1 AI浏览器与插件

| 产品 | 官网 | 功能 | 平台 |
|------|------|------|------|
| **Arc Browser** | [arc.net](https://arc.net/) | AI浏览器 | macOS/iOS |
| **SigmaOS** | [sigmaos.com](https://sigmaos.com/) | AI工作流浏览器 | macOS |
| **Brave Leo** | [brave.com/leo-ai](https://brave.com/leo-ai/) | 隐私AI助手 | 全平台 |
| **Edge Copilot** | [microsoft.com/edge](https://www.microsoft.com/edge/) | 内置AI | 全平台 |
| **Monica** | [monica.im](https://monica.im/) | 浏览器AI助手 | Chrome/Edge |
| **Sider** | [sider.ai](https://sider.ai/) | 侧边栏AI | Chrome/Edge |
| **ChatGPT Sidebar** | [chatgpt-sidebar.com](https://chatgpt-sidebar.com/) | 网页AI助手 | Chrome |

### 5.2 AI移动应用

| 应用 | 平台 | 功能 | 特点 |
|------|------|------|------|
| **ChatGPT App** | iOS/Android | 对话助手 | 语音、GPT-4o |
| **Claude App** | iOS/Android | 对话助手 | 长文、代码 |
| **Perplexity App** | iOS/Android | AI搜索 | 实时信息 |
| **豆包App** | iOS/Android | 多模态助手 | 字节生态 |
| **文心一言App** | iOS/Android | 中文助手 | 百度生态 |
| **讯飞星火App** | iOS/Android | 语音助手 | 语音交互 |
| **Replika** | iOS/Android | AI伴侣 | 情感陪伴 |
| **Character.AI** | iOS/Android | AI角色 | 娱乐对话 |

### 5.3 AI硬件

| 产品 | 官网 | 类型 | 价格 |
|------|------|------|------|
| **Rabbit R1** | [rabbit.tech](https://www.rabbit.tech/) | AI掌机 | $199 |
| **Humane Ai Pin** | [humane.com](https://humane.com/) | AI胸针 | $699 |
| **Ray-Ban Meta** | [ray-ban.com/meta](https://www.ray-ban.com/meta/) | AI眼镜 | $299 |
| **Plaud Note** | [plaud.ai](https://www.plaud.ai/) | AI录音笔 | $159 |
| **Rewind Pendant** | [rewind.ai/pendant](https://www.rewind.ai/pendant) | 记忆助手 | 待定 |

## 六、支撑体系：学习与成长

### 6.1 学习资源

| 资源 | 官网 | 类型 | 适合人群 |
|------|------|------|----------|
| **Fast.ai** | [fast.ai](https://www.fast.ai/) | 免费课程 | 初学者 |
| **DeepLearning.AI** | [deeplearning.ai](https://www.deeplearning.ai/) | 系统课程 | 进阶者 |
| **Hugging Face Learn** | [huggingface.co/learn](https://huggingface.co/learn/) | NLP/LLM | 开发者 |
| **LangChain Academy** | [academy.langchain.com](https://academy.langchain.com/) | LLM应用 | 工程师 |
| **吴恩达AI课程** | [coursera.org/instructor/andrewng](https://www.coursera.org/instructor/andrewng) | 经典课程 | 所有人 |
| **李沐《动手学深度学习》** | [zh.d2l.ai](https://zh.d2l.ai/) | 中文教材 | 中文学习者 |
| **Prompt Engineering Guide** | [promptingguide.ai](https://www.promptingguide.ai/) | 提示工程 | 所有人 |
| **LLM University** | [cohere.com/llm-university](https://docs.cohere.com/docs/llmu/) | LLM基础 | 初学者 |

### 6.2 社区与论坛

| 社区 | 官网 | 特点 | 活跃度 |
|------|------|------|--------|
| **Hugging Face** | [huggingface.co](https://huggingface.co/) | 模型社区 | ⭐⭐⭐⭐⭐ |
| **Reddit r/MachineLearning** | [reddit.com/r/MachineLearning](https://www.reddit.com/r/MachineLearning/) | 学术讨论 | ⭐⭐⭐⭐ |
| **Papers with Code** | [paperswithcode.com](https://paperswithcode.com/) | 论文+代码 | ⭐⭐⭐⭐⭐ |
| **Kaggle** | [kaggle.com](https://www.kaggle.com/) | 竞赛社区 | ⭐⭐⭐⭐⭐ |
| **GitHub Trending** | [github.com/trending](https://github.com/trending) | 开源项目 | ⭐⭐⭐⭐⭐ |
| **Discord AI Communities** | - | 实时讨论 | ⭐⭐⭐⭐ |
| **Twitter/X AI** | [x.com](https://x.com/) | 最新动态 | ⭐⭐⭐⭐⭐ |
| **即刻AI圈子** | [okjike.com](https://okjike.com/) | 中文社区 | ⭐⭐⭐⭐ |
| **知乎AI话题** | [zhihu.com/topic/19559450](https://www.zhihu.com/topic/19559450) | 中文问答 | ⭐⭐⭐⭐ |

### 6.3 行业报告与资讯

| 来源 | 官网 | 类型 | 更新频率 |
|------|------|------|----------|
| **State of AI Report** | [stateof.ai](https://www.stateof.ai/) | 年度综合报告 | 年度 |
| **Epoch AI** | [epoch.ai](https://epoch.ai/) | AI趋势分析 | 持续 |
| **AI Index Report** | [aiindex.stanford.edu](https://aiindex.stanford.edu/) | 斯坦福指数 | 年度 |
| **Import AI** | [importai.substack.com](https://importai.substack.com/) | 新闻简报 | 周刊 |
| **The Batch** | [deeplearning.ai/the-batch](https://www.deeplearning.ai/the-batch/) | 深度新闻 | 周刊 |
| **机器之心** | [jiqizhixin.com](https://www.jiqizhixin.com/) | 中文媒体 | 日更 |
| **量子位** | [qbitai.com](https://www.qbitai.com/) | 中文媒体 | 日更 |
| **InfoQ AI** | [infoq.cn/ai](https://www.infoq.cn/ai/) | 技术媒体 | 持续 |

## 七、知识图谱可视化

### 7.1 AI技术栈思维导图

```
AI技术栈
│
├── 基础层
│   ├── 数学基础
│   │   ├── 线性代数
│   │   ├── 概率统计
│   │   ├── 微积分
│   │   └── 优化理论
│   │
│   ├── 编程基础
│   │   ├── Python
│   │   ├── PyTorch/TensorFlow
│   │   ├── CUDA编程
│   │   └── 数据处理
│   │
│   └── 机器学习基础
│       ├── 监督学习
│       ├── 无监督学习
│       ├── 强化学习
│       └── 深度学习
│
├── 模型层
│   ├── 大语言模型
│   │   ├── Transformer架构
│   │   ├── 预训练技术
│   │   ├── 对齐技术(RLHF)
│   │   └── 推理优化
│   │
│   ├── 多模态模型
│   │   ├── 视觉-语言
│   │   ├── 语音-文本
│   │   └── 视频理解
│   │
│   └── 垂直领域模型
│       ├── 代码模型
│       ├── 科学计算
│       └── 行业专用
│
├── 应用层
│   ├── 对话系统
│   ├── 内容生成
│   ├── 代码助手
│   ├── 知识管理
│   ├── 数据分析
│   └── 智能体
│
└── 工程层
    ├── 模型部署
    ├── 推理优化
    ├── MLOps
    ├── 数据工程
    └── 产品化
```

### 7.2 AI产品选择决策树

```
你需要什么类型的AI工具？
│
├── 编程开发
│   ├── 需要完整IDE体验 → Cursor/Windsurf
│   ├── IDE插件即可 → GitHub Copilot/通义灵码
│   └── 免费方案 → Codeium/CodeGeeX
│
├── 内容创作
│   ├── 文本写作 → ChatGPT/Claude/文心一言
│   ├── 图像生成 → Midjourney/Stable Diffusion/通义万相
│   ├── 视频生成 → Runway/Pika/可灵
│   └── 音乐生成 → Suno/Udio
│
├── 知识管理
│   ├── 全能工作空间 → Notion
│   ├── 本地优先 → Obsidian
│   └── 团队协作 → 飞书/语雀
│
├── 搜索研究
│   ├── 需要引用来源 → Perplexity
│   ├── 开发者专用 → Devv/Phind
│   └── 中文内容 → 秘塔AI/Kimi
│
└── 模型API
    ├── 国际模型 → OpenAI/Anthropic/Google
    ├── 国产模型 → 阿里云/火山引擎/智谱
    └── 开源模型 → Ollama/Together AI
```

### 7.3 AI学习路径图

```
阶段1: 入门 (1-2个月)
│
├── 了解AI基础概念
│   ├── 什么是机器学习
│   ├── 什么是深度学习
│   └── 什么是大语言模型
│
├── 掌握基础工具
│   ├── ChatGPT/Claude日常使用
│   ├── 提示工程基础
│   └── 简单API调用
│
└── 完成入门课程
    ├── Fast.ai Practical Deep Learning
    └── Andrew Ng Machine Learning

阶段2: 进阶 (3-6个月)
│
├── 深入学习
│   ├── Transformer架构
│   ├── Fine-tuning技术
│   └── RAG应用开发
│
├── 动手实践
│   ├── 构建第一个LLM应用
│   ├── 参与开源项目
│   └── 复现经典论文
│
└── 专项深入
    ├── NLP/LLM方向
    ├── 计算机视觉方向
    └── AI工程方向

阶段3: 精通 (6-12个月)
│
├── 前沿跟进
│   ├── 阅读最新论文
│   ├── 参加顶级会议
│   └── 关注SOTA模型
│
├── 实际项目
│   ├── 完整产品落地
│   ├── 性能优化经验
│   └── 大规模部署
│
└── 领域专家
    ├── 特定领域深耕
    ├── 技术影响力建设
    └── 商业应用探索
```

## 八、快速索引表

### 8.1 按场景快速查找

| 场景 | 推荐工具 | 官网 |
|------|----------|------|
| **日常对话** | ChatGPT/Claude/通义千问 | chatgpt.com / claude.ai / tongyi.aliyun.com |
| **编程开发** | Cursor/通义灵码 | cursor.com / tongyi.aliyun.com/lingma |
| **AI绘画** | Midjourney/Stable Diffusion | midjourney.com / stability.ai |
| **视频生成** | Runway/可灵 | runwayml.com / klingai.com |
| **音乐生成** | Suno/Udio | suno.ai / udio.com |
| **知识管理** | Notion/Obsidian | notion.so / obsidian.md |
| **AI搜索** | Perplexity/秘塔 | perplexity.ai / metaso.cn |
| **语音合成** | ElevenLabs/讯飞 | elevenlabs.io / xfyun.cn |
| **本地模型** | Ollama/LM Studio | ollama.com / lmstudio.ai |
| **API服务** | OpenAI/阿里云/火山 | platform.openai.com / dashscope.aliyun.com |

### 8.2 按价格快速查找

| 价格区间 | 工具类型 | 代表产品 |
|----------|----------|----------|
| **免费** | 开源模型 | Llama/Qwen/DeepSeek |
| **免费** | 国产助手 | 通义千问/文心一言/Kimi/豆包 |
| **免费额度** | API服务 | OpenAI/Google/阿里云 |
| **$10-20/月** | 专业工具 | ChatGPT Plus/Cursor/Copilot |
| **$20-50/月** | 高级功能 | Midjourney/Runway/Claude Pro |
| **企业定制** | 私有化部署 | 联系厂商 |

## 九、结语：构建你的AI知识体系

AI领域发展太快，不可能掌握所有工具。关键是：

1. **建立知识框架**：理解全链路，知道什么工具解决什么问题
2. **选择核心工具**：每个场景选1-2个主力工具深入学习
3. **保持好奇心**：定期尝试新工具，但不盲目追新
4. **实践出真知**：用AI解决实际问题，而不是收藏即学会

这张知识图谱不是终点，而是起点。

建议你将本文收藏，作为AI工具查询手册。
每当你需要新工具时，先来这里看看。

AI时代，工具是杠杆，选择比努力更重要。

---

**最后更新**: 2026年2月20日  
**维护说明**: 本文将持续更新，欢迎反馈新工具信息

**相关阅读**:
- [Cursor完全指南](097-cursor-guide)
- [Vibe Coding与AI Coding](096-vibe-ai-coding)
- [AI辅助编码工具盘点](056-ai-code-tools)
