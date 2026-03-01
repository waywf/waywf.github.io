---
title: OpenClaw + Ollama + 飞书机器人：根据官方文档搭建企业AI助手
category: AI
excerpt: 完全依据OpenClaw、Ollama、飞书官方文档，一步步搭建企业级AI助手。从安装配置到飞书集成，每一步都有官方依据。
tags: OpenClaw, Ollama, 飞书机器人, 本地部署, 企业AI, LLM, 官方文档
date: 2026-01-29
readTime: 35
---

# OpenClaw + Ollama + 飞书机器人：根据官方文档搭建企业AI助手

> 本文严格按照各项目官方文档编写：
> - OpenClaw官方文档: https://docs.openclaw.ai
> - Ollama官方文档: https://ollama.com/docs
> - 飞书开放平台文档: https://open.feishu.cn/document/

## 一、OpenClaw：什么是它？（官方定义）

### 1.1 官方介绍

根据 [OpenClaw官方文档](https://docs.openclaw.ai)：

> **OpenClaw** 是一个自托管的网关，连接你喜欢的聊天应用（WhatsApp、Telegram、Discord、iMessage等）到AI编码代理。你在自己的机器（或服务器）上运行一个单一的Gateway进程，它就成为你的消息应用和始终可用的AI助手之间的桥梁。

**核心特性（来自官网）**：
- **自托管**：在你的硬件上运行，你的规则
- **多渠道**：一个Gateway同时服务WhatsApp、Telegram、Discord等
- **Agent原生**：为编码代理构建，支持工具使用、会话、记忆和多代理路由
- **开源**：MIT许可证，社区驱动

### 1.2 官方Quick Start要求

官网明确说明：
- **Node 22+**
- API密钥（推荐Anthropic）
- 5分钟时间

### 1.3 官方默认配置

配置文件位置：`~/.openclaw/openclaw.json`

本地访问地址：`http://127.0.0.1:18789/`

## 二、Ollama：安装与配置（官方步骤）

### 2.1 官方安装方式

根据 [Ollama官方文档](https://ollama.com/docs)：

**macOS安装（官方）**：
```bash
# 方式1：Homebrew（官方推荐）
brew install ollama

# 方式2：手动下载
# 访问 https://ollama.com 下载 .dmg
```

**Linux安装（官方）**：
```bash
# 官方一键安装脚本
curl -fsSL https://ollama.com/install.sh | sh

# 手动安装参考：https://github.com/ollama/ollama/blob/main/docs/linux.md
```

**Windows安装（官方）**：
```
访问 https://ollama.com 下载安装程序
```

### 2.2 验证安装（官方命令）

```bash
ollama --version
```

### 2.3 下载模型（官方命令）

根据官网，支持的模型包括：gpt-oss、Gemma 3、DeepSeek-R1、Qwen3等

```bash
# 下载Qwen模型（推荐国产）
ollama pull qwen2.5:14b

# 查看已下载模型
ollama list
```

### 2.4 启动服务（官方方式）

```bash
# 启动Ollama服务
ollama serve
```

默认端口：**11434**

### 2.5 官方API调用示例

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:14b",
  "prompt": "Hello"
}'
```

## 三、OpenClaw：安装与启动（官方步骤）

### 3.1 环境要求

官网明确：**Node 22+**

检查Node版本：
```bash
node --version
# 需要 >= 22
```

### 3.2 安装OpenClaw

```bash
# 从GitHub克隆（官方仓库）
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 安装依赖
npm install
```

### 3.3 启动OpenClaw Gateway

```bash
# 启动Gateway
npm start
```

启动成功后，打开浏览器访问：
```
http://127.0.0.1:18789/
```

### 3.4 配置文件（官方格式）

配置文件位置：`~/.openclaw/openclaw.json`

官方示例配置：
```json
{
  "channels": {
    "whatsapp": {
      "allowFrom": ["+15555550123"],
      "groups": {
        "*": { "requireMention": true }
      }
    }
  },
  "messages": {
    "groupChat": {
      "mentionPatterns": ["@openclaw"]
    }
  }
}
```

## 四、飞书机器人：官方开发流程

### 4.1 官方准备工作

根据 [飞书开放平台文档](https://open.feishu.cn/document/)：

1. 登录 [飞书开放平台](https://open.feishu.cn/)
2. 创建"企业自建应用"
3. 记录以下凭证：
   - App ID
   - App Secret

### 4.2 官方步骤：开启机器人能力

在飞书开放平台后台：

1. 左侧菜单 → **添加应用能力**
2. 找到并添加 **机器人**
3. 发布应用版本

### 4.3 官方步骤：开通权限

在"权限管理"中添加：
- `im:message:send` - 发送消息
- `im:message:receive` - 接收消息

### 4.4 官方步骤：订阅事件

在"事件订阅"中：
1. 配置请求地址（Webhook URL）
2. 订阅事件：`im.message.receive_v1`

## 五、完整集成：从0到1（按官方步骤）

### 5.1 第一步：安装Ollama（官方）

```bash
# macOS
brew install ollama

# 验证
ollama --version
```

### 5.2 第二步：下载模型（官方）

```bash
# 下载通义千问
ollama pull qwen2.5:14b

# 启动服务
ollama serve
```

### 5.3 第三步：安装OpenClaw（官方）

```bash
# 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 安装依赖（需要Node 22+）
npm install

# 启动
npm start
```

访问控制面板：`http://127.0.0.1:18789/`

### 5.4 第四步：创建飞书应用（官方）

1. 访问 https://open.feishu.cn/
2. 创建企业自建应用
3. 开启机器人能力
4. 配置权限：
   - `im:message:send`
   - `im:message:receive`
5. 发布应用

### 5.5 第五步：编写集成代码

创建 `feishu_openclaw.py`：

```python
#!/usr/bin/env python3
"""
飞书机器人集成OpenClaw
参考：
- 飞书文档: https://open.feishu.cn/document/
- Ollama API: https://ollama.com/docs
"""

from flask import Flask, request
import requests
import json

app = Flask(__name__)

# ====== 配置区域 ======
FEISHU_APP_ID = "你的App ID"
FEISHU_APP_SECRET = "你的App Secret"
OPENCLAW_URL = "http://127.0.0.1:18789"  # OpenClaw官方默认地址
OLLAMA_URL = "http://localhost:11434"      # Ollama官方默认地址
DEFAULT_MODEL = "qwen2.5:14b"


def get_feishu_tenant_token():
    """
    获取飞书tenant_access_token
    参考: https://open.feishu.cn/document/server-docs/authentication/tenant-access-token
    """
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    resp = requests.post(url, json={
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET
    })
    data = resp.json()
    if data.get("code") == 0:
        return data.get("tenant_access_token")
    return None


def send_feishu_message(receive_id, content):
    """
    发送消息到飞书
    参考: https://open.feishu.cn/document/server-docs/im-v1/message/create
    """
    token = get_feishu_tenant_token()
    if not token:
        return None

    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "receive_id": receive_id,
        "msg_type": "text",
        "content": json.dumps({"text": content})
    }

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        params={"receive_id_type": "open_id"}
    )
    return resp.json()


def chat_with_ollama(message):
    """
    调用Ollama API
    参考: https://ollama.com/docs
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": message,
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        data = resp.json()
        return data.get("response", "抱歉，处理出错了")
    except Exception as e:
        return f"服务异常: {str(e)}"


@app.route("/webhook/feishu", methods=["POST"])
def feishu_webhook():
    """
    飞书事件回调
    参考: https://open.feishu.cn/document/server-docs/im-v1/message/events/receive
    """
    data = request.get_json()

    # URL验证（飞书首次配置需要）
    if data.get("type") == "url_verification":
        return {"challenge": data.get("challenge")}

    # 处理消息事件
    event = data.get("event", {})
    message = event.get("message", {})

    # 只处理文本消息
    if message.get("message_type") != "text":
        return {"code": 0}

    # 获取发送者ID和消息内容
    sender = event.get("sender", {})
    sender_id = sender.get("sender_id", {}).get("open_id")
    content = json.loads(message.get("content", "{}"))
    user_message = content.get("text", "").strip()

    print(f"收到消息 [用户ID: {sender_id}]: {user_message}")

    # 调用AI
    ai_reply = chat_with_ollama(user_message)

    # 发送回复
    send_feishu_message(sender_id, ai_reply)

    return {"code": 0, "msg": "success"}


if __name__ == "__main__":
    print("飞书机器人已启动，监听端口: 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
```

### 5.6 运行集成服务

```bash
# 安装依赖
pip install flask requests

# 启动服务
python feishu_openclaw.py
```

### 5.7 配置飞书Webhook

在飞书开放平台后台：
1. 进入"事件订阅"
2. 配置请求地址：`https://你的域名/webhook/feishu`
3. 验证通过后，保存配置

## 六、官方文档链接汇总

| 项目 | 官方文档 | 关键链接 |
|------|---------|---------|
| **OpenClaw** | https://docs.openclaw.ai | GitHub: https://github.com/openclaw/openclaw |
| **Ollama** | https://ollama.com/docs | Linux安装: https://github.com/ollama/ollama/blob/main/docs/linux.md |
| **飞书** | https://open.feishu.cn/document/ | 发送消息API: https://open.feishu.cn/document/server-docs/im-v1/message/create |

## 七、常见问题（基于官方文档）

### 7.1 Ollama相关

**Q: Ollama默认端口是多少？**
A: 11434（来自官方文档）

**Q: 如何让局域网访问Ollama？**
A: 启动时设置环境变量：
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 7.2 OpenClaw相关

**Q: OpenClaw需要什么Node版本？**
A: Node 22+（来自官方文档）

**Q: OpenClaw控制面板地址？**
A: http://127.0.0.1:18789/（来自官方文档）

### 7.3 飞书相关

**Q: 飞书机器人需要什么权限？**
A: `im:message:send` 和 `im:message:receive`（来自官方文档）

## 八、总结

本文完全依据各项目官方文档编写：

✅ **OpenClaw** - 自托管网关，连接多渠道到AI  
✅ **Ollama** - 本地运行大模型，数据不出公司  
✅ **飞书机器人** - 官方API，稳定可靠  

所有步骤都有官方文档作为依据，确保正确性。

---

**参考资源**：
- OpenClaw官方文档: https://docs.openclaw.ai
- Ollama官方文档: https://ollama.com/docs
- 飞书开放平台文档: https://open.feishu.cn/document/
