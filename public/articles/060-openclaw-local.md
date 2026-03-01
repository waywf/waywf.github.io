---
title: OpenClaw + Ollama + 飞书机器人：打造企业级AI助手的完整实战
category: AI
excerpt: 深入探索OpenClaw与Ollama的强强联合，从本地模型部署到飞书机器人集成，手把手教你搭建企业级AI助手，实现数据隐私与智能服务的完美平衡。
tags: OpenClaw, Ollama, 飞书机器人, 本地部署, 企业AI, LLM, 国产模型
date: 2026-01-29
readTime: 35
---

## 一、OpenClaw是什么？企业AI的瑞士军刀

### 1.1 从OpenAI到OpenClaw：自主可控的AI之路

还记得2023年吗？那时候企业想要AI能力，只有一条路：**调用OpenAI API**。

但这条路有三大痛点：
- **数据隐私**：公司机密要传到国外服务器
- **成本不可控**：Token用量像流水，月底账单吓一跳
- **网络依赖**：内网环境、跨国延迟、API限流

OpenClaw的出现，就像给企业发了一台**"AI发电机"**——把大模型部署在自己服务器上，数据不出内网，成本可控，响应飞快。

### 1.2 OpenClaw vs Ollama：双剑合璧

很多人问：有了Ollama为什么还要OpenClaw？

| 维度 | Ollama | OpenClaw |
|------|--------|----------|
| **定位** | 本地模型运行工具 | 企业级AI服务框架 |
| **模型支持** | 开源模型为主 | 国产商用模型 +
| **并发能力** | 单机单用户 | 企业级高并发 |
| **管理功能** | 基础CLI | 完整管理后台 |
| **扩展性** | 插件机制 | 企业集成API |
| **适用场景** | 个人/小团队 | 中大型企业 |

**最佳实践**：
```
Ollama负责：本地模型运行、快速原型验证
OpenClaw负责：企业级部署、多模型管理、业务集成
```

### 1.3 OpenClaw的核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      OpenClaw 架构图                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API Gateway 层                        │   │
│  │  • RESTful API  • WebSocket  • 认证鉴权  • 限流熔断      │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │                   Model Manager 层                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ 通义千问 │ │ 文心一言 │ │ DeepSeek│ │  Llama  │       │   │
│  │  │  Qwen   │ │  ERNIE  │ │   V3    │ │   3.1   │       │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │       └─────────────┴─────────────┴──────────┘           │   │
│  │                    统一调度接口                           │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │                  Inference Engine 层                     │   │
│  │  • vLLM  • TensorRT-LLM  • llama.cpp  • 自定义后端      │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │                    Storage 层                            │   │
│  │  • 模型仓库  • 对话历史  • 知识库  • 配置中心            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 二、环境准备：从零开始搭建

### 2.1 硬件配置建议

**开发测试环境**：
```yaml
CPU: Intel i7-12700 / AMD Ryzen 7 5800X
内存: 32GB DDR4
GPU: NVIDIA RTX 3090 (24GB显存)
存储: 500GB NVMe SSD
网络: 内网千兆
```

**生产环境（支持50并发）**：
```yaml
CPU: Intel Xeon Gold 6348 / AMD EPYC 7543
内存: 128GB DDR4 ECC
GPU: 2x NVIDIA A100 40GB
存储: 2TB NVMe SSD RAID1
网络: 万兆内网 + 负载均衡
```

### 2.2 软件环境搭建

**Step 1: 安装Docker和Docker Compose**

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**Step 2: 安装NVIDIA Container Toolkit**

```bash
# 添加NVIDIA仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Step 3: 验证GPU可用性**

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## 三、OpenClaw + Ollama 联合部署

### 3.1 为什么需要联合部署？

OpenClaw专注于**企业级管理和调度**，Ollama专注于**本地模型运行**。两者结合：

- OpenClaw提供统一API和管理界面
- Ollama作为后端推理引擎之一
- 支持多模型热切换和负载均衡

### 3.2 Docker Compose 部署配置

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  # Ollama 服务
  ollama:
    image: ollama/ollama:latest
    container_name: openclaw-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_HOST=0.0.0.0
    networks:
      - openclaw-network

  # OpenClaw 核心服务
  openclaw:
    image: openclaw/openclaw:latest
    container_name: openclaw-core
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "8081:8081"  # 管理后台
    volumes:
      - openclaw_data:/app/data
      - openclaw_models:/app/models
      - ./config:/app/config:ro
    environment:
      - OPENCLAW_MODE=production
      - OPENCLAW_DB_URL=postgresql://openclaw:password@postgres:5432/openclaw
      - OPENCLAW_REDIS_URL=redis://redis:6379/0
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - postgres
      - redis
      - ollama
    networks:
      - openclaw-network

  # PostgreSQL 数据库
  postgres:
    image: postgres:15-alpine
    container_name: openclaw-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: openclaw
      POSTGRES_PASSWORD: password
      POSTGRES_DB: openclaw
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - openclaw-network

  # Redis 缓存
  redis:
    image: redis:7-alpine
    container_name: openclaw-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - openclaw-network

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    container_name: openclaw-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - openclaw
    networks:
      - openclaw-network

volumes:
  ollama_data:
  openclaw_data:
  openclaw_models:
  postgres_data:
  redis_data:

networks:
  openclaw-network:
    driver: bridge
```

### 3.3 启动服务

```bash
# 创建配置目录
mkdir -p config ssl

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 验证服务状态
docker-compose ps
```

### 3.4 配置国产模型

创建 `config/models.yml`：

```yaml
models:
  # 通义千问 2.5
  qwen2.5:
    name: "通义千问 2.5"
    provider: ollama
    model_id: qwen2.5:72b
    max_tokens: 8192
    temperature: 0.7
    context_window: 32768
    capabilities:
      - chat
      - code
      - analysis
    priority: 1

  # DeepSeek V3
  deepseek-v3:
    name: "DeepSeek V3"
    provider: ollama
    model_id: deepseek-v3
    max_tokens: 8192
    temperature: 0.7
    context_window: 64000
    capabilities:
      - chat
      - code
      - reasoning
    priority: 2

  # 本地轻量级模型（备用）
  qwen2.5-7b:
    name: "通义千问 2.5 (7B轻量版)"
    provider: ollama
    model_id: qwen2.5:7b
    max_tokens: 4096
    temperature: 0.7
    context_window: 32768
    capabilities:
      - chat
      - quick_response
    priority: 3

# 路由策略
routing:
  default_model: qwen2.5
  fallback_enabled: true
  load_balance: round_robin
```

### 3.5 拉取模型

```bash
# 进入Ollama容器
docker exec -it openclaw-ollama bash

# 拉取通义千问
ollama pull qwen2.5:72b

# 拉取DeepSeek
ollama pull deepseek-v3

# 拉取轻量版备用
ollama pull qwen2.5:7b

# 查看已安装模型
ollama list
```

## 四、飞书机器人集成实战

### 4.1 飞书机器人创建

**Step 1: 创建企业自建应用**

1. 进入 [飞书开放平台](https://open.feishu.cn/)
2. 点击"创建企业自建应用"
3. 填写应用名称："OpenClaw AI助手"
4. 选择应用类型："机器人"

**Step 2: 获取凭证**

在"凭证与基础信息"页面获取：
- `App ID` (app_id)
- `App Secret` (app_secret)
- `Verification Token` (verify_token)
- `Encrypt Key` (encrypt_key)

**Step 3: 配置权限**

在"权限管理"中添加以下权限：
- `im:chat:readonly` - 读取群组信息
- `im:message:send` - 发送消息
- `im:message:receive` - 接收消息
- `im:message.group_msg` - 接收群消息

**Step 4: 配置事件订阅**

在"事件订阅"中设置：
- 请求地址：`https://your-domain.com/webhook/feishu`
- 订阅事件：
  - `im.message.receive_v1` - 接收消息
  - `im.chat.member.user.added_v1` - 被添加进群

### 4.2 开发飞书机器人服务

创建 `feishu_bot.py`：

```python
#!/usr/bin/env python3
"""
OpenClaw 飞书机器人服务
实现与飞书的消息收发和OpenClaw的集成
"""

import asyncio
import json
import logging
import aiohttp
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import base64

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FeishuConfig:
    """飞书配置"""
    app_id: str
    app_secret: str
    verify_token: str
    encrypt_key: Optional[str] = None
    openclaw_base_url: str = "http://localhost:8080"
    default_model: str = "qwen2.5"


class OpenClawClient:
    """OpenClaw API 客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat(
        self,
        message: str,
        model: str = "qwen2.5",
        conversation_id: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """发送聊天请求到OpenClaw"""
        
        url = f"{self.base_url}/api/v1/chat"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ],
            "stream": stream,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"OpenClaw API错误: {response.status} - {error_text}")
                    return {
                        "error": f"API错误: {response.status}",
                        "content": "抱歉，服务暂时不可用，请稍后重试。"
                    }
        except Exception as e:
            logger.error(f"请求OpenClaw失败: {e}")
            return {
                "error": str(e),
                "content": "抱歉，连接服务失败，请检查网络。"
            }


class FeishuBot:
    """飞书机器人核心类"""
    
    def __init__(self, config: FeishuConfig):
        self.config = config
        self.access_token: Optional[str] = None
        self.token_expire_time: Optional[datetime] = None
        self.openclaw = OpenClawClient(config.openclaw_base_url)
        
        # 会话管理
        self.conversations: Dict[str, str] = {}  # user_id -> conversation_id
    
    async def get_access_token(self) -> str:
        """获取飞书访问令牌"""
        
        # 检查令牌是否有效
        if self.access_token and self.token_expire_time:
            if datetime.now() < self.token_expire_time:
                return self.access_token
        
        # 请求新令牌
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret
            }) as response:
                data = await response.json()
                
                if data.get("code") == 0:
                    self.access_token = data["tenant_access_token"]
                    # 令牌有效期2小时，提前5分钟刷新
                    self.token_expire_time = datetime.now().timestamp() + data["expire"] - 300
                    return self.access_token
                else:
                    raise Exception(f"获取访问令牌失败: {data}")
    
    def verify_signature(self, timestamp: str, nonce: str, body: str, signature: str) -> bool:
        """验证飞书请求签名"""
        
        # 构造签名字符串
        sign_str = f"{timestamp}\n{nonce}\n{body}\n"
        
        # 计算签名
        computed = hmac.new(
            self.config.encrypt_key.encode(),
            sign_str.encode(),
            hashlib.sha256
        ).digest()
        computed_b64 = base64.b64encode(computed).decode()
        
        return computed_b64 == signature
    
    async def send_message(
        self,
        receive_id: str,
        content: str,
        msg_type: str = "text",
        receive_id_type: str = "open_id"
    ):
        """发送消息到飞书"""
        
        token = await self.get_access_token()
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # 构造消息内容
        if msg_type == "text":
            content_json = json.dumps({"text": content})
        elif msg_type == "markdown":
            content_json = json.dumps({"content": content})
        else:
            content_json = content
        
        params = {"receive_id_type": receive_id_type}
        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": content_json
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                params=params,
                json=payload
            ) as response:
                data = await response.json()
                
                if data.get("code") != 0:
                    logger.error(f"发送消息失败: {data}")
                else:
                    logger.info(f"消息发送成功: {receive_id}")
    
    async def handle_message(self, event: Dict[str, Any]):
        """处理收到的消息"""
        
        message = event.get("message", {})
        sender = event.get("sender", {})
        
        # 获取发送者信息
        sender_id = sender.get("sender_id", {}).get("open_id")
        sender_name = sender.get("sender_id", {}).get("name", "用户")
        
        # 获取消息内容
        msg_type = message.get("message_type")
        content = json.loads(message.get("content", "{}"))
        
        # 只处理文本消息
        if msg_type != "text":
            await self.send_message(
                sender_id,
                "目前我只支持文本消息哦～",
                receive_id_type="open_id"
            )
            return
        
        user_message = content.get("text", "").strip()
        
        # 忽略空消息
        if not user_message:
            return
        
        logger.info(f"收到消息 from {sender_name}: {user_message[:50]}...")
        
        # 获取或创建会话ID
        conversation_id = self.conversations.get(sender_id)
        
        # 显示"正在输入"
        await self.send_message(
            sender_id,
            "🤔 正在思考中...",
            receive_id_type="open_id"
        )
        
        # 调用OpenClaw
        async with self.openclaw:
            response = await self.openclaw.chat(
                message=user_message,
                model=self.config.default_model,
                conversation_id=conversation_id
            )
        
        # 保存会话ID
        if "conversation_id" in response:
            self.conversations[sender_id] = response["conversation_id"]
        
        # 发送回复
        reply_content = response.get("content", "抱歉，处理您的请求时出现了问题。")
        
        # 添加引用格式
        formatted_reply = f"💬 **回复**\n\n{reply_content}\n\n---\n*Powered by OpenClaw + {self.config.default_model}*"
        
        await self.send_message(
            sender_id,
            formatted_reply,
            msg_type="markdown",
            receive_id_type="open_id"
        )
    
    async def handle_event(self, event: Dict[str, Any]):
        """处理飞书事件"""
        
        event_type = event.get("header", {}).get("event_type")
        
        if event_type == "im.message.receive_v1":
            await self.handle_message(event.get("event", {}))
        elif event_type == "im.chat.member.user.added_v1":
            # 被添加进群
            chat_id = event.get("event", {}).get("chat_id")
            await self.send_message(
                chat_id,
                "👋 大家好！我是OpenClaw AI助手，\n"
                "可以直接@我提问，我会尽力帮助您！\n"
                "支持功能：问答、代码、分析、写作",
                receive_id_type="chat_id"
            )


# Flask Webhook服务
from flask import Flask, request, jsonify

app = Flask(__name__)

# 初始化机器人
config = FeishuConfig(
    app_id="cli_xxxxxxxxxxxxxxxx",  # 替换为你的App ID
    app_secret="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 替换为你的App Secret
    verify_token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 替换为你的Verify Token
    encrypt_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 替换为你的Encrypt Key
    openclaw_base_url="http://localhost:8080",
    default_model="qwen2.5"
)

bot = FeishuBot(config)


@app.route("/webhook/feishu", methods=["POST"])
def feishu_webhook():
    """飞书Webhook入口"""
    
    data = request.get_json()
    
    # 处理URL验证
    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        return jsonify({"challenge": challenge})
    
    # 验证签名（生产环境建议开启）
    # timestamp = request.headers.get("X-Lark-Request-Timestamp")
    # nonce = request.headers.get("X-Lark-Request-Nonce")
    # signature = request.headers.get("X-Lark-Signature")
    # body = request.get_data(as_text=True)
    # 
    # if not bot.verify_signature(timestamp, nonce, body, signature):
    #     return jsonify({"code": 403, "msg": "Invalid signature"}), 403
    
    # 处理事件
    event = data.get("event")
    if event:
        asyncio.run(bot.handle_event(event))
    
    return jsonify({"code": 0, "msg": "success"})


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "OpenClaw Feishu Bot",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

### 4.3 部署机器人服务

创建 `Dockerfile.bot`：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY feishu_bot.py .

# 暴露端口
EXPOSE 5000

# 启动服务
CMD ["python", "feishu_bot.py"]
```

创建 `requirements.txt`：

```
flask==3.0.0
aiohttp==3.9.0
```

添加到 `docker-compose.yml`：

```yaml
  feishu-bot:
    build:
      context: .
      dockerfile: Dockerfile.bot
    container_name: openclaw-feishu-bot
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      - FEISHU_APP_ID=cli_xxxxxxxxxxxxxxxx
      - FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      - FEISHU_VERIFY_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      - FEISHU_ENCRYPT_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      - OPENCLAW_BASE_URL=http://openclaw:8080
      - DEFAULT_MODEL=qwen2.5
    depends_on:
      - openclaw
    networks:
      - openclaw-network
```

### 4.4 高级功能：知识库集成

让机器人能够回答企业内部知识：

```python
class KnowledgeBase:
    """企业知识库"""
    
    def __init__(self, openclaw_url: str):
        self.openclaw_url = openclaw_url
        self.documents = []
    
    async def add_document(self, title: str, content: str, metadata: Dict = None):
        """添加文档到知识库"""
        
        url = f"{self.openclaw_url}/api/v1/knowledge/documents"
        
        payload = {
            "title": title,
            "content": content,
            "metadata": metadata or {}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                return await response.json()
    
    async def query(self, question: str, top_k: int = 3) -> List[Dict]:
        """检索相关知识"""
        
        url = f"{self.openclaw_url}/api/v1/knowledge/query"
        
        payload = {
            "query": question,
            "top_k": top_k
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                return data.get("documents", [])
    
    async def chat_with_knowledge(
        self,
        message: str,
        model: str = "qwen2.5"
    ) -> str:
        """基于知识库回答"""
        
        # 检索相关知识
        relevant_docs = await self.query(message)
        
        # 构建增强提示
        context = "\n\n".join([
            f"文档{i+1}: {doc['title']}\n{doc['content'][:500]}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        enhanced_prompt = f"""基于以下参考资料回答问题：

{context}

用户问题：{message}

请根据参考资料回答，如果资料不足以回答，请说明。"""
        
        # 调用模型
        async with aiohttp.ClientSession() as session:
            url = f"{self.openclaw_url}/api/v1/chat"
            async with session.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": enhanced_prompt}]
            }) as response:
                data = await response.json()
                return data.get("content", "")


# 在FeishuBot中添加知识库支持
class FeishuBotWithKB(FeishuBot):
    def __init__(self, config: FeishuConfig):
        super().__init__(config)
        self.kb = KnowledgeBase(config.openclaw_base_url)
    
    async def handle_message(self, event: Dict[str, Any]):
        """增强版消息处理，支持知识库"""
        
        message = event.get("message", {})
        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {}).get("open_id")
        
        content = json.loads(message.get("content", "{}"))
        user_message = content.get("text", "").strip()
        
        # 检查是否触发知识库模式
        if user_message.startswith("/kb "):
            # 知识库查询模式
            query = user_message[4:]
            await self.send_message(sender_id, "🔍 正在查询知识库...")
            
            response = await self.kb.chat_with_knowledge(query)
            
            await self.send_message(
                sender_id,
                f"📚 **知识库回答**\n\n{response}",
                msg_type="markdown"
            )
        else:
            # 普通对话模式
            await super().handle_message(event)
```

## 五、生产环境优化

### 5.1 性能监控

创建监控配置 `prometheus.yml`：

```yaml
scrape_configs:
  - job_name: 'openclaw'
    static_configs:
      - targets: ['openclaw:8080']
  
  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
```

### 5.2 负载均衡配置

```nginx
# nginx.conf
upstream openclaw_backend {
    least_conn;
    server openclaw:8080 weight=5;
    server openclaw-backup:8080 backup;
}

server {
    listen 80;
    server_name ai.yourcompany.com;
    
    location / {
        proxy_pass http://openclaw_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # WebSocket支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 5.3 备份策略

```bash
#!/bin/bash
# backup.sh - 每日备份脚本

BACKUP_DIR="/backup/openclaw/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 备份数据库
docker exec openclaw-postgres pg_dump -U openclaw openclaw > $BACKUP_DIR/database.sql

# 备份配置
cp -r config $BACKUP_DIR/

# 备份模型（可选，模型文件较大）
# cp -r ollama_data $BACKUP_DIR/

# 压缩
 tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
 rm -rf $BACKUP_DIR

# 保留最近7天备份
find /backup/openclaw -name "*.tar.gz" -mtime +7 -delete
```

## 六、常见问题与解决方案

### 6.1 模型加载失败

**问题**：Ollama无法加载大模型，显存不足

**解决**：
```bash
# 使用量化版本
ollama pull qwen2.5:14b-q4_K_M

# 或启用CPU推理
OLLAMA_CPU_ONLY=1 ollama serve
```

### 6.2 飞书消息延迟

**问题**：消息响应慢，用户体验差

**解决**：
```python
# 添加异步处理和流式响应
async def stream_response(self, message: str, sender_id: str):
    """流式响应，提升用户体验"""
    
    # 先发送"正在输入"
    await self.send_message(sender_id, "🤔 思考中...")
    
    # 流式获取响应
    buffer = ""
    last_update = time.time()
    
    async for chunk in self.openclaw.stream_chat(message):
        buffer += chunk
        
        # 每2秒更新一次消息
        if time.time() - last_update > 2:
            await self.update_message(
                message_id,
                f"💬 回复中...\n\n{buffer}..."
            )
            last_update = time.time()
    
    # 发送最终回复
    await self.update_message(message_id, buffer)
```

### 6.3 高并发处理

**问题**：多人同时使用时响应慢

**解决**：
```yaml
# docker-compose.yml 扩展
services:
  openclaw:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 16G
    
  ollama:
    deploy:
      replicas: 2
```

## 七、总结：企业AI的自主可控之路

通过OpenClaw + Ollama + 飞书机器人的组合，我们实现了：

✅ **数据安全**：所有数据留在企业内网  
✅ **成本可控**：无需按Token付费，一次性投入  
✅ **响应快速**：内网延迟<50ms  
✅ **灵活扩展**：支持多种国产模型和业务集成  
✅ **用户体验**：与飞书无缝集成，零学习成本

这不仅是技术的胜利，更是**企业AI自主可控**的实践。

当其他公司还在为OpenAI的API限流发愁时，你已经拥有了自己的AI基础设施。

当其他公司担心数据泄露时，你的数据安全地跑在自己的服务器上。

这就是OpenClaw + Ollama带来的**企业级AI自由**。

---

**项目地址**：https://github.com/openclaw/openclaw  
**文档中心**：https://docs.openclaw.io  
**社区论坛**：https://forum.openclaw.io

**相关阅读**：
- [Ollama深度解析](095-ollama-deep-dive.md)
- [AI全链路知识图谱](098-ai-knowledge-map.md)
