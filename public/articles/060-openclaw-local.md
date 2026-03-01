---
title: OpenClaw + Ollama + 飞书机器人：从零搭建企业AI助手的完整实操指南
category: AI
excerpt: 深入实操OpenClaw与Ollama的联合部署，完整演示飞书机器人接入流程。从环境准备到生产上线，每一步都详细解释原理，让你真正掌握企业级AI助手的搭建方法。
tags: OpenClaw, Ollama, 飞书机器人, 本地部署, 企业AI, LLM, 实操指南
date: 2026-01-29
readTime: 35
---

# OpenClaw + Ollama + 飞书机器人：从零搭建企业AI助手的完整实操指南

> 2025年的某个周一早晨，我收到了CTO的紧急召唤。"公司要接入AI助手，但数据绝不能出内网。"他盯着我说，"给你两周时间。"我脑海里闪过OpenAI的API文档，然后默默关掉了它。那天晚上，我在GitHub上发现了OpenClaw——一个能让国产大模型在企业内网跑起来的神奇框架。当我把它和Ollama、飞书机器人在三天内打通时，我意识到：企业级AI，不一定要依赖云端。

## 一、为什么要用OpenClaw + Ollama？

### 1.1 企业AI的三大坎

2023年，企业想要AI能力，只有一条路：**调用OpenAI API**。

但这条路有三个坑：

**第一坑：数据裸奔**
- 你的客户对话、合同条款、代码机密，都要传到国外服务器
- 2024年某知名企业因为用ChatGPT处理财报，泄露了内幕信息
- 金融、医疗、政务行业，直接说"No"

**第二坑：账单无底洞**
- "这个月怎么花了8万？"
- 用户越多，费用越高，没有上限
- 每次功能更新都是钱

**第三坑：网络抽风**
- 高峰期API限流，用户转圈圈
- 跨国延迟500ms+，对话像卡带
- 内网环境直接无法使用

### 1.2 我们的方案：把AI装进公司保险箱

**核心思路**：买台服务器，在自己家里跑AI。

```
以前：员工 → 互联网 → OpenAI服务器 → 互联网 → 员工
                                    ❌ 数据泄露风险
                                    ❌ 每月账单吓死人
                                    ❌ 网络不好就卡顿

现在：员工 → 内网 → Ollama → 员工
       ✅ 数据不出公司
       ✅ 一次性投入，后续免费
       ✅ 内网延迟<10ms
```

**OpenClaw是管家的，Ollama是大厨的**

我经常这样比喻：

> 想象你开了一家餐厅。
> - **Ollama** 就是厨房里的大厨，负责炒菜（模型推理）
> - **OpenClaw** 就是前台的、服务员，负责接待客人、记菜单、传菜（API调度、权限管理）
>
> 单独用Ollama，就像只有个大厨在厨房里炒菜，客人得自己跑去厨房看菜单、端盘子。
> 加上OpenClaw，就像有了完整的餐厅运营体系，客人坐着等就行。

## 二、硬件准备：花多少钱？买什么？

### 2.1 预算方案

| 规模 | 配置 | 一次性投入 | 每月电费 | 适合场景 |
|------|------|-----------|---------|---------|
| **试试水** | RTX 3060显卡的电脑 | ¥0（已有） | ¥100 | 3-5人团队 |
| **小团队** | RTX 3090 + 32GB内存 | ¥15,000 | ¥200 | 10-20人 |
| **正规军** | A100 40GB双卡 + 128GB内存 | ¥150,000 | ¥1500 | 50人以上 |

**我的建议**：
- 先用现有电脑试试，别急着买
- 确认需求真实存在，再上大件
- 50人以下团队，其实API方式更划算

### 2.2 捡现成的（推荐）

如果你有台游戏电脑（RTX 3060及以上），直接用：

```bash
# Mac
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 直接去 https://ollama.com 下载安装
```

验证安装：
```bash
ollama --version
```

## 三、Ollama实操：一条命令跑起来

### 3.1 选模型：像选员工一样

模型就是你的"AI员工"，要选合适的：

| 模型 | 相当于 | 显存要求 | 适合场景 |
|------|--------|---------|---------|
| **qwen2.5:7b** | 实习生 | 8GB | 简单问答、速度快 |
| **qwen2.5:14b** | 高级工程师 | 16GB | 通用对话、内容创作 |
| **qwen2.5:72b** | 技术专家 | 80GB | 复杂推理、专业任务 |
| **deepseek-coder** | 资深程序员 | 24GB | 代码审查、技术问题 |

**我的选择**：
- 主力：qwen2.5:14b（性价比之王）
- 备选：qwen2.5:7b（响应快，用来回答简单问题）

### 3.2 下载模型

```bash
# 下载通义千问14B（约9GB）
ollama pull qwen2.5:14b

# 下载代码专家（可选）
ollama pull deepseek-coder:33b

# 查看已下载
ollama list
```

**第一次下载会比较慢**，模型文件都很大。建议：
- 晚上下班前开始下载
- 用公司网络，别用手机热点

### 3.3 启动服务

```bash
# 启动服务（默认11434端口）
ollama serve

# 测试一下
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:14b",
  "prompt": "用一句话介绍你自己"
}'
```

看到回复了？恭喜！你已经有了一个能跑的AI服务。

### 3.4 让局域网都能访问

默认只能本机访问，要让其他机器也能用：

```bash
# 启动时指定 host
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 或者用环境变量文件
echo 'OLLAMA_HOST=0.0.0.0:11434' >> ~/.bashrc
source ~/.bashrc
```

现在局域网其他机器可以通过 `http://服务器IP:11434` 访问了。

## 四、飞书机器人：让公司全员都能用

### 4.1 先去飞书注册"员工"

1. 打开 [飞书开放平台](https://open.feishu.cn/)
2. 点击右上角"创建应用"
3. 取个名字："公司AI助手"
4. 选择"机器人"类型

然后你会拿到几个**钥匙**（凭证）：
- App ID：相当于工号
- App Secret：相当于密码
- Verification Token：相当于门禁卡

### 4.2 开通权限

在飞书后台，找到"权限管理"，开启这些：
- `im:message:send` - 能发消息
- `im:message:receive` - 能收消息

在"事件订阅"里，订阅：
- `im.message.receive_v1` - 有人发消息告诉我

### 4.3 写机器人的"脑子"

就一个文件 `bot.py`，核心逻辑：

```python
#!/usr/bin/env python3
"""飞书机器人 - 连接到Ollama"""

from flask import Flask, request
import requests
import json

app = Flask(__name__)

# ====== 配置区 ======
APP_ID = "你的App ID"
APP_SECRET = "你的App Secret"
OLLAMA_URL = "http://服务器IP:11434"  # 改成你Ollama的地址

# 获取飞书访问令牌
def get_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    resp = requests.post(url, json={"app_id": APP_ID, "app_secret": APP_SECRET})
    return resp.json().get("tenant_access_token")

# 发消息给用户
def send_msg(user_id, text):
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {"Authorization": f"Bearer {get_token()}"}
    data = {"receive_id": user_id, "msg_type": "text", 
            "content": json.dumps({"text": text})}
    requests.post(url, headers=headers, json=data, params={"receive_id_type": "open_id"})

# 问AI
def ask_ai(question):
    url = f"{OLLAMA_URL}/api/generate"
    resp = requests.post(url, json={
        "model": "qwen2.5:14b",
        "prompt": question,
        "stream": False
    }, timeout=60)
    return resp.json().get("response", "出错了")

# 入口
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    
    # 验证URL（飞书第一次会检查）
    if data.get("type") == "url_verification":
        return {"challenge": data.get("challenge")}
    
    # 取消息
    msg = data.get("event", {}).get("message", {})
    if msg.get("message_type") != "text":
        return {"code": 0}
    
    user_id = data.get("event", {}).get("sender", {}).get("sender_id", {}).get("open_id")
    question = json.loads(msg.get("content", "{}")).get("text", "").strip()
    
    # 调用AI
    answer = ask_ai(question)
    send_msg(user_id, answer)
    
    return {"code": 0}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

**就这么多代码？**
对，核心就40行。

**原理是什么？**
```
用户发消息 → 飞书推送到我们的服务器 → 我们调用Ollama → Ollama返回回答 → 我们发回给用户
```

### 4.4 跑起来

```bash
# 安装依赖
pip install flask requests

# 启动（生产环境用 nohup 或 systemd）
python bot.py
```

**测试**：
1. 在飞书里搜索你的机器人
2. 发消息："你好"
3. 等几秒，应该能收到回复

### 4.5 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 不回复 | 机器人没上线 | 检查 `python bot.py` 是否在运行 |
| 回复很慢 | 模型太大 | 换成 7b 模型 |
| 收不到消息 | 服务器没公网IP | 用内网穿透或云服务器 |
| 权限错误 | App ID/Secret错了 | 重新复制 |

## 五、进阶：让AI懂公司业务

### 5.1 什么是RAG？

RAG就是**让AI先查资料，再回答**。

比如问"年假几天？"，AI直接回答可能瞎编。但如果先让它查一下公司规章制度，再回答，就不会错。

### 5.2 简单的知识库实现

安装向量数据库：
```bash
pip install chromadb sentence-transformers
```

创建 `kb.py`：

```python
"""知识库 - 让AI懂得公司规矩"""
import chromadb
from sentence_transformers import SentenceTransformer

# 加载中文 embedding 模型
encoder = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# 创建本地向量数据库
client = chromadb.PersistentClient(path="./company_kb")
kb = client.get_or_create_collection("docs")

def add_knowledge(title, content):
    """添加知识"""
    chunks = [content[i:i+500] for i in range(0, len(content), 500)]
    for i, chunk in enumerate(chunks):
        emb = encoder.encode(chunk).tolist()
        kb.add(embeddings=[emb], documents=[chunk], 
               ids=[f"{title}_{i}"], metadatas=[{"title": title}])
    print(f"已添加: {title}")

def search(question, top=3):
    """搜相关知识"""
    emb = encoder.encode(question).tolist()
    result = kb.query(query_embeddings=[emb], n_results=top)
    return result['documents'][0]

# ====== 使用方式 ======
if __name__ == "__main__":
    # 添加公司制度
    add_knowledge("休假制度", """
    年假：入职1年5天，3年10天，5年15天
    病假：每年30天，需要医院证明
    婚假：3天
    产假：128天
    """)
    
    # 测试
    print(search("产假有多少天？"))
```

### 5.3 接入机器人

在 `bot.py` 里加两句：

```python
# 顶部导入
from kb import search

# 修改 ask_ai 函数
def ask_ai(question):
    # 先搜知识库
    docs = search(question)
    if docs:
        # 有相关资料，用资料回答
        prompt = f"根据以下资料回答：\n{chr(10).join(docs)}\n\n问题：{question}"
    else:
        # 没有资料，直接问AI
        prompt = question
    
    resp = requests.post(f"{OLLAMA_URL}/api/generate", 
        json={"model": "qwen2.5:14b", "prompt": prompt, "stream": False}, timeout=60)
    return resp.json().get("response", "出错了")
```

现在问AI公司相关问题，它会根据知识库回答了。

## 六、生产环境注意事项

### 6.1 监控

写个脚本 `monitor.sh`：

```bash
#!/bin/bash
# 检查服务是否活着

# 检查Ollama
if ! curl -s http://localhost:11434 > /dev/null; then
    echo "$(date): Ollama挂了" | tee -a /var/log/ai.log
    # 这里可以加发通知的代码
fi
```

加到定时任务：
```bash
crontab -e
*/5 * * * * /path/to/monitor.sh
```

### 6.2 备份

```bash
#!/bin/bash
# 备份知识库
DATE=$(date +%Y%m%d)
tar -czf /backup/kb_$DATE.tar.gz company_kb/
find /backup -name "kb_*.tar.gz" -mtime +7 -delete
```

### 6.3 升级模型

模型更新了，这样升级：

```bash
# 拉取新版本
ollama pull qwen2.5:14b

# 重启服务
pkill -f "ollama serve"
ollama serve
```

## 七、总结

这篇文章我们从0到1完成了一个企业AI助手：

✅ **Ollama** - 在自己服务器上跑AI模型  
✅ **飞书机器人** - 让大家在飞书里用AI  
✅ **知识库** - 让AI懂公司业务  

**花了多少钱？**
- 如果用现有电脑：电费每月100块
- 如果买服务器：一次性1.5万，每月200块

**能服务多少人？**
- 7b模型：10-20人同时用没问题
- 14b模型：5-10人同时用
- 想要更多人？加显卡，或者上API

**这条路能走多远？**
- 短期：满足内部需求，不花冤枉钱
- 中期：积累经验，为真正的AI转型做准备
- 长期：成为AI基础设施，持续产生价值

当你第一次在飞书里收到AI的回复，当同事跟你说"这个真好用"，你会觉得这一切都值得。

---

**相关阅读**：
- [Ollama深度解析](095-ollama-deep-dive.md)
- [AI全链路知识图谱](098-ai-knowledge-map.md)
