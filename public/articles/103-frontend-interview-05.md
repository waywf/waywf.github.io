---
title: 05.前端高频面试题深度解析：从 SSR 到系统设计的巅峰对决
category: 前端开发
excerpt: 前端面试题第五弹！深入解析 SSR/SSG、WebWorker、前端测试、WebSocket、PWA、系统设计等高频考点。冲刺前端架构师岗位的必备秘籍！
tags: 前端面试, SSR, WebWorker, 前端测试, WebSocket, PWA, 系统设计
date: 2026-03-01
readTime: 55
---

# 05.前端高频面试题深度解析：从 SSR 到系统设计的巅峰对决

> 面试官："说说 SSR 的原理和优化策略。"
> 
> 你："就是...服务端渲染...更快的首屏..."
> 
> 面试官："那 SSR 的缺点呢？服务端压力大怎么办？CSR、SSR、SSG、ISR 的区别是什么？"
> 
> 你："..."
> 
> 恭喜你来到前端面试的巅峰战场！这次我们深入 SSR/SSG、WebWorker、前端测试、WebSocket、PWA、系统设计等高级领域。准备好了吗？

## 一、服务端渲染（SSR）篇

### 1.1 SSR 原理与实现

**面试题：什么是 SSR？请解释 SSR 的工作原理，并手写一个简单的 SSR。**

#### SSR vs CSR 对比

```
────────────────────────────────────────────────────────────────
                      CSR (客户端渲染)
────────────────────────────────────────────────────────────────

浏览器                                  服务器
  |                                       |
  |───────────────── GET / ─────────────→|
  |                                       |
  |←─────────── index.html ──────────────|
  |   (只有一个 div#app)                  |
  |                                       |
  |←─────────── app.js ──────────────────|
  |←─────────── vendor.js ───────────────|
  |                                       |
  |   执行 JS，请求数据                    |
  |───────────────── API ───────────────→|
  |                                       |
  |←─────────── JSON 数据 ───────────────|
  |                                       |
  |   渲染页面                            |
  |   用户看到内容（首屏时间较长）         |
  |                                       |
────────────────────────────────────────────────────────────────
                      SSR (服务端渲染)
────────────────────────────────────────────────────────────────

浏览器                                  服务器
  |                                       |
  |───────────────── GET / ─────────────→|
  |                                       |
  |   执行 SSR：                          |
  |   1. 请求数据                         |
  |   2. 渲染 HTML 字符串                 |
  |                                       |
  |←───────── 完整的 HTML ───────────────|
  |   (用户立即看到内容)                   |
  |                                       |
  |←─────────── app.js ──────────────────|
  |←─────────── vendor.js ───────────────|
  |                                       |
  |   水合（Hydration）：                 |
  |   将静态 HTML 变成可交互页面          |
  |                                       |
────────────────────────────────────────────────────────────────
```

#### 手写简单 SSR

```javascript
// 服务端代码（Express + React SSR）
const express = require('express');
const React = require('react');
const { renderToString } = require('react-dom/server');
const App = require('./src/App');

const app = express();

app.get('/', async (req, res) => {
  // 1. 获取数据
  const data = await fetchData();
  
  // 2. 渲染 React 组件为 HTML 字符串
  const html = renderToString(
    React.createElement(App, { data })
  );
  
  // 3. 拼接完整的 HTML
  const fullHtml = `
    <!DOCTYPE html>
    <html>
      <head>
        <title>SSR Demo</title>
        <script>window.__INITIAL_DATA__ = ${JSON.stringify(data)}</script>
      </head>
      <body>
        <div id="app">${html}</div>
        <script src="/app.js"></script>
      </body>
    </html>
  `;
  
  res.send(fullHtml);
});

app.listen(3000);

// 客户端代码（水合）
const React = require('react');
const { hydrateRoot } = require('react-dom/client');
const App = require('./App');

const root = document.getElementById('app');
const initialData = window.__INITIAL_DATA__;

hydrateRoot(root, React.createElement(App, { data: initialData }));
```

### 1.2 SSR 生态与优化

**面试题：SSR 的优缺点是什么？有哪些优化策略？CSR、SSR、SSG、ISR 有什么区别？**

#### 优缺点分析

```javascript
// SSR 优点
1. 更快的首屏加载（FCP）
2. 更好的 SEO（搜索引擎可以直接抓取内容）
3. 更好的用户体验（减少白屏时间）

// SSR 缺点
1. 服务端压力大（需要执行渲染）
2. 开发复杂度高
3. 更多的服务器资源消耗
4. 某些库不支持 SSR（依赖 window、document）
```

#### 优化策略

```javascript
// 1. 流式渲染（Streaming SSR）
const { renderToPipeableStream } = require('react-dom/server');

app.get('/', (req, res) => {
  const { pipe } = renderToPipeableStream(
    React.createElement(App),
    {
      onAllReady() {
        res.setHeader('Content-Type', 'text/html');
        pipe(res);
      }
    }
  );
});

// 2. 组件级缓存
const cache = new Map();

async function renderComponent(name, props) {
  const key = `${name}-${JSON.stringify(props)}`;
  
  if (cache.has(key)) {
    return cache.get(key);
  }
  
  const html = renderToString(React.createElement(Component, props));
  cache.set(key, html);
  
  // 设置过期时间
  setTimeout(() => cache.delete(key), 60000);
  
  return html;
}

// 3. 数据预取与缓存
// 服务端数据预取
// 组件级别：getServerSideProps（Next.js）
// 路由级别：在路由守卫中预取

// 4. 按需加载与代码分割
// 使用 dynamic import，只加载需要的代码
const LazyComponent = React.lazy(() => import('./LazyComponent'));

// 5. 减少水合时间
// 只给需要交互的元素添加事件监听
// 使用部分水合（Partial Hydration）
```

#### CSR vs SSR vs SSG vs ISR

```javascript
// CSR (Client-Side Rendering) - 客户端渲染
// 适用于：后台管理系统、需要频繁交互的应用
// 特点：首屏慢，但后续交互快；SEO 较差

// SSR (Server-Side Rendering) - 服务端渲染
// 适用于：电商首页、新闻资讯
// 特点：首屏快，SEO 好；服务器压力大

// SSG (Static Site Generation) - 静态生成
// 适用于：博客、文档、营销页
// 特点：构建时生成 HTML，部署 CDN；无法处理动态数据

// ISR (Incremental Static Regeneration) - 增量静态再生成
// 适用于：更新频率不高的动态内容
// 特点：结合 SSG 和 SSR 的优点，定期更新页面

// 对比表格
/*
┌────────────┬──────────┬──────┬──────────┬────────────┐
│   类型     │ 首屏时间 │ SEO  │ 服务器   │ 适用场景   │
├────────────┼──────────┼──────┼──────────┼────────────┤
│   CSR      │    慢    │  差  │    低    │ 后台系统   │
│   SSR      │    快    │  好  │    高    │ 电商首页   │
│   SSG      │   最快   │  好  │    低    │ 博客文档   │
│   ISR      │   快     │  好  │   中     │ 新闻资讯   │
└────────────┴──────────┴──────┴──────────┴────────────┘
*/
```

## 二、WebWorker 篇

### 2.1 WebWorker 原理

**面试题：什么是 WebWorker？WebWorker 可以做什么？有什么限制？**

#### WebWorker 概念

```javascript
// JavaScript 是单线程的，WebWorker 让我们可以使用多线程
// 主线程负责：DOM 操作、用户交互
// Worker 线程负责：耗时计算、数据处理

// 工作原理
┌─────────────────────────────────────────────────────────┐
│                    主线程 (Main Thread)                   │
│  ┌─────────────────┐     ┌─────────────────┐          │
│  │   DOM 操作      │     │   UI 更新       │          │
│  └────────┬────────┘     └────────┬────────┘          │
│           │                         │                    │
│  ┌───────────────────────────────────────────────┐      │
│  │         postMessage()  ←──────────→           │      │
│  │              ↓                     ↑           │      │
│  │  ┌────────────────────────────────────────┐  │      │
│  │  │     Worker 线程 (Worker Thread)        │  │      │
│  │  │  ┌────────────────────────────────┐   │  │      │
│  │  │  │   耗时计算、数据处理            │   │  │      │
│  │  │  └────────────────────────────────┘   │  │      │
│  │  └────────────────────────────────────────┘  │      │
│  └───────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

#### WebWorker 限制

```javascript
// ❌ 不能访问的
1. DOM API
2. window 对象
3. document 对象
4. parent 对象

// ✓ 可以访问的
1. navigator 对象
2. location 对象（只读）
3. XMLHttpRequest / fetch
4. setTimeout / setInterval
5. IndexedDB
6. WebSocket
```

#### WebWorker 实战

```javascript
// 主线程代码
const worker = new Worker('worker.js');

// 发送数据给 Worker
worker.postMessage({ type: 'calculate', data: 1000000 });

// 接收 Worker 返回的数据
worker.onmessage = function(event) {
  console.log('计算结果:', event.data);
  worker.terminate();  // 终止 Worker
};

// 错误处理
worker.onerror = function(error) {
  console.error(`Worker 错误: ${error.message}`);
  worker.terminate();
};

// worker.js (Worker 线程代码)
self.onmessage = function(event) {
  const { type, data } = event.data;
  
  if (type === 'calculate') {
    const result = heavyCalculation(data);
    self.postMessage(result);
  }
};

function heavyCalculation(n) {
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.sqrt(i);
  }
  return sum;
}

// 使用 SharedWorker（多标签共享）
// main.js
const sharedWorker = new SharedWorker('shared-worker.js');

sharedWorker.port.onmessage = function(event) {
  console.log('来自 SharedWorker 的消息:', event.data);
};

sharedWorker.port.start();
sharedWorker.port.postMessage('Hello SharedWorker!');

// shared-worker.js
let connections = 0;

self.onconnect = function(event) {
  connections++;
  const port = event.ports[0];
  
  port.postMessage(`你是第 ${connections} 个连接`);
  
  port.onmessage = function(event) {
    port.postMessage(`收到: ${event.data}`);
  };
};
```

### 2.2 WebWorker 应用场景

```javascript
// 场景 1：复杂计算
// 斐波那契数列、矩阵运算、图像处理

// 场景 2：大数据处理
// 处理大量数据排序、筛选、聚合
const processLargeData = (data) => {
  const worker = new Worker('data-processor.js');
  
  worker.postMessage(data);
  
  return new Promise((resolve) => {
    worker.onmessage = (event) => {
      resolve(event.data);
      worker.terminate();
    };
  });
};

// 场景 3：后台任务
// 定期轮询、日志上传、数据同步
class BackgroundWorker {
  constructor() {
    this.worker = new Worker('background.js');
  }
  
  startPolling(interval) {
    this.worker.postMessage({ type: 'poll', interval });
  }
  
  stop() {
    this.worker.postMessage({ type: 'stop' });
  }
}

// 场景 4：文件处理
// 压缩、解压、解析大文件
const compressFile = (file) => {
  return new Promise((resolve) => {
    const worker = new Worker('compressor.js');
    
    worker.postMessage(file);
    
    worker.onmessage = (event) => {
      resolve(event.data);
      worker.terminate();
    };
  });
};
```

## 三、前端测试篇

### 3.1 测试体系

**面试题：前端测试有哪些类型？请解释单元测试、集成测试、E2E 测试的区别。**

#### 测试金字塔

```
                    ┌─────────────────────┐
                    │   E2E 测试 (10%)    │
                    │   (端到端测试)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  集成测试 (30%)     │
                    │  (组件/模块集成)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  单元测试 (60%)     │
                    │  (函数/类测试)       │
                    └─────────────────────┘
```

#### 测试类型详解

```javascript
// 1. 单元测试 (Unit Test)
// 测试最小可测试单元（函数、类、组件）
// 使用：Jest, Vitest, Mocha

// 示例：测试一个工具函数
// utils.js
export function add(a, b) {
  return a + b;
}

// utils.test.js
import { add } from './utils';

test('adds 1 + 2 to equal 3', () => {
  expect(add(1, 2)).toBe(3);
});

test('adds negative numbers', () => {
  expect(add(-1, -1)).toBe(-2);
});

// 2. 组件测试 (Component Test)
// 使用：React Testing Library, Vue Test Utils

// Button.jsx
export function Button({ onClick, children }) {
  return (
    <button onClick={onClick} data-testid="button">
      {children}
    </button>
  );
}

// Button.test.jsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

test('renders button with text', () => {
  render(<Button>Click Me</Button>);
  expect(screen.getByText('Click Me')).toBeInTheDocument();
});

test('calls onClick when clicked', () => {
  const handleClick = jest.fn();
  render(<Button onClick={handleClick}>Click</Button>);
  
  fireEvent.click(screen.getByTestId('button'));
  expect(handleClick).toHaveBeenCalledTimes(1);
});

// 3. 集成测试 (Integration Test)
// 测试多个模块一起工作的情况
// 示例：测试登录流程
test('login flow', async () => {
  render(<App />);
  
  // 输入用户名
  fireEvent.change(screen.getByLabelText('用户名'), {
    target: { value: 'testuser' }
  });
  
  // 输入密码
  fireEvent.change(screen.getByLabelText('密码'), {
    target: { value: 'password123' }
  });
  
  // 点击登录
  fireEvent.click(screen.getByText('登录'));
  
  // 等待跳转
  await waitFor(() => {
    expect(screen.getByText('欢迎回来')).toBeInTheDocument();
  });
});

// 4. E2E 测试 (End-to-End)
// 测试整个应用流程，模拟真实用户操作
// 使用：Cypress, Playwright, Puppeteer

// cypress/e2e/login.cy.js
describe('登录流程', () => {
  it('成功登录', () => {
    cy.visit('/login');
    
    cy.get('[data-testid="username"]').type('testuser');
    cy.get('[data-testid="password"]').type('password123');
    cy.get('[data-testid="submit"]').click();
    
    cy.url().should('include', '/dashboard');
    cy.contains('欢迎回来');
  });
});

// Playwright 示例
import { test, expect } from '@playwright/test';

test('登录测试', async ({ page }) => {
  await page.goto('/login');
  
  await page.fill('[data-testid="username"]', 'testuser');
  await page.fill('[data-testid="password"]', 'password123');
  await page.click('[data-testid="submit"]');
  
  await expect(page).toHaveURL(/dashboard/);
  await expect(page.locator('text=欢迎回来')).toBeVisible();
});
```

### 3.2 测试最佳实践

```javascript
// 1. AAA 模式 (Arrange-Act-Assert)
test('测试示例', () => {
  // Arrange: 准备数据
  const a = 1;
  const b = 2;
  
  // Act: 执行操作
  const result = add(a, b);
  
  // Assert: 断言结果
  expect(result).toBe(3);
});

// 2. Mock 外部依赖
import axios from 'axios';
jest.mock('axios');

test('获取用户数据', async () => {
  const mockUser = { id: 1, name: '张三' };
  
  // Mock axios.get
  axios.get.mockResolvedValue({ data: mockUser });
  
  // 调用函数
  const user = await fetchUser(1);
  
  // 验证
  expect(axios.get).toHaveBeenCalledWith('/api/user/1');
  expect(user).toEqual(mockUser);
});

// 3. 测试覆盖率
// 运行 jest --coverage 查看覆盖率
// 目标：至少 80% 的覆盖率

// 4. 测试应该独立
// 每个测试不应该依赖其他测试的执行顺序
// 使用 beforeEach/afterEach 清理

describe('计数器测试', () => {
  let counter;
  
  beforeEach(() => {
    counter = new Counter();  // 每个测试前重置
  });
  
  test('初始值为 0', () => {
    expect(counter.value).toBe(0);
  });
  
  test('increment 加 1', () => {
    counter.increment();
    expect(counter.value).toBe(1);
  });
});
```

## 四、实时通信篇

### 4.1 WebSocket

**面试题：WebSocket 和 HTTP 有什么区别？请实现一个简单的 WebSocket 聊天应用。**

#### WebSocket vs HTTP

```javascript
// HTTP：短连接，请求-响应模式
// 每次请求都需要建立新的 TCP 连接（或复用）
// 适用于：数据获取、表单提交

// WebSocket：长连接，全双工通信
// 一次握手，持续连接
// 适用于：实时聊天、通知、在线游戏

// 对比
/*
┌──────────────┬──────────────────┬──────────────────┐
│     特性     │       HTTP       │     WebSocket    │
├──────────────┼──────────────────┼──────────────────┤
│   连接方式   │  短连接/复用     │   长连接         │
│   通信方式   │  请求-响应       │   全双工         │
│   协议开销   │  较大（Header）  │   很小           │
│   适用场景   │  数据获取        │   实时通信       │
└──────────────┴──────────────────┴──────────────────┘
*/
```

#### WebSocket 实现

```javascript
// 服务端（Node.js + ws）
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

// 存储所有连接的客户端
const clients = new Set();

wss.on('connection', (ws) => {
  console.log('新客户端连接');
  clients.add(ws);
  
  // 给新客户端发送欢迎消息
  ws.send(JSON.stringify({
    type: 'system',
    message: '欢迎加入聊天室！'
  }));
  
  // 广播给其他客户端
  broadcast({
    type: 'system',
    message: '有新用户加入了聊天室'
  }, ws);
  
  // 接收客户端消息
  ws.on('message', (data) => {
    const message = JSON.parse(data);
    console.log('收到消息:', message);
    
    // 广播给所有客户端
    broadcast(message);
  });
  
  // 连接关闭
  ws.on('close', () => {
    console.log('客户端断开连接');
    clients.delete(ws);
    
    broadcast({
      type: 'system',
      message: '有用户离开了聊天室'
    });
  });
  
  // 错误处理
  ws.on('error', (error) => {
    console.error('WebSocket 错误:', error);
  });
});

// 广播消息
function broadcast(message, excludeClient = null) {
  clients.forEach((client) => {
    if (client !== excludeClient && client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(message));
    }
  });
}

// 客户端代码
class ChatClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }
  
  connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.onopen = () => {
      console.log('连接成功');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.onMessage(message);
    };
    
    this.ws.onclose = (event) => {
      console.log('连接关闭，尝试重连...');
      this.reconnect();
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket 错误:', error);
    };
  }
  
  reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      
      setTimeout(() => {
        console.log(`第 ${this.reconnectAttempts} 次重连...`);
        this.connect();
      }, delay);
    }
  }
  
  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
  
  onMessage(message) {
    console.log('收到消息:', message);
    // 在这里处理消息
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 使用
const client = new ChatClient('ws://localhost:8080');
client.connect();

client.send({
  type: 'chat',
  user: '张三',
  message: '大家好！'
});
```

### 4.2 SSE（Server-Sent Events）

```javascript
// SSE：服务器向客户端推送消息（单向）
// 适用于：实时通知、数据更新

// 服务端（Node.js + Express）
app.get('/events', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  // 发送事件
  const sendEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };
  
  // 定期发送心跳
  const heartbeat = setInterval(() => {
    sendEvent('heartbeat', { time: Date.now() });
  }, 30000);
  
  // 发送消息
  sendEvent('message', { text: '欢迎！' });
  
  // 清理
  req.on('close', () => {
    clearInterval(heartbeat);
  });
});

// 客户端
const eventSource = new EventSource('/events');

eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  console.log('收到消息:', data);
});

eventSource.addEventListener('heartbeat', (event) => {
  const data = JSON.parse(event.data);
  console.log('心跳:', data.time);
});

eventSource.onerror = (error) => {
  console.error('EventSource 错误:', error);
  eventSource.close();
};
```

## 五、PWA 篇

### 5.1 Service Worker

**面试题：什么是 Service Worker？它可以做什么？请实现一个简单的离线缓存。**

#### Service Worker 概念

```javascript
// Service Worker 是一种独立于网页运行的脚本
// 位于浏览器和网络之间，可以拦截请求、缓存资源
// 生命周期：install → activate → fetch

/*
┌─────────────────────────────────────────────────────────┐
│                     浏览器                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Service Worker                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│  │  │ Install  │→ │ Activate │→ │  Fetch   │     │  │
│  │  └──────────┘  └──────────┘  └──────────┘     │  │
│  └───────────────────────────────────────────────────┘  │
│         ↑                        ↓                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │                  缓存存储 (Cache)                  │  │
│  └───────────────────────────────────────────────────┘  │
│         ↑                        ↓                        │
│  ┌──────────┐         ┌──────────────────────┐         │
│  │  网页    │         │      网络            │         │
│  └──────────┘         └──────────────────────┘         │
└─────────────────────────────────────────────────────────┘
*/
```

#### Service Worker 实现

```javascript
// service-worker.js
const CACHE_NAME = 'my-pwa-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/index.html',
  '/styles.css',
  '/app.js',
  '/offline.html'
];

// Install：安装时缓存资源
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service Worker: Caching assets');
        return cache.addAll(ASSETS_TO_CACHE);
      })
      .then(() => {
        console.log('Service Worker: Install completed');
        return self.skipWaiting();  // 立即激活
      })
  );
});

// Activate：清理旧缓存
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME) {
              console.log('Service Worker: Deleting old cache', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('Service Worker: Activate completed');
        return self.clients.claim();  // 立即控制所有客户端
      })
  );
});

// Fetch：拦截请求
self.addEventListener('fetch', (event) => {
  console.log('Service Worker: Fetching', event.request.url);
  
  // 策略 1：缓存优先（静态资源）
  if (event.request.destination === 'image' || 
      event.request.destination === 'style' || 
      event.request.destination === 'script') {
    event.respondWith(
      caches.match(event.request)
        .then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          return fetch(event.request)
            .then((response) => {
              // 缓存新资源
              return caches.open(CACHE_NAME)
                .then((cache) => {
                  cache.put(event.request, response.clone());
                  return response;
                });
            });
        })
        .catch(() => {
          // 网络和缓存都失败，返回离线页面
          if (event.request.destination === 'document') {
            return caches.match('/offline.html');
          }
        })
    );
  }
  
  // 策略 2：网络优先（动态数据）
  else if (event.request.url.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          // 缓存 API 响应
          return caches.open(CACHE_NAME)
            .then((cache) => {
              cache.put(event.request, response.clone());
              return response;
            });
        })
        .catch(() => {
          // 网络失败，返回缓存
          return caches.match(event.request);
        })
    );
  }
  
  // 策略 3：Stale-While-Revalidate（快速响应，后台更新）
  else {
    event.respondWith(
      caches.match(event.request)
        .then((cachedResponse) => {
          // 后台更新缓存
          const fetchPromise = fetch(event.request)
            .then((response) => {
              caches.open(CACHE_NAME)
                .then((cache) => {
                  cache.put(event.request, response.clone());
                });
              return response;
            });
          
          // 优先返回缓存
          return cachedResponse || fetchPromise;
        })
    );
  }
});

// 主线程注册 Service Worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then((registration) => {
        console.log('Service Worker 注册成功:', registration);
        
        // 监听更新
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing;
          console.log('发现新版本');
          
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              console.log('新版本已安装，等待激活');
              // 提示用户刷新
            }
          });
        });
      })
      .catch((error) => {
        console.error('Service Worker 注册失败:', error);
      });
  });
}
```

### 5.2 PWA 完整配置

```javascript
// manifest.json - Web App Manifest
{
  "name": "我的 PWA 应用",
  "short_name": "PWA",
  "description": "一个很棒的 PWA 应用",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#4285f4",
  "orientation": "portrait-primary",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "categories": ["productivity", "utilities"],
  "screenshots": [
    {
      "src": "/screenshot1.png",
      "sizes": "1280x720",
      "type": "image/png"
    }
  ],
  "share_target": {
    "action": "/share",
    "method": "GET",
    "enctype": "application/x-www-form-urlencoded",
    "params": {
      "title": "title",
      "text": "text",
      "url": "url"
    }
  }
}

// HTML 中添加
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#4285f4">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
```

## 六、系统设计篇

### 6.1 前端架构设计

**面试题：请设计一个电商首页的前端架构。**

#### 需求分析

```
功能需求：
1. 顶部导航栏（搜索、分类、购物车、用户中心）
2. 轮播图展示
3. 商品分类导航
4. 推荐商品列表
5. 底部信息
6. 响应式设计（PC/移动）

非功能需求：
1. 首屏加载快
2. SEO 友好
3. 可扩展性强
4. 易维护
```

#### 技术选型

```javascript
/*
技术栈选择：
- 框架：Next.js（React + SSR）
- 状态管理：Zustand（轻量级）
- UI 组件库：Ant Design
- CSS：Tailwind CSS
- 数据请求：React Query（缓存 + 预取）
- 测试：Jest + React Testing Library
- 部署：Vercel
*/
```

#### 架构设计

```
my-ecommerce/
├── public/                    # 静态资源
│   ├── images/
│   └── manifest.json
├── src/
│   ├── components/           # 可复用组件
│   │   ├── common/          # 通用组件
│   │   │   ├── Header/
│   │   │   ├── Footer/
│   │   │   └── SearchBar/
│   │   ├── home/            # 首页组件
│   │   │   ├── Carousel/
│   │   │   ├── CategoryNav/
│   │   │   └── ProductList/
│   │   └── ui/              # UI 原子组件
│   │       ├── Button/
│   │       └── Card/
│   ├── pages/               # 页面（Next.js）
│   │   ├── index.tsx       # 首页
│   │   └── _app.tsx
│   ├── hooks/               # 自定义 Hooks
│   │   ├── useCart.ts
│   │   └── useProducts.ts
│   ├── store/               # 状态管理
│   │   ├── cart.ts
│   │   └── user.ts
│   ├── services/            # API 服务
│   │   ├── api.ts
│   │   └── products.ts
│   ├── types/               # TypeScript 类型
│   │   └── index.ts
│   ├── utils/               # 工具函数
│   │   ├── format.ts
│   │   └── validate.ts
│   └── styles/              # 全局样式
└── tests/                   # 测试
    ├── components/
    └── pages/
```

#### 数据获取策略

```javascript
// 首页数据预取（SSR）
import { GetServerSideProps } from 'next';
import { dehydrate, QueryClient } from 'react-query';
import { fetchProducts, fetchCategories } from '@/services/products';

export const getServerSideProps: GetServerSideProps = async () => {
  const queryClient = new QueryClient();
  
  // 预取商品数据
  await queryClient.prefetchQuery(['products'], fetchProducts);
  
  // 预取分类数据
  await queryClient.prefetchQuery(['categories'], fetchCategories);
  
  return {
    props: {
      dehydratedState: dehydrate(queryClient)
    }
  };
};

// 首页组件
import { useQuery } from 'react-query';

export default function Home() {
  const { data: products } = useQuery(['products'], fetchProducts);
  const { data: categories } = useQuery(['categories'], fetchCategories);
  
  return (
    <div className="home">
      <Header />
      <Carousel />
      <CategoryNav categories={categories} />
      <ProductList products={products} />
      <Footer />
    </div>
  );
}
```

### 6.2 性能优化方案

```javascript
// 1. 图片优化
import Image from 'next/image';

// 使用 Next.js Image 组件
<Image
  src={product.image}
  alt={product.name}
  width={300}
  height={300}
  quality={80}
  loading="lazy"  // 懒加载
  placeholder="blur"  // 模糊占位
/>

// 2. 组件懒加载
import dynamic from 'next/dynamic';

// 非首屏关键组件延迟加载
const Footer = dynamic(() => import('@/components/Footer'), {
  loading: () => <div>加载中...</div>,
  ssr: false  // 非关键组件可以禁用 SSR
});

// 3. 数据缓存
import { QueryClient, QueryClientProvider } from 'react-query';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,  // 1 分钟不重新获取
      cacheTime: 5 * 60 * 1000,  // 5 分钟缓存
      refetchOnWindowFocus: false
    }
  }
});

// 4. 预加载关键资源
<link rel="preload" href="/fonts/main.woff2" as="font" crossorigin>
<link rel="preconnect" href="https://api.example.com">
<link rel="dns-prefetch" href="https://cdn.example.com">
```

## 七、总结

五篇面试系列文章，我们覆盖了：

**第一篇**：JavaScript 核心、Vue/React 原理、浏览器原理、性能优化、手写代码  
**第二篇**：HTTP/HTTPS、Webpack、TypeScript、设计模式、网络安全  
**第三篇**：DOM 操作、CSS 布局、异步编程、微前端、低代码  
**第四篇**：Node.js 核心、事件循环、缓存策略、安全攻防、监控运维  
**第五篇**：SSR/SSG、WebWorker、前端测试、WebSocket、PWA、系统设计

### 前端架构师的核心能力

1. **技术深度**：深入理解底层原理，能够解决复杂技术问题
2. **技术广度**：了解全栈技术栈，能够进行技术选型
3. **架构能力**：能够设计高可用、可扩展的系统架构
4. **工程能力**：能够搭建工程化体系，提升开发效率
5. **软技能**：能够带领团队，沟通协作，推动项目进展

### 面试终极建议

1. **建立知识体系**：不要零散记忆，建立完整的知识网络
2. **多做项目实践**：在项目中应用所学知识，积累经验
3. **阅读优秀源码**：学习优秀的代码设计和架构
4. **关注技术趋势**：保持学习，了解最新技术发展
5. **保持平常心**：面试是双向选择，展示真实的自己

祝你面试顺利，早日成为前端架构师！

---

**系列文章**：
- [第一篇：JavaScript 核心与框架原理](099-frontend-interview.md)
- [第二篇：网络协议与工程化](100-frontend-interview-02.md)
- [第三篇：DOM 操作与架构设计](101-frontend-interview-03.md)
- [第四篇：Node.js 与工程化架构](102-frontend-interview-04.md)

**相关阅读**：
- [React SSR 实战](035-react-ssr.md)
- [WebWorker 指南](045-webworker-guide.md)
- [前端测试实践](055-frontend-testing.md)
- [PWA 入门到精通](065-pwa-guide.md)
