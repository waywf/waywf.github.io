---
title: 04.前端高频面试题深度解析：从 Node.js 到工程化架构的终极挑战
category: 前端开发
excerpt: 前端面试题第四弹！深入解析Node.js核心、事件循环、缓存策略、安全攻防、监控运维等进阶考点。结合大厂真实面试场景，带你冲刺高级前端工程师岗位。
tags: 前端面试, Node.js, 事件循环, 缓存, 安全, 监控, 架构
date: 2026-07-01 10:00:00
readTime: 52
---

# 04.前端高频面试题深度解析：从 Node.js 到工程化架构的终极挑战

> 面试官："说说你对 Node.js 事件循环的理解。"
> 
> 你："就是...宏任务...微任务...然后...那个..."
> 
> 面试官："那 setTimeout 和 Promise 的执行顺序你能解释一下吗？为什么 Node.js 和浏览器的事件循环不一样？"
> 
> 你："..."
> 
> 恭喜你来到前端面试的终极战场！这次我们深入 Node.js 核心、缓存策略、安全攻防、监控运维等高级领域。准备好了吗？

## 一、Node.js 核心篇

### 1.1 事件循环（Event Loop）

**面试题：Node.js 事件循环和浏览器事件循环有什么区别？请详细解释事件循环的各个阶段。**

#### Node.js 事件循环架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Node.js 事件循环                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐                                           │
│   │   Timers    │  ── setTimeout, setInterval               │
│   └──────┬──────┘                                           │
│          ↓                                                    │
│   ┌─────────────┐                                           │
│   │  Pending    │  ── I/O callbacks (系统级)                │
│   │  callbacks  │                                           │
│   └──────┬──────┘                                           │
│          ↓                                                    │
│   ┌─────────────┐                                           │
│   │  Idle,      │  ── Node.js 内部使用                      │
│   │  prepare    │                                           │
│   └──────┬──────┘                                           │
│          ↓                                                    │
│   ┌─────────────┐                                           │
│   │   Poll     │  ── 获取新 I/O 事件                         │
│   │            │    执行 I/O 相关的回调                      │
│   └──────┬──────┘                                           │
│          ↓                                                    │
│   ┌─────────────┐                                           │
│   │    Check   │  ── setImmediate 回调                      │
│   └──────┬──────┘                                           │
│          ↓                                                    │
│   ┌─────────────┐                                           │
│   │ Close      │  ── socket.on('close') 等                 │
│   │ callbacks  │                                           │
│   └─────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 宏任务与微任务

```javascript
// 执行顺序分析
console.log('1. 全局代码开始');

setTimeout(() => {
  console.log('2. setTimeout 回调');
}, 0);

Promise.resolve().then(() => {
  console.log('3. Promise.then 回调 1');
});

setImmediate(() => {
  console.log('4. setImmediate 回调');
});

process.nextTick(() => {
  console.log('5. process.nextTick 回调');
});

new Promise((resolve) => {
  console.log('6. Promise 构造函数同步代码');
  resolve();
}).then(() => {
  console.log('7. Promise.then 回调 2');
});

console.log('8. 全局代码结束');

// 输出顺序：
// 1. 全局代码开始
// 6. Promise 构造函数同步代码
// 8. 全局代码结束
// 5. process.nextTick 回调
// 3. Promise.then 回调 1
// 7. Promise.then 回调 2
// 2. setTimeout 回调
// 4. setImmediate 回调
```

#### Node.js vs 浏览器

```javascript
// 浏览器环境
setTimeout(() => console.log('timeout'), 0);
Promise.resolve().then(() => console.log('promise'));
console.log('end');

// 输出：end -> promise -> timeout

// Node.js 环境
setTimeout(() => console.log('timeout'), 0);
Promise.resolve().then(() => console.log('promise'));
console.log('end');

// 输出：end -> promise -> timeout

// 但 setImmediate 和 setTimeout(0) 的顺序不确定
setTimeout(() => console.log('timeout'), 0);
setImmediate(() => console.log('immediate'));
// 可能：timeout -> immediate 或 immediate -> timeout

// 关键区别：Node.js 有 process.nextTick 和 setImmediate
// - process.nextTick：在当前阶段完成后立即执行，优先级高于其他微任务
// - setImmediate：在 Check 阶段执行
```

#### 深入理解 timers 和 check 阶段

```javascript
// I/O 事件后的执行顺序
const fs = require('fs');

fs.readFile(__filename, () => {
  setTimeout(() => console.log('timeout'), 0);
  setImmediate(() => console.log('immediate'));
});

// 输出：
// immediate
// timeout
// 原因：I/O 回调在 Poll 阶段执行完后，会进入 Check 阶段执行 setImmediate
// 然后才是 timers 阶段（因为 setTimeout 延迟为 0）
```

#### microTask 在各阶段之间执行

```javascript
setTimeout(() => {
  console.log('timeout');
  
  Promise.resolve().then(() => {
    console.log('promise in timeout');
  });
  
  process.nextTick(() => {
    console.log('nextTick in timeout');
  });
}, 0);

Promise.resolve().then(() => {
  console.log('promise 1');
  setTimeout(() => console.log('timeout in promise'), 0);
});

process.nextTick(() => {
  console.log('nextTick 1');
});

// 输出顺序：
// nextTick 1
// promise 1
// timeout
// nextTick in timeout
// promise in timeout
// timeout in promise
```

### 1.2 模块系统

**面试题：Node.js 的模块加载机制是什么？请解释 require 和 import 的区别。**

#### CommonJS 模块加载流程

```javascript
// 1. 路径解析
const path = require('path');

// 2. 模块查找策略
// 绝对路径 -> 相对路径 -> node_modules

// 3. 模块缓存
// 第一次 require 后会缓存，后续直接返回缓存

// 4. 循环引用处理
// a.js
console.log('a 开始');
exports.done = false;
const b = require('./b');
console.log('a: b.done =', b.done);
exports.done = true;
console.log('a 结束');

// b.js
console.log('b 开始');
const a = require('./a');
console.log('b: a.done =', a.done);
exports.done = true;
console.log('b 结束');

// main.js
const a = require('./a');
console.log('main: a.done =', a.done);

// 输出：
// a 开始
// b 开始
// b: a.done = false
// b 结束
// a: b.done = true
// a 结束
// main: a.done = true

// 关键：循环引用时，部分导出的模块可以被访问
```

#### ES Module vs CommonJS

```javascript
// ES Module (ESM) - import/export
// 特点：
// 1. 静态导入（在解析阶段确定依赖）
// 2. 绑定导入（导入的是只读引用）
// 3. 自动严格模式
// 4. CORS 限制（浏览器）

// math.js
export const add = (a, b) => a + b;
export const subtract = (a, b) => a - b;
export default multiply;

// main.js
import multiply, { add, subtract } from './math.js';

// CommonJS (CJS) - require/module.exports
// 特点：
// 1. 动态导入（在运行时确定依赖）
// 2. 值拷贝（导出的是值的拷贝）
// 3. 可以混合使用

// polyfill.js
const add = (a, b) => a + b;
module.exports = { add };

// 注意：Node.js 中混用需要明确转换
```

#### 手写 require 函数

```javascript
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function myRequire(id) {
  // 1. 解析绝对路径
  const absolutePath = path.resolve(id);
  
  // 2. 检查缓存
  if (require.cache[absolutePath]) {
    return require.cache[absolutePath].exports;
  }
  
  // 3. 读取文件
  const code = fs.readFileSync(absolutePath, 'utf-8');
  
  // 4. 包装代码（模拟模块作用域）
  const wrapper = `(function(module, exports, require) {
    ${code}
  })`;
  
  // 5. 创建模块对象
  const module = {
    id: absolutePath,
    filename: absolutePath,
    exports: {},
    loaded: false
  };
  
  // 6. 缓存
  require.cache[absolutePath] = module;
  
  // 7. 执行
  const fn = vm.runInThisContext(wrapper);
  fn(module.exports, module.exports, myRequire);
  
  // 8. 标记加载完成
  module.loaded = true;
  
  return module.exports;
}

myRequire.cache = {};

module.exports = myRequire;
```

### 1.3 Buffer 与流

**面试题：Node.js 中的 Buffer 是什么？如何处理大文件？**

#### Buffer 基础

```javascript
// 创建 Buffer
const buf1 = Buffer.alloc(10);           // 初始化为 0
const buf2 = Buffer.allocUnsafe(10);      // 未初始化，速度快
const buf3 = Buffer.from('hello');       // 从字符串创建
const buf4 = Buffer.from([0x68, 0x69]);  // 从数组创建

// 字符串编码
const buf = Buffer.from('你好', 'utf8');
console.log(buf.length);        // 6
console.log(buf.toString());   // 你好

// 十六进制
console.log(buf.toString('hex'));  // e4bda0e5a5bd

// base64
console.log(buf.toString('base64'));  // 5L2g5aW9
```

#### Stream 流操作

```javascript
const fs = require('fs');

// 读取流
const readStream = fs.createReadStream('largefile.txt', {
  highWaterMark: 64 * 1024  // 64KB
});

readStream.on('data', (chunk) => {
  console.log('收到数据块:', chunk.length);
});

readStream.on('end', () => {
  console.log('读取完成');
});

readStream.on('error', (err) => {
  console.error('读取错误:', err);
});

// 写入流
const writeStream = fs.createWriteStream('output.txt');

writeStream.write('第一行\n');
writeStream.write('第二行\n');
writeStream.end('最后一行');

writeStream.on('finish', () => {
  console.log('写入完成');
});

// 管道流（推荐）
const zlib = require('zlib');

// 压缩大文件
fs.createReadStream('input.txt')
  .pipe(zlib.createGzip())
  .pipe(fs.createWriteStream('input.txt.gz'));
```

#### 处理大文件的最佳实践

```javascript
const fs = require('fs');
const readline = require('readline');

// 逐行读取大文件
async function processLargeFile(filePath) {
  const fileStream = fs.createReadStream(filePath);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });
  
  let lineCount = 0;
  
  for await (const line of rl) {
    lineCount++;
    // 处理每一行
    console.log(`第 ${lineCount} 行: ${line.substring(0, 50)}...`);
  }
  
  console.log(`总计 ${lineCount} 行`);
}

// 分割大文件
function splitFile(inputFile, outputDir, chunkSize = 1000) {
  let lineIndex = 0;
  let fileIndex = 1;
  let writeStream;
  
  const readStream = fs.createReadStream(inputFile, {
    encoding: 'utf8',
    highWaterMark: 1024 * 1024  // 1MB
  });
  
  readStream.on('data', (chunk) => {
    const lines = chunk.split('\n');
    
    lines.forEach((line) => {
      if (lineIndex % chunkSize === 0) {
        if (writeStream) writeStream.end();
        const outputFile = path.join(outputDir, `part_${fileIndex}.txt`);
        writeStream = fs.createWriteStream(outputFile);
        fileIndex++;
      }
      
      if (writeStream) {
        writeStream.write(line + '\n');
      }
      lineIndex++;
    });
  });
  
  readStream.on('end', () => {
    if (writeStream) writeStream.end();
    console.log(`分割完成，共 ${fileIndex - 1} 个文件`);
  });
}
```

## 二、缓存策略篇

### 2.1 浏览器缓存机制

**面试题：浏览器缓存有哪些？请解释强缓存和协商缓存的区别。**

#### 缓存类型总览

```
┌─────────────────────────────────────────────────────────────┐
│                      浏览器缓存机制                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   请求 ──→ 缓存检查                                          │
│            │                                                │
│            ├── 强缓存（Cache-Control / Expires）            │
│            │   ├── 命中：直接使用缓存，不请求服务器           │
│            │   └── 未命中：继续检查协商缓存                   │
│            │                                                │
│            └── 协商缓存（ETag / Last-Modified）            │
│                ├── 命中：304，使用缓存                       │
│                └── 未命中：200，返回新资源                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 强缓存

```javascript
// 响应头设置
// 1. Expires（HTTP/1.0）
Expires: Wed, 21 Oct 2026 07:28:00 GMT
// 绝对时间，依赖本地时钟，可能不准确

// 2. Cache-Control（HTTP/1.1）- 推荐
Cache-Control: max-age=3600           // 相对时间，秒
Cache-Control: max-age=3600, public  // 公共缓存，可被 CDN 缓存
Cache-control: max-age=31536000, immutable  // 永久缓存
Cache-Control: no-cache              // 每次都需要验证
Cache-Control: no-store              // 不缓存任何内容
Cache-Control: private               // 私有缓存，仅浏览器缓存

// 3. 实际应用（Express）
app.use((req, res, next) => {
  // 静态资源：长期缓存
  if (req.url.match(/\.(js|css|png|jpg)$/)) {
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
  } else {
    // HTML：协商缓存
    res.setHeader('Cache-Control', 'no-cache');
  }
  next();
});
```

#### 协商缓存

```javascript
// 1. Last-Modified / If-Modified-Since
// 服务器返回
Last-Modified: Wed, 21 Oct 2026 07:28:00 GMT

// 客户端请求
If-Modified-Since: Wed, 21 Oct 2026 07:28:00 GMT

// 服务器判断
// 如果资源未修改：返回 304 Not Modified
// 如果资源已修改：返回 200 + 新资源

// 2. ETag / If-None-Match（推荐）- 解决文件内容未变但时间戳变化的问题
// 服务器返回
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"

// 客户端请求
If-None-Match: "33a64df551425fcc55e4d42a148795d9f25f89d4"

// 服务器判断
// 匹配：返回 304
// 不匹配：返回 200 + 新资源 + 新 ETag

// 3. Express 实现
const etag = require('etag');
const fs = require('fs');

app.get('/static/:file', (req, res) => {
  const filePath = path.join(__dirname, 'public', req.params.file);
  const stats = fs.statSync(filePath);
  
  // 设置 ETag
  res.set('ETag', etag(stats));
  
  // 检查 If-None-Match
  if (req.fresh) {
    return res.status(304).end();
  }
  
  // 返回完整内容
  res.sendFile(filePath);
});
```

### 2.2 CDN 缓存

**面试题：CDN 缓存如何配置？有哪些注意事项？**

#### CDN 缓存策略

```javascript
// 1. 静态资源：长期缓存 + 版本化
// URL: /static/js/app.v1.0.0.js
// Cache-Control: public, max-age=31536000

// 2. 动态内容：短期缓存或协商缓存
// Cache-Control: public, max-age=0
// Cache-Control: no-cache

// 3. 用户相关：私有缓存
// Cache-Control: private, max-age=3600

// 4. HTML 页面：no-cache（必须验证）
// Cache-Control: no-cache, must-revalidate
```

#### CDN 配置最佳实践

```nginx
# Nginx 配置示例

# 静态资源 - 长期缓存
location /static/ {
    # 开启 gzip
    gzip on;
    gzip_types text/plain application/javascript text/css;
    
    # 缓存一年
    expires 1y;
    add_header Cache-Control "public, immutable";
    
    # 跨域配置
    add_header Access-Control-Allow-Origin *;
}

# API - 不缓存
location /api/ {
    proxy_pass http://backend;
    add_header Cache-Control "no-store, no-cache, must-revalidate";
}

# HTML - 协商缓存
location / {
    proxy_pass http://frontend;
    add_header Cache-Control "no-cache";
}
```

### 2.3 前端缓存方案

```javascript
// 1. LocalStorage
class LocalCache {
  set(key, value, expire = null) {
    const data = {
      value,
      expire: expire ? Date.now() + expire : null
    };
    localStorage.setItem(key, JSON.stringify(data));
  }
  
  get(key) {
    const item = localStorage.getItem(key);
    if (!item) return null;
    
    const { value, expire } = JSON.parse(item);
    
    if (expire && Date.now() > expire) {
      this.delete(key);
      return null;
    }
    
    return value;
  }
  
  delete(key) {
    localStorage.removeItem(key);
  }
}

// 2. SessionStorage（标签页关闭清除）
// 3. IndexedDB（大量结构化数据）
// 4. Cache API（Service Worker）

// Service Worker 缓存策略
const CACHE_NAME = 'my-app-v1';
const urlsToCache = ['/', '/index.html', '/static/js/main.js'];

// 缓存优先
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // 命中缓存
        if (response) {
          return response;
        }
        // 请求网络
        return fetch(event.request).then(response => {
          // 缓存新资源
          if (!response || response.status !== 200) {
            return response;
          }
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then(cache => {
            cache.put(event.request, responseToCache);
          });
          return response;
        });
      })
  );
});
```

## 三、安全攻防篇

### 3.1 XSS 攻击深入

**面试题：XSS 攻击有哪些类型？如何防御？**

#### XSS 攻击类型

```javascript
// 1. 反射型 XSS
// URL: http://example.com/search?q=<script>alert('xss')</script>
// 服务器直接将参数返回到 HTML 中

// 2. 存储型 XSS
// 恶意脚本存储到数据库，所有访问该页面的用户都会受害

// 3. DOM 型 XSS
// 不经过服务器，完全在客户端执行
// <div id="content"></div>
// <script>document.getElementById('content').innerHTML = location.hash.slice(1)</script>
// URL: http://example.com/#<img src=x onerror=alert('xss')>
```

#### 防御方案

```javascript
// 1. HTML 转义
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;'
  };
  return text.replace(/[&<>"'/]/g, char => map[char]);
}

// 2. JavaScript 转义
function escapeJs(str) {
  return str
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/"/g, '\\"')
    .replace(/`/g, '\\`')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r');
}

// 3. URL 参数转义
function encodeURIComponentSafe(str) {
  return encodeURIComponent(str)
    .replace(/[!'()*]/g, c => '%' + c.charCodeAt(0).toString(16).toUpperCase());
}

// 4. React 自动转义
// React 会自动转义 JSX 中的内容，防止 XSS

// 5. CSP（内容安全策略）
// 服务器响应头
Content-Security-Policy: 
  default-src 'self';
  script-src 'self' 'nonce-xxx';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  connect-src 'self' https://api.example.com;

// 6. HTTP Only Cookie
// 防止 JavaScript 读取 Cookie
Set-Cookie: sessionId=xxx; HttpOnly; Secure; SameSite=Strict
```

### 3.2 CSRF 攻击

**面试题：什么是 CSRF？如何防御？**

#### CSRF 攻击原理

```
用户登录银行网站 → 获取 Cookie
                    ↓
攻击者诱导访问恶意网站 → 自动发送银行转账请求
                    ↓
浏览器自动携带 Cookie → 服务器执行转账
                    ↓
用户损失财产
```

#### 防御方案

```javascript
// 1. CSRF Token
// 服务器生成 Token
app.get('/transfer', (req, res) => {
  const csrfToken = crypto.randomBytes(32).toString('hex');
  req.session.csrfToken = csrfToken;
  res.json({ csrfToken });
});

// 服务器验证 Token
app.post('/transfer', (req, res) => {
  const { csrfToken, to, amount } = req.body;
  
  if (req.session.csrfToken !== csrfToken) {
    return res.status(403).json({ error: 'CSRF token 无效' });
  }
  
  // 执行转账
  transferMoney(to, amount);
  res.json({ success: true });
});

// 前端发送请求时携带 Token
fetch('/transfer', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-CSRF-Token': csrfToken  // 自定义头
  },
  body: JSON.stringify({ to: '123', amount: 1000 })
});

// 2. SameSite Cookie
Set-Cookie: sessionId=xxx; SameSite=Strict
Set-Cookie: sessionId=xxx; SameSite=Lax  // 默认

// 3. 验证 Referer 或 Origin
app.use((req, res, next) => {
  const origin = req.headers.origin;
  const referer = req.headers.referer;
  
  if (!origin && !referer) {
    return res.status(403).json({ error: '缺少来源信息' });
  }
  
  if (!origin.includes('mybank.com') && !referer.includes('mybank.com')) {
    return res.status(403).json({ error: '非法来源' });
  }
  
  next();
});

// 4. 双重提交 Cookie
// 将 Token 放在 Cookie 中，同时在请求体中也提交
// 服务器比较两者是否一致
```

### 3.3 其他安全问题

```javascript
// 1. SQL 注入
// ❌ 危险
const query = `SELECT * FROM users WHERE name = '${username}'`;

// ✓ 安全：参数化查询
const query = 'SELECT * FROM users WHERE name = $1';
const result = await client.query(query, [username]);

// 2. 命令注入
// ❌ 危险
const cmd = `grep ${username} /etc/passwd`;

// ✓ 安全
const { execFile } = require('child_process');
execFile('grep', [username, '/etc/passwd']);

// 3. JSON 解析安全
// ❌ 危险：可能耗尽内存
const data = JSON.parse(userInput);

// ✓ 安全：限制解析深度
function safeJsonParse(str, depth = 10) {
  try {
    return JSON.parse(str);
  } catch (e) {
    if (e.message.includes('too deep')) {
      throw new Error('JSON 嵌套过深');
    }
    throw e;
  }
}

// 4. 路径遍历
// ❌ 危险
const file = fs.readFileSync('./uploads/' + filename);

// ✓ 安全
const path = require('path');
const safePath = path.join('./uploads', filename);
const normalized = path.normalize(safePath);
if (!normalized.startsWith('./uploads/')) {
  throw new Error('非法路径');
}
const file = fs.readFileSync(normalized);
```

## 四、监控与运维篇

### 4.1 前端监控体系

**面试题：如何构建前端监控体系？需要监控哪些指标？**

#### 监控体系架构

```
┌─────────────────────────────────────────────────────────────┐
│                     前端监控体系                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│   │ 性能监控 │   │ 错误监控 │   │ 行为监控 │              │
│   └─────┬────┘   └─────┬────┘   └─────┬────┘              │
│         │              │              │                     │
│         └──────────────┼──────────────┘                    │
│                        ↓                                    │
│              ┌─────────────────────┐                         │
│              │    数据采集 SDK     │                         │
│              └──────────┬──────────┘                         │
│                         ↓                                    │
│              ┌─────────────────────┐                         │
│              │    数据上报服务      │                         │
│              └──────────┬──────────┘                         │
│                         ↓                                    │
│              ┌─────────────────────┐                         │
│              │    数据存储与分析    │                         │
│              └──────────┬──────────┘                         │
│                         ↓                                    │
│              ┌─────────────────────┐                         │
│              │    可视化展示平台    │                         │
│              └─────────────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 性能监控

```javascript
class PerformanceMonitor {
  constructor() {
    this.metrics = {};
  }
  
  // Core Web Vitals
  init() {
    // LCP (Largest Contentful Paint)
    this.observeLCP();
    
    // FID (First Input Delay)
    this.observeFID();
    
    // CLS (Cumulative Layout Shift)
    this.observeCLS();
    
    // FCP (First Contentful Paint)
    this.observeFCP();
    
    // TTFB (Time To First Byte)
    this.observeTTFB();
  }
  
  observeLCP() {
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.metrics.lcp = lastEntry.renderTime || lastEntry.loadTime;
    }).observe({ entryTypes: ['largest-contentful-paint'] });
  }
  
  observeFID() {
    new PerformanceObserver((list) => {
      const firstEntry = list.getEntries()[0];
      this.metrics.fid = firstEntry.processingStart - firstEntry.startTime;
    }).observe({ entryTypes: ['first-input'] });
  }
  
  observeCLS() {
    let clsValue = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      this.metrics.cls = clsValue;
    }).observe({ entryTypes: ['layout-shift'] });
  }
  
  observeFCP() {
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const fcpEntry = entries.find(e => e.name === 'first-contentful-paint');
      if (fcpEntry) {
        this.metrics.fcp = fcpEntry.startTime;
      }
    }).observe({ entryTypes: ['paint'] });
  }
  
  observeTTFB() {
    const [navigation] = performance.getEntriesByType('navigation');
    if (navigation) {
      this.metrics.ttfb = navigation.responseStart - navigation.requestStart;
    }
  }
  
  report() {
    console.log('Performance Metrics:', this.metrics);
    
    // 上报数据
    this.sendMetrics(this.metrics);
  }
  
  sendMetrics(data) {
    navigator.sendBeacon('/api/metrics', JSON.stringify(data));
  }
}
```

#### 错误监控

```javascript
class ErrorMonitor {
  constructor() {
    this.errors = [];
  }
  
  init() {
    // 全局未捕获错误
    window.addEventListener('error', (event) => {
      this.handleError({
        type: 'error',
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack
      });
    });
    
    // Promise 未捕获错误
    window.addEventListener('unhandledrejection', (event) => {
      this.handleError({
        type: 'unhandledrejection',
        message: event.reason?.message || event.reason,
        stack: event.reason?.stack
      });
    });
    
    // Vue 错误
    if (window.Vue) {
      window.Vue.config.errorHandler = (err, vm, info) => {
        this.handleError({
          type: 'vue',
          message: err.message,
          stack: err.stack,
          componentName: vm.$options.name,
          hookInfo: info
        });
      };
    }
    
    // React 错误边界
    // class ErrorBoundary extends React.Component {
    //   componentDidCatch(error, errorInfo) {
    //     this.handleError({ type: 'react', ...errorInfo });
    //   }
    // }
  }
  
  handleError(error) {
    const errorData = {
      ...error,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      sessionId: this.getSessionId()
    };
    
    this.errors.push(errorData);
    
    // 上报错误
    this.reportError(errorData);
  }
  
  reportError(error) {
    // 使用 sendBeacon 确保页面关闭也能发送
    navigator.sendBeacon('/api/errors', JSON.stringify(error));
  }
  
  getSessionId() {
    let sessionId = sessionStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      sessionStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  }
}
```

### 4.2 日志系统

```javascript
class Logger {
  constructor(options = {}) {
    this.level = options.level || 'info';
    this.endpoint = options.endpoint || '/api/logs';
    this.enableConsole = options.enableConsole !== false;
    this.buffer = [];
    this.bufferSize = options.bufferSize || 10;
  }
  
  log(level, message, meta = {}) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      meta,
      userId: this.getUserId(),
      url: window.location.href,
      userAgent: navigator.userAgent
    };
    
    if (this.enableConsole) {
      console[level](message, meta);
    }
    
    this.buffer.push(entry);
    
    if (this.buffer.length >= this.bufferSize) {
      this.flush();
    }
  }
  
  info(message, meta) { this.log('info', message, meta); }
  warn(message, meta) { this.log('warn', message, meta); }
  error(message, meta) { this.log('error', message, meta); }
  debug(message, meta) { this.log('debug', message, meta); }
  
  flush() {
    if (this.buffer.length === 0) return;
    
    const entries = [...this.buffer];
    this.buffer = [];
    
    navigator.sendBeacon(this.endpoint, JSON.stringify(entries));
  }
  
  getUserId() {
    let userId = localStorage.getItem('user_id');
    if (!userId) {
      userId = crypto.randomUUID();
      localStorage.setItem('user_id', userId);
    }
    return userId;
  }
}

// 使用
const logger = new Logger({
  level: 'info',
  bufferSize: 5
});

// 自动刷新
window.addEventListener('beforeunload', () => logger.flush());
setInterval(() => logger.flush(), 30000);
```

## 五、工程化架构篇

### 5.1 构建优化

**面试题：如何优化 Webpack/Vite 构建速度？**

#### 构建优化策略

```javascript
// webpack.config.js

module.exports = {
  // 1. 并行构建
  parallelism: 4,
  
  // 2. 缓存
  cache: {
    type: 'filesystem',
    buildDependencies: {
      config: [__filename]
    }
  },
  
  // 3. 代码分割
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          priority: 10
        },
        common: {
          minChunks: 2,
          priority: 5,
          reuseExistingChunk: true
        }
      }
    },
    
    // Tree Shaking
    usedExports: true,
    
    // 压缩
    minimizer: [
      new TerserPlugin({
        parallel: true,
        terserOptions: {
          compress: {
            drop_console: true
          }
        }
      })
    ]
  },
  
  // 4. 懒加载
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            cacheDirectory: true
          }
        }
      }
    ]
  }
};
```

#### Vite 构建优化

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  build: {
    // 目标浏览器
    target: 'es2015',
    
    // 输出目录
    outDir: 'dist',
    
    // 代码分割
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-utils': ['lodash', 'axios']
        }
      },
      
      plugins: [
        // 打包分析
        visualizer({
          filename: 'dist/stats.html',
          open: true,
          gzipSize: true
        })
      ]
    },
    
    // 压缩
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    
    // CSS 代码分割
    cssCodeSplit: true,
    
    // 资源内联阈值
    assetsInlineLimit: 4096
  },
  
  // 开发优化
  optimizeDeps: {
    include: ['react', 'react-dom', 'lodash'],
    exclude: ['vue']
  },
  
  // SSR
  ssr: {
    noExternal: ['vant']
  }
});
```

### 5.2 微前端进阶

```javascript
// qiankun 沙箱隔离实现

// 1. 快照沙箱
class SnapshotSandbox {
  constructor() {
    this.proxy = window;
    this.modifyMap = {};  // 存储修改
    this.active = false;
  }
  
  active() {
    this.modifyMap = {};
    // 记录当前 window 属性
    for (let key in window) {
      this.modifyMap[key] = window[key];
    }
    this.active = true;
  }
  
  inactive() {
    // 恢复 window 属性
    for (let key in this.modifyMap) {
      if (window[key] !== this.modifyMap[key]) {
        window[key] = this.modifyMap[key];
      }
    }
    this.active = false;
  }
}

// 2. 代理沙箱（推荐）
class ProxySandbox {
  constructor() {
    this.isRunning = false;
    this.modifyMap = new Map();
    
    this.proxy = new Proxy(window, {
      get: (target, key) => {
        if (!this.isRunning) {
          return target[key];
        }
        
        const value = this.modifyMap.get(key);
        if (value !== undefined) {
          return value;
        }
        
        return target[key];
      },
      
      set: (target, key, value) => {
        if (!this.isRunning) {
          return true;
        }
        
        this.modifyMap.set(key, value);
        return true;
      }
    });
  }
  
  active() {
    this.isRunning = true;
  }
  
  inactive() {
    this.isRunning = false;
    this.modifyMap.clear();
  }
}
```

## 六、算法与数据结构篇

### 6.1 手写算法题

```javascript
// 1. 防抖
function debounce(fn, delay, immediate = false) {
  let timer = null;
  
  return function(...args) {
    const context = this;
    
    if (timer) clearTimeout(timer);
    
    if (immediate && !timer) {
      fn.apply(context, args);
    }
    
    timer = setTimeout(() => {
      if (!immediate) {
        fn.apply(context, args);
      }
      timer = null;
    }, delay);
  };
}

// 2. 节流
function throttle(fn, delay) {
  let lastTime = 0;
  let timer = null;
  
  return function(...args) {
    const context = this;
    const now = Date.now();
    
    if (now - lastTime >= delay) {
      fn.apply(context, args);
      lastTime = now;
    } else if (!timer) {
      timer = setTimeout(() => {
        fn.apply(context, args);
        timer = null;
        lastTime = Date.now();
      }, delay - (now - lastTime));
    }
  };
}

// 3. 深拷贝
function deepClone(obj, hash = new WeakMap()) {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  
  if (hash.has(obj)) {
    return hash.get(obj);
  }
  
  const clone = Array.isArray(obj) ? [] : {};
  hash.set(obj, clone);
  
  for (let key in obj) {
    if (obj.hasOwnProperty(key)) {
      clone[key] = deepClone(obj[key], hash);
    }
  }
  
  return clone;
}

// 4. LRU 缓存
class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();
  }
  
  get(key) {
    if (!this.cache.has(key)) return -1;
    
    const value = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, value);
    
    return value;
  }
  
  put(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.capacity) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    this.cache.set(key, value);
  }
}
```

## 七、总结

四篇面试系列文章，我们覆盖了：

**第一篇**：JavaScript 核心、Vue/React 原理、浏览器原理、性能优化、手写代码  
**第二篇**：HTTP/HTTPS、Webpack、TypeScript、设计模式、网络安全  
**第三篇**：DOM 操作、CSS 布局、异步编程、微前端、低代码  
**第四篇**：Node.js 核心、事件循环、缓存策略、安全攻防、监控运维

### 高级前端工程师的核心能力

1. **基础扎实**：JavaScript、TypeScript、计算机网络、操作系统
2. **工程能力**：构建工具、性能优化、自动化测试、CI/CD
3. **架构思维**：微前端、服务端渲染、跨端开发
4. **软实力**：沟通能力、问题分析、项目推进
5. **学习能力**：新技术跟进、源码阅读、技术视野

### 面试成功的关键

1. **理解原理**：不只是会用，要理解为什么
2. **能举一反三**：一个知识点能关联到其他知识点
3. **有深度**：不仅知道表面，还能深入原理
4. **有广度**：了解前端生态和相邻领域
5. **会表达**：清晰的思路和良好的沟通

祝你面试顺利，早日成为高级前端工程师！

---

**系列文章**：
- [第一篇：JavaScript 核心与框架原理](099-frontend-interview.md)
- [第二篇：网络协议与工程化](100-frontend-interview-02.md)
- [第三篇：DOM 操作与架构设计](101-frontend-interview-03.md)

**相关阅读**：
- [Node.js 深入浅出](030-nodejs-deep.md)
- [前端工程化实践](040-fp-engineering.md)
- [安全实战：XSS 与 CSRF](050-web-security.md)
