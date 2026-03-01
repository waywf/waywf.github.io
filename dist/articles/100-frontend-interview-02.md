---
title: 02.前端高频面试题深度解析：从原理到实战的进阶之路
category: 前端开发
excerpt: 前端面试题第二弹！深入解析HTTP/HTTPS、Webpack优化、TypeScript高级类型、设计模式、安全攻防等高频考点。不只是背答案，而是理解背后的工程实践和架构思维。
tags: 前端面试, HTTP, Webpack, TypeScript, 设计模式, 网络安全, 工程化
date: 2026-03-01
readTime: 50
---

# 02.前端高频面试题深度解析：从原理到实战的进阶之路

> 面试官："说说 HTTPS 的握手过程。"
> 
> 你："就是...加密传输...有证书..."
> 
> 面试官："具体怎么加密的？为什么需要两次握手？"
> 
> 你："..."
> 
> 欢迎来到前端面试的第二战场！这次我们深入网络协议、构建工具、TypeScript 类型系统等更硬核的领域。准备好了吗？

## 一、网络协议篇

### 1.1 HTTP/HTTPS：Web 的基石

**面试题：HTTP 和 HTTPS 有什么区别？HTTPS 的握手过程是怎样的？**

#### HTTP 的"裸奔"时代

HTTP 是明文传输的，就像你寄明信片，邮递员、分拣员都能看到内容：

```javascript
// HTTP 请求（明文，任何人可见）
GET /api/user HTTP/1.1
Host: example.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

// 中间人可以轻松窃取 token、密码等敏感信息
```

**HTTP 的三大问题**：
1. **窃听风险**：数据明文传输
2. **篡改风险**：无法验证数据完整性
3. **冒充风险**：无法验证服务器身份

#### HTTPS 的加密艺术

HTTPS = HTTP + SSL/TLS，通过加密和证书解决上述问题。

**HTTPS 握手过程（简化版）**：

```
客户端                          服务器
  |                              |
  |--- ClientHello ------------>|
  |   - 支持的加密套件            |
  |   - 随机数 A                  |
  |                              |
  |<-- ServerHello -------------|
  |   - 选定的加密套件            |
  |   - 随机数 B                  |
  |   - 服务器证书                |
  |                              |
  |--- 验证证书 ---------------->|
  |   - 检查证书有效期            |
  |   - 检查颁发机构              |
  |   - 检查域名匹配              |
  |                              |
  |<-- 公钥 ---------------------|
  |                              |
  |--- 预主密钥（用公钥加密）--->|
  |   （用服务器公钥加密）        |
  |                              |
  |<-- 确认 ---------------------|
  |                              |
  |=== 生成会话密钥 =============|
  |   (A + B + 预主密钥)          |
  |                              |
  |=== 加密通信开始 =============|
```

**为什么需要两次随机数？**

如果只用服务器的随机数，攻击者截获后可以用相同随机数重放攻击。客户端也提供随机数，确保每次会话都是唯一的。

#### 对称加密 vs 非对称加密

```
HTTPS 的巧妙设计：
1. 握手阶段：使用非对称加密（RSA/ECC）
   - 优点：安全，公钥加密只有私钥能解密
   - 缺点：慢，计算复杂度高

2. 通信阶段：使用对称加密（AES）
   - 优点：快，加解密效率高
   - 缺点：密钥传输不安全

3. 最佳组合：
   - 用非对称加密传输对称密钥
   - 用对称密钥加密实际数据
```

#### HTTP/2 和 HTTP/3 的进化

| 版本 | 核心特性 | 性能提升 |
|------|---------|---------|
| **HTTP/1.1** | 文本协议、长连接 | 基础版本 |
| **HTTP/2** | 二进制协议、多路复用、头部压缩 | 减少延迟 |
| **HTTP/3** | 基于 QUIC（UDP）、0-RTT 连接 | 更快握手 |

**HTTP/2 的多路复用**：

```javascript
// HTTP/1.1：队头阻塞
请求 1: [======]
请求 2:         [======]
请求 3:                 [======]

// HTTP/2：多路复用
请求 1: [==][==][==]
请求 2: [==][==][==]
请求 3: [==][==][==]
        同一连接同时传输
```

### 1.2 TCP 三次握手与四次挥手

**面试题：为什么 TCP 建立连接是三次握手，断开是四次挥手？**

#### 三次握手：确保双方都能收发

```
客户端                    服务器
  |                         |
  |--- SYN=1, seq=x ------->|
  |   我准备好了             |
  |                         |
  |<-- SYN=1, ACK=1, seq=y -|
  |   我也准备好了，确认你   |
  |                         |
  |--- ACK=1, seq=x+1 ----->|
  |   确认收到               |
  |                         |
  |====== 连接建立 ==========|
```

**为什么不是两次？**

如果只有两次，可能出现"已失效的连接请求"问题：
- 客户端第一次 SYN 延迟到达
- 服务器收到，建立连接
- 但客户端已经不需要这个连接了
- 服务器一直等待，浪费资源

第三次握手确认客户端还活着，且确实想建立连接。

#### 四次挥手：优雅地告别

```
客户端                    服务器
  |                         |
  |--- FIN=1, seq=u ------->|
  |   我没数据要发了         |
  |                         |
  |<-- ACK=1, seq=v --------|
  |   收到，但我还有数据     |
  |                         |
  |    [服务器继续发送数据]  |
  |                         |
  |<-- FIN=1, ACK=1, seq=w -|
  |   我也发完了             |
  |                         |
  |--- ACK=1, seq=u+1 ----->|
  |   确认，关闭连接         |
  |                         |
  |====== 连接关闭 ==========|
```

**为什么是四次？**

因为 TCP 是全双工的，每个方向都要单独关闭：
- 第一次：客户端说"我不发了"
- 第二次：服务器确认"知道了"
- 第三次：服务器说"我也不发了"
- 第四次：客户端确认"好的"

### 1.3 浏览器缓存策略

**面试题：强缓存和协商缓存的区别？Cache-Control 有哪些值？**

#### 缓存决策流程

```
请求资源
  ↓
检查强缓存（Cache-Control/Expires）
  ↓
命中？--是--> 直接使用，不请求服务器
  ↓ 否
发送请求（带 If-None-Match/If-Modified-Since）
  ↓
服务器检查
  ↓
未修改？--是--> 返回 304，使用本地缓存
  ↓ 否
返回 200 + 新资源 + 新缓存头
```

#### Cache-Control 详解

```javascript
// 常见值及含义
Cache-Control: public, max-age=31536000
// 公共缓存，缓存 1 年

Cache-Control: private, max-age=600
// 仅浏览器缓存，10 分钟

Cache-Control: no-cache
// 不直接使用，先协商

Cache-Control: no-store
// 完全不缓存

Cache-Control: max-age=0, must-revalidate
// 过期后必须验证

Cache-Control: max-stale=3600
// 客户端愿意接受过期 1 小时内的资源
```

#### 实战配置

```nginx
# Nginx 配置示例

# 静态资源（带 hash 的文件）
location ~* \.(js|css|png|jpg)$ {
  if ($request_filename ~* "(\.[0-9a-f]{8}\.)") {
    add_header Cache-Control "public, max-age=31536000, immutable";
  }
}

# HTML 文件（不缓存）
location ~* \.html$ {
  add_header Cache-Control "no-cache, no-store, must-revalidate";
}

# API 接口（协商缓存）
location /api/ {
  add_header Cache-Control "no-cache";
}
```

## 二、Webpack 工程化篇

### 2.1 Webpack 构建流程

**面试题：Webpack 的构建流程是什么？Loader 和 Plugin 的区别？**

#### Webpack 的核心概念

```javascript
// webpack.config.js
module.exports = {
  entry: './src/index.js',      // 入口
  output: {                      // 出口
    filename: 'bundle.[contenthash].js',
    path: path.resolve(__dirname, 'dist')
  },
  module: {
    rules: [                     // Loader 配置
      {
        test: /\.js$/,
        use: 'babel-loader'
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      }
    ]
  },
  plugins: [                     // Plugin 配置
    new HtmlWebpackPlugin({
      template: './src/index.html'
    }),
    new MiniCssExtractPlugin({
      filename: '[name].[contenthash].css'
    })
  ]
};
```

#### 构建流程详解

```
1. 初始化参数
   - 读取 webpack.config.js
   - 合并默认配置
   - 验证配置合法性

2. 创建 Compiler
   - 单例，贯穿整个构建过程
   - 持有所有配置信息

3. 确定入口
   - 从 entry 开始
   - 递归解析依赖

4. 编译模块
   - 调用 Loader 转换文件
   - 提取 AST（抽象语法树）
   - 分析依赖

5. 完成模块编译
   - 得到每个模块的最终代码
   - 构建依赖图

6. 输出资源
   - 根据依赖图生成 chunks
   - 调用 Plugin 处理
   - 写入文件系统
```

#### Loader vs Plugin

| 特性 | Loader | Plugin |
|------|--------|--------|
| **作用** | 文件转换器 | 功能扩展器 |
| **执行时机** | 模块加载时 | 整个构建过程 |
| **示例** | babel-loader, css-loader | HtmlWebpackPlugin, TerserPlugin |
| **原理** | 接收源文件，返回转换后代码 | 监听 Webpack 事件流，在特定时机执行 |

#### 手写 Loader

```javascript
// simple-loader.js
module.exports = function(source) {
  // source 是原始文件内容
  console.log('处理文件:', this.resourcePath);
  
  // 简单示例：将所有 console.log 替换为自定义函数
  const modified = source.replace(
    /console\.log\(/g, 
    'window.customLog('
  );
  
  return modified;
};

// 使用
module.exports = {
  module: {
    rules: [{
      test: /\.js$/,
      use: './simple-loader.js'
    }]
  }
};
```

#### 手写 Plugin

```javascript
// copyright-plugin.js
class CopyrightPlugin {
  apply(compiler) {
    // 监听 emit 事件（生成资源到 output 目录之前）
    compiler.hooks.emit.tapAsync(
      'CopyrightPlugin',
      (compilation, callback) => {
        // 遍历所有输出文件
        for (let filename in compilation.assets) {
          if (filename.endsWith('.js')) {
            // 添加版权信息
            const content = compilation.assets[filename].source();
            const copyright = `/* Copyright © ${new Date().getFullYear()} MyCompany */\n`;
            compilation.assets[filename] = {
              source: () => copyright + content,
              size: () => copyright.length + content.length
            };
          }
        }
        callback();
      }
    );
  }
}

// 使用
plugins: [new CopyrightPlugin()]
```

### 2.2 Webpack 性能优化

**面试题：如何优化 Webpack 的构建速度和打包体积？**

#### 构建速度优化

```javascript
module.exports = {
  // 1. 缩小文件搜索范围
  module: {
    rules: [{
      test: /\.js$/,
      include: path.resolve(__dirname, 'src'),
      exclude: /node_modules/,
      use: 'babel-loader'
    }]
  },
  
  // 2. 缓存 Loader 结果
  module: {
    rules: [{
      test: /\.js$/,
      use: 'cache-loader'  // 缓存到磁盘
    }]
  },
  
  // 3. 多线程构建
  module: {
    rules: [{
      test: /\.js$/,
      use: 'thread-loader'  // 多进程执行
    }]
  },
  
  // 4. 减少入口数量
  // 多个 entry 会创建多个 compilation，减慢构建
  entry: {
    main: './src/index.js'
    // 避免过多 entry
  },
  
  // 5. 使用持久化缓存（Webpack 5）
  cache: {
    type: 'filesystem',
    cacheDirectory: path.resolve(__dirname, '.webpack_cache')
  }
};
```

#### 打包体积优化

```javascript
module.exports = {
  // 1. 代码分割
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendors: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          priority: 10
        },
        commons: {
          name: 'commons',
          minChunks: 2,
          priority: 5
        }
      }
    }
  },
  
  // 2. Tree Shaking
  mode: 'production',  // 自动启用
  optimization: {
    usedExports: true,  // 标记未使用代码
    sideEffects: true   // 移除未使用代码
  },
  
  // 3. 压缩代码
  optimization: {
    minimize: true,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true,  // 移除 console
            drop_debugger: true  // 移除 debugger
          }
        }
      }),
      new CssMinimizerPlugin()
    ]
  },
  
  // 4. 分析打包结果
  // 运行：webpack --profile --json > stats.json
  // 访问：https://webpack.github.io/analyse
};
```

#### 按需加载

```javascript
// 动态导入（推荐）
const handleClick = async () => {
  const module = await import('./heavy-module.js');
  module.doSomething();
};

// 魔法注释
const module = await import(
  /* webpackChunkName: "heavy-module" */
  /* webpackPrefetch: true */
  './heavy-module.js'
);

// React.lazy
const LazyComponent = React.lazy(() => 
  import('./HeavyComponent')
);
```

## 三、TypeScript 高级篇

### 3.1 泛型：类型的"函数"

**面试题：什么是泛型？如何设计灵活的泛型？**

#### 泛型的本质

泛型就是"类型参数化"，让类型也能像函数参数一样灵活：

```typescript
// 没有泛型：不灵活
function identity(arg: number): number {
  return arg;
}
// 只能处理 number

// 有泛型：灵活
function identity<T>(arg: T): T {
  return arg;
}

identity<number>(5);      // T = number
identity<string>('hello'); // T = string
identity({ name: 'Tom' }); // T = { name: string }，类型推断
```

#### 泛型约束

```typescript
// 问题：访问不存在的属性
function getProperty<T>(obj: T, key: string) {
  return obj[key];  // 错误：T 可能没有这个属性
}

// 解决：约束 T 必须包含该属性
function getProperty<T, K extends keyof T>(obj: T, key: K) {
  return obj[key];  // ✓ 正确
}

const user = { name: 'Tom', age: 18 };
getProperty(user, 'name');  // ✓
getProperty(user, 'email'); // ✗ 编译错误
```

#### 实战：工具类型

```typescript
// 1. Partial：所有属性变为可选
type Partial<T> = {
  [P in keyof T]?: T[P];
};

// 2. Required：所有属性变为必填
type Required<T> = {
  [P in keyof T]-?: T[P];
};

// 3. Pick：选择部分属性
type Pick<T, K extends keyof T> = {
  [P in K]: T[P];
};

// 4. Omit：排除部分属性
type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// 5. Record：构造对象类型
type Record<K extends string | number | symbol, T> = {
  [P in K]: T;
};

// 使用示例
interface User {
  id: number;
  name: string;
  email: string;
}

type UserUpdate = Partial<User>;  // { id?: number; name?: string; email?: string }
type UserName = Pick<User, 'name'>;  // { name: string }
type UserWithoutId = Omit<User, 'id'>;  // { name: string; email: string }
type UserMap = Record<number, User>;  // { [key: number]: User }
```

### 3.2 条件类型与类型推断

```typescript
// 条件类型：类型的三元表达式
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;   // true
type B = IsString<number>;   // false

// 内置工具类型
type NonNullable<T> = T extends null | undefined ? never : T;

type C = NonNullable<string | null>;  // string
type D = NonNullable<number | undefined>;  // number

// 类型推断：infer 关键字
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : any;

function getUser() {
  return { name: 'Tom', age: 18 };
}

type User = ReturnType<typeof getUser>;  // { name: string; age: number }
```

## 四、设计模式篇

### 4.1 单例模式：全局唯一实例

**面试题：如何实现一个单例模式？有什么应用场景？**

#### 经典实现

```javascript
// ES5 版本
function Singleton(name) {
  this.name = name;
  this.instance = null;
}

Singleton.getInstance = function(name) {
  if (!this.instance) {
    this.instance = new Singleton(name);
  }
  return this.instance;
};

const s1 = Singleton.getInstance('test');
const s2 = Singleton.getInstance('test');
console.log(s1 === s2);  // true
```

#### 现代实现

```javascript
// ES6 Class
class Singleton {
  constructor(name) {
    if (Singleton.instance) {
      return Singleton.instance;
    }
    this.name = name;
    Singleton.instance = this;
  }
}

// 使用 Proxy 实现（更优雅）
const createSingleton = (Constructor) => {
  let instance = null;
  return new Proxy(Constructor, {
    construct(target, args) {
      if (!instance) {
        instance = Reflect.construct(target, args);
      }
      return instance;
    }
  });
};

class Database {
  constructor() {
    this.connection = 'connected';
  }
}

const DB = createSingleton(Database);
const db1 = new DB();
const db2 = new DB();
console.log(db1 === db2);  // true
```

#### 应用场景

1. **全局状态管理**：Vuex、Redux store
2. **数据库连接池**：避免重复创建连接
3. **日志记录器**：统一日志输出
4. **配置管理**：全局配置对象

### 4.2 观察者模式：发布订阅

```javascript
// 实现一个简单的 EventBus
class EventEmitter {
  constructor() {
    this.events = {};
  }
  
  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
    return this;  // 支持链式调用
  }
  
  off(event, callback) {
    if (!this.events[event]) return this;
    
    if (!callback) {
      delete this.events[event];
    } else {
      this.events[event] = this.events[event].filter(
        cb => cb !== callback
      );
    }
    return this;
  }
  
  emit(event, ...args) {
    if (!this.events[event]) return false;
    
    this.events[event].forEach(callback => {
      callback(...args);
    });
    return true;
  }
  
  once(event, callback) {
    const wrapper = (...args) => {
      callback(...args);
      this.off(event, wrapper);
    };
    return this.on(event, wrapper);
  }
}

// 使用
const emitter = new EventEmitter();

emitter.on('click', (e) => console.log('clicked', e));
emitter.emit('click', { x: 100, y: 200 });
```

### 4.3 策略模式：消除条件分支

```javascript
// 问题：大量 if-else
function calculateDiscount(type, price) {
  if (type === 'normal') {
    return price;
  } else if (type === 'member') {
    return price * 0.9;
  } else if (type === 'vip') {
    return price * 0.7;
  } else if (type === 'svip') {
    return price * 0.5;
  }
}

// 优化：策略模式
const strategies = {
  normal: (price) => price,
  member: (price) => price * 0.9,
  vip: (price) => price * 0.7,
  svip: (price) => price * 0.5
};

function calculateDiscount(type, price) {
  const strategy = strategies[type];
  if (!strategy) {
    throw new Error(`Unknown type: ${type}`);
  }
  return strategy(price);
}

// 更灵活：支持运行时添加策略
const DiscountCalculator = {
  strategies: {},
  
  register(type, strategy) {
    this.strategies[type] = strategy;
  },
  
  calculate(type, price) {
    const strategy = this.strategies[type];
    if (!strategy) {
      throw new Error(`Unknown type: ${type}`);
    }
    return strategy(price);
  }
};

// 使用
DiscountCalculator.register('normal', p => p);
DiscountCalculator.register('member', p => p * 0.9);
DiscountCalculator.register('blackfriday', p => p * 0.3);  // 随时添加
```

## 五、网络安全篇

### 5.1 XSS 攻击与防御

**面试题：什么是 XSS 攻击？如何防御？**

#### XSS 的三种类型

1. **存储型 XSS**：恶意脚本存储在服务器
```javascript
// 攻击：在评论区插入恶意脚本
POST /comment
content: <script>stealCookie()</script>

// 受害者访问页面
<div class="comment">
  <script>stealCookie()</script>  // 执行！
</div>
```

2. **反射型 XSS**：恶意脚本在 URL 中
```javascript
// 攻击：诱导用户点击恶意链接
https://example.com/search?q=<script>stealCookie()</script>

// 页面直接渲染
<div>搜索结果：<script>stealCookie()</script></div>
```

3. **DOM 型 XSS**：不经过服务器，纯前端问题
```javascript
// 不安全的代码
const hash = location.hash.slice(1);
document.body.innerHTML = `<h1>${hash}</h1>`;

// 攻击 URL
https://example.com/#<script>stealCookie()</script>
```

#### 防御策略

```javascript
// 1. 转义 HTML 特殊字符
function escapeHTML(str) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return str.replace(/[&<>"']/g, m => map[m]);
}

// 2. 使用安全的 API
// ❌ 危险
element.innerHTML = userInput;

// ✓ 安全
element.textContent = userInput;

// 3. 使用框架的自动转义
// Vue/React 默认转义
<div>{userInput}</div>  // React 自动转义

// 4. CSP（内容安全策略）
// HTTP 头
Content-Security-Policy: default-src 'self'; script-src 'self'

// 5. HttpOnly Cookie
Set-Cookie: session=xxx; HttpOnly
// JavaScript 无法读取
```

### 5.2 CSRF 攻击与防御

**面试题：什么是 CSRF？如何防御？**

#### CSRF 原理

CSRF（跨站请求伪造）：利用用户已登录的身份，在用户不知情的情况下执行操作。

```javascript
// 攻击流程：
// 1. 用户登录银行网站，有 session cookie
// 2. 攻击者诱导用户访问恶意网站
// 3. 恶意网站自动发起转账请求

// 恶意网站代码
<img src="https://bank.com/transfer?to=attacker&amount=1000" 
     style="display:none">

// 因为浏览器自动带上 cookie，转账成功！
```

#### 防御策略

```javascript
// 1. CSRF Token（最有效）
// 服务端生成随机 token
<form>
  <input type="hidden" name="csrf_token" value="random123">
  <button type="submit">转账</button>
</form>

// 服务端验证 token
if (req.body.csrf_token !== session.csrf_token) {
  throw new Error('CSRF attack detected');
}

// 2. SameSite Cookie
Set-Cookie: session=xxx; SameSite=Strict
// 跨站请求不发送 cookie

// 3. 验证 Referer/Origin
if (!req.headers.referer.startsWith('https://bank.com')) {
  throw new Error('Invalid origin');
}

// 4. 二次验证
// 敏感操作需要密码、短信等
```

### 5.3 其他安全要点

```javascript
// 1. SQL 注入
// ❌ 危险
const sql = `SELECT * FROM users WHERE name = '${userInput}'`;

// ✓ 安全：参数化查询
const sql = 'SELECT * FROM users WHERE name = ?';
db.execute(sql, [userInput]);

// 2. 点击劫持
// 防御：X-Frame-Options
X-Frame-Options: DENY
// 或
X-Frame-Options: SAMEORIGIN

// 3. 中间人攻击
// 防御：强制 HTTPS
Strict-Transport-Security: max-age=31536000; includeSubDomains

// 4. 敏感信息泄露
// 不要在前端代码中硬编码密钥
// ❌ 危险
const API_KEY = 'sk-xxxxxxxx';

// ✓ 安全：通过后端代理
```

## 六、实战编程题

### 6.1 实现一个防抖函数

```javascript
/**
 * 防抖函数
 * @param {Function} fn - 要执行的函数
 * @param {number} delay - 延迟时间
 * @param {boolean} immediate - 是否立即执行
 */
function debounce(fn, delay, immediate = false) {
  let timer = null;
  
  return function(...args) {
    const callNow = immediate && !timer;
    
    if (timer) {
      clearTimeout(timer);
    }
    
    timer = setTimeout(() => {
      timer = null;
      if (!immediate) {
        fn.apply(this, args);
      }
    }, delay);
    
    if (callNow) {
      fn.apply(this, args);
    }
  };
}

// 使用场景：搜索框
const searchInput = document.querySelector('#search');
const handleSearch = debounce((e) => {
  console.log('搜索:', e.target.value);
}, 300);

searchInput.addEventListener('input', handleSearch);
```

### 6.2 实现数组扁平化

```javascript
// 方法 1：递归
function flatten(arr) {
  let result = [];
  
  for (let item of arr) {
    if (Array.isArray(item)) {
      result = result.concat(flatten(item));
    } else {
      result.push(item);
    }
  }
  
  return result;
}

// 方法 2：reduce
function flatten(arr) {
  return arr.reduce((acc, item) => {
    return acc.concat(
      Array.isArray(item) ? flatten(item) : item
    );
  }, []);
}

// 方法 3：flat（ES2019）
const result = [1, [2, [3, [4]]]].flat(Infinity);

// 方法 4：迭代（性能更好）
function flatten(arr) {
  const stack = [...arr];
  const result = [];
  
  while (stack.length) {
    const item = stack.pop();
    if (Array.isArray(item)) {
      stack.push(...item);
    } else {
      result.unshift(item);
    }
  }
  
  return result;
}
```

### 6.3 实现函数柯里化

```javascript
/**
 * 函数柯里化
 * add(1)(2)(3) = 6
 */
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    } else {
      return function(...args2) {
        return curried.apply(this, args.concat(args2));
      };
    }
  };
}

// 使用
function add(a, b, c) {
  return a + b + c;
}

const curriedAdd = curry(add);
console.log(curriedAdd(1)(2)(3));  // 6
console.log(curriedAdd(1, 2)(3));  // 6
console.log(curriedAdd(1)(2, 3));  // 6
```

## 七、总结

面试的本质不是考记忆力，而是考察：

1. **知识深度**：是否理解原理，而不是只会用
2. **工程能力**：是否有实战经验，能否解决实际问题
3. **学习能力**：是否持续学习，跟上技术发展
4. **沟通表达**：能否清晰表达复杂概念

### 面试技巧

**STAR 法则回答问题**：
- **S**ituation：什么场景下遇到这个问题
- **T**ask：需要完成什么任务
- **A**ction：采取了什么方案，为什么
- **R**esult：结果如何，有什么收获

**遇到不会的问题**：
1. 不要慌，先冷静思考
2. 尝试从已知知识推导
3. 诚实承认，但展示学习意愿
4. 事后及时补上知识盲点

### 进阶路线

1. **阅读源码**：Vue、React、Webpack、TypeScript
2. **实践项目**：用新技术重构现有项目
3. **技术输出**：写博客、做分享、参与开源
4. **关注前沿**：W3C 规范、TC39 提案、浏览器新特性

记住：面试是双向选择，不仅是公司面你，也是你面公司。保持自信，展示真实的自己！

---

**上一篇**：[前端高频面试题深度解析（一）](099-frontend-interview.md)  
**相关阅读**：
- [JavaScript 深入系列](020-this.md)
- [Webpack 性能优化](064-vite-deep-dive.md)
- [TypeScript 高级类型](065-typescript-advanced.md)
