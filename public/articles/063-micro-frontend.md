---
title: 微前端架构深度解析：从原理到实战的完整指南
date: 2025-06-15
category: 前端开发
tags: 微前端, qiankun, Module Federation, iframe, 架构设计
excerpt: 深入探索微前端架构的奥秘，从iframe到qiankun再到Module Federation，详解各种方案的实现原理、适用场景和实战案例，助你构建可扩展的大型前端应用。
readTime: 25
---

> 想象一下：你正在维护一个拥有百万行代码的巨型前端应用，每次发版都需要协调十几个团队，一个简单的按钮修改可能要等两周才能上线。这时候，微前端就像一把瑞士军刀，帮你把这个庞然大物拆解成独立运作的小块，每个团队都能自主迭代、独立部署。听起来很美好？但别急，微前端不是银弹，它有自己的适用场景和坑点。今天，我们就来深入聊聊微前端的那些事儿。

## 一、什么是微前端？为什么要用它？

### 1.1 微前端的定义

微前端（Micro-Frontends）是一种将前端应用拆分成更小、更易于管理的独立单元的架构风格。它的核心思想是：**将庞大的前端应用拆分成多个独立开发、独立部署、独立运行的小型应用，这些小型应用可以像搭积木一样组合在一起**。

这个概念借鉴了微服务的思想。如果说微服务是后端架构的解耦利器，那微前端就是前端领域的"分而治之"策略。

### 1.2 为什么需要微前端？

让我们先看一个真实的场景：

> 某电商平台的运营后台，集成了商品管理、订单处理、用户分析、营销活动、财务结算等十几个业务模块。每个模块都由不同的团队维护，技术栈也各不相同：有的用Vue 2，有的用Vue 3，还有的用React。每次大版本升级都是一场噩梦，团队之间的协调成本极高。

这就是典型的**巨石应用（Monolithic Application）**问题。微前端的出现，就是为了解决这些痛点：

| 痛点 | 微前端的解决方案 |
|------|-----------------|
| 代码库庞大，构建速度慢 | 拆分成小应用，独立构建 |
| 多团队协作困难 | 团队自治，独立开发和部署 |
| 技术栈难以升级 | 各应用可以独立选择技术栈 |
| 一个模块出问题影响全局 | 应用隔离，故障不扩散 |
| 新功能上线周期长 | 独立部署，快速迭代 |

### 1.3 微前端的核心价值

1. **技术栈无关**：每个微应用可以选择最适合自己的技术栈
2. **独立开发**：团队可以并行开发，互不干扰
3. **独立部署**：每个微应用可以独立发布，降低风险
4. **渐进式迁移**：可以逐步将老系统迁移到新架构
5. **故障隔离**：一个微应用出错不会影响其他应用

## 二、微前端的实现方案全景图

目前业界主要有以下几种微前端实现方案：

```
微前端方案
├── 简单方案
│   ├── iframe
│   ├── Web Components
│   └── 路由分发
├── 进阶方案
│   ├── qiankun（阿里开源）
│   ├── Module Federation（Webpack 5）
│   ├── single-spa
│   └── Garfish（字节跳动）
└── 新兴方案
    ├── Micro App（京东）
    ├── wujie（腾讯）
    └── 原生ESM导入
```

接下来，我们将深入剖析每种方案的原理、优缺点和实战案例。

## 三、iframe：最简单的微前端方案

### 3.1 原理揭秘

iframe（内联框架）是HTML原生提供的标签，它可以在当前页面中嵌入另一个独立的HTML文档。每个iframe都有自己的**独立窗口（window）、独立文档（document）和独立执行环境**。

```html
<!-- 主应用 -->
<!DOCTYPE html>
<html>
<head>
  <title>主应用</title>
</head>
<body>
  <h1>这是主应用</h1>
  <!-- 嵌入子应用 -->
  <iframe 
    id="sub-app"
    src="https://sub-app.example.com"
    width="100%"
    height="600"
    frameborder="0"
  ></iframe>
</body>
</html>
```

### 3.2 iframe的隔离机制

iframe提供了**完美的隔离性**：

1. **JavaScript隔离**：每个iframe有自己的全局对象，变量不会相互污染
2. **CSS隔离**：iframe内部的样式不会影响外部，反之亦然
3. **DOM隔离**：iframe有自己的DOM树，完全独立
4. **存储隔离**：localStorage、sessionStorage、cookie都是独立的

```javascript
// 主应用和iframe的window对象完全不同
console.log(window === document.getElementById('sub-app').contentWindow); 
// false

// 主应用无法直接访问iframe的变量
// iframe内部：var secret = '这是子应用的秘密';
console.log(window.secret); // undefined
```

### 3.3 iframe通信方案

虽然iframe是隔离的，但主应用和子应用之间可以通过`postMessage`进行通信：

```javascript
// 主应用向子应用发送消息
const iframe = document.getElementById('sub-app');
iframe.contentWindow.postMessage({
  type: 'UPDATE_USER_INFO',
  payload: { name: '张三', id: 123 }
}, 'https://sub-app.example.com');

// 子应用接收消息
window.addEventListener('message', (event) => {
  // 安全检查：验证消息来源
  if (event.origin !== 'https://main-app.example.com') return;
  
  const { type, payload } = event.data;
  if (type === 'UPDATE_USER_INFO') {
    console.log('收到用户信息：', payload);
  }
});
```

### 3.4 iframe的优缺点

**优点：**
- ✅ 实现简单，原生支持
- ✅ 隔离性最强，完全独立
- ✅ 技术栈完全无关
- ✅ 子应用可以独立部署

**缺点：**
- ❌ 体验问题：弹窗无法覆盖整个页面、路由同步困难
- ❌ 性能问题：每个iframe都是独立的渲染进程，内存占用高
- ❌ SEO不友好：搜索引擎难以抓取iframe内容
- ❌ 通信复杂：需要通过postMessage进行跨窗口通信

### 3.5 实战案例：iframe微前端架构

**场景**：某金融公司的管理系统，包含多个独立的业务系统（CRM、ERP、BI），每个系统由不同团队维护。

**架构设计**：

```
主应用（导航框架）
├── 顶部导航栏（用户信息、消息通知）
├── 左侧菜单（各系统入口）
└── 内容区域（iframe容器）
    ├── iframe#crm → CRM系统（Vue 2）
    ├── iframe#erp → ERP系统（React 16）
    └── iframe#bi → BI系统（Angular 10）
```

**核心代码实现**：

```js
<!-- 主应用：App.vue -->
<template>
  <div class="main-layout">
    <TopNav :user="userInfo" />
    <div class="content-wrapper">
      <SideMenu @switch="switchApp" />
      <div class="iframe-container">
        <iframe
          v-for="app in apps"
          :key="app.name"
          v-show="currentApp === app.name"
          :src="app.url"
          :name="app.name"
          @load="onIframeLoad(app.name)"
        />
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      currentApp: 'crm',
      apps: [
        { name: 'crm', url: 'https://crm.company.com' },
        { name: 'erp', url: 'https://erp.company.com' },
        { name: 'bi', url: 'https://bi.company.com' }
      ],
      userInfo: { name: '张三', role: 'admin' }
    };
  },
  mounted() {
    // 监听子应用的消息
    window.addEventListener('message', this.handleMessage);
    // 初始化时发送用户信息
    this.broadcastToAllApps({
      type: 'INIT_USER_INFO',
      payload: this.userInfo
    });
  },
  methods: {
    switchApp(appName) {
      this.currentApp = appName;
    },
    onIframeLoad(appName) {
      console.log(`${appName} 加载完成`);
      // 向特定应用发送初始化数据
      this.sendToApp(appName, {
        type: 'APP_INIT',
        payload: { theme: 'dark', lang: 'zh-CN' }
      });
    },
    sendToApp(appName, data) {
      const iframe = document.querySelector(`iframe[name="${appName}"]`);
      if (iframe) {
        iframe.contentWindow.postMessage(data, '*');
      }
    },
    broadcastToAllApps(data) {
      this.apps.forEach(app => {
        this.sendToApp(app.name, data);
      });
    },
    handleMessage(event) {
      // 处理子应用发来的消息
      const { type, payload } = event.data;
      switch (type) {
        case 'NAVIGATE':
          // 子应用请求导航到另一个系统
          this.switchApp(payload.targetApp);
          break;
        case 'SHOW_MESSAGE':
          // 子应用请求显示全局消息
          this.$message.info(payload.message);
          break;
      }
    }
  }
};
</script>
```

**子应用适配**：

```javascript
// 子应用（CRM系统）的入口文件
class MicroAppAdapter {
  constructor() {
    this.parentOrigin = 'https://main-app.company.com';
    this.init();
  }

  init() {
    // 监听父应用消息
    window.addEventListener('message', (event) => {
      if (event.origin !== this.parentOrigin) return;
      this.handleParentMessage(event.data);
    });

    // 通知父应用：子应用已就绪
    this.notifyParent({
      type: 'APP_READY',
      payload: { appName: 'crm', version: '2.1.0' }
    });
  }

  handleParentMessage(data) {
    const { type, payload } = data;
    switch (type) {
      case 'INIT_USER_INFO':
        // 存储用户信息到全局状态
        window.$userInfo = payload;
        break;
      case 'APP_INIT':
        // 初始化主题、语言等
        this.applyTheme(payload.theme);
        break;
    }
  }

  notifyParent(data) {
    window.parent.postMessage(data, this.parentOrigin);
  }

  // 子应用跳转到其他系统
  navigateTo(appName, path) {
    this.notifyParent({
      type: 'NAVIGATE',
      payload: { targetApp: appName, path }
    });
  }
}

// 初始化适配器
new MicroAppAdapter();
```

## 四、qiankun：阿里开源的微前端框架

### 4.1 qiankun简介

qiankun（乾坤）是阿里巴巴开源的微前端框架，它基于single-spa进行了封装，提供了更简洁的API和更完善的功能。qiankun的名字来源于"乾坤"，寓意"天行健，君子以自强不息"，象征着微前端架构的强大生命力。

### 4.2 qiankun的核心原理

qiankun的核心机制包括：

1. **JS沙箱（Sandbox）**：隔离子应用的JavaScript执行环境
2. **样式隔离**：确保子应用的CSS不会影响其他应用
3. **应用加载**：通过fetch加载子应用的HTML、JS、CSS资源
4. **生命周期管理**：统一子应用的挂载、卸载流程

```
qiankun架构图

主应用（基座）
├── 路由管理（匹配子应用）
├── 应用加载器（fetch资源）
├── JS沙箱（Proxy隔离）
├── 样式隔离（Shadow DOM / CSS前缀）
└── 生命周期管理
    ├── bootstrap（初始化）
    ├── mount（挂载）
    ├── unmount（卸载）
    └── update（更新）

子应用1（Vue）  子应用2（React）  子应用3（Angular）
```

### 4.3 JS沙箱实现原理

qiankun使用**Proxy**创建了一个虚拟的全局对象，子应用对window的修改都被代理到这个虚拟对象上：

```javascript
// 简化的沙箱实现原理
class ProxySandbox {
  constructor() {
    // 创建虚拟的window对象
    this.fakeWindow = {};
    this.proxy = new Proxy(window, {
      get: (target, prop) => {
        // 优先从虚拟window读取
        if (prop in this.fakeWindow) {
          return this.fakeWindow[prop];
        }
        return target[prop];
      },
      set: (target, prop, value) => {
        // 所有修改都写入虚拟window
        this.fakeWindow[prop] = value;
        return true;
      }
    });
  }

  active() {
    // 激活沙箱，子应用使用proxy作为全局对象
    // 实际实现更复杂，需要处理全局事件、定时器等
  }

  inactive() {
    // 卸载沙箱，清理副作用
    // 恢复被修改的全局状态
  }
}
```

### 4.4 样式隔离方案

qiankun提供了两种样式隔离方案：

**方案一：Shadow DOM（严格隔离）**

```javascript
import { loadMicroApp } from 'qiankun';

loadMicroApp({
  name: 'vue-app',
  entry: '//localhost:7101',
  container: '#container',
  // 启用Shadow DOM
  sandbox: {
    strictStyleIsolation: true
  }
});
```

这种方式将子应用渲染在Shadow DOM中，样式完全隔离，但可能有一些兼容性问题。

**方案二：CSS前缀（实验性）**

```javascript
loadMicroApp({
  name: 'vue-app',
  entry: '//localhost:7101',
  container: '#container',
  sandbox: {
    experimentalStyleIsolation: true
  }
});
```

qiankun会自动为子应用的CSS选择器添加前缀，例如：

```css
/* 原始样式 */
.button { color: red; }

/* 转换后 */
[data-qiankun-vue-app] .button { color: red; }
```

### 4.5 qiankun实战：从零搭建微前端应用

**项目结构**：

```
micro-frontend-demo/
├── main-app/          # 主应用（基座）
│   ├── src/
│   │   ├── main.js
│   │   ├── App.vue
│   │   └── micro-apps.js
│   └── package.json
├── vue-sub-app/       # Vue子应用
│   ├── src/
│   │   ├── main.js
│   │   └── public-path.js
│   └── vue.config.js
├── react-sub-app/     # React子应用
│   ├── src/
│   │   ├── index.js
│   │   └── public-path.js
│   └── config-overrides.js
└── package.json
```

**主应用实现**：

```javascript
// main-app/src/main.js
import { createApp } from 'vue';
import { registerMicroApps, start, setDefaultMountApp } from 'qiankun';
import App from './App.vue';
import router from './router';

const app = createApp(App);
app.use(router);
app.mount('#app');

// 注册子应用
registerMicroApps([
  {
    name: 'vue-sub-app',
    entry: '//localhost:7101',
    container: '#subapp-container',
    activeRule: '/vue',
    props: {
      routerBase: '/vue',
      // 传递全局状态
      globalState: {
        user: { name: '张三', id: 123 },
        theme: 'dark'
      }
    }
  },
  {
    name: 'react-sub-app',
    entry: '//localhost:7102',
    container: '#subapp-container',
    activeRule: '/react',
    props: {
      routerBase: '/react'
    }
  }
], {
  // 生命周期钩子
  beforeLoad: (app) => {
    console.log('before load', app.name);
    return Promise.resolve();
  },
  afterMount: (app) => {
    console.log('after mount', app.name);
    return Promise.resolve();
  }
});

// 设置默认子应用
setDefaultMountApp('/vue');

// 启动qiankun
start({
  sandbox: {
    strictStyleIsolation: false,
    experimentalStyleIsolation: true
  }
});
```

```js
<!-- main-app/src/App.vue -->
<template>
  <div class="main-app">
    <header class="header">
      <div class="logo">Micro Frontend Demo</div>
      <nav class="nav">
        <router-link to="/">首页</router-link>
        <router-link to="/vue">Vue子应用</router-link>
        <router-link to="/react">React子应用</router-link>
      </nav>
      <div class="user-info">
        欢迎，{{ userInfo.name }}
      </div>
    </header>
    
    <main class="main-content">
      <!-- 主应用自己的路由内容 -->
      <router-view v-show="!isMicroApp"></router-view>
      
      <!-- 子应用挂载点 -->
      <div id="subapp-container" v-show="isMicroApp"></div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRoute } from 'vue-router';

export default {
  setup() {
    const route = useRoute();
    const isMicroApp = computed(() => {
      return route.path.startsWith('/vue') || route.path.startsWith('/react');
    });
    
    return {
      isMicroApp,
      userInfo: { name: '张三' }
    };
  }
};
</script>
```

**Vue子应用改造**：

```javascript
// vue-sub-app/src/public-path.js
// 动态设置publicPath，确保资源加载正确
if (window.__POWERED_BY_QIANKUN__) {
  __webpack_public_path__ = window.__INJECTED_PUBLIC_PATH_BY_QIANKUN__;
}
```

```javascript
// vue-sub-app/src/main.js
import './public-path';
import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import store from './store';

let instance = null;

// 渲染函数
function render(props = {}) {
  const { container, globalState } = props;
  
  instance = createApp(App);
  instance.use(router);
  instance.use(store);
  
  // 接收主应用传递的状态
  if (globalState) {
    instance.provide('globalState', globalState);
  }
  
  instance.mount(container ? container.querySelector('#app') : '#app');
}

// 独立运行时直接渲染
if (!window.__POWERED_BY_QIANKUN__) {
  render();
}

// qiankun生命周期钩子
export async function bootstrap() {
  console.log('vue sub app bootstraped');
}

export async function mount(props) {
  console.log('vue sub app mount', props);
  render(props);
}

export async function unmount() {
  console.log('vue sub app unmount');
  instance.unmount();
  instance = null;
}
```

```javascript
// vue-sub-app/vue.config.js
const { defineConfig } = require('@vue/cli-service');
const packageName = require('./package.json').name;

module.exports = defineConfig({
  devServer: {
    port: 7101,
    headers: {
      // 允许跨域
      'Access-Control-Allow-Origin': '*',
    },
  },
  configureWebpack: {
    output: {
      // 打包成umd格式
      library: `${packageName}-[name]`,
      libraryTarget: 'umd',
      chunkLoadingGlobal: `webpackJsonp_${packageName}`,
    },
  },
});
```

**React子应用改造**：

```javascript
// react-sub-app/src/public-path.js
if (window.__POWERED_BY_QIANKUN__) {
  __webpack_public_path__ = window.__INJECTED_PUBLIC_PATH_BY_QIANKUN__;
}
```

```javascript
// react-sub-app/src/index.js
import './public-path';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

let root = null;

function render(props) {
  const { container, globalState } = props;
  const dom = container ? container.querySelector('#root') : document.getElementById('root');
  
  root = ReactDOM.createRoot(dom);
  root.render(
    <React.StrictMode>
      <App globalState={globalState} />
    </React.StrictMode>
  );
}

if (!window.__POWERED_BY_QIANKUN__) {
  render({});
}

export async function bootstrap() {
  console.log('react app bootstraped');
}

export async function mount(props) {
  console.log('react app mount', props);
  render(props);
}

export async function unmount() {
  console.log('react app unmount');
  root.unmount();
  root = null;
}
```

### 4.6 qiankun的通信机制

qiankun提供了`initGlobalState`API来实现主应用和子应用之间的状态共享：

```javascript
// 主应用：初始化全局状态
import { initGlobalState, MicroAppStateActions } from 'qiankun';

const initialState = {
  user: { name: '张三', role: 'admin' },
  theme: 'dark',
  messages: []
};

const actions = initGlobalState(initialState);

// 监听状态变化
actions.onGlobalStateChange((state, prev) => {
  console.log('主应用：状态变化', state, prev);
});

// 修改状态
actions.setGlobalState({ theme: 'light' });
```

```javascript
// 子应用：接收和使用全局状态
export async function mount(props) {
  const { onGlobalStateChange, setGlobalState } = props;
  
  // 监听全局状态
  onGlobalStateChange((state, prev) => {
    console.log('子应用：状态变化', state, prev);
    // 更新本地状态
    store.commit('updateGlobalState', state);
  });
  
  // 修改全局状态
  setGlobalState({
    messages: [...state.messages, { text: '来自子应用的消息' }]
  });
}
```

### 4.7 qiankun的优缺点

**优点：**
- ✅ 开箱即用，API简洁
- ✅ 完善的沙箱机制
- ✅ 支持多种技术栈
- ✅ 社区活跃，文档完善
- ✅ 阿里系大规模验证

**缺点：**
- ❌ 需要改造子应用，添加生命周期钩子
- ❌ 某些库（如Chart.js）可能在沙箱中运行异常
- ❌ 样式隔离方案各有局限
- ❌ 子应用加载需要额外网络请求

## 五、Module Federation：Webpack 5的革命性方案

### 5.1 什么是Module Federation？

Module Federation（模块联邦）是Webpack 5引入的一项革命性特性，它允许JavaScript应用**在运行时动态加载其他应用的代码模块**。

与qiankun等方案不同，Module Federation不是"应用级别"的微前端，而是"模块级别"的共享。你可以把它理解为：**把多个应用构建成一个"分布式的大型应用"，它们可以像调用本地模块一样调用彼此的代码**。

### 5.2 Module Federation的核心概念

```
Module Federation架构

┌─────────────────────────────────────────────────────────────┐
│                      运行时容器（Container）                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │   Host应用    │◄──────►│ Remote应用A  │                  │
│  │  (消费者)     │  共享   │  (提供者)    │                  │
│  │              │  模块   │              │                  │
│  │ import('remoteA/Button') │              │                  │
│  └──────┬───────┘        └──────────────┘                  │
│         │                                                   │
│         │              ┌──────────────┐                    │
│         └─────────────►│ Remote应用B  │                    │
│                        │  (提供者)    │                    │
│                        └──────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心术语：**

1. **Host（消费者）**：消费其他应用模块的应用
2. **Remote（提供者）**：提供模块给其他应用消费的应用
3. **Shared（共享模块）**：多个应用共享的依赖（如React、Vue）
4. **Container**：运行时模块容器

### 5.3 Module Federation的工作原理

Module Federation的神奇之处在于它的**运行时模块加载机制**：

```javascript
// 简化的原理说明

// 1. Remote应用打包时，会生成一个manifest文件
// remoteEntry.js - 描述了这个应用提供了哪些模块
var remoteApp = {
  get: (moduleName) => {
    // 动态加载模块
    return __webpack_require__(moduleName);
  },
  init: (shareScope) => {
    // 初始化共享作用域
    // 确保共享依赖版本兼容
  }
};

// 2. Host应用在运行时加载remoteEntry.js
const remoteApp = await import('remoteApp/remoteEntry.js');
await remoteApp.init(__webpack_share_scopes__.default);

// 3. 像使用本地模块一样使用远程模块
const Button = await import('remoteApp/Button');
```

### 5.4 Module Federation配置详解

**Remote应用配置（提供者）**：

```javascript
// remote-app/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  entry: './src/index',
  mode: 'development',
  devServer: {
    port: 3001,
  },
  plugins: [
    new ModuleFederationPlugin({
      // 应用名称，唯一标识
      name: 'remoteApp',
      
      // 远程入口文件名
      filename: 'remoteEntry.js',
      
      // 暴露的模块
      exposes: {
        // 键：模块的公开名称
        // 值：模块的实际路径
        './Button': './src/components/Button',
        './utils': './src/utils/index',
        './store': './src/store',
      },
      
      // 共享依赖
      shared: {
        react: {
          singleton: true,  // 单例模式，确保只有一个实例
          requiredVersion: '^18.0.0',
        },
        'react-dom': {
          singleton: true,
          requiredVersion: '^18.0.0',
        },
        // 共享状态管理
        '@reduxjs/toolkit': {
          singleton: true,
        },
      },
    }),
  ],
};
```

**Host应用配置（消费者）**：

```javascript
// host-app/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  entry: './src/index',
  mode: 'development',
  devServer: {
    port: 3000,
  },
  plugins: [
    new ModuleFederationPlugin({
      name: 'hostApp',
      
      // 引用的远程应用
      remotes: {
        // 键：引用时的别名
        // 值：应用名@远程入口URL
        remoteApp: 'remoteApp@http://localhost:3001/remoteEntry.js',
        dashboardApp: 'dashboardApp@http://localhost:3002/remoteEntry.js',
      },
      
      // 共享依赖（版本要匹配）
      shared: {
        react: {
          singleton: true,
          eager: true,  // 立即加载，不异步
          requiredVersion: '^18.0.0',
        },
        'react-dom': {
          singleton: true,
          eager: true,
          requiredVersion: '^18.0.0',
        },
      },
    }),
  ],
};
```

### 5.5 Module Federation实战：组件共享

**场景**：公司有一个通用的UI组件库，需要在多个项目中共享使用。

**Remote应用（组件库）**：

```javascript
// component-lib/src/components/Button/index.jsx
import React from 'react';
import './Button.css';

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'medium',
  onClick,
  disabled = false
}) => {
  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

export default Button;
```

```javascript
// component-lib/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  entry: './src/index',
  mode: 'production',
  output: {
    publicPath: 'https://cdn.company.com/component-lib/',
  },
  plugins: [
    new ModuleFederationPlugin({
      name: 'componentLib',
      filename: 'remoteEntry.js',
      exposes: {
        './Button': './src/components/Button',
        './Input': './src/components/Input',
        './Modal': './src/components/Modal',
        './Table': './src/components/Table',
        './theme': './src/theme/index',
      },
      shared: {
        react: { singleton: true },
        'react-dom': { singleton: true },
      },
    }),
  ],
};
```

**Host应用（业务系统A）**：

```javascript
// app-a/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'appA',
      remotes: {
        // 引用组件库
        components: 'componentLib@https://cdn.company.com/component-lib/remoteEntry.js',
      },
      shared: {
        react: { singleton: true },
        'react-dom': { singleton: true },
      },
    }),
  ],
};
```

```jsx
// app-a/src/pages/UserList.jsx
import React, { lazy, Suspense } from 'react';

// 动态导入远程组件
const Button = lazy(() => import('components/Button'));
const Table = lazy(() => import('components/Table'));

const UserList = () => {
  const columns = [
    { title: '姓名', dataIndex: 'name' },
    { title: '邮箱', dataIndex: 'email' },
    { title: '角色', dataIndex: 'role' },
  ];

  return (
    <div className="user-list">
      <h1>用户管理</h1>
      
      <Suspense fallback={<div>加载组件中...</div>}>
        <div className="actions">
          <Button variant="primary" onClick={() => console.log('新增用户')}>
            新增用户
          </Button>
          <Button variant="secondary">
            批量导入
          </Button>
        </div>
        
        <Table 
          columns={columns}
          dataSource={users}
          pagination={{ pageSize: 10 }}
        />
      </Suspense>
    </div>
  );
};

export default UserList;
```

### 5.6 Module Federation的高级用法：状态共享

Module Federation不仅可以共享UI组件，还可以共享状态管理：

```javascript
// shared-store/src/store/index.js
import { configureStore, createSlice } from '@reduxjs/toolkit';

const userSlice = createSlice({
  name: 'user',
  initialState: {
    info: null,
    permissions: [],
    isLogin: false,
  },
  reducers: {
    setUserInfo: (state, action) => {
      state.info = action.payload;
      state.isLogin = true;
    },
    logout: (state) => {
      state.info = null;
      state.isLogin = false;
      state.permissions = [];
    },
  },
});

export const { setUserInfo, logout } = userSlice.actions;

export const store = configureStore({
  reducer: {
    user: userSlice.reducer,
  },
});

// 导出hooks供外部使用
export const useUserInfo = () => {
  return useSelector((state) => state.user.info);
};
```

```javascript
// shared-store/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'sharedStore',
      filename: 'remoteEntry.js',
      exposes: {
        './store': './src/store',
        './UserProvider': './src/components/UserProvider',
      },
      shared: {
        '@reduxjs/toolkit': { singleton: true },
        'react-redux': { singleton: true },
        react: { singleton: true },
      },
    }),
  ],
};
```

```jsx
// app-a/src/App.jsx
import React, { lazy, Suspense } from 'react';

// 导入共享的store
const { store, UserProvider } = lazy(() => import('sharedStore/store'));

const App = () => {
  return (
    <Suspense fallback={<div>加载中...</div>}>
      <UserProvider store={store}>
        <Router>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/users" element={<UserList />} />
          </Routes>
        </Router>
      </UserProvider>
    </Suspense>
  );
};
```

### 5.7 Module Federation的优缺点

**优点：**
- ✅ 真正的运行时模块共享
- ✅ 不需要iframe或复杂的沙箱
- ✅ 共享依赖，减少重复加载
- ✅ 可以共享任意模块（组件、工具、状态）
- ✅ 类型安全（配合TypeScript）

**缺点：**
- ❌ 必须使用Webpack 5
- ❌ 所有应用需要统一构建工具
- ❌ 版本兼容性需要仔细管理
- ❌ 调试相对复杂
- ❌ 不支持Vue 2等旧版本框架

## 六、微前端方案对比与选型建议

### 6.1 方案对比表

| 特性 | iframe | qiankun | Module Federation |
|------|--------|---------|-------------------|
| 实现复杂度 | ⭐ 简单 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 复杂 |
| 隔离性 | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐ 良好 | ⭐⭐ 较弱 |
| 性能 | ⭐⭐ 较差 | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 优秀 |
| 技术栈限制 | 无限制 | 需改造 | 需Webpack 5 |
| 通信复杂度 | ⭐⭐⭐⭐ 复杂 | ⭐⭐ 简单 | ⭐ 极简 |
| 样式隔离 | 完美 | 需配置 | 需自行处理 |
| SEO友好度 | ⭐ 差 | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 优秀 |
| 适用场景 | 简单集成 | 中大型应用 | 组件共享 |

### 6.2 选型建议

**选择iframe，如果你：**
- 需要快速集成第三方系统
- 对隔离性要求极高
- 不介意体验上的一些妥协
- 子应用完全独立，很少需要通信

**选择qiankun，如果你：**
- 正在构建一个大型的多团队前端应用
- 需要渐进式迁移老系统
- 团队技术栈不统一
- 需要完善的沙箱隔离

**选择Module Federation，如果你：**
- 需要共享组件和模块
- 所有应用都用Webpack 5
- 追求极致的性能
- 团队技术栈相对统一

## 七、微前端的最佳实践

### 7.1 应用拆分原则

1. **按业务领域拆分**：每个微应用对应一个业务领域
2. **保持应用自治**：每个应用可以独立开发、测试、部署
3. **最小化通信**：减少应用间的依赖和通信
4. **统一规范**：制定统一的接口规范、路由规范、样式规范

### 7.2 样式隔离最佳实践

```css
/* 为每个微应用添加命名空间 */
[data-app="vue-app"] {
  /* Vue子应用的样式 */
}

[data-app="react-app"] {
  /* React子应用的样式 */
}

/* 使用CSS Modules */
.button { /* 自动转换为 .button__xxx */ }

/* 使用CSS-in-JS */
const Button = styled.button`
  /* 样式只作用于当前组件 */
`;
```

### 7.3 通信机制设计

```javascript
// 建立统一的通信规范
const EVENT_TYPES = {
  USER_LOGIN: 'user:login',
  USER_LOGOUT: 'user:logout',
  THEME_CHANGE: 'theme:change',
  ROUTE_CHANGE: 'route:change',
  ERROR_REPORT: 'error:report',
};

// 封装通信层
class MicroAppBus {
  emit(event, payload) {
    // 实现...
  }
  
  on(event, callback) {
    // 实现...
  }
  
  off(event, callback) {
    // 实现...
  }
}
```

### 7.4 性能优化

1. **懒加载**：按需加载微应用
2. **预加载**：预测用户行为，提前加载
3. **缓存策略**：合理缓存资源
4. **代码分割**：每个微应用独立打包

```javascript
// qiankun预加载示例
import { prefetchApps } from 'qiankun';

// 用户登录后预加载常用应用
prefetchApps([
  { name: 'dashboard', entry: '//localhost:7101' },
  { name: 'settings', entry: '//localhost:7102' },
]);
```

## 八、总结与展望

微前端不是银弹，但它确实为大型前端应用的开发和维护提供了一种可行的解决方案。从最简单的iframe到最先进的Module Federation，每种方案都有其适用场景。

**关键要点：**

1. **没有最好的方案，只有最适合的方案**
2. **微前端增加了复杂度，不要为了拆分而拆分**
3. **良好的架构设计比技术选型更重要**
4. **团队规范和协作是微前端成功的关键**

**未来趋势：**

- **原生ESM**：浏览器原生模块加载越来越成熟
- **Web Components**：标准化的组件封装方案
- **Server Components**：React正在探索的服务端组件
- **低代码集成**：微前端与低代码平台的结合

微前端的旅程才刚刚开始，期待看到更多创新的实践和方案！

---

**延伸阅读：**
- [qiankun官方文档](https://qiankun.umijs.org/)
- [Webpack Module Federation](https://webpack.js.org/concepts/module-federation/)
- [single-spa文档](https://single-spa.js.org/)
- [Micro-Frontends.org](https://micro-frontends.org/)
