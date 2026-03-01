---
title: 03.前端高频面试题深度解析：从 DOM 操作到架构设计的全面突破
category: 前端开发
excerpt: 前端面试题第三弹！深入解析 DOM 操作、CSS 布局、异步编程、微前端架构、低代码平台等高频考点。结合真实面试场景，带你从原理到实战全面突破。
tags: 前端面试, DOM, CSS, 异步编程, 微前端, 低代码, 架构设计
date: 2025-06-21
readTime: 48
---

# 03.前端高频面试题深度解析：从 DOM 操作到架构设计的全面突破

> 面试官："实现一个深拷贝。"
> 
> 你："这个简单，JSON.parse(JSON.stringify(obj))！"
> 
> 面试官："那如果对象里有函数、Date、RegExp、循环引用呢？"
> 
> 你："..."
> 
> 欢迎来到前端面试的第三战场！这次我们深入 DOM 操作、CSS 布局、异步编程、架构设计等更全面的领域。准备好了吗？

## 一、DOM 操作篇

### 1.1 DOM 事件机制

**面试题：事件冒泡和事件捕获有什么区别？如何阻止事件传播？**

#### 事件传播的三个阶段

```
点击 div 内的 button：

1. 捕获阶段（从上到下）
   window → document → html → body → div → button

2. 目标阶段
   button（事件目标）

3. 冒泡阶段（从下到上）
   button → div → body → html → document → window
```

#### addEventListener 的第三个参数

```javascript
// 语法
element.addEventListener(event, handler, options);

// options 可以是布尔值或对象
// 布尔值：true = 捕获阶段，false = 冒泡阶段（默认）
element.addEventListener('click', handler, true);

// 对象：更灵活的控制
element.addEventListener('click', handler, {
  capture: false,      // 冒泡阶段
  once: true,          // 只执行一次
  passive: true        // 不会调用 preventDefault()
});
```

#### 阻止事件传播

```javascript
// 方法 1：阻止冒泡（不影响捕获）
element.addEventListener('click', function(e) {
  e.stopPropagation();
  // 或者旧版 IE
  // e.cancelBubble = true;
});

// 方法 2：阻止默认行为
element.addEventListener('click', function(e) {
  e.preventDefault();
  // 或者旧版 IE
  // e.returnValue = false;
});

// 方法 3：阻止后续所有处理（包括捕获和冒泡）
element.addEventListener('click', function(e) {
  e.stopImmediatePropagation();
});
```

#### 事件委托：性能优化的利器

```javascript
// 问题：给 1000 个 li 绑定点击事件
const lis = document.querySelectorAll('li');
lis.forEach(li => {
  li.addEventListener('click', handleClick);
});
// 内存占用大，性能差

// 优化：事件委托
const ul = document.querySelector('ul');
ul.addEventListener('click', function(e) {
  if (e.target.tagName === 'LI') {
    handleClick.call(e.target, e);
  }
});
// 只绑定一个事件，性能优

// 实战：动态列表
const list = document.querySelector('#todo-list');

list.addEventListener('click', function(e) {
  const target = e.target;
  
  // 删除按钮
  if (target.classList.contains('delete-btn')) {
    const li = target.closest('li');
    li.remove();
  }
  // 编辑按钮
  else if (target.classList.contains('edit-btn')) {
    const li = target.closest('li');
    editTodo(li);
  }
  // 复选框
  else if (target.classList.contains('checkbox')) {
    const li = target.closest('li');
    toggleTodo(li);
  }
});
```

### 1.2 DOM 操作性能优化

**面试题：如何优化 DOM 操作的性能？**

#### 减少重排和重绘

```javascript
// ❌ 糟糕的做法：多次触发重排
element.style.width = '100px';
element.style.height = '100px';
element.style.padding = '10px';
element.style.margin = '10px';

// ✓ 优化：批量修改
element.style.cssText = `
  width: 100px;
  height: 100px;
  padding: 10px;
  margin: 10px;
`;

// 或者使用 class
element.className = 'new-style';
```

#### 使用 DocumentFragment

```javascript
// ❌ 糟糕：每次 append 都触发重排
const list = document.querySelector('#list');
for (let i = 0; i < 1000; i++) {
  const li = document.createElement('li');
  li.textContent = `Item ${i}`;
  list.appendChild(li);
}

// ✓ 优化：使用 DocumentFragment
const fragment = document.createDocumentFragment();
for (let i = 0; i < 1000; i++) {
  const li = document.createElement('li');
  li.textContent = `Item ${i}`;
  fragment.appendChild(li);
}
list.appendChild(fragment);  // 只触发一次重排
```

#### 离线 DOM 操作

```javascript
// 方法 1：display: none
const element = document.querySelector('#myElement');
const display = element.style.display;

element.style.display = 'none';
// 进行大量 DOM 操作
element.innerHTML = '...';
element.style.width = '100px';
// ...

element.style.display = display;  // 只触发一次重排

// 方法 2：cloneNode
const clone = element.cloneNode(true);
// 在 clone 上操作
clone.innerHTML = '...';
// 替换原元素
element.parentNode.replaceChild(clone, element);
```

#### 使用 requestAnimationFrame

```javascript
// ❌ 糟糕：使用 setTimeout
setTimeout(() => {
  element.style.width = '200px';
}, 100);

// ✓ 优化：使用 requestAnimationFrame
requestAnimationFrame(() => {
  element.style.width = '200px';
});
// 在浏览器下一次重绘前执行，更流畅
```

### 1.3 手写 DOM 操作

```javascript
// 实现一个简单的虚拟 DOM
class VNode {
  constructor(tag, props, children) {
    this.tag = tag;
    this.props = props;
    this.children = children;
  }
}

function h(tag, props = {}, children = []) {
  return new VNode(tag, props, children);
}

function render(vnode) {
  if (typeof vnode === 'string') {
    return document.createTextNode(vnode);
  }
  
  const el = document.createElement(vnode.tag);
  
  // 设置属性
  for (let key in vnode.props) {
    setAttr(el, key, vnode.props[key]);
  }
  
  // 渲染子节点
  vnode.children.forEach(child => {
    el.appendChild(render(child));
  });
  
  return el;
}

function setAttr(el, key, value) {
  if (key.startsWith('on')) {
    const event = key.slice(2).toLowerCase();
    el.addEventListener(event, value);
  } else if (key in el) {
    el[key] = value;
  } else {
    el.setAttribute(key, value);
  }
}

// 使用
const vnode = h('div', { id: 'app' }, [
  h('h1', {}, ['Hello']),
  h('p', { class: 'text' }, ['World']),
  h('button', { onclick: () => alert('Hi') }, ['Click'])
]);

const realDOM = render(vnode);
document.body.appendChild(realDOM);
```

## 二、CSS 布局篇

### 2.1 Flexbox 布局

**面试题：Flexbox 有哪些常用属性？如何实现水平垂直居中？**

#### Flexbox 核心属性

```css
.container {
  /* 主轴方向 */
  flex-direction: row | row-reverse | column | column-reverse;
  
  /* 换行 */
  flex-wrap: nowrap | wrap | wrap-reverse;
  
  /* 主轴对齐方式 */
  justify-content: flex-start | flex-end | center | space-between | space-around | space-evenly;
  
  /* 交叉轴对齐方式 */
  align-items: stretch | flex-start | flex-end | center | baseline;
  
  /* 多行时的对齐 */
  align-content: flex-start | flex-end | center | space-between | space-around | stretch;
}

.item {
  /* 放大比例 */
  flex-grow: 0;  /* 默认不放大 */
  
  /* 缩小比例 */
  flex-shrink: 1;  /* 默认缩小 */
  
  /* 固定大小 */
  flex-basis: auto;  /* 默认内容大小 */
  
  /* 简写 */
  flex: 0 1 auto;
  
  /* 单独的对齐 */
  align-self: auto | flex-start | flex-end | center | baseline | stretch;
  
  /* 排序 */
  order: 0;
}
```

#### 水平垂直居中的 N 种方式

```css
/* 方法 1：Flexbox（推荐） */
.container {
  display: flex;
  justify-content: center;
  align-items: center;
}

/* 方法 2：Grid */
.container {
  display: grid;
  place-items: center;
}

/* 方法 3：绝对定位 + transform */
.container {
  position: relative;
}
.item {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

/* 方法 4：绝对定位 + margin auto */
.container {
  position: relative;
}
.item {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  margin: auto;
  width: 100px;
  height: 100px;
}
```

### 2.2 BFC（块级格式化上下文）

**面试题：什么是 BFC？如何触发 BFC？有什么应用？**

#### BFC 的特性

```
1. 内部的 Box 会在垂直方向上一个接一个放置
2. Box 垂直方向的距离由 margin 决定，属于同一个 BFC 的两个相邻 Box 的 margin 会发生重叠
3. 每个元素的 margin box 的左边，与包含块 border box 的左边相接触
4. BFC 的区域不会与 float box 重叠
5. BFC 就是页面上的一个隔离的独立容器，容器里面的子元素不会影响到外面的元素
6. 计算 BFC 的高度时，浮动元素也参与计算
```

#### 触发 BFC 的方式

```css
/* 1. 根元素 */
html

/* 2. float 不为 none */
.element {
  float: left | right;
}

/* 3. position 不为 static 或 relative */
.element {
  position: absolute | fixed;
}

/* 4. display 为特定值 */
.element {
  display: inline-block | table-cell | table-caption | flex | inline-flex | grid | inline-grid;
}

/* 5. overflow 不为 visible */
.element {
  overflow: hidden | auto | scroll;
}
```

#### BFC 的应用

**1. 清除浮动**

```css
/* 方法 1：overflow 触发 BFC */
.clearfix {
  overflow: hidden;
}

/* 方法 2：伪元素（推荐） */
.clearfix::after {
  content: '';
  display: table;
  clear: both;
}
```

**2. 防止 margin 重叠**

```css
/* 问题：相邻元素 margin 重叠 */
.box1 { margin-bottom: 20px; }
.box2 { margin-top: 20px; }
/* 实际间距是 20px，不是 40px */

/* 解决：用 BFC 隔离 */
.wrapper {
  overflow: hidden;  /* 触发 BFC */
}
.box1 { margin-bottom: 20px; }
.wrapper + .box2 { margin-top: 20px; }
```

**3. 自适应两栏布局**

```css
.left {
  float: left;
  width: 200px;
}

.right {
  overflow: hidden;  /* 触发 BFC，不会与 float 重叠 */
  /* 自动填满剩余空间 */
}
```

### 2.3 CSS 性能优化

```css
/* 1. 使用 transform 和 opacity（GPU 加速） */
/* ❌ 差性能 */
.element {
  top: 100px;
  left: 100px;
}

/* ✓ 好性能 */
.element {
  transform: translate(100px, 100px);
}

/* 2. 避免通用选择器 */
/* ❌ 慢 */
* { margin: 0; }
div * { color: red; }

/* ✓ 快 */
.element { margin: 0; }

/* 3. 减少选择器层级 */
/* ❌ 慢 */
ul li a span.icon { }

/* ✓ 快 */
.icon { }

/* 4. 使用 will-change（谨慎使用） */
.element {
  will-change: transform, opacity;
}
/* 提前告知浏览器优化，但会占用内存 */
```

## 三、异步编程篇

### 3.1 Promise 进阶

**面试题：Promise.all 和 Promise.race 有什么区别？如何实现 Promise.all？**

#### Promise.all 实现

```javascript
function promiseAll(promises) {
  return new Promise((resolve, reject) => {
    if (!Array.isArray(promises)) {
      return reject(new TypeError('Argument must be an array'));
    }
    
    const results = [];
    let completed = 0;
    
    if (promises.length === 0) {
      return resolve(results);
    }
    
    promises.forEach((promise, index) => {
      Promise.resolve(promise).then(result => {
        results[index] = result;
        completed++;
        
        if (completed === promises.length) {
          resolve(results);
        }
      }).catch(reject);
    });
  });
}

// 测试
promiseAll([
  Promise.resolve(1),
  Promise.resolve(2),
  3,  // 非 Promise 值
  Promise.reject('error')
]).then(console.log).catch(console.error);
```

#### Promise.race 实现

```javascript
function promiseRace(promises) {
  return new Promise((resolve, reject) => {
    if (!Array.isArray(promises)) {
      return reject(new TypeError('Argument must be an array'));
    }
    
    promises.forEach(promise => {
      Promise.resolve(promise).then(resolve).catch(reject);
    });
  });
}
```

#### Promise.allSettled

```javascript
function promiseAllSettled(promises) {
  return new Promise((resolve, reject) => {
    if (!Array.isArray(promises)) {
      return reject(new TypeError('Argument must be an array'));
    }
    
    const results = [];
    let completed = 0;
    
    if (promises.length === 0) {
      return resolve(results);
    }
    
    promises.forEach((promise, index) => {
      Promise.resolve(promise).then(result => {
        results[index] = { status: 'fulfilled', value: result };
      }).catch(reason => {
        results[index] = { status: 'rejected', reason };
      }).finally(() => {
        completed++;
        if (completed === promises.length) {
          resolve(results);
        }
      });
    });
  });
}
```

### 3.2 async/await 原理

**面试题：async/await 的本质是什么？如何处理错误？**

#### async/await 的本质

```javascript
// async/await 是 Generator + Promise 的语法糖

// Generator 版本
function* getData() {
  const user = yield fetch('/api/user');
  const posts = yield fetch(`/api/posts/${user.id}`);
  return posts;
}

// async/await 版本
async function getData() {
  const user = await fetch('/api/user');
  const posts = await fetch(`/api/posts/${user.id}`);
  return posts;
}

// 转换后
function getData() {
  return fetch('/api/user')
    .then(user => fetch(`/api/posts/${user.id}`))
    .then(posts => posts);
}
```

#### 错误处理

```javascript
// 方法 1：try-catch
async function getData() {
  try {
    const user = await fetch('/api/user');
    const posts = await fetch(`/api/posts/${user.id}`);
    return posts;
  } catch (error) {
    console.error('请求失败:', error);
    throw error;  // 继续抛出
  }
}

// 方法 2：.catch()
async function getData() {
  const user = await fetch('/api/user').catch(err => null);
  if (!user) return [];
  
  const posts = await fetch(`/api/posts/${user.id}`);
  return posts;
}

// 方法 3：统一错误处理（推荐）
async function asyncHandler(promise) {
  try {
    const data = await promise;
    return [null, data];
  } catch (error) {
    return [error, null];
  }
}

// 使用
const [error, posts] = await asyncHandler(fetch('/api/posts'));
if (error) {
  console.error(error);
  return;
}
```

### 3.3 手写异步任务控制

```javascript
// 实现一个并发限制的异步任务队列
class AsyncQueue {
  constructor(limit) {
    this.limit = limit;
    this.running = 0;
    this.queue = [];
  }
  
  add(task) {
    return new Promise((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this.run();
    });
  }
  
  async run() {
    if (this.running >= this.limit || this.queue.length === 0) {
      return;
    }
    
    this.running++;
    const { task, resolve, reject } = this.queue.shift();
    
    try {
      const result = await task();
      resolve(result);
    } catch (error) {
      reject(error);
    } finally {
      this.running--;
      this.run();  // 继续执行下一个
    }
  }
}

// 使用
const queue = new AsyncQueue(3);  // 最多并发 3 个

for (let i = 0; i < 10; i++) {
  queue.add(async () => {
    console.log(`任务 ${i} 开始`);
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log(`任务 ${i} 完成`);
  });
}
```

## 四、架构设计篇

### 4.1 微前端架构

**面试题：什么是微前端？有什么优缺点？如何实现？**

#### 微前端的核心思想

```
将大型前端应用拆分成多个小型、独立的应用，可以：
1. 独立开发：不同团队使用不同技术栈
2. 独立部署：单独发布，不影响其他应用
3. 渐进式升级：逐步迁移，降低风险
```

#### 实现方案对比

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **iframe** | 用 iframe 隔离 | 完全隔离，简单 | 通信复杂，性能差，URL 不同步 |
| **路由分发** | Nginx 反向代理 | 简单，性能好 | 切换页面刷新，状态不共享 |
| **Web Components** | 原生组件标准 | 标准化，隔离好 | 兼容性问题，生态不成熟 |
| **single-spa** | JS 沙箱 + 路由 | 灵活，生态好 | 配置复杂，需要改造 |
| **qiankun** | single-spa 封装 | 开箱即用，功能全 | 体积大，有一定学习成本 |

#### qiankun 实战

```javascript
// 主应用配置
import { registerMicroApps, start } from 'qiankun';

registerMicroApps([
  {
    name: 'app1',
    entry: '//localhost:7100',
    container: '#sub-app-container',
    activeRule: '/app1',
    props: {
      name: '主应用传递的数据'
    }
  },
  {
    name: 'app2',
    entry: '//localhost:7101',
    container: '#sub-app-container',
    activeRule: '/app2'
  }
]);

start({
  prefetch: true,  // 预加载
  sandbox: true,   // 沙箱隔离
  singular: false  // 是否单例
});
```

```javascript
// 子应用配置（Vue）
export const qiankun = {
  // 渲染函数
  render(props) {
    const { container } = props || {};
    new Vue({
      el: container ? container.querySelector('#app') : '#app',
      router,
      store,
      render: h => h(App)
    });
  },
  
  // 生命周期钩子
  async bootstrap() {
    console.log('子应用启动');
  },
  
  async mount(props) {
    console.log('子应用挂载', props);
    this.render(props);
  },
  
  async unmount() {
    console.log('子应用卸载');
    // 清理副作用
  }
};
```

### 4.2 低代码平台设计

**面试题：低代码平台的核心原理是什么？如何设计组件描述协议？**

#### 核心架构

```
┌─────────────────────────────────────────┐
│           可视化编辑器                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ 组件库  │ │ 画布    │ │ 属性面板│   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           Schema 描述协议                │
│  {                                        │
│    "component": "Button",                │
│    "props": { "type": "primary" },       │
│    "children": [...]                     │
│  }                                        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           渲染引擎                       │
│  function render(schema) {               │
│    const Comp = components[schema.comp]; │
│    return <Comp {...schema.props}>       │
│      {schema.children.map(render)}       │
│    </Comp>;                              │
│  }                                        │
└─────────────────────────────────────────┘
```

#### 组件描述协议

```javascript
// Schema 示例
const schema = {
  id: 'root',
  component: 'Container',
  props: {
    width: '100%',
    padding: '20px'
  },
  style: {
    backgroundColor: '#f5f5f5'
  },
  events: {
    onClick: {
      type: 'action',
      actions: [
        { type: 'navigate', url: '/detail' }
      ]
    }
  },
  children: [
    {
      id: 'btn1',
      component: 'Button',
      props: {
        type: 'primary',
        text: '点击我'
      }
    },
    {
      id: 'text1',
      component: 'Text',
      props: {
        content: 'Hello World'
      }
    }
  ]
};
```

#### 渲染引擎实现

```javascript
// 简易渲染引擎
class LowCodeRenderer {
  constructor(components) {
    this.components = components;
  }
  
  render(schema, context = {}) {
    const { component, props = {}, style = {}, children = [], events = {} } = schema;
    
    const Comp = this.components[component];
    if (!Comp) {
      console.warn(`Component ${component} not found`);
      return null;
    }
    
    // 处理事件
    const handlers = {};
    for (let event in events) {
      handlers[event] = this.handleEvent(events[event], context);
    }
    
    // 渲染子节点
    const renderedChildren = children.map(child => 
      this.render(child, { ...context, parent: props })
    );
    
    // 创建组件
    return h(Comp, {
      ...props,
      style,
      ...handlers
    }, renderedChildren);
  }
  
  handleEvent(eventConfig, context) {
    return async (...args) => {
      for (const action of eventConfig.actions) {
        await this.executeAction(action, context, args);
      }
    };
  }
  
  async executeAction(action, context, args) {
    switch (action.type) {
      case 'navigate':
        window.location.href = action.url;
        break;
      case 'api':
        const data = await fetch(action.url, action.options);
        context[action.resultKey] = await data.json();
        break;
      case 'custom':
        await action.handler(...args, context);
        break;
    }
  }
}

// 使用
const renderer = new LowCodeRenderer({
  Container: ({ children, ...props }) => h('div', props, children),
  Button: ({ text, ...props }) => h('button', props, text),
  Text: ({ content }) => h('span', {}, content)
});

const vnode = renderer.render(schema);
```

## 五、实战编程题

### 5.1 实现一个图片懒加载

```javascript
class LazyImage {
  constructor(selector, options = {}) {
    this.images = document.querySelectorAll(selector);
    this.placeholder = options.placeholder || 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    this.loadedClass = options.loadedClass || 'loaded';
    this.errorClass = options.errorClass || 'error';
    
    this.init();
  }
  
  init() {
    // 设置占位图
    this.images.forEach(img => {
      img.dataset.src = img.src;
      img.src = this.placeholder;
    });
    
    // 监听滚动
    this.observe();
  }
  
  observe() {
    if ('IntersectionObserver' in window) {
      // 使用 IntersectionObserver（推荐）
      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.loadImage(entry.target);
            observer.unobserve(entry.target);
          }
        });
      });
      
      this.images.forEach(img => observer.observe(img));
    } else {
      // 降级方案：监听 scroll
      const check = this.debounce(() => {
        this.images.forEach(img => {
          if (this.isInViewport(img)) {
            this.loadImage(img);
          }
        });
      }, 200);
      
      window.addEventListener('scroll', check);
      check();
    }
  }
  
  loadImage(img) {
    if (img.src !== this.placeholder) return;
    
    const src = img.dataset.src;
    const image = new Image();
    
    image.onload = () => {
      img.src = src;
      img.classList.add(this.loadedClass);
    };
    
    image.onerror = () => {
      img.classList.add(this.errorClass);
    };
    
    image.src = src;
  }
  
  isInViewport(img) {
    const rect = img.getBoundingClientRect();
    return rect.top < window.innerHeight && rect.bottom > 0;
  }
  
  debounce(fn, delay) {
    let timer = null;
    return function(...args) {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  }
}

// 使用
new LazyImage('img.lazy', {
  loadedClass: 'fade-in',
  errorClass: 'error-state'
});
```

### 5.2 实现一个简易的虚拟列表

```javascript
class VirtualList {
  constructor(container, options) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    this.itemHeight = options.itemHeight || 50;
    this.bufferSize = options.bufferSize || 10;
    this.data = options.data || [];
    
    this.init();
  }
  
  init() {
    // 创建滚动容器
    this.container.style.overflow = 'auto';
    this.container.style.position = 'relative';
    
    // 占位元素（撑开高度）
    this.placeholder = document.createElement('div');
    this.placeholder.style.position = 'absolute';
    this.placeholder.style.top = 0;
    this.placeholder.style.left = 0;
    this.placeholder.style.right = 0;
    this.container.appendChild(this.placeholder);
    
    // 可视区域容器
    this.viewport = document.createElement('div');
    this.viewport.style.position = 'absolute';
    this.viewport.style.top = 0;
    this.viewport.style.left = 0;
    this.viewport.style.right = 0;
    this.container.appendChild(this.viewport);
    
    // 监听滚动
    this.container.addEventListener('scroll', () => this.render());
    
    // 初始渲染
    this.render();
  }
  
  setData(data) {
    this.data = data;
    this.updatePlaceholder();
    this.render();
  }
  
  updatePlaceholder() {
    const totalHeight = this.data.length * this.itemHeight;
    this.placeholder.style.height = `${totalHeight}px`;
  }
  
  render() {
    const scrollTop = this.container.scrollTop;
    const viewportHeight = this.container.clientHeight;
    
    // 计算可视范围
    const startIndex = Math.max(0, Math.floor(scrollTop / this.itemHeight) - this.bufferSize);
    const endIndex = Math.min(
      this.data.length,
      Math.ceil((scrollTop + viewportHeight) / this.itemHeight) + this.bufferSize
    );
    
    // 渲染可见项
    this.viewport.innerHTML = '';
    this.viewport.style.transform = `translateY(${startIndex * this.itemHeight}px)`;
    
    for (let i = startIndex; i < endIndex; i++) {
      const item = document.createElement('div');
      item.style.height = `${this.itemHeight}px`;
      item.style.display = 'flex';
      item.style.alignItems = 'center';
      item.style.padding = '0 16px';
      item.style.borderBottom = '1px solid #eee';
      item.textContent = this.data[i];
      this.viewport.appendChild(item);
    }
  }
}

// 使用
const list = new VirtualList('#container', {
  itemHeight: 50,
  bufferSize: 5,
  data: Array.from({ length: 10000 }, (_, i) => `Item ${i + 1}`)
});
```

## 六、总结

三篇面试系列文章，我们覆盖了：

**第一篇**：JavaScript 核心、Vue/React 原理、浏览器原理、性能优化、手写代码  
**第二篇**：HTTP/HTTPS、Webpack、TypeScript、设计模式、网络安全  
**第三篇**：DOM 操作、CSS 布局、异步编程、微前端、低代码架构

### 面试的本质

1. **基础扎实**：JavaScript、CSS、浏览器原理
2. **原理深入**：框架源码、构建工具、网络协议
3. **工程能力**：性能优化、架构设计、问题解决
4. **持续学习**：新技术、新趋势、新思维

### 最后的建议

1. **不要死记硬背**：理解原理，能举一反三
2. **多动手实践**：手写代码，做项目，踩坑
3. **学会表达**：清晰阐述思路，展示思考过程
4. **保持好奇心**：技术更新快，持续学习是关键

祝你面试顺利，拿到心仪的 offer！

---

**系列文章**：
- [第一篇：JavaScript 核心与框架原理](099-frontend-interview.md)
- [第二篇：网络协议与工程化](100-frontend-interview-02.md)

**相关阅读**：
- [JavaScript 深入系列](020-this.md)
- [Vue2.0源码解读](023-vue-reactive.md)
- [React Hooks 深度解析](066-react-hooks-deep-dive.md)
- [微前端架构](063-micro-frontend.md)
