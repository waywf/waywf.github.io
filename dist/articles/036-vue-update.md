---
title: Vue2.0源码解读-渲染更新原理
excerpt: 深入解析Vue2.0渲染更新原理
category: 技术
date: 2026-02-25
readTime: 30
tags: JavaScript, 底层系列, Vue
---

## 前言

此篇主要手写 Vue2.0 源码-渲染更新原理上一篇咱们主要介绍了 Vue 初始渲染原理 完成了数据到视图层的映射过程 但是当我们改变数据的时候发现页面并不会自动更新 我们知道 Vue 的一个特性就是数据驱动 当数据改变的时候 我们无需手动操作 dom 视图会自动更新 回顾第一篇 响应式数据原理 此篇主要采用观察者模式 定义 Watcher 和 Dep 完成依赖收集和派发更新 从而实现渲染更新

提示：此篇难度稍大 是整个 Vue 源码非常核心的内容 后续的计算属性和自定义 watcher 以及$set $delete 等 Api 的实现 都需要理解此篇的思路 小编看源码这块也是看了有好几遍才搞懂 希望大家克服困难一起去实现一遍吧！

## 正文

```javascript
<script>
  // Vue实例化
  let vm = new Vue({
    el: "#app",
    data() {
      return {
        a: 123,
      };
    },
    // render(h) {
    //   return h('div',{id:'a'},'hello')
    // },
    template: `<div id="a">hello {{a}}</div>`,
  });
  // 我们在这里模拟更新
  setTimeout(() => {
    vm.a = 456;
    // 此方法是刷新视图的核心
    vm._update(vm._render());
  }, 1000);
</script>
```

上段代码 我们在 setTimeout 里面调用 vm._update(vm._render())来实现更新功能 因为从上一篇初始渲染的原理可知 此方法就是渲染的核心

但是我们不可能每次数据变化都要求用户自己去调用渲染方法更新视图 我们需要一个机制在数据变动的时候自动去更新

## 1.定义 Watcher

```javascript
// src/observer/watcher.js
// 全局变量id  每次new Watcher都会自增
let id = 0;

export default class Watcher {
  constructor(vm, exprOrFn, cb, options) {
    this.vm = vm;
    this.exprOrFn = exprOrFn;
    this.cb = cb; //回调函数 比如在watcher更新之前可以执行beforeUpdate方法
    this.options = options; //额外的选项 true代表渲染watcher
    this.id = id++; // watcher的唯一标识
    // 如果表达式是一个函数
    if (typeof exprOrFn === "function") {
      this.getter = exprOrFn;
    }
    // 实例化就会默认调用get方法
    this.get();
  }
  get() {
    this.getter();
  }
}
```

在 observer 文件夹下新建 watcher.js 代表和观察者相关 这里首先介绍 Vue 里面使用到的观察者模式 我们可以把 Watcher 当做观察者 它需要订阅数据的变动 当数据变动之后 通知它去执行某些方法 其实本质就是一个构造函数 初始化的时候会去执行 get 方法

## 2.创建渲染 Watcher

```javascript
// src/lifecycle.js
export function mountComponent(vm, el) {
  //   _update和._render方法都是挂载在Vue原型的方法  类似_init
  // 引入watcher的概念 这里注册一个渲染watcher 执行vm._update(vm._render())方法渲染视图
  let updateComponent = () => {
    console.log("刷新页面");
    vm._update(vm._render());
  };
  new Watcher(vm, updateComponent, null, true);
}
```

我们在组件挂载方法里面 定义一个渲染 Watcher 主要功能就是执行核心渲染页面的方法

## 3.定义 Dep

```javascript
// src/observer/dep.js
// dep和watcher是多对多的关系
// 每个属性都有自己的dep
let id = 0; //dep实例的唯一标识
export default class Dep {
  constructor() {
    this.id = id++;
    this.subs = []; // 这个是存放watcher的容器
  }
}

// 默认Dep.target为null
Dep.target = null;
```

Dep 也是一个构造函数 可以把他理解为观察者模式里面的被观察者 在 subs 里面收集 watcher 当数据变动的时候通知自身 subs 所有的 watcher 更新

Dep.target 是一个全局 Watcher 指向 初始状态是 null

## 4.对象的依赖收集

```javascript
// src/observer/index.js
// Object.defineProperty数据劫持核心 兼容性在ie9以及以上
function defineReactive(data, key, value) {
  observe(value);
  let dep = new Dep(); // 为每个属性实例化一个Dep
  Object.defineProperty(data, key, {
    get() {
      // 页面取值的时候 可以把watcher收集到dep里面--依赖收集
      if (Dep.target) {
        // 如果有watcher dep就会保存watcher 同时watcher也会保存dep
        dep.depend();
      }
      return value;
    },
    set(newValue) {
      if (newValue === value) return;
      // 如果赋值的新值也是一个对象  需要观测
      observe(newValue);
      value = newValue;
      dep.notify(); // 通知渲染watcher去更新--派发更新
    },
  });
}
```

上诉代码就是依赖收集和派发更新的核心 其实就是在数据被访问的时候 把我们定义好的渲染 Watcher 放到 dep 的 subs 数组里面 同时把 dep 实例对象也放到渲染 Watcher 里面去 数据更新时就可以通知 dep 的 subs 存储的 watcher 更新

## 5.完善 watcher

```javascript
// src/observer/watcher.js
import { pushTarget, popTarget } from "./dep";

// 全局变量id  每次new Watcher都会自增
let id = 0;

export default class Watcher {
  constructor(vm, exprOrFn, cb, options) {
    this.vm = vm;
    this.exprOrFn = exprOrFn;
    this.cb = cb; //回调函数 比如在watcher更新之前可以执行beforeUpdate方法
    this.options = options; //额外的选项 true代表渲染watcher
    this.id = id++; // watcher的唯一标识
    this.deps = []; // 存放dep的容器
    this.depsId = new Set(); // 用来去重dep
    // 如果表达式是一个函数
    if (typeof exprOrFn === "function") {
      this.getter = exprOrFn;
    }
    // 实例化就会默认调用get方法
    this.get();
  }
  get() {
    pushTarget(this); // 把当前watcher实例推到全局Dep.target上
    this.getter.call(this.vm); // 执行渲染函数
    popTarget(); // 弹出当前watcher实例
  }
  addDep(dep) {
    let id = dep.id;
    if (!this.depsId.has(id)) {
      this.deps.push(dep);
      this.depsId.add(id);
      dep.addSub(this); // 把当前watcher添加到dep的subs数组中
    }
  }
  update() {
    this.get(); // 直接调用get方法更新视图
  }
}
```

## 6.完善 dep

```javascript
// src/observer/dep.js
// dep和watcher是多对多的关系
// 每个属性都有自己的dep
let id = 0; //dep实例的唯一标识
export default class Dep {
  constructor() {
    this.id = id++;
    this.subs = []; // 这个是存放watcher的容器
  }
  addSub(watcher) {
    this.subs.push(watcher); // 把watcher添加到subs数组中
  }
  depend() {
    if (Dep.target) {
      Dep.target.addDep(this); // 调用watcher的addDep方法
    }
  }
  notify() {
    this.subs.forEach((watcher) => {
      watcher.update(); // 调用watcher的update方法
    });
  }
}

// 默认Dep.target为null
Dep.target = null;

// 栈结构用来存watcher
let targetStack = [];

export function pushTarget(watcher) {
  targetStack.push(watcher);
  Dep.target = watcher;
}

export function popTarget() {
  targetStack.pop();
  Dep.target = targetStack[targetStack.length - 1];
}
```

## 总结

通过以上步骤 我们实现了 Vue 的渲染更新原理 主要采用观察者模式 定义 Watcher 和 Dep 完成依赖收集和派发更新 从而实现渲染更新

当数据变动的时候 会触发 set 方法 然后调用 dep.notify() 通知所有的 watcher 去更新 最后调用 watcher.update() 方法更新视图

这就是 Vue 数据驱动视图的核心原理

希望通过本文的介绍 大家能够理解 Vue 的渲染更新原理 并在实际开发中灵活运用

如果你还有其他疑问 欢迎在评论区留言 我会尽量为大家解答