---
title: Vue2.0源码解读-全局api原理
excerpt: 深入解析Vue2.0全局api原理
category: 前端开发
date: 2026-02-25
readTime: 30
tags: JavaScript, 底层系列, Vue2
---

## 前言

此篇主要手写 Vue2.0 源码-全局 api 原理上一篇咱们主要介绍了 Vue 计算属性原理 知道了计算属性缓存的特点是怎么实现的 到目前为止整个 Vue 源码核心内容 咱们已经基本手写了一遍 那么此篇来梳理下 Vue 的全局 api

适用人群：
1.想要深入理解 vue 源码更好的进行日常业务开发
2.想要在简历写上精通 vue 框架源码（再也不怕面试官的连环夺命问 哈哈）
3.没时间去看官方源码或者初看源码觉得难以理解的同学

## 1 Vue.util

```javascript
// src/global-api/index.js
// exposed util methods.
// NOTE: these are not considered part of the public API - avoid relying on
// them unless you are aware of the risk.
Vue.util = {
  warn,
  extend,
  mergeOptions,
  defineReactive,
};
```

Vue.util 是 Vue 内部的工具方法 不推荐业务组件去使用 因为可能随着版本发生变动 如果咱们不开发第三方 Vue 插件确实使用会比较少

## 2 Vue.set / Vue.delete

```javascript
export function set(target: Array<any> | Object, key: any, val: any): any {
  // 如果是数组 直接调用我们重写的splice方法 可以刷新视图
  if (Array.isArray(target) && isValidArrayIndex(key)) {
    target.length = Math.max(target.length, key);
    target.splice(key, 1, val);
    return val;
  }
  // 如果是对象本身的属性，则直接添加即可
  if (key in target && !(key in Object.prototype)) {
    target[key] = val;
    return val;
  }
  const ob = (target: any).__ob__;
  // 如果对象本身就不是响应式 不需要将其定义成响应式属性
  if (!ob) {
    target[key] = val;
    return val;
  }
  // 利用defineReactive   实际就是Object.defineProperty 将新增的属性定义成响应式的
  defineReactive(ob.value, key, val);
  ob.dep.notify(); // 通知视图更新
  return val;
}
```

```javascript
export function del(target: Array<any> | Object, key: any) {
  // 如果是数组依旧调用splice方法
  if (Array.isArray(target) && isValidArrayIndex(key)) {
    target.splice(key, 1);
    return;
  }
  const ob = (target: any).__ob__;
  // 如果对象本身就没有这个属性 什么都不做
  if (!hasOwn(target, key)) {
    return;
  }
  // 直接使用delete  删除这个属性
  delete target[key];
  //   如果对象本身就不是响应式 直接返回
  if (!ob) {
    return;
  }
  ob.dep.notify(); //通知视图更新
}
```

这两个 api 其实在实际业务场景使用还是很多的 set 方法用来新增响应式数据 delete 方法用来删除响应式数据 因为 Vue 整个响应式过程是依赖 Object.defineProperty 这一底层 api 的 但是这个 api 只能对当前已经声明过的对象属性进行劫持 所以新增的属性不是响应式数据 另外直接修改数组下标也不会引发视图更新 这个是考虑到性能原因 所以我们需要使用$set 和$delete 来进行操作 对响应式原理不熟悉的可以看手写 Vue2.0 源码（一）-响应式数据原理

## 3 Vue.nextTick

```javascript
let callbacks = []; //回调函数
let pending = false;

function flushCallbacks() {
  pending = false; //把标志还原为false
  // 依次执行回调
  for (let i = 0; i < callbacks.length; i++) {
    callbacks[i]();
  }
}

let timerFunc; //先采用微任务并按照优先级优雅降级的方式实现异步刷新

if (typeof Promise !== "undefined") {
  // 如果支持promise
  const p = Promise.resolve();
  timerFunc = () => {
    p.then(flushCallbacks);
  };
} else if (typeof MutationObserver !== "undefined") {
  // MutationObserver 主要是监听dom变化 也是一个异步方法
  let counter = 1;
  const observer = new MutationObserver(flushCallbacks);
  const textNode = document.createTextNode(String(counter));
  observer.observe(textNode, {
    characterData: true,
  });
  timerFunc = () => {
    counter = (counter + 1) % 2;
    textNode.data = String(counter);
  };
} else if (typeof setImmediate !== "undefined") {
  // 如果前面都不支持 判断setImmediate
  timerFunc = () => {
    setImmediate(flushCallbacks);
  };
} else {
  // 最后降级采用setTimeout
  timerFunc = () => {
    setTimeout(flushCallbacks, 0);
  };
}

export function nextTick(cb) {
  // 除了渲染watcher  还有用户自己手动调用的nextTick 一起被收集到数组
  callbacks.push(cb);
  if (!pending) {
    // 如果多次调用nextTick  只会执行一次异步 等异步队列清空之后再把标志变为false
    pending = true;
    timerFunc();
  }
}
```

nextTick 是 Vue 实现异步更新的核心 此 api 在实际业务使用频次也很高 一般用作在数据改变之后立马要获取 dom 节点相关的属性 那么就可以把这样的方法放在 nextTick 中去实现 异步更新原理可以看手写 Vue2.0 源码（五）-异步更新原理

## 4 Vue.observable

```javascript
Vue.observable = <T>(obj: T): T => {
  observe(obj);
  return obj;
};
```

核心就是调用 observe 方法将传入的数据变成响应式对象 可用于制造全局变量在组件共享数据 具体 observe 方法可以看响应式数据原理-对象的数据劫持

## 5 Vue.options

```javascript
Vue.options = Object.create(null);
ASSET_TYPES.forEach((type) => {
  Vue.options[type
```

## 总结

通过以上步骤 我们梳理了 Vue 的全局 api 包括 Vue.util、Vue.set / Vue.delete、Vue.nextTick、Vue.observable 和 Vue.options

这些 api 在 Vue 源码中起着重要的作用 了解它们的原理可以帮助我们更好地理解 Vue 的内部机制

希望通过本文的介绍 大家能够理解 Vue 的全局 api 原理 并在实际开发中灵活运用

如果你还有其他疑问 欢迎在评论区留言 我会尽量为大家解答