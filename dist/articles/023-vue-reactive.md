---
title: Vue2.0源码解读-响应式数据原理
excerpt: 深入解析Vue2.0响应式数据原理
category: 前端开发
date: 2021-10-20
readTime: 25
tags: JavaScript, 底层系列, Vue2
---

## 前言

大家都知道 Vue 的一个核心特点是数据驱动 如果按照以往 Jquery 的思想 咱们数据变化了想要同步到视图就必须要手动操作 dom 更新 但是 Vue 帮我们做到了数据变动自动更新视图的功能 那在 Vue 内部就一定有一个机制能监听到数据变化然后触发更新 本篇主要介绍响应式数据的原理

## 数据初始化

```javascript
new Vue({
  el: "#app",
  router,
  store,
  render: (h) => h(App),
});
```

这段代码 大家一定非常熟悉 这就是 Vue 实例化的过程 从 new 操作符 咱们可以看出 Vue 其实就是一个构造函数 没啥特别的 传入的参数就是一个对象 我们叫做 options（选项）

```javascript
// src/index.js
import { initMixin } from "./init.js";

// Vue就是一个构造函数 通过new关键字进行实例化
function Vue(options) {
  // 这里开始进行Vue初始化工作
  this._init(options);
}

// _init方法是挂载在Vue原型的方法 通过引入文件的方式进行原型挂载需要传入Vue
// 此做法有利于代码分割
initMixin(Vue);

export default Vue;
```

因为在 Vue 初始化可能会处理很多事情 比如数据处理 事件处理 生命周期处理等等 所以划分不同文件引入利于代码分割

```javascript
// src/init.js
import { initState } from "./state";

export function initMixin(Vue) {
  Vue.prototype._init = function (options) {
    const vm = this;
    // 这里的this代表调用_init方法的对象(实例对象)
    //  this.$options就是用户new Vue的时候传入的属性
    vm.$options = options;
    // 初始化状态
    initState(vm);
  };
}
```

initMixin 把_init 方法挂载在 Vue 原型 供 Vue 实例调用

```javascript
// src/state.js
import { observe } from "./observer/index.js";

// 初始化状态 注意这里的顺序 比如我经常面试会问到 是否能在data里面直接使用prop的值 为什么？
// 这里初始化的顺序依次是 prop>methods>data>computed>watchexport function initState(vm) {
  // 获取传入的数据对象
  const opts = vm.$options;
  if (opts.props) {
    initProps(vm);
  }
  if (opts.methods) {
    initMethod(vm);
  }
  if (opts.data) {
    // 初始化data
    initData(vm);
  }
  if (opts.computed) {
    initComputed(vm);
  }
  if (opts.watch) {
    initWatch(vm);
  }
}

// 初始化data数据
function initData(vm) {
  let data = vm.$options.data;
  // 实例的_data属性就是传入的data
  // vue组件data推荐使用函数 防止数据在组件之间共享
  data = vm._data = typeof data === "function" ? data.call(vm) : data || {};
  // 把data数据代理到vm 也就是Vue实例上面 我们可以使用this.a来访问this._data.a
  for (let key in data) {
    proxy(vm, `_data`, key);
  }
  // 对数据进行观测 --响应式数据核心
  observe(data);
}

// 数据代理
function proxy(object, sourceKey, key) {
  Object.defineProperty(object, key, {
    get() {
      return object[sourceKey][key];
    },
    set(newValue) {
      object[sourceKey][key] = newValue;
    },
  });
}
```

initState 咱们主要关注 initData 里面的 observe 是响应式数据核心 所以另建 observer 文件夹来专注响应式逻辑 其次我们还做了一层数据代理 把data代理到实例对象this上

## 对象的数据劫持

```javascript
// src/obserber/index.js
class Observer {
  // 观测值
  constructor(value) {
    this.walk(value);
  }
  walk(data) {
    // 对象上的所有属性依次进行观测
    let keys = Object.keys(data);
    for (let i = 0; i < keys.length; i++) {
      let key = keys[i];
      let value = data[key];
      defineReactive(data, key, value);
    }
  }
}

// Object.defineProperty数据劫持核心 兼容性在ie9以及以上
function defineReactive(data, key, value) {
  observe(value); // 递归关键
  // --如果value还是一个对象会继续走一遍odefineReactive 层层遍历一直到value不是对象才停止
  //   思考？如果Vue数据嵌套层级过深 >>性能会受影响
  Object.defineProperty(data, key, {
    get() {
      console.log("获取值");
      return value;
    },
    set(newValue) {
      if (newValue === value) return;
      console.log("设置值");
      value = newValue;
    },
  });
}

export function observe(value) {
  // 如果传过来的是对象或者数组 进行属性劫持
  if (
    Object.prototype.toString.call(value) === "[object Object]" ||
    Array.isArray(value)
  ) {
    return new Observer(value);
  }
}
```

数据劫持核心是 defineReactive 函数 主要使用 Object.defineProperty 来对数据 get 和 set 进行劫持 这里就解决了之前的问题 为啥数据变动了会自动更新视图 我们可以在 set 里面去通知视图更新

## 思考

1.这样的数据劫持方式对数组有什么影响？

这样递归的方式其实无论是对象还是数组都进行了观测 但是我们想一下此时如果 data 包含数组比如 a:[1,2,3,4,5] 那么我们根据下标可以直接修改数组的元素 比如 this.a[0] = 100 这样的操作能否被 Vue 监听到？

答案是否定的 因为 Object.defineProperty 无法监听到数组下标的变化 所以 Vue 对数组的一些方法进行了重写 比如 push pop shift unshift splice sort reverse 这些方法 当我们调用这些方法时 Vue 会监听到并触发视图更新

2.如果 Vue 数据嵌套层级过深 性能会受影响吗？

答案是肯定的 因为递归遍历会消耗大量的性能 所以在开发中我们应该尽量避免数据嵌套层级过深

## 数组的响应式处理

为了解决数组下标的变化无法被监听到的问题 Vue 对数组的一些方法进行了重写

```javascript
// src/observer/array.js
const arrayProto = Array.prototype;

export const arrayMethods = Object.create(arrayProto);

// 要重写的数组方法
const methodsToPatch = [
  'push',
  'pop',
  'shift',
  'unshift',
  'splice',
  'sort',
  'reverse'
];

methodsToPatch.forEach(function (method) {
  // 缓存原始方法
  const original = arrayProto[method];
  // 重写方法
  def(arrayMethods, method, function mutator(...args) {
    // 调用原始方法
    const result = original.apply(this, args);
    // 通知视图更新
    const ob = this.__ob__;
    let inserted;
    switch (method) {
      case 'push':
      case 'unshift':
        inserted = args;
        break;
      case 'splice':
        inserted = args.slice(2);
        break;
    }
    if (inserted) ob.observeArray(inserted);
    // 通知更新
    ob.dep.notify();
    return result;
  });
});

function def(obj, key, val, enumerable) {
  Object.defineProperty(obj, key, {
    value: val,
    enumerable: !!enumerable,
    writable: true,
    configurable: true
  });
}
```

通过重写数组的方法 Vue 可以监听到数组的变化并触发视图更新

## 总结

Vue 的响应式数据原理主要是通过 Object.defineProperty 来对数据进行劫持 当数据发生变化时 会通知视图更新 对于数组 Vue 则通过重写数组的方法来实现响应式

希望通过本文的介绍 大家能够理解 Vue 的响应式数据原理 并在实际开发中灵活运用

