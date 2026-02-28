---
title: Vue2.0源码解读-计算属性原理
excerpt: 深入解析Vue2.0计算属性原理
category: 前端开发
date: 2022-07-18
readTime: 25
tags: JavaScript, 底层系列, Vue2
---

## 前言

此篇主要手写 Vue2.0 源码-计算属性上一篇咱们主要介绍了 Vue 侦听属性原理 知道了用户定义的 watch 如何被创建 此篇我们介绍他的兄弟-计算属性 主要特性是如果计算属性依赖的值不发生变化 页面更新的时候不会重新计算 计算结果会被缓存 可以用此 api 来优化性能

适用人群：
1.想要深入理解 vue 源码更好的进行日常业务开发
2.想要在简历写上精通 vue 框架源码（再也不怕面试官的连环夺命问 哈哈）
3.没时间去看官方源码或者初看源码觉得难以理解的同学

## 正文

```javascript
<script>
  // Vue实例化
  let vm = new Vue({
    el: "#app",
    data() {
      return {
        aa: 1,
        bb: 2,
        cc: 3,
      };
    },
    // render(h) {
    //   return h('div',{id:'a'},'hello')
    // },
    template: `<div id="a">hello 这是我自己写的Vue{{computedName}}{{cc}}</div>`,
    computed: {
      computedName() {
        return this.aa + this.bb;
      },
    },
  });
  // 当我们每一次改变数据的时候  渲染watcher都会执行一次 这个是影响性能的
  setTimeout(() => {
    vm.cc = 4;
  }, 2000);
  console.log(vm);
</script>
```

上述例子 就是计算属性的基础用法 我们在两秒之后改变了模板里面的 cc 但是计算属性依赖的 aa 和 bb 都没变化 所以计算属性不会重新计算 还是保留的上次计算结果

## 1.计算属性的初始化

```javascript
// src/state.js
function initComputed(vm) {
  const computed = vm.$options.computed;
  const watchers = (vm._computedWatchers = {}); //用来存放计算watcher
  for (let k in computed) {
    const userDef = computed[k]; //获取用户定义的计算属性
    const getter = typeof userDef === "function" ? userDef : userDef.get; //创建计算属性watcher使用
    // 创建计算watcher  lazy设置为true
    watchers[k] = new Watcher(vm, getter, () => {}, { lazy: true });
    defineComputed(vm, k, userDef);
  }
}
```

计算属性可以写成一个函数也可以写成一个对象 对象的形式 get 属性就代表的是计算属性依赖的值 set 代表修改计算属性的依赖项的值 我们主要关心 get 属性 然后类似侦听属性 我们把 lazy:true 传给构造函数 Watcher 用来创建计算属性 Watcher 那么 defineComputed 是什么意思呢

思考？ 计算属性是可以缓存计算结果的 我们应该怎么做？

## 2.对计算属性进行属性劫持

```javascript
//  src/state.js
// 定义普通对象用来劫持计算属性
const sharedPropertyDefinition = {
  enumerable: true,
  configurable: true,
  get: () => {},
  set: () => {},
};

// 重新定义计算属性  对get和set劫持
function defineComputed(target, key, userDef) {
  if (typeof userDef === "function") {
    // 如果是一个函数  需要手动赋值到get上
    sharedPropertyDefinition.get = createComputedGetter(key);
  } else {
    sharedPropertyDefinition.get = createComputedGetter(key);
    sharedPropertyDefinition.set = userDef.set;
  }
  //   利用Object.defineProperty来对计算属性的get和set进行劫持
  Object.defineProperty(target, key, sharedPropertyDefinition);
}

// 重写计算属性的get方法 来判断是否需要进行重新计算
function createComputedGetter(key) {
  return function () {
    const watcher = this._computedWatchers[key]; //获取对应的计算属性watcher
    if (watcher) {
      if (watcher.dirty) {
        watcher.evaluate(); //计算属性取值的时候 如果是脏的  需要重新求值
      }
      return watcher.value;
    }
  };
}
```

defineComputed 方法主要是重新定义计算属性 其实最主要的是劫持 get 方法 也就是计算属性依赖的值 为啥要劫持呢 因为我们需要根据依赖值是否发生变化来判断计算属性是否需要重新计算

createComputedGetter 方法就是判断计算属性依赖的值是否变化的核心了 我们在计算属性创建的 Watcher 增加 dirty 标志位 如果标志变为 true 代表需要调用 watcher.evaluate 来进行重新计算了

## 3.Watcher 改造

```javascript
// src/observer/watcher.js
// import { pushTarget, popTarget } from "./dep";
// import { queueWatcher } from "./scheduler";
// import {isObject} from '../util/index'
// // 全局变量id  每次new Watcher都会自增
// let id = 0;

export default class Watcher {
  constructor(vm, exprOrFn, cb, options) {
    // this.vm = vm;
    // this.exprOrFn = exprOrFn;
    // this.cb = cb; //回调函数 比如在watcher更新之前可以执行beforeUpdate方法
    // this.options = options; //额外的选项 true代表渲染watcher
    // this.id = id++; // watcher的唯一标识
    // this.deps = []; //存放dep的容器
    // this.depsId = new Set(); //用来去重dep
    // this.user = options.user; //标识用户watcher
    this.lazy = options.lazy; //标识计算属性watcher
    this.dirty = this.lazy; //dirty可变  表示计算watcher是否需要重新计算 默认值是true
    // 如果表达式是一个函数
    // if (typeof exprOrFn === "function") {
    //   this.getter = exprOrFn;
    // } else {
    //   this.getter = function () {
    //     //用户watcher传过来的可能是一个字符串
```

## 总结

通过以上步骤 我们实现了 Vue 的计算属性原理 主要是利用 initComputed 方法初始化计算属性 然后利用 defineComputed 方法对计算属性进行属性劫持 最后利用 Watcher 改造实现计算属性的缓存

计算属性的特点是如果计算属性依赖的值不发生变化 页面更新的时候不会重新计算 计算结果会被缓存 可以用此 api 来优化性能

希望通过本文的介绍 大家能够理解 Vue 的计算属性原理 并在实际开发中灵活运用

