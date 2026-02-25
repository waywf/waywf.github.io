---
title: Vue2.0源码解读-初始渲染原理
excerpt: 深入解析Vue2.0初始渲染原理
category: 技术
date: 2026-02-25
readTime: 25
tags: JavaScript, 底层系列, Vue
---

## 前言

此篇主要手写 Vue2.0 源码-初始渲染原理上一篇咱们主要介绍了 Vue 模板编译原理 它是 Vue 生成虚拟 dom 的基础 模板编译最后转化成了 render 函数 之后又如何能生成真实的 dom 节点去替换掉 el 选项配置呢 那么通过此篇的学习就可以知道 Vue 初始渲染的流程 此篇主要包含虚拟 dom 以及真实 dom 的生成

## 组件挂载入口

```javascript
// src/init.js
Vue.prototype.$mount = function (el) {
  const vm = this;
  const options = vm.$options;
  el = document.querySelector(el);
  // 如果不存在render属性
  if (!options.render) {
    // 如果存在template属性
    let template = options.template;
    if (!template && el) {
      // 如果不存在render和template 但是存在el属性 直接将模板赋值到el所在的外层html结构（就是el本身 并不是父元素）
      template = el.outerHTML;
    }
    // 最终需要把tempalte模板转化成render函数
    if (template) {
      const render = compileToFunctions(template);
      options.render = render;
    }
  }
  // 将当前组件实例挂载到真实的el节点上面
  return mountComponent(vm, el);
};
```

接着看$mount 方法 我们主要关注最后一句话 mountComponent 就是组件实例挂载的入口函数

这个方法放在源码的 lifecycle 文件里面 代表了与生命周期相关 因为我们组件初始渲染前后对应有 beforeMount 和 mounted 生命周期钩子

## 组件挂载核心方法 mountComponent

```javascript
// src/lifecycle.js
export function mountComponent(vm, el) {
  // 上一步模板编译解析生成了render函数
  // 下一步就是执行vm._render()方法 调用生成的render函数 生成虚拟dom
  // 最后使用vm._update()方法把虚拟dom渲染到页面
  // 真实的el选项赋值给实例的$el属性 为之后虚拟dom产生的新的dom替换老的dom做铺垫
  vm.$el = el;
  //   _update和._render方法都是挂载在Vue原型的方法  类似_init
  vm._update(vm._render());
}
```

新建 lifecycle.js 文件 表示生命周期相关功能 核心导出 mountComponent 函数 主要使用 vm._update(vm._render())方法进行实例挂载

## render 函数转化成虚拟 dom 核心方法 _render

```javascript
// src/render.js
import { createElement, createTextNode } from "./vdom/index";

export function renderMixin(Vue) {
  Vue.prototype._render = function () {
    const vm = this;
    // 获取模板编译生成的render方法
    const { render } = vm.$options;
    // 生成vnode--虚拟dom
    const vnode = render.call(vm);
    return vnode;
  };
  // render函数里面有_c _v _s方法需要定义
  Vue.prototype._c = function (...args) {
    // 创建虚拟dom元素
    return createElement(...args);
  };
  Vue.prototype._v = function (text) {
    // 创建虚拟dom文本
    return createTextNode(text);
  };
  Vue.prototype._s = function (val) {
    // 如果模板里面的是一个对象  需要JSON.stringify
    return val == null ? "" : typeof val === "object" ? JSON.stringify(val) : val;
  };
}
```

主要在原型定义了_render 方法 然后执行了 render 函数 我们知道模板编译出来的 render 函数核心代码主要 return 了 类似于_c('div',{id:"app"},_c('div',undefined,_v("hello"+_s(name)),_c('span',undefined,_v("world"))))这样的代码 那么我们还需要定义一下_c _v _s 这些函数才能最终转化成为虚拟 dom

```javascript
// src/vdom/index.js
// 定义Vnode类
export default class Vnode {
  constructor(tag, data, key, children, text) {
    this.tag = tag;
    this.data = data;
    this.key = key;
    this.children = children;
    this.text = text;
  }
}

// 创建元素vnode 等于render函数里面的 h=>h(App)
export function createElement(tag, data = {}, ...children) {
  let key = data.key;
  return new Vnode(tag, data, key, children);
}

// 创建文本vnode
export function createTextNode(text) {
  return new Vnode(undefined, undefined, undefined, undefined, text);
}
```

新建 vdom 文件夹 代表虚拟 dom 相关功能 定义 Vnode 类 以及 createElement 和 createTextNode 方法最后都返回 vnode

## 虚拟 dom 转化成真实 dom 核心方法 _update

```javascript
// src/lifecycle.js
import { patch } from "./vdom/patch";

export function lifecycleMixin(Vue) {
  // 把_update挂载在Vue的原型
  Vue.prototype._update = function (vnode) {
    const vm = this;
    // patch是渲染vnode为真实dom核心
    patch(vm.$el, vnode);
  };
}
```

```javascript
// src/vdom/patch.js
// patch用来渲染和更新视图 今天只介绍初次渲染的逻辑
export function patch(oldVnode, vnode) {
  // 判断传入的oldVnode是否是一个真实元素
  // 这里很关键  初次渲染 传入的vm.$el就是咱们传入的el选项  所以是真实dom
  // 如果不是初始渲染而是视图更新的时候  vm.$el就被替换成了更新之前的老的虚拟dom
  const isRealElement = oldVnode.nodeType;
  if (isRealElement) {
    // 这里是初次渲染的逻辑
    const oldElm = oldVnode;
    const parentElm = oldElm.parentNode;
    // 调用createElm方法将虚拟dom转化成真实dom
    const elm = createElm(vnode);
    // 插入到父元素中
    parentElm.insertBefore(elm, oldElm.nextSibling);
    // 移除老的dom元素
    parentElm.removeChild(oldElm);
    // 返回新的dom元素
    return elm;
  }
}

function createElm(vnode) {
  let { tag, data, children, text } = vnode;
  if (typeof tag === 'string') {
    // 创建元素
    vnode.el = document.createElement(tag);
    // 处理属性
    updateProperties(vnode.el, data);
    // 处理子元素
    children.forEach((child) => {
      vnode.el.appendChild(createElm(child));
    });
  } else {
    // 创建文本节点
    vnode.el = document.createTextNode(text);
  }
  return vnode.el;
}

function updateProperties(el, props) {
  for (let key in props) {
    if (key === 'style') {
      // 处理样式
      for (let styleName in props.style) {
        el.style[styleName] = props.style[styleName];
      }
    } else if (key === 'class') {
      // 处理类名
      el.className = props.class;
    } else {
      // 处理其他属性
      el.setAttribute(key, props[key]);
    }
  }
}
```

## 总结

Vue 的初始渲染原理主要是将模板编译成 render 函数 然后执行 render 函数生成虚拟 dom 最后将虚拟 dom 渲染成真实 dom

希望通过本文的介绍 大家能够理解 Vue 的初始渲染原理 并在实际开发中灵活运用

如果你还有其他疑问 欢迎在评论区留言 我会尽量为大家解答