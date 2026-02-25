---
title: Vue2.0源码解读-模板编译原理
excerpt: 深入解析Vue2.0模板编译原理
category: 技术
date: 2026-02-25
readTime: 30
tags: JavaScript, 底层系列, Vue
---

## 前言

此篇主要手写 Vue2.0 源码-模板编译原理上一篇咱们主要介绍了 Vue 数据的响应式原理 对于中高级前端来说 响应式原理基本是面试 Vue 必考的源码基础类 如果不是很清楚的话基本就被 pass 了 那么今天咱们手写的模板编译原理也是 Vue 面试比较频繁的一个点 而且复杂程度是高于响应式原理的 里面主要涉及到 ast 以及大量正则匹配 大家学习完可以看着思维导图一起手写一遍加深印象哈

## 正文

```javascript
// Vue实例化
new Vue({
  el: "#app",
  data() {
    return {
      a: 111,
    };
  },
  // render(h) {
  //   return h('div',{id:'a'},'hello')
  // },
  // template:`<div id="a">hello</div>`
});
```

上面这段代码 大家一定不陌生 按照官网给出的生命周期图 咱们传入的 options 选项里面可以手动配置 template 或者是 render

注意一：平常开发中 我们使用的是不带编译版本的 Vue 版本（runtime-only）直接在 options 传入 template 选项 在开发环境报错

注意二：这里传入的 template 选项不要和.vue 文件里面的<template>模板搞混淆了 vue 单文件组件的 template 是需要 vue-loader 进行处理的

我们传入的 el 或者 template 选项最后都会被解析成 render 函数 这样才能保持模板解析的一致性

## 模板编译入口

```javascript
// src/init.js
import { initState } from "./state";
import { compileToFunctions } from "./compiler/index";

export function initMixin(Vue) {
  Vue.prototype._init = function (options) {
    const vm = this;
    // 这里的this代表调用_init方法的对象(实例对象)
    //  this.$options就是用户new Vue的时候传入的属性
    vm.$options = options;
    // 初始化状态
    initState(vm);
    // 如果有el属性 进行模板渲染
    if (vm.$options.el) {
      vm.$mount(vm.$options.el);
    }
  };
  // 这块代码在源码里面的位置其实是放在entry-runtime-with-compiler.js里面
  // 代表的是Vue源码里面包含了compile编译功能 这个和runtime-only版本需要区分开
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
  };
}
```

咱们主要关心$mount 方法 最终将处理好的 template 模板转成 render 函数

## 模板转化核心方法 compileToFunctions

```javascript
// src/compiler/index.js
import { parse } from "./parse";
import { generate } from "./codegen";

export function compileToFunctions(template) {
  // 我们需要把html字符串变成render函数
  // 1.把html代码转成ast语法树  ast用来描述代码本身形成树结构 不仅可以描述html 也能描述css以及js语法
  // 很多库都运用到了ast 比如 webpack babel eslint等等
  let ast = parse(template);
  // 2.优化静态节点
  // 这个有兴趣的可以去看源码  不影响核心功能就不实现了
  //   if (options.optimize !== false) {
  //     optimize(ast, options);
  //   }
  // 3.通过ast 重新生成代码
  // 我们最后生成的代码需要和render函数一样
  // 类似_c('div',{id:"app"},_c('div',undefined,_v("hello"+_s(name)),_c('span',undefined,_v("world"))))
  // _c代表创建元素 _v代表创建文本 _s代表文Json.stringify--把对象解析成文本
  let code = generate(ast);
  //   使用with语法改变作用域为this  之后调用render函数可以使用call改变this 方便code里面的变量取值
  let renderFn = new Function(`with(this){return ${code}}`);
  return renderFn;
}
```

新建 compiler 文件夹 表示编译相关功能 核心导出 compileToFunctions 函数 主要有三个步骤 1.生成 ast 2.优化静态节点 3.根据 ast 生成 render 函数

## 解析 html 并生成 ast

```javascript
// src/compiler/parse.js
// 以下为源码的正则  对正则表达式不清楚的同学可以参考小编之前写的文章(前端进阶高薪必看 - 正则篇);
const ncname = `[a-zA-Z_][-.0-9_a-zA-Z]*`; //匹配标签名 形如 abc-123
const qnameCapture = `((?:${ncname}:)?${ncname})`; //匹配特殊标签 形如 abc:234 前面的abc:可有可无
const startTagOpen = new RegExp(`^<${qnameCapture}`); // 匹配标签开始 形如 <abc-123 捕获里面的标签名
const startTagClose = /^s*(/?)>/; // 匹配标签结束  >
const endTag = new RegExp(`^</${qnameCapture}[^>]*>`); // 匹配标签结尾 如 </abc-123> 捕获里面的标签名
const attribute = /^s*([^s="']+?)s*(?:=s*(?:(["'])(.*?)\2|([^s="']+)))?/; // 匹配属性 形如 id="app" 或者 id=app
const defaultTagRE = /{{((?:.|n)+?)}}/g; // 匹配{{}} 形如 {{name}}

let root = null;
let currentParent = null;
let stack = [];

function createASTElement(tag, attrs) {
  return {
    tag: tag,
    type: 1,
    attrs: attrs,
    children: [],
    parent: null,
  };
}

function start(tag, attrs) {
  let element = createASTElement(tag, attrs);
  if (!root) {
    root = element;
  }
  currentParent = element;
  stack.push(element);
}

function end(tag) {
  let element = stack.pop();
  currentParent = stack[stack.length - 1];
  if (currentParent) {
    element.parent = currentParent;
    currentParent.children.push(element);
  }
}

function chars(text) {
  text = text.trim();
  if (text) {
    currentParent.children.push({
      type: 3,
      text: text,
    });
  }
}

export function parse(html) {
  while (html) {
    let textEnd = html.indexOf('<');
    if (textEnd === 0) {
      // 解析开始标签
      const startTagMatch = parseStartTag();
      if (startTagMatch) {
        start(startTagMatch.tagName, startTagMatch.attrs);
        continue;
      }
      // 解析结束标签
      const endTagMatch = html.match(endTag);
      if (endTagMatch) {
        advance(endTagMatch[0].length);
        end(endTagMatch[1]);
        continue;
      }
    }
    // 解析文本
    let text = html.substring(0, textEnd);
    if (text) {
      advance(text.length);
      chars(text);
    }
  }
  return root;
}

function parseStartTag() {
  const start = html.match(startTagOpen);
  if (start) {
    const match = {
      tagName: start[1],
      attrs: [],
    };
    advance(start[0].length);
    let end, attr;
    while (!(end = html.match(startTagClose)) && (attr = html.match(attribute))) {
      advance(attr[0].length);
      match.attrs.push({
        name: attr[1],
        value: attr[3] || attr[4] || '',
      });
    }
    if (end) {
      advance(end[0].length);
      return match;
    }
  }
}

function advance(n) {
  html = html.substring(n);
}
```

## 生成 render 函数

```javascript
// src/compiler/codegen.js
function gen(node) {
  if (node.type === 1) {
    return genElement(node);
  } else {
    return genText(node);
  }
}

function genElement(node) {
  let children = genChildren(node.children);
  let code = `_c('${node.tag}',${genProps(node.attrs)}${children ? `,${children}` : ''})`;
  return code;
}

function genProps(attrs) {
  let str = '';
  for (let i = 0; i < attrs.length; i++) {
    let attr = attrs[i];
    if (attr.name === 'style') {
      let obj = {};
      attr.value.split(';').forEach((item) => {
        let [key, value] = item.split(':');
        obj[key] = value;
      });
      attr.value = obj;
    }
    str += `${attr.name}:${JSON.stringify(attr.value)},`;
  }
  return `{${str.slice(0, -1)}}`;
}

function genChildren(children) {
  return children.map((child) => gen(child)).join(',');
}

function genText(node) {
  return `_v(${JSON.stringify(node.text)})`;
}

export function generate(ast) {
  let code = gen(ast);
  return code;
}
```

## 总结

Vue 的模板编译原理主要是将 html 模板解析成 ast 语法树 然后根据 ast 生成 render 函数 这样 Vue 就可以通过 render 函数生成虚拟 dom 并最终渲染成真实 dom

希望通过本文的介绍 大家能够理解 Vue 的模板编译原理 并在实际开发中灵活运用

如果你还有其他疑问 欢迎在评论区留言 我会尽量为大家解答