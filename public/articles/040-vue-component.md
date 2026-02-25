---
title: Vue2.0源码解读-组件原理
excerpt: 深入解析Vue2.0组件原理
category: 技术
date: 2026-02-25
readTime: 30
tags: JavaScript, 底层系列, Vue
---

## 前言

此篇主要手写 Vue2.0 源码-组件原理上一篇咱们主要介绍了 Vue Mixin原理 是Vue初始化选项合并核心的api 大家都知道 Vue 的一大特色就是组件化 此篇主要介绍整个组件创建和渲染流程 其中 Vue.extend 这一 api 是创建组件的核心

适用人群：
1.想要深入理解 vue 源码更好的进行日常业务开发
2.想要在简历写上精通 vue 框架源码（再也不怕面试官的连环夺命问 哈哈）
3.没时间去看官方源码或者初看源码觉得难以理解的同学

## 正文

```javascript
<script>
  // 全局组件
  Vue.component("parent-component", {
    template: `<div>我是全局组件</div>`,
  });
  // Vue实例化
  let vm = new Vue({
    el: "#app",
    data() {
      return {
        aa: 1,
      };
    },
    // render(h) {
    //   return h('div',{id:'a'},'hello')
    // },
    template: `<div id="a">
      hello 这是我自己写的Vue{{aa}}
      <parent-component><parent-component>
      <child-component></child-component>
      </div>`,
    // 局部组件
    components: {
      "child-component": {
        template: `<div>我是局部组件</div>`,
      },
    },
  });
</script>
```

上面演示了最基础的全局组件和局部组件的用法 其实我们每一个组件都是一个继承自 Vue 的子类 能够使用 Vue 的原型方法

## 1.全局组件注册

```javascript
// src/global-api/index.js
import initExtend from "./initExtend";
import initAssetRegisters from "./assets";

const ASSETS_TYPE = ["component", "directive", "filter"];

export function initGlobalApi(Vue) {
  Vue.options = {}; // 全局的组件 指令 过滤器
  ASSETS_TYPE.forEach((type) => {
    Vue.options[type + "s"] = {};
  });
  Vue.options._base = Vue; //_base指向Vue
  initExtend(Vue); // extend方法定义
  initAssetRegisters(Vue); //assets注册方法 包含组件 指令和过滤器
}
```

initGlobalApi方法主要用来注册Vue的全局方法 比如之前写的Vue.Mixin 和今天的Vue.extend Vue.component等

```javascript
// src/global-api/asset.js
const ASSETS_TYPE = ["component", "directive", "filter"];

export default function initAssetRegisters(Vue) {
  ASSETS_TYPE.forEach((type) => {
    Vue[type] = function (id, definition) {
      if (type === "component") {
        //   this指向Vue
        // 全局组件注册
        // 子组件可能也有extend方法  VueComponent.component方法
        definition = this.options._base.extend(definition);
      }
      this.options[type + "s"][id] = definition;
    };
  });
}
```

this.options._base 就是指代 Vue 可见所谓的全局组件就是使用 Vue.extend 方法把传入的选项处理之后挂载到了 Vue.options.components 上面

## 2.Vue.extend 定义

```javascript
//  src/global-api/initExtend.js
import { mergeOptions } from "../util/index";

export default function initExtend(Vue) {
  let cid = 0; //组件的唯一标识
  // 创建子类继承Vue父类 便于属性扩展
  Vue.extend = function (extendOptions) {
    // 创建子类的构造函数 并且调用初始化方法
    const Sub = function VueComponent(options) {
      this._init(options); //调用Vue初始化方法
    };
    Sub.cid = cid++;
    Sub.prototype = Object.create(this.prototype); // 子类原型指向父类
    Sub.prototype.constructor = Sub; //constructor指向自己
    Sub.options = mergeOptions(this.options, extendOptions); //合并自己的options和父类的options
    return Sub;
  };
}
```

Vue.extend 核心思路就是使用原型继承的方法返回了 Vue 的子类 并且利用 mergeOptions 把传入组件的 options 和父类的 options 进行了合并

## 3.组件的合并策略

```javascript
// src/init.js
Vue.prototype._init = function (options) {
  const vm = this;
  vm.$options = mergeOptions(vm.constructor.options, options); //合并options
};
```

还记得我们 Vue 初始化的时候合并 options 吗 全局组件挂载在 Vue.options.components 上 局部组件也定义在自己的 options.components 上面 那我们怎么处理全局组件和局部组件的合并呢

```javascript
// src/util/index.js
const ASSETS_TYPE = ["component", "directive", "filter"];

// 组件 指令 过滤器的合并策略
function mergeAssets(parentVal, childVal) {
  const res = Object.create(parentVal); //比如有同名的全局组件和自己定义的局部组件 那么parentVal代表全局组件 自己定义的组件是childVal  首先会查找自已局部组件有就用自己的  没有就从原型继承全局组件  res.__proto__===parentVal
  if (childVal) {
    for (let k in childVal) {
      res[k] = childVal[k];
    }
  }
  return res;
}

// 定义组件的合并策略
ASSETS_TYPE.forEach((type) => {
  strats[type + "s"] = mergeAssets;
});
```

这里又使用到了原型继承的方式来进行组件合并 组件内部优先查找自己局部定义的组件 找不到会向上查找原型中定义的组件

## 4.创建组件 Vnode

```javascript
// src/util/index.jsexport function isObject(data) {
  //判断是否是对象
  if (typeof data !== "object" || data == null) {
    return false;
  }
  return true;
}

export function isReservedTag(tagName) {
  //判断是不是常规html标签
  const reservedTags = 'a,div,span,p,img,input,button,ul,ol,li,table,tr,td,th,form,select,option,textarea,label,h1,h2,h3,h4,h5,h6,header,footer,nav,section,article,aside,main,canvas,video,audio,iframe,embed,object,param,script,style,link,meta,title,base,br,hr,col,colgroup,tbody,thead,tfoot,tr,td,th,caption,fieldset,legend,dl,dt,dd,address,blockquote,cite,code,pre,samp,kbd,var,del,ins,sup,sub,q,abbr,acronym,dfn,time,mark,ruby,rt,rp,bdi,bdo,small,big,strong,em,i,b,u,s,strike,samp,kbd,var,code,pre,tt,listing,xmp,plaintext'.split(',');
  return reservedTags.includes(tagName);
}
```

## 总结

通过以上步骤 我们实现了 Vue 的组件原理 主要是利用 Vue.extend 方法创建组件的构造函数 并且利用 mergeOptions 把传入组件的 options 和父类的 options 进行了合并

组件的合并策略是使用原型继承的方式来进行组件合并 组件内部优先查找自己局部定义的组件 找不到会向上查找原型中定义的组件

这样可以实现组件的复用 提高开发效率

希望通过本文的介绍 大家能够理解 Vue 的组件原理 并在实际开发中灵活运用

如果你还有其他疑问 欢迎在评论区留言 我会尽量为大家解答