---
title: Vue2.0源码解读-diff算法原理
excerpt: 深入解析Vue2.0 diff算法原理
category: 前端开发
date: 2022-03-30
readTime: 30
tags: JavaScript, 底层系列, Vue2
---

## 前言

此篇主要手写 Vue2.0 源码-diff 算法原理上一篇咱们主要介绍了 Vue 异步更新原理 是对视图更新的性能优化 此篇同样是对渲染更新的优化 当模板发生变化之后 我们可以利用 diff 算法 对比新老虚拟 dom 看是否能进行节点复用 diff 算法也是 vue 面试比较热门的考点哈

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
        a: 123,
      };
    },
    template: `<div id="a">hello {{a}}</div>`,
  });
  setTimeout(() => {
    vm.a = 1;
  }, 1000);
</script>
```

大家思考一下 如果我们当初始渲染完成 1 秒后改变了一下模板里面 a 的值 vue 会怎么处理来显示最新的值呢?

1.把上次渲染的真实 dom 删除 然后重新渲染一个新的 dom 节点来应用最新的 a 的值
2.把老的 dom 进行复用 改变一下内部文本节点的 textContent 的值

这两种方案 很明显后者的性能开销更小 一起来看看 vue 怎么使用 diff 算法来进行渲染更新的吧

## 1.patch 核心渲染方法改写

```javascript
// src/vdom/patch.js
export function patch(oldVnode, vnode) {
  const isRealElement = oldVnode.nodeType;
  if (isRealElement) {
    // oldVnode是真实dom元素 就代表初次渲染
  } else {
    // oldVnode是虚拟dom 就是更新过程 使用diff算法
    if (oldVnode.tag !== vnode.tag) {
      // 如果新旧标签不一致 用新的替换旧的 oldVnode.el代表的是真实dom节点--同级比较
      oldVnode.el.parentNode.replaceChild(createElm(vnode), oldVnode.el);
    }
    // 如果旧节点是一个文本节点
    if (!oldVnode.tag) {
      if (oldVnode.text !== vnode.text) {
        oldVnode.el.textContent = vnode.text;
      }
    }
    // 不符合上面两种 代表标签一致 并且不是文本节点
    // 为了节点复用 所以直接把旧的虚拟dom对应的真实dom赋值给新的虚拟dom的el属性
    const el = (vnode.el = oldVnode.el);
    updateProperties(vnode, oldVnode.data); // 更新属性
    const oldCh = oldVnode.children || []; // 老的儿子
    const newCh = vnode.children || []; // 新的儿子
    if (oldCh.length > 0 && newCh.length > 0) {
      // 新老都存在子节点
      updateChildren(el, oldCh, newCh);
    } else if (oldCh.length) {
      // 老的有儿子新的没有
      el.innerHTML = "";
    } else if (newCh.length) {
      // 新的有儿子
      for (let i = 0; i < newCh.length; i++) {
        const child = newCh[i];
        el.appendChild(createElm(child));
      }
    }
  }
}
```

我们直接看 else 分支 代表的是渲染更新过程 可以分为以下几步

1.diff 只进行同级比较
2.根据新老 vnode 子节点不同情况分别处理

## 2.updateProperties 更新属性

```javascript
//  src/vdom/patch.js
// 解析vnode的data属性 映射到真实dom上
function updateProperties(vnode, oldProps = {}) {
  const newProps = vnode.data || {}; //新的vnode的属性
  const el = vnode.el; // 真实节点
  // 如果新的节点没有 需要把老的节点属性移除
  for (const k in oldProps) {
    if (!newProps[k]) {
      el.removeAttribute(k);
    }
  }
  // 对style样式做特殊处理 如果新的没有 需要把老的style值置为空
  const newStyle = newProps.style || {};
  const oldStyle = oldProps.style || {};
  for (const key in oldStyle) {
    if (!newStyle[key]) {
      el.style[key] = "";
    }
  }
  // 遍历新的属性 进行增加操作
  for (const key in newProps) {
    if (key === "style") {
      for (const styleName in newProps.style) {
        el.style[styleName] = newProps.style[styleName];
      }
    } else if (key === "class") {
      el.className = newProps.class;
    } else {
      // 给这个元素添加属性 值就是对应的值
      el.setAttribute(key, newProps[key]);
    }
  }
}
```

对比新老 vnode 进行属性更新

## 3.updateChildren 更新子节点-diff 核心方法

```javascript
// src/vdom/patch.js
// 判断两个vnode的标签和key是否相同 如果相同 就可以认为是同一节点就地复用
function isSameVnode(oldVnode, newVnode) {
  return oldVnode.tag === newVnode.tag && oldVnode.key === newVnode.key;
}

// diff算法核心 采用双指针的方式 对比新老vnode的儿子节点
function updateChildren(parent, oldCh, newCh) {
  let oldStartIndex = 0; //老儿子的起始下标
  let oldStartVnode = oldCh[0]; //老儿子的第一个节点
  let oldEndIndex = oldCh.length - 1; //老儿子的结束下标
  let oldEndVnode = oldCh[oldEndIndex]; //老儿子的起结束节点
  let newStartIndex = 0; //同上  新儿子的
  let newStartVnode = newCh[0];
  let newEndIndex = newCh.length - 1;
  let newEndVnode = newCh[newEndIndex];

  // 根据key来创建老的儿子的index映射表  类似 {'a':0,'b':1} 代表key为'a'的节点在第一个位置 key为'b'的节点在第二个位置
  function makeIndexByKey(children) {
    let map = {};
    children.forEach((item, index) => {
      map[item.key] = index;
    });
    return map;
  }

  let map = makeIndexByKey(oldCh);

  // 循环比对 直到有一方循环完毕
  while (oldStartIndex <= oldEndIndex && newStartIndex <= newEndIndex) {
    // 老的节点被移动了
    if (!oldStartVnode) {
      oldStartVnode = oldCh[++oldStartIndex];
    } else if (!oldEndVnode) {
      oldEndVnode = oldCh[--oldEndIndex];
    } else if (isSameVnode(oldStartVnode, newStartVnode)) {
      // 头头比对
      patch(oldStartVnode, newStartVnode);
      oldStartVnode = oldCh[++oldStartIndex];
      newStartVnode = newCh[++newStartIndex];
    } else if (isSameVnode(oldEndVnode, newEndVnode)) {
      // 尾尾比对
      patch(oldEndVnode, newEndVnode);
      oldEndVnode = oldCh[--oldEndIndex];
      newEndVnode = newCh[--newEndIndex];
    } else if (isSameVnode(oldStartVnode, newEndVnode)) {
      // 头尾比对
      patch(oldStartVnode, newEndVnode);
      // 把老的头节点移动到老的尾节点后面
      parent.insertBefore(oldStartVnode.el, oldEndVnode.el.nextSibling);
      oldStartVnode = oldCh[++oldStartIndex];
      newEndVnode = newCh[--newEndIndex];
    } else if (isSameVnode(oldEndVnode, newStartVnode)) {
      // 尾头比对
      patch(oldEndVnode, newStartVnode);
      // 把老的尾节点移动到老的头节点前面
      parent.insertBefore(oldEndVnode.el, oldStartVnode.el);
      oldEndVnode = oldCh[--oldEndIndex];
      newStartVnode = newCh[++newStartIndex];
    } else {
      // 四种比对都没有命中
      // 用新的节点的key去老的节点的映射表里面查找
      let moveIndex = map[newStartVnode.key];
      if (!moveIndex) {
        // 如果没有找到 就创建新的节点
        parent.insertBefore(createElm(newStartVnode), oldStartVnode.el);
      } else {
        // 如果找到了 就把老的节点移动到新的位置
        let moveVnode = oldCh[moveIndex];
        oldCh[moveIndex] = undefined; // 把老的节点置为undefined
        parent.insertBefore(moveVnode.el, oldStartVnode.el);
        patch(moveVnode, newStartVnode);
      }
      newStartVnode = newCh[++newStartIndex];
    }
  }

  // 如果新的节点还有剩余 就把剩余的节点插入到老的节点的后面
  while (newStartIndex <= newEndIndex) {
    parent.insertBefore(createElm(newCh[newStartIndex]), oldStartVnode.el);
    newStartIndex++;
  }

  // 如果老的节点还有剩余 就把剩余的节点删除
  while (oldStartIndex <= oldEndIndex) {
    let child = oldCh[oldStartIndex];
    if (child) {
      parent.removeChild(child.el);
    }
    oldStartIndex++;
  }
}
```

## 总结

diff 算法的核心是采用双指针的方式 对比新老 vnode 的儿子节点 进行四种比对

1.头头比对
2.尾尾比对
3.头尾比对
4.尾头比对

如果四种比对都没有命中 就用新的节点的 key 去老的节点的映射表里面查找

如果找到了 就把老的节点移动到新的位置

如果没有找到 就创建新的节点

这样可以最大限度的复用节点 减少 dom 操作 提高性能

希望通过本文的介绍 大家能够理解 Vue 的 diff 算法原理 并在实际开发中灵活运用

