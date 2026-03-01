---
title: 01.前端高频面试题深度解析：从"八股文"到真功夫的修炼之路
category: 前端开发
excerpt: 深入剖析2024-2025年前端高频面试题，从JavaScript核心原理到Vue/React框架源码，从浏览器原理到性能优化。不只是背答案，而是真正理解背后的设计哲学和工程实践。
tags: 前端面试, JavaScript, Vue, React, 浏览器原理, 性能优化, 源码解析
date: 2025-06-01
readTime: 45
---

> 面试官："说说闭包吧。"
> 
> 你："闭包就是...函数内部返回函数...可以访问外部变量..."
> 
> 面试官："然后呢？"
> 
> 你："..."
> 
> 这样的场景是否似曾相识？本文不是面试题的简单罗列，而是带你深入每个问题背后的设计原理、工程实践和源码实现。让我们一起，把"八股文"练成真功夫。

## 一、JavaScript核心原理篇

### 1.1 闭包：不只是"函数套函数"

**面试题：什么是闭包？闭包有什么作用？有什么缺点？**

#### 从一道经典面试题开始

```javascript
for (var i = 0; i < 5; i++) {
  setTimeout(() => {
    console.log(i);
  }, 1000);
}
// 输出什么？
```

答案是：5个5。为什么？

#### 深入理解：作用域链与词法环境

闭包的本质是**词法作用域的延续**。当函数被创建时，它会记住自己被定义时的环境，而不是执行时的环境。

```javascript
function createCounter() {
  let count = 0;  // 这个变量被"闭"在函数内部
  
  return {
    increment: function() {
      count++;
      return count;
    },
    decrement: function() {
      count--;
      return count;
    },
    getCount: function() {
      return count;
    }
  };
}

const counter = createCounter();
console.log(counter.increment()); // 1
console.log(counter.increment()); // 2
console.log(counter.getCount());  // 2
```

**为什么count没有被垃圾回收？**

因为返回的对象中的方法仍然引用着`count`变量，形成了一个闭包。JavaScript引擎看到还有引用，就不会回收这块内存。

#### 闭包的实际应用场景

**1. 数据私有化（模拟私有变量）**

```javascript
const User = (function() {
  // 真正的私有变量
  const users = new Map();
  let idCounter = 0;
  
  return class User {
    constructor(name) {
      this._id = ++idCounter;
      users.set(this._id, { name, createdAt: new Date() });
    }
    
    getInfo() {
      return users.get(this._id);
    }
    
    static getUserCount() {
      return users.size;
    }
  };
})();

const user1 = new User('张三');
console.log(user1.getInfo());        // { name: '张三', createdAt: ... }
console.log(User.getUserCount());    // 1
console.log(user1.users);            // undefined，无法直接访问
```

**2. 函数柯里化与偏函数应用**

```javascript
// 柯里化：将多参数函数转为单参数链
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    } else {
      return function(...args2) {
        return curried.apply(this, args.concat(args2));
      };
    }
  };
}

const add = curry((a, b, c) => a + b + c);
console.log(add(1)(2)(3));  // 6
console.log(add(1, 2)(3));  // 6
console.log(add(1)(2, 3));  // 6
```

**3. 防抖与节流**

```javascript
// 防抖：事件停止触发后才执行
function debounce(fn, delay) {
  let timer = null;
  
  return function(...args) {
    if (timer) clearTimeout(timer);
    
    timer = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

// 节流：固定时间内只执行一次
function throttle(fn, limit) {
  let inThrottle = false;
  
  return function(...args) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}
```

#### 闭包的缺点与解决方案

**1. 内存泄漏风险**

```javascript
// 不好的做法：无意中持有大量数据
function processBigData() {
  const hugeArray = new Array(1000000).fill('x');
  
  return {
    getLength: function() {
      // 只返回长度，但闭包持有了整个数组
      return hugeArray.length;
    }
  };
}

// 好的做法：只保留需要的数据
function processBigDataBetter() {
  const hugeArray = new Array(1000000).fill('x');
  const length = hugeArray.length;  // 只保存需要的数据
  
  // 手动解除引用（可选，通常不需要）
  hugeArray.length = 0;
  
  return {
    getLength: function() {
      return length;  // 只闭包了length，不是整个数组
    }
  };
}
```

**2. 性能影响**

闭包会导致变量无法被优化，因为引擎不知道这些变量什么时候会被使用。在性能敏感的代码中要注意。

### 1.2 this指向：JavaScript的"玄学"

**面试题：this的指向规则是什么？箭头函数的this有什么不同？**

#### this的四种绑定规则

```javascript
// 1. 默认绑定（严格模式下是undefined，非严格是global）
function foo() {
  console.log(this);
}
foo(); // window（浏览器）或 global（Node.js）

// 2. 隐式绑定：谁调用，this就指向谁
const obj = {
  name: 'obj',
  foo: function() {
    console.log(this.name);
  }
};
obj.foo(); // 'obj'

// 3. 显式绑定：call、apply、bind
function bar() {
  console.log(this.name);
}
const obj2 = { name: 'obj2' };
bar.call(obj2);  // 'obj2'
bar.apply(obj2); // 'obj2'
const boundBar = bar.bind(obj2);
boundBar();      // 'obj2'

// 4. new绑定
function Person(name) {
  this.name = name;
}
const person = new Person('张三');
console.log(person.name); // '张三'
```

#### 箭头函数：没有自己的this

```javascript
const obj = {
  name: 'obj',
  regularFunc: function() {
    console.log('regular:', this.name);
    
    setTimeout(function() {
      console.log('callback:', this.name); // undefined！
    }, 100);
    
    setTimeout(() => {
      console.log('arrow:', this.name); // 'obj' ✓
    }, 100);
  },
  arrowFunc: () => {
    console.log('arrowFunc:', this.name); // 继承外层this
  }
};

obj.regularFunc();
```

**箭头函数的原理**：箭头函数没有自己的`this`，它会捕获定义时的`this`值。这通过词法作用域实现，而不是动态绑定。

#### 一道让人崩溃的面试题

```javascript
const obj = {
  name: 'obj',
  foo: function() {
    console.log(this.name);
  }
};

const foo = obj.foo;
foo(); // 输出什么？
```

答案是：`undefined`（严格模式）或空字符串/全局对象的name（非严格模式）。

**为什么？** 因为`foo()`是独立调用，不是`obj.foo()`。this的绑定只跟调用方式有关，跟定义位置无关。

### 1.3 原型与继承：JavaScript的"血统"

**面试题：JavaScript的继承方式有哪些？原型链是什么？**

#### 原型链的本质

```javascript
function Person(name) {
  this.name = name;
}

Person.prototype.sayHello = function() {
  console.log(`Hello, I'm ${this.name}`);
};

const person = new Person('张三');
person.sayHello(); // Hello, I'm 张三

// 原型链
console.log(person.__proto__ === Person.prototype); // true
console.log(Person.prototype.__proto__ === Object.prototype); // true
console.log(Object.prototype.__proto__); // null
```

**原型链查找过程**：
1. 先在对象自身属性查找
2. 没找到，去`__proto__`（即构造函数的prototype）查找
3. 还没找到，继续沿`__proto__`向上查找
4. 直到`Object.prototype.__proto__`（null），返回undefined

#### 继承的演变史

**1. 原型链继承（不推荐）**

```javascript
function Parent() {
  this.colors = ['red', 'blue'];
}

function Child() {}
Child.prototype = new Parent();

const child1 = new Child();
child1.colors.push('green');

const child2 = new Child();
console.log(child2.colors); // ['red', 'blue', 'green']  // 共享了引用类型！
```

**问题**：所有实例共享父类的引用类型属性。

**2. 借用构造函数（不推荐）**

```javascript
function Parent(name) {
  this.name = name;
  this.colors = ['red', 'blue'];
}

function Child(name, age) {
  Parent.call(this, name);  // 借用父类构造函数
  this.age = age;
}

const child1 = new Child('张三', 18);
const child2 = new Child('李四', 20);

child1.colors.push('green');
console.log(child2.colors); // ['red', 'blue']  // 不共享，✓
```

**问题**：方法都在构造函数中定义，无法复用；父类原型上的方法无法继承。

**3. 组合继承（经典方式）**

```javascript
function Parent(name) {
  this.name = name;
  this.colors = ['red', 'blue'];
}

Parent.prototype.sayName = function() {
  console.log(this.name);
};

function Child(name, age) {
  Parent.call(this, name);  // 第二次调用Parent
  this.age = age;
}

Child.prototype = new Parent();  // 第一次调用Parent
Child.prototype.constructor = Child;

Child.prototype.sayAge = function() {
  console.log(this.age);
};
```

**问题**：父类构造函数被调用了两次。

**4. 寄生组合继承（最佳实践）**

```javascript
function inheritPrototype(child, parent) {
  const prototype = Object.create(parent.prototype);
  prototype.constructor = child;
  child.prototype = prototype;
}

function Parent(name) {
  this.name = name;
  this.colors = ['red', 'blue'];
}

Parent.prototype.sayName = function() {
  console.log(this.name);
};

function Child(name, age) {
  Parent.call(this, name);
  this.age = age;
}

inheritPrototype(Child, Parent);

Child.prototype.sayAge = function() {
  console.log(this.age);
};
```

**ES6 Class语法糖**：

```javascript
class Parent {
  constructor(name) {
    this.name = name;
    this.colors = ['red', 'blue'];
  }
  
  sayName() {
    console.log(this.name);
  }
}

class Child extends Parent {
  constructor(name, age) {
    super(name);
    this.age = age;
  }
  
  sayAge() {
    console.log(this.age);
  }
}
```

**注意**：ES6的class本质上还是原型继承的语法糖，`typeof Child`仍然是`"function"`。

### 1.4 事件循环：异步的"指挥家"

**面试题：说说JavaScript的事件循环机制？宏任务和微任务的区别？**

#### 为什么需要事件循环？

JavaScript是单线程的（为了操作DOM的简洁性），但又要处理异步操作（网络请求、定时器等）。事件循环就是协调同步代码和异步代码的机制。

#### 事件循环的执行顺序

```javascript
console.log('1');

setTimeout(() => {
  console.log('2');
}, 0);

Promise.resolve().then(() => {
  console.log('3');
});

console.log('4');

// 输出：1 4 3 2
```

**为什么是这个顺序？**

```
1. 执行同步代码：console.log('1')、console.log('4')
2. 同步代码执行完毕，检查微任务队列：Promise.then → 输出3
3. 微任务队列清空，检查宏任务队列：setTimeout → 输出2
```

#### 宏任务 vs 微任务

| 类型 | 包含 | 优先级 |
|------|------|--------|
| **宏任务** | script整体代码、setTimeout、setInterval、I/O、UI渲染 | 低 |
| **微任务** | Promise.then/catch/finally、MutationObserver、queueMicrotask | 高 |

**关键规则**：
- 每次事件循环，先执行一个宏任务
- 然后执行所有微任务
- 微任务清空后，才执行下一个宏任务

#### 复杂的例子

```javascript
async function async1() {
  console.log('async1 start');
  await async2();
  console.log('async1 end');
}

async function async2() {
  console.log('async2');
}

console.log('script start');

setTimeout(() => {
  console.log('setTimeout');
}, 0);

async1();

new Promise((resolve) => {
  console.log('promise1');
  resolve();
}).then(() => {
  console.log('promise2');
});

console.log('script end');

// 输出：
// script start
// async1 start
// async2
// promise1
// script end
// async1 end
// promise2
// setTimeout
```

**解析**：
1. `await async2()`会立即执行`async2()`，但后面的代码会被放入微任务队列
2. `Promise.then`也是微任务
3. 微任务按顺序执行：`async1 end`先于`promise2`，因为`await`先进入队列

#### Node.js的事件循环

Node.js的事件循环更复杂，有6个阶段：

```
   ┌───────────────────────────┐
┌─>│           timers          │  setTimeout/setInterval
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
│  │     pending callbacks     │  系统回调
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
│  │       idle, prepare       │  内部使用
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
│  │           poll            │  获取新的I/O事件
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
│  │           check           │  setImmediate
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
└──┤      close callbacks      │  socket.on('close', ...)
   └───────────────────────────┘
```

**关键区别**：
- 浏览器：微任务在每个宏任务后执行
- Node.js：微任务在每个阶段后执行

## 二、Vue原理篇

### 2.1 响应式系统：Vue的"魔法"

**面试题：Vue2和Vue3的响应式原理有什么区别？为什么Vue3要用Proxy？**

#### Vue2的响应式：Object.defineProperty

```javascript
// 简化版实现
function defineReactive(obj, key, val) {
  const dep = new Dep();  // 依赖收集器
  
  Object.defineProperty(obj, key, {
    get() {
      if (Dep.target) {
        dep.depend();  // 收集依赖
      }
      return val;
    },
    set(newVal) {
      if (newVal !== val) {
        val = newVal;
        dep.notify();  // 通知更新
      }
    }
  });
}

// 依赖收集器
class Dep {
  constructor() {
    this.subs = [];
  }
  
  depend() {
    if (Dep.target) {
      this.subs.push(Dep.target);
    }
  }
  
  notify() {
    this.subs.forEach(watcher => watcher.update());
  }
}
```

**Vue2的问题**：
1. **无法检测新增/删除属性**：`this.obj.newProp = 'xxx'`不会触发更新
2. **数组问题**：通过索引修改数组`arr[0] = 1`不会触发更新
3. **深度递归**：初始化时需要递归遍历所有属性，性能差

#### Vue3的响应式：Proxy

```javascript
// Vue3的简化实现
function reactive(target) {
  return new Proxy(target, {
    get(target, key, receiver) {
      const result = Reflect.get(target, key, receiver);
      track(target, key);  // 依赖收集
      
      // 深度响应式：懒递归
      if (isObject(result)) {
        return reactive(result);
      }
      
      return result;
    },
    set(target, key, value, receiver) {
      const oldValue = target[key];
      const result = Reflect.set(target, key, value, receiver);
      
      if (oldValue !== value) {
        trigger(target, key);  // 触发更新
      }
      
      return result;
    },
    deleteProperty(target, key) {
      const hadKey = hasOwn(target, key);
      const result = Reflect.deleteProperty(target, key);
      
      if (hadKey && result) {
        trigger(target, key);  // 删除也触发更新
      }
      
      return result;
    }
  });
}
```

**Proxy的优势**：
1. **可以监听新增/删除属性**
2. **可以监听数组索引和length变化**
3. **懒递归**：只在访问时才创建Proxy，初始化更快
4. **可以监听更多操作**：has、ownKeys等

### 2.2 虚拟DOM与Diff算法

**面试题：虚拟DOM是什么？Diff算法的过程是怎样的？key的作用是什么？**

#### 为什么需要虚拟DOM？

直接操作真实DOM很慢：
```javascript
// 真实DOM操作：触发回流和重绘
element.style.width = '100px';  // 回流
element.style.height = '100px'; // 回流
element.style.background = 'red'; // 重绘
```

虚拟DOM是真实DOM的JavaScript对象表示：
```javascript
// 虚拟DOM
const vnode = {
  tag: 'div',
  props: { id: 'app', class: 'container' },
  children: [
    { tag: 'h1', props: {}, children: ['Hello'] },
    { tag: 'p', props: {}, children: ['World'] }
  ]
};
```

**流程**：修改数据 → 生成新VNode → Diff对比 → 计算最小更新 → 批量更新真实DOM

#### Diff算法的核心策略

Vue的Diff算法采用**双端比较**，时间复杂度O(n)：

```javascript
// 简化版Diff
function patch(oldVnode, newVnode) {
  // 1. 如果类型不同，直接替换
  if (oldVnode.tag !== newVnode.tag) {
    replaceNode(oldVnode, newVnode);
    return;
  }
  
  // 2. 类型相同，比较属性
  patchProps(oldVnode.props, newVnode.props);
  
  // 3. 比较子节点（核心）
  patchChildren(oldVnode.children, newVnode.children);
}

function patchChildren(oldCh, newCh) {
  let oldStartIdx = 0, oldEndIdx = oldCh.length - 1;
  let newStartIdx = 0, newEndIdx = newCh.length - 1;
  
  while (oldStartIdx <= oldEndIdx && newStartIdx <= newEndIdx) {
    // 四种比较方式
    if (sameVnode(oldCh[oldStartIdx], newCh[newStartIdx])) {
      // 头头比较
      patch(oldCh[oldStartIdx++], newCh[newStartIdx++]);
    } else if (sameVnode(oldCh[oldEndIdx], newCh[newEndIdx])) {
      // 尾尾比较
      patch(oldCh[oldEndIdx--], newCh[newEndIdx--]);
    } else if (sameVnode(oldCh[oldStartIdx], newCh[newEndIdx])) {
      // 头尾比较（说明有元素被移动到后面）
      patch(oldCh[oldStartIdx], newCh[newEndIdx]);
      moveNode(oldCh[oldStartIdx], oldEndIdx + 1);
      oldStartIdx++;
      newEndIdx--;
    } else if (sameVnode(oldCh[oldEndIdx], newCh[newStartIdx])) {
      // 尾头比较（说明有元素被移动到前面）
      patch(oldCh[oldEndIdx], newCh[newStartIdx]);
      moveNode(oldCh[oldEndIdx], oldStartIdx);
      oldEndIdx--;
      newStartIdx++;
    } else {
      // 都没有匹配，用key查找
      // ...
    }
  }
}
```

#### key的重要性

**没有key的问题**：
```javascript
// 旧列表：[A, B, C]
// 新列表：[A, C, B]

// 没有key时，Diff会认为：
// B → C（更新）
// C → B（更新）
// 实际上只需要移动位置！
```

**有key时**：
```javascript
// 旧列表：[A(key=1), B(key=2), C(key=3)]
// 新列表：[A(key=1), C(key=3), B(key=2)]

// Diff识别出：
// A相同
// C从位置2移动到位置2
// B从位置1移动到位置3
// 只需移动，无需更新DOM内容！
```

**key的使用注意事项**：
1. **不要用index作为key**：如果列表会增删，会导致错误的复用
2. **key要唯一且稳定**：不要用随机数
3. **同一层级key要唯一**：不同层级可以重复

## 三、React原理篇

### 3.1 Fiber架构：React的"时间切片"

**面试题：React的Fiber架构是什么？解决了什么问题？**

#### React 15的问题：递归更新阻塞主线程

```javascript
// React 15的更新是递归的，不可中断
function updateComponent(component) {
  const newVNode = component.render();
  const childComponent = newVNode.type;
  updateComponent(childComponent);  // 递归，无法中断
  // ...
}

// 如果组件树很深，会长时间占用主线程
// 导致动画卡顿、输入延迟
```

#### Fiber的核心思想

Fiber是React 16引入的新的 reconciliation 架构，核心特性：

1. **可中断的更新**：将大任务拆分成小任务，每执行完一个Fiber节点就检查是否需要让出主线程
2. **优先级调度**：高优先级任务（如用户输入）可以打断低优先级任务（如列表渲染）
3. **双缓冲**：同时维护两棵Fiber树，current（当前屏幕）和workInProgress（正在构建）

#### Fiber节点的结构

```javascript
// Fiber节点的简化结构
const fiber = {
  // 类型信息
  type: 'div',  // 或组件函数/类
  tag: WorkTag.HostComponent,
  
  // 树形结构
  return: parentFiber,    // 父节点
  child: firstChildFiber, // 第一个子节点
  sibling: nextSiblingFiber, // 下一个兄弟节点
  
  // 状态
  pendingProps: newProps,
  memoizedProps: oldProps,
  memoizedState: oldState,
  
  // 副作用
  effectTag: EffectTag.Placement | EffectTag.Update,
  nextEffect: nextFiberWithEffect,
  
  // 调度相关
  expirationTime: priority,
  mode: ConcurrentMode
};
```

#### Fiber的工作流程

```javascript
// 简化版Fiber工作流程
function workLoop(deadline) {
  while (nextUnitOfWork && !deadline.timeRemaining()) {
    // 执行一个Fiber单元
    nextUnitOfWork = performUnitOfWork(nextUnitOfWork);
  }
  
  if (!nextUnitOfWork && wipRoot) {
    // 所有Fiber都处理完了，提交更新
    commitRoot();
  }
  
  // 请求下一次调度
  requestIdleCallback(workLoop);
}

function performUnitOfWork(fiber) {
  // 1. 创建或更新组件
  if (fiber.tag === FunctionComponent) {
    updateFunctionComponent(fiber);
  } else {
    updateHostComponent(fiber);
  }
  
  // 2. 返回下一个要处理的Fiber（深度优先）
  if (fiber.child) {
    return fiber.child;
  }
  
  let nextFiber = fiber;
  while (nextFiber) {
    if (nextFiber.sibling) {
      return nextFiber.sibling;
    }
    nextFiber = nextFiber.return;
  }
}
```

### 3.2 Hooks原理：为什么不能在循环中调用

**面试题：Hooks的实现原理是什么？为什么Hooks有调用顺序的限制？**

#### Hooks的存储结构

React用链表来存储Hooks的状态：

```javascript
// 简化版实现
let hookIndex = 0;
const hooks = [];  // 实际上存储在fiber.memoizedState中

function useState(initialValue) {
  const state = hooks[hookIndex] || initialValue;
  hooks[hookIndex] = state;
  
  const setState = (newValue) => {
    hooks[hookIndex] = newValue;
    reRender();  // 触发重新渲染
  };
  
  hookIndex++;
  return [state, setState];
}

function MyComponent() {
  hookIndex = 0;  // 每次渲染重置索引
  
  const [count, setCount] = useState(0);      // index 0
  const [name, setName] = useState('React');  // index 1
  
  // ...
}
```

#### 为什么不能在循环/条件中调用

```javascript
// ❌ 错误：在条件中调用
function BadComponent({ condition }) {
  if (condition) {
    const [state, setState] = useState(0);  // 条件调用！
  }
  const [other, setOther] = useState(1);
  // ...
}

// 第一次渲染：condition为true
// hooks[0] = state
// hooks[1] = other

// 第二次渲染：condition为false
// useState(0)被跳过！
// hooks[0] = other（错误！）
```

**核心原因**：Hooks通过索引来对应状态，调用顺序改变会导致状态错乱。

#### useEffect的实现

```javascript
function useEffect(callback, deps) {
  const oldHook = hooks[hookIndex];
  const hasChanged = deps ? 
    !oldHook || !deps.every((dep, i) => dep === oldHook.deps[i]) :
    true;
  
  const hook = {
    callback: hasChanged ? callback : null,  // 有变化才执行
    deps
  };
  
  hooks[hookIndex] = hook;
  hookIndex++;
}

// 在commit阶段执行effect
function commitEffects() {
  hooks.forEach(hook => {
    if (hook.callback) {
      hook.callback();
    }
  });
}
```

## 四、浏览器原理篇

### 4.1 从输入URL到页面显示

**面试题：在浏览器地址栏输入URL后，发生了什么？**

这是一个经典问题，考察对Web全链路的理解：

```
1. URL解析
   ↓
2. DNS解析（缓存查询 → 递归查询）
   ↓
3. 建立TCP连接（三次握手）
   ↓
4. 发送HTTP请求
   ↓
5. 服务器处理请求
   ↓
6. 接收HTTP响应
   ↓
7. 浏览器解析渲染
   - 构建DOM树
   - 构建CSSOM树
   - 合并为Render Tree
   - Layout（布局）
   - Paint（绘制）
   - Composite（合成）
```

#### 关键细节：渲染流水线

```javascript
// 触发回流的属性（几何属性）
element.offsetHeight;  // 读取，强制回流
element.style.width = '100px';  // 写入，触发回流

// 触发重绘的属性（外观属性）
element.style.color = 'red';
element.style.backgroundColor = 'blue';

// 合成属性（GPU加速）
element.style.transform = 'translateX(100px)';
element.style.opacity = 0.5;
```

**优化策略**：
1. **减少回流**：使用`class`批量修改样式，而不是逐个修改
2. **读写分离**：先批量读取，再批量写入
3. **使用transform**：开启GPU加速
4. **虚拟滚动**：只渲染可见区域

### 4.2 跨域问题及解决方案

**面试题：什么是跨域？如何解决跨域问题？**

#### 跨域的本质

浏览器的**同源策略**（Same-Origin Policy）：协议、域名、端口三者相同才是同源。

```javascript
// 同源检查
http://www.example.com:80/page1
http://www.example.com:80/page2  // ✅ 同源

http://www.example.com/page1
https://www.example.com/page1    // ❌ 协议不同

http://www.example.com/page1
http://api.example.com/page1     // ❌ 域名不同

http://www.example.com/page1
http://www.example.com:8080/page1 // ❌ 端口不同
```

#### 解决方案对比

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **CORS** | 服务端设置Access-Control-Allow-Origin | 标准方案，支持各种请求 | 需要服务端配合 |
| **JSONP** | 利用script标签不受同源限制 | 兼容旧浏览器 | 只支持GET，不安全 |
| **代理服务器** | 同源策略只针对浏览器 | 无需服务端修改 | 需要额外服务器 |
| **postMessage** | HTML5跨文档通信 | 灵活，iframe通信 | 需要目标窗口配合 |
| **WebSocket** | 不受同源策略限制 | 实时双向通信 | 需要服务端支持 |

#### CORS的预检请求

```javascript
// 复杂请求会先发送OPTIONS预检
fetch('https://api.example.com/data', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-Custom-Header': 'value'
  },
  body: JSON.stringify({ name: 'test' })
});

// 浏览器自动发送：
// OPTIONS /data HTTP/1.1
// Origin: http://localhost:3000
// Access-Control-Request-Method: POST
// Access-Control-Request-Headers: content-type,x-custom-header

// 服务端响应：
// Access-Control-Allow-Origin: http://localhost:3000
// Access-Control-Allow-Methods: POST, GET, OPTIONS
// Access-Control-Allow-Headers: content-type,x-custom-header
```

## 五、性能优化篇

### 5.1 性能指标与测量

**面试题：如何衡量网页性能？Core Web Vitals是什么？**

#### Core Web Vitals（核心网页指标）

Google提出的三个关键指标：

| 指标 | 含义 | 良好标准 |
|------|------|---------|
| **LCP** (Largest Contentful Paint) | 最大内容绘制时间 | ≤2.5s |
| **FID** (First Input Delay) | 首次输入延迟 | ≤100ms |
| **CLS** (Cumulative Layout Shift) | 累积布局偏移 | ≤0.1 |

**LCP优化**：
- 优化服务器响应时间
- 预加载关键资源`<link rel="preload">`
- 优化图片（WebP、响应式图片）
- 使用CDN

**FID优化**：
- 减少JavaScript执行时间
- 代码分割，懒加载非关键JS
- 使用Web Workers

**CLS优化**：
- 图片设置width/height属性
- 广告位预留空间
- 避免在内容上方插入动态内容

### 5.2 代码分割与懒加载

```javascript
// React.lazy + Suspense
const LazyComponent = React.lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <LazyComponent />
    </Suspense>
  );
}

// Vue异步组件
const AsyncComponent = defineAsyncComponent(() => 
  import('./HeavyComponent.vue')
);

// 路由懒加载
const routes = [
  {
    path: '/about',
    component: () => import('./views/About.vue')  // 按需加载
  }
];
```

### 5.3 缓存策略

```javascript
// HTTP缓存头
Cache-Control: public, max-age=31536000, immutable  // 永久缓存（带hash的资源）
Cache-Control: public, max-age=3600                  // 1小时缓存
Cache-Control: no-cache                              // 协商缓存
Cache-Control: no-store                              // 不缓存

// Service Worker缓存
// 可以实现离线访问、后台同步等高级功能
```

## 六、手写代码篇

### 6.1 Promise实现

```javascript
class MyPromise {
  constructor(executor) {
    this.state = 'pending';
    this.value = undefined;
    this.reason = undefined;
    this.onFulfilledCallbacks = [];
    this.onRejectedCallbacks = [];
    
    const resolve = (value) => {
      if (this.state === 'pending') {
        this.state = 'fulfilled';
        this.value = value;
        this.onFulfilledCallbacks.forEach(fn => fn());
      }
    };
    
    const reject = (reason) => {
      if (this.state === 'pending') {
        this.state = 'rejected';
        this.reason = reason;
        this.onRejectedCallbacks.forEach(fn => fn());
      }
    };
    
    try {
      executor(resolve, reject);
    } catch (e) {
      reject(e);
    }
  }
  
  then(onFulfilled, onRejected) {
    // 返回新Promise，支持链式调用
    const promise2 = new MyPromise((resolve, reject) => {
      if (this.state === 'fulfilled') {
        setTimeout(() => {
          try {
            const x = onFulfilled(this.value);
            resolvePromise(promise2, x, resolve, reject);
          } catch (e) {
            reject(e);
          }
        }, 0);
      } else if (this.state === 'rejected') {
        setTimeout(() => {
          try {
            const x = onRejected(this.reason);
            resolvePromise(promise2, x, resolve, reject);
          } catch (e) {
            reject(e);
          }
        }, 0);
      } else {
        // pending状态，存入回调队列
        this.onFulfilledCallbacks.push(() => {
          setTimeout(() => {
            try {
              const x = onFulfilled(this.value);
              resolvePromise(promise2, x, resolve, reject);
            } catch (e) {
              reject(e);
            }
          }, 0);
        });
        
        this.onRejectedCallbacks.push(() => {
          setTimeout(() => {
            try {
              const x = onRejected(this.reason);
              resolvePromise(promise2, x, resolve, reject);
            } catch (e) {
              reject(e);
            }
          }, 0);
        });
      }
    });
    
    return promise2;
  }
}

// 处理then返回值的函数
function resolvePromise(promise2, x, resolve, reject) {
  if (promise2 === x) {
    return reject(new TypeError('Chaining cycle detected'));
  }
  
  if (x !== null && (typeof x === 'object' || typeof x === 'function')) {
    try {
      const then = x.then;
      if (typeof then === 'function') {
        then.call(x, y => resolvePromise(promise2, y, resolve, reject), reject);
      } else {
        resolve(x);
      }
    } catch (e) {
      reject(e);
    }
  } else {
    resolve(x);
  }
}
```

### 6.2 深拷贝实现

```javascript
function deepClone(obj, map = new WeakMap()) {
  // 基本类型直接返回
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  
  // 处理Date
  if (obj instanceof Date) {
    return new Date(obj.getTime());
  }
  
  // 处理RegExp
  if (obj instanceof RegExp) {
    return new RegExp(obj);
  }
  
  // 处理循环引用
  if (map.has(obj)) {
    return map.get(obj);
  }
  
  // 创建新对象
  const clone = Array.isArray(obj) ? [] : {};
  map.set(obj, clone);
  
  // 递归拷贝
  for (let key in obj) {
    if (obj.hasOwnProperty(key)) {
      clone[key] = deepClone(obj[key], map);
    }
  }
  
  return clone;
}

// 测试
const obj = {
  a: 1,
  b: { c: 2 },
  d: [1, 2, 3],
  e: new Date(),
  f: /abc/g
};
obj.circular = obj;  // 循环引用

const cloned = deepClone(obj);
console.log(cloned);
console.log(cloned.b === obj.b);  // false，深拷贝成功
```

## 七、总结

面试不是背答案，而是展示你的**思考深度**和**工程能力**。

### 回答问题的STAR法则

- **S**ituation（场景）：这个问题在什么场景下出现？
- **T**ask（任务）：需要解决什么问题？
- **A**ction（行动）：采取了什么方案？
- **R**esult（结果）：效果如何？有什么优缺点？

### 进阶学习路线

1. **阅读源码**：Vue、React、Lodash的源码
2. **实践项目**：用所学知识优化现有项目
3. **关注标准**：ECMAScript、W3C规范
4. **工程化**：Webpack、Vite、CI/CD流程

记住：技术是手段，解决问题才是目的。祝你面试顺利！

---

**推荐阅读**：
- [JavaScript深入系列](020-this.md)
- [Vue2.0源码解读](023-vue-reactive.md)
- [React Hooks深度解析](066-react-hooks-deep-dive.md)
