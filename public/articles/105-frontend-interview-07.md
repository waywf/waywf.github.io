---
title: 07.前端高频手写面试题深度解析：从 Promise 到组件的全链路实战
category: 前端开发
excerpt: 前端面试题第七弹！80+手写题，涵盖Promise、数组、对象、字符串、算法、设计模式、Vue/React核心等。看完这篇，手写题从此不再怕！
tags: 前端面试, 手写题, Promise, 算法, 设计模式, Vue, React
date: 2025-07-18
readTime: 60
---

# 07.前端高频手写面试题深度解析

各位前端er，欢迎来到手写题专场！这是一场没有硝烟的战争，也是大厂筛选候选人的终极利器。今天，我将带大家横扫80+高频手写题，让你在面试中提笔就写，下笔如有神！

## 一、Promise 家族

### 1.1 手写 Promise

```javascript
class MyPromise {
  static PENDING = 'pending';
  static FULFILLED = 'fulfilled';
  static REJECTED = 'rejected';

  constructor(executor) {
    this.state = MyPromise.PENDING;
    this.value = null;
    this.reason = null;
    this.onFulfilledCallbacks = [];
    this.onRejectedCallbacks = [];

    const resolve = (value) => {
      if (this.state !== MyPromise.PENDING) return;
      this.state = MyPromise.FULFILLED;
      this.value = value;
      this.onFulfilledCallbacks.forEach(fn => fn());
    };

    const reject = (reason) => {
      if (this.state !== MyPromise.PENDING) return;
      this.state = MyPromise.REJECTED;
      this.reason = reason;
      this.onRejectedCallbacks.forEach(fn => fn());
    };

    try {
      executor(resolve, reject);
    } catch (error) {
      reject(error);
    }
  }

  then(onFulfilled, onRejected) {
    onFulfilled = typeof onFulfilled === 'function' ? onFulfilled : val => val;
    onRejected = typeof onRejected === 'function' ? onRejected : err => { throw err };

    const promise2 = new MyPromise((resolve, reject) => {
      if (this.state === MyPromise.FULFILLED) {
        setTimeout(() => {
          try {
            const x = onFulfilled(this.value);
            this.resolvePromise(promise2, x, resolve, reject);
          } catch (e) {
            reject(e);
          }
        });
      } else if (this.state === MyPromise.REJECTED) {
        setTimeout(() => {
          try {
            const x = onRejected(this.reason);
            this.resolvePromise(promise2, x, resolve, reject);
          } catch (e) {
            reject(e);
          }
        });
      } else {
        this.onFulfilledCallbacks.push(() => {
          setTimeout(() => {
            try {
              const x = onFulfilled(this.value);
              this.resolvePromise(promise2, x, resolve, reject);
            } catch (e) {
              reject(e);
            }
          });
        });
        this.onRejectedCallbacks.push(() => {
          setTimeout(() => {
            try {
              const x = onRejected(this.reason);
              this.resolvePromise(promise2, x, resolve, reject);
            } catch (e) {
              reject(e);
            }
          });
        });
      }
    });

    return promise2;
  }

  resolvePromise(promise2, x, resolve, reject) {
    if (promise2 === x) {
      return reject(new TypeError('Chaining cycle detected'));
    }

    if (x instanceof MyPromise) {
      x.then(y => {
        this.resolvePromise(promise2, y, resolve, reject);
      }, reject);
    } else if (x !== null && (typeof x === 'object' || typeof x === 'function')) {
      let called = false;
      try {
        const then = x.then;
        if (typeof then === 'function') {
          then.call(x, y => {
            if (called) return;
            called = true;
            this.resolvePromise(promise2, y, resolve, reject);
          }, r => {
            if (called) return;
            called = true;
            reject(r);
          });
        } else {
          resolve(x);
        }
      } catch (e) {
        if (called) return;
        called = true;
        reject(e);
      }
    } else {
      resolve(x);
    }
  }

  static resolve(value) {
    if (value instanceof MyPromise) return value;
    return new MyPromise(resolve => resolve(value));
  }

  static reject(reason) {
    return new MyPromise((resolve, reject) => reject(reason));
  }

  catch(onRejected) {
    return this.then(null, onRejected);
  }

  finally(callback) {
    return this.then(
      value => MyPromise.resolve(callback()).then(() => value),
      reason => MyPromise.resolve(callback()).then(() => { throw reason; })
    );
  }
}
```

### 1.2 手写 Promise.all

```javascript
MyPromise.all = function(promises) {
  return new MyPromise((resolve, reject) => {
    const results = [];
    let count = 0;
    
    if (promises.length === 0) {
      resolve(results);
      return;
    }

    promises.forEach((promise, index) => {
      MyPromise.resolve(promise).then(value => {
        results[index] = value;
        count++;
        if (count === promises.length) {
          resolve(results);
        }
      }).catch(reject);
    });
  });
};
```

### 1.3 手写 Promise.race

```javascript
MyPromise.race = function(promises) {
  return new MyPromise((resolve, reject) => {
    promises.forEach(promise => {
      MyPromise.resolve(promise).then(resolve, reject);
    });
  });
};
```

### 1.4 手写 Promise.allSettled

```javascript
MyPromise.allSettled = function(promises) {
  return new MyPromise((resolve) => {
    const results = [];
    let count = 0;

    if (promises.length === 0) {
      resolve(results);
      return;
    }

    promises.forEach((promise, index) => {
      MyPromise.resolve(promise)
        .then(value => {
          results[index] = { status: 'fulfilled', value };
        })
        .catch(reason => {
          results[index] = { status: 'rejected', reason };
        })
        .finally(() => {
          count++;
          if (count === promises.length) {
            resolve(results);
          }
        });
    });
  });
};
```

### 1.5 手写 Promise.any

```javascript
MyPromise.any = function(promises) {
  return new MyPromise((resolve, reject) => {
    const errors = [];
    let count = 0;

    if (promises.length === 0) {
      reject(new AggregateError([], 'All promises were rejected'));
      return;
    }

    promises.forEach((promise, index) => {
      MyPromise.resolve(promise)
        .then(resolve)
        .catch(reason => {
          errors[index] = reason;
          count++;
          if (count === promises.length) {
            reject(new AggregateError(errors, 'All promises were rejected'));
          }
        });
    });
  });
};
```

## 二、数组方法

### 2.1 手写 forEach

```javascript
Array.prototype.myForEach = function(callback, thisArg) {
  if (this == null) {
    throw new TypeError('this is null or not defined');
  }
  if (typeof callback !== 'function') {
    throw new TypeError(callback + ' is not a function');
  }

  const arr = Object(this);
  const len = arr.length >>> 0;

  for (let i = 0; i < len; i++) {
    if (i in arr) {
      callback.call(thisArg, arr[i], i, arr);
    }
  }
};
```

### 2.2 手写 map

```javascript
Array.prototype.myMap = function(callback, thisArg) {
  if (this == null) throw new TypeError('this is null or not defined');
  if (typeof callback !== 'function') throw new TypeError(callback + ' is not a function');

  const arr = Object(this);
  const len = arr.length >>> 0;
  const result = new Array(len);

  for (let i = 0; i < len; i++) {
    if (i in arr) {
      result[i] = callback.call(thisArg, arr[i], i, arr);
    }
  }

  return result;
};
```

### 2.3 手写 filter

```javascript
Array.prototype.myFilter = function(callback, thisArg) {
  if (this == null) throw new TypeError('this is null or not defined');
  if (typeof callback !== 'function') throw new TypeError(callback + ' is not a function');

  const arr = Object(this);
  const len = arr.length >>> 0;
  const result = [];

  for (let i = 0; i < len; i++) {
    if (i in arr) {
      if (callback.call(thisArg, arr[i], i, arr)) {
        result.push(arr[i]);
      }
    }
  }

  return result;
};
```

### 2.4 手写 reduce

```javascript
Array.prototype.myReduce = function(callback, initialValue) {
  if (this == null) throw new TypeError('this is null or not defined');
  if (typeof callback !== 'function') throw new TypeError(callback + ' is not a function');

  const arr = Object(this);
  const len = arr.length >>> 0;
  let accumulator = initialValue;
  let i = 0;

  if (arguments.length < 2) {
    if (len === 0) throw new TypeError('Reduce of empty array with no initial value');
    while (i < len && !(i in arr)) i++;
    accumulator = arr[i++];
  }

  for (; i < len; i++) {
    if (i in arr) {
      accumulator = callback(accumulator, arr[i], i, arr);
    }
  }

  return accumulator;
};
```

### 2.5 手写 some

```javascript
Array.prototype.mySome = function(callback, thisArg) {
  if (this == null) throw new TypeError('this is null or not defined');
  if (typeof callback !== 'function') throw new TypeError(callback + ' is not a function');

  const arr = Object(this);
  const len = arr.length >>> 0;

  for (let i = 0; i < len; i++) {
    if (i in arr && callback.call(thisArg, arr[i], i, arr)) {
      return true;
    }
  }

  return false;
};
```

### 2.6 手写 every

```javascript
Array.prototype.myEvery = function(callback, thisArg) {
  if (this == null) throw new TypeError('this is null or not defined');
  if (typeof callback !== 'function') throw new TypeError(callback + ' is not a function');

  const arr = Object(this);
  const len = arr.length >>> 0;

  for (let i = 0; i < len; i++) {
    if (i in arr && !callback.call(thisArg, arr[i], i, arr)) {
      return false;
    }
  }

  return true;
};
```

### 2.7 手写 find

```javascript
Array.prototype.myFind = function(callback, thisArg) {
  if (this == null) throw new TypeError('this is null or not defined');
  if (typeof callback !== 'function') throw new TypeError(callback + ' is not a function');

  const arr = Object(this);
  const len = arr.length >>> 0;

  for (let i = 0; i < len; i++) {
    if (i in arr && callback.call(thisArg, arr[i], i, arr)) {
      return arr[i];
    }
  }

  return undefined;
};
```

### 2.8 手写 findIndex

```javascript
Array.prototype.myFindIndex = function(callback, thisArg) {
  if (this == null) throw new TypeError('this is null or not defined');
  if (typeof callback !== 'function') throw new TypeError(callback + ' is not a function');

  const arr = Object(this);
  const len = arr.length >>> 0;

  for (let i = 0; i < len; i++) {
    if (i in arr && callback.call(thisArg, arr[i], i, arr)) {
      return i;
    }
  }

  return -1;
};
```

### 2.9 手写 flat

```javascript
Array.prototype.myFlat = function(depth = 1) {
  const result = [];
  const flatten = (arr, currentDepth) => {
    for (let i = 0; i < arr.length; i++) {
      if (Array.isArray(arr[i]) && currentDepth < depth) {
        flatten(arr[i], currentDepth + 1);
      } else {
        result.push(arr[i]);
      }
    }
  };
  flatten(this, 0);
  return result;
};
```

### 2.10 手写 unique（数组去重）

```javascript
Array.prototype.myUnique = function() {
  const seen = new Set();
  return this.filter(item => {
    if (!seen.has(item)) {
      seen.add(item);
      return true;
    }
    return false;
  });
};

Array.prototype.myUnique2 = function() {
  return [...new Set(this)];
};

Array.prototype.myUnique3 = function() {
  const result = [];
  this.forEach(item => {
    if (!result.includes(item)) {
      result.push(item);
    }
  });
  return result;
};
```

## 三、对象方法

### 3.1 手写深拷贝

```javascript
function deepClone(obj, hash = new WeakMap()) {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj);
  }

  if (obj instanceof RegExp) {
    return new RegExp(obj.source, obj.flags);
  }

  if (hash.has(obj)) {
    return hash.get(obj);
  }

  const cloneObj = new obj.constructor();
  hash.set(obj, cloneObj);

  for (let key in obj) {
    if (obj.hasOwnProperty(key)) {
      cloneObj[key] = deepClone(obj[key], hash);
    }
  }

  return cloneObj;
}
```

### 3.2 手写 instanceof

```javascript
function myInstanceof(left, right) {
  if (left === null || (typeof left !== 'object' && typeof left !== 'function')) {
    return false;
  }

  let proto = Object.getPrototypeOf(left);
  while (proto !== null) {
    if (proto === right.prototype) {
      return true;
    }
    proto = Object.getPrototypeOf(proto);
  }

  return false;
}
```

### 3.3 手写 new

```javascript
function myNew(constructor, ...args) {
  const obj = Object.create(constructor.prototype);
  const result = constructor.apply(obj, args);
  return result instanceof Object ? result : obj;
}
```

### 3.4 手写 Object.create

```javascript
Object.myCreate = function(proto, propertiesObject) {
  if (typeof proto !== 'object' && typeof proto !== 'function') {
    throw new TypeError('Object prototype may only be an Object or null');
  }

  function F() {}
  F.prototype = proto;
  const obj = new F();

  if (propertiesObject !== undefined) {
    Object.defineProperties(obj, propertiesObject);
  }

  return obj;
};
```

### 3.5 手写 Object.assign

```javascript
Object.myAssign = function(target, ...sources) {
  if (target == null) {
    throw new TypeError('Cannot convert undefined or null to object');
  }

  const to = Object(target);

  sources.forEach(source => {
    if (source != null) {
      for (let key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
          to[key] = source[key];
        }
      }

      const symbols = Object.getOwnPropertySymbols(source);
      symbols.forEach(symbol => {
        if (Object.prototype.propertyIsEnumerable.call(source, symbol)) {
          to[symbol] = source[symbol];
        }
      });
    }
  });

  return to;
};
```

### 3.6 手写 Object.freeze

```javascript
Object.myFreeze = function(obj) {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  Object.getOwnPropertyNames(obj).forEach(prop => {
    if (Object.getOwnPropertyDescriptor(obj, prop).configurable) {
      Object.defineProperty(obj, prop, {
        writable: false,
        configurable: false
      });
    }
  });

  Object.getOwnPropertySymbols(obj).forEach(symbol => {
    if (Object.getOwnPropertyDescriptor(obj, symbol).configurable) {
      Object.defineProperty(obj, symbol, {
        writable: false,
        configurable: false
      });
    }
  });

  Object.setPrototypeOf(obj, null);
  return obj;
};
```

## 四、函数方法

### 4.1 手写 call

```javascript
Function.prototype.myCall = function(context, ...args) {
  context = context || window;
  const fn = Symbol('fn');
  context[fn] = this;
  const result = context[fn](...args);
  delete context[fn];
  return result;
};
```

### 4.2 手写 apply

```javascript
Function.prototype.myApply = function(context, args) {
  context = context || window;
  const fn = Symbol('fn');
  context[fn] = this;
  const result = context[fn](...(args || []));
  delete context[fn];
  return result;
};
```

### 4.3 手写 bind

```javascript
Function.prototype.myBind = function(context, ...args1) {
  const self = this;
  const F = function() {};

  const bound = function(...args2) {
    return self.apply(
      this instanceof F ? this : context,
      [...args1, ...args2]
    );
  };

  if (this.prototype) {
    F.prototype = this.prototype;
  }
  bound.prototype = new F();

  return bound;
};
```

### 4.4 手写柯里化

```javascript
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    } else {
      return function(...moreArgs) {
        return curried.apply(this, [...args, ...moreArgs]);
      };
    }
  };
}

function curry2(fn, ...args) {
  return args.length >= fn.length 
    ? fn(...args) 
    : (...moreArgs) => curry2(fn, ...args, ...moreArgs);
}
```

### 4.5 手写防抖

```javascript
function debounce(fn, delay = 300, immediate = false) {
  let timer = null;

  return function(...args) {
    const context = this;
    
    if (timer) clearTimeout(timer);

    if (immediate) {
      const callNow = !timer;
      timer = setTimeout(() => timer = null, delay);
      if (callNow) fn.apply(context, args);
    } else {
      timer = setTimeout(() => fn.apply(context, args), delay);
    }
  };
}
```

### 4.6 手写节流

```javascript
function throttle(fn, delay = 300) {
  let lastTime = 0;
  let timer = null;

  return function(...args) {
    const context = this;
    const now = Date.now();

    if (now - lastTime >= delay) {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      fn.apply(context, args);
      lastTime = now;
    } else if (!timer) {
      timer = setTimeout(() => {
        fn.apply(context, args);
        lastTime = Date.now();
        timer = null;
      }, delay - (now - lastTime));
    }
  };
}

function throttle2(fn, delay = 300) {
  let lastTime = 0;
  return function(...args) {
    const now = Date.now();
    if (now - lastTime >= delay) {
      fn.apply(this, args);
      lastTime = now;
    }
  };
}
```

### 4.7 手写 compose

```javascript
function compose(...fns) {
  if (fns.length === 0) return arg => arg;
  if (fns.length === 1) return fns[0];
  return fns.reduce((a, b) => (...args) => a(b(...args)));
}

function pipe(...fns) {
  if (fns.length === 0) return arg => arg;
  if (fns.length === 1) return fns[0];
  return fns.reduce((a, b) => (...args) => b(a(...args)));
}
```

### 4.8 手写函数记忆化

```javascript
function memoize(fn) {
  const cache = new Map();
  return function(...args) {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  };
}
```

## 五、字符串方法

### 5.1 手写 trim

```javascript
String.prototype.myTrim = function() {
  return this.replace(/^\s+|\s+$/g, '');
};

String.prototype.myTrim2 = function() {
  let start = 0;
  let end = this.length - 1;

  while (start <= end && /\s/.test(this[start])) {
    start++;
  }

  while (end >= start && /\s/.test(this[end])) {
    end--;
  }

  return this.slice(start, end + 1);
};
```

### 5.2 手写模板字符串

```javascript
function renderTemplate(template, data) {
  return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
    return data[key] !== undefined ? data[key] : match;
  });
}

const template = '你好，我是{{name}}，今年{{age}}岁';
const data = { name: '张三', age: 25 };
console.log(renderTemplate(template, data));
```

### 5.3 手写千分位格式化

```javascript
function formatNumber(num) {
  const str = num.toString();
  const parts = str.split('.');
  parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return parts.join('.');
}

function formatNumber2(num) {
  let result = '';
  let count = 0;
  const str = num.toString();
  const [integer, decimal] = str.split('.');

  for (let i = integer.length - 1; i >= 0; i--) {
    if (count && count % 3 === 0) {
      result = ',' + result;
    }
    result = integer[i] + result;
    count++;
  }

  return decimal ? result + '.' + decimal : result;
}
```

### 5.4 手写URL参数解析

```javascript
function parseUrlParams(url) {
  const params = {};
  const search = url.split('?')[1];
  if (!search) return params;

  search.split('&').forEach(pair => {
    const [key, value] = pair.split('=');
    params[decodeURIComponent(key)] = decodeURIComponent(value || '');
  });

  return params;
}

function parseUrlParams2(url) {
  const searchParams = new URLSearchParams(url.split('?')[1]);
  const params = {};
  for (const [key, value] of searchParams) {
    params[key] = value;
  }
  return params;
}
```

## 六、算法

### 6.1 手写快排

```javascript
function quickSort(arr) {
  if (arr.length <= 1) return arr;

  const pivot = arr[Math.floor(arr.length / 2)];
  const left = [];
  const right = [];
  const middle = [];

  for (let num of arr) {
    if (num < pivot) left.push(num);
    else if (num > pivot) right.push(num);
    else middle.push(num);
  }

  return [...quickSort(left), ...middle, ...quickSort(right)];
}

function quickSort2(arr, left = 0, right = arr.length - 1) {
  if (left < right) {
    const pivotIndex = partition(arr, left, right);
    quickSort2(arr, left, pivotIndex - 1);
    quickSort2(arr, pivotIndex + 1, right);
  }
  return arr;
}

function partition(arr, left, right) {
  const pivot = arr[right];
  let i = left - 1;

  for (let j = left; j < right; j++) {
    if (arr[j] <= pivot) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  [arr[i + 1], arr[right]] = [arr[right], arr[i + 1]];
  return i + 1;
}
```

### 6.2 手写归并排序

```javascript
function mergeSort(arr) {
  if (arr.length <= 1) return arr;

  const mid = Math.floor(arr.length / 2);
  const left = arr.slice(0, mid);
  const right = arr.slice(mid);

  return merge(mergeSort(left), mergeSort(right));
}

function merge(left, right) {
  const result = [];
  let i = 0, j = 0;

  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }

  return [...result, ...left.slice(i), ...right.slice(j)];
}
```

### 6.3 手写二分查找

```javascript
function binarySearch(arr, target) {
  let left = 0;
  let right = arr.length - 1;

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    
    if (arr[mid] === target) {
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return -1;
}

function binarySearchRecursive(arr, target, left = 0, right = arr.length - 1) {
  if (left > right) return -1;

  const mid = Math.floor((left + right) / 2);

  if (arr[mid] === target) {
    return mid;
  } else if (arr[mid] < target) {
    return binarySearchRecursive(arr, target, mid + 1, right);
  } else {
    return binarySearchRecursive(arr, target, left, mid - 1);
  }
}
```

### 6.4 手写链表反转

```javascript
function reverseList(head) {
  let prev = null;
  let curr = head;

  while (curr !== null) {
    const next = curr.next;
    curr.next = prev;
    prev = curr;
    curr = next;
  }

  return prev;
}

function reverseListRecursive(head) {
  if (head === null || head.next === null) {
    return head;
  }

  const newHead = reverseListRecursive(head.next);
  head.next.next = head;
  head.next = null;

  return newHead;
}
```

### 6.5 手写二叉树遍历

```javascript
function preorderTraversal(root) {
  const result = [];
  const stack = [];

  if (root) stack.push(root);

  while (stack.length > 0) {
    const node = stack.pop();
    result.push(node.val);

    if (node.right) stack.push(node.right);
    if (node.left) stack.push(node.left);
  }

  return result;
}

function inorderTraversal(root) {
  const result = [];
  const stack = [];
  let curr = root;

  while (curr !== null || stack.length > 0) {
    while (curr !== null) {
      stack.push(curr);
      curr = curr.left;
    }

    curr = stack.pop();
    result.push(curr.val);
    curr = curr.right;
  }

  return result;
}

function postorderTraversal(root) {
  const result = [];
  const stack = [];
  let last = null;
  let curr = root;

  while (curr !== null || stack.length > 0) {
    while (curr !== null) {
      stack.push(curr);
      curr = curr.left;
    }

    curr = stack[stack.length - 1];

    if (curr.right === null || curr.right === last) {
      result.push(curr.val);
      stack.pop();
      last = curr;
      curr = null;
    } else {
      curr = curr.right;
    }
  }

  return result;
}

function levelOrder(root) {
  if (!root) return [];

  const result = [];
  const queue = [root];

  while (queue.length > 0) {
    const level = [];
    const levelSize = queue.length;

    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift();
      level.push(node.val);

      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }

    result.push(level);
  }

  return result;
}
```

### 6.6 手写LRU缓存

```javascript
class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();
  }

  get(key) {
    if (!this.cache.has(key)) {
      return -1;
    }

    const value = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  put(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    this.cache.set(key, value);

    if (this.cache.size > this.capacity) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
  }
}
```

## 七、设计模式

### 7.1 手写单例模式

```javascript
class Singleton {
  constructor() {
    if (Singleton.instance) {
      return Singleton.instance;
    }
    Singleton.instance = this;
  }

  static getInstance() {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }
}

const singleton1 = new Singleton();
const singleton2 = new Singleton();
console.log(singleton1 === singleton2);
```

### 7.2 手写观察者模式

```javascript
class Subject {
  constructor() {
    this.observers = [];
  }

  subscribe(observer) {
    this.observers.push(observer);
  }

  unsubscribe(observer) {
    this.observers = this.observers.filter(o => o !== observer);
  }

  notify(data) {
    this.observers.forEach(observer => observer.update(data));
  }
}

class Observer {
  constructor(name) {
    this.name = name;
  }

  update(data) {
    console.log(`${this.name}收到通知:`, data);
  }
}

const subject = new Subject();
const observer1 = new Observer('观察者1');
const observer2 = new Observer('观察者2');

subject.subscribe(observer1);
subject.subscribe(observer2);
subject.notify('Hello World!');
```

### 7.3 手写发布订阅模式

```javascript
class EventEmitter {
  constructor() {
    this.events = {};
  }

  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }

  off(event, callback) {
    if (!this.events[event]) return;

    if (callback) {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    } else {
      delete this.events[event];
    }
  }

  emit(event, ...args) {
    if (!this.events[event]) return;
    this.events[event].forEach(callback => callback(...args));
  }

  once(event, callback) {
    const onceCallback = (...args) => {
      callback(...args);
      this.off(event, onceCallback);
    };
    this.on(event, onceCallback);
  }
}

const emitter = new EventEmitter();
emitter.on('click', (data) => console.log('点击事件:', data));
emitter.emit('click', { x: 100, y: 200 });
```

### 7.4 手写工厂模式

```javascript
class Product {
  constructor(name) {
    this.name = name;
  }
}

class ConcreteProductA extends Product {
  constructor() {
    super('产品A');
  }
}

class ConcreteProductB extends Product {
  constructor() {
    super('产品B');
  }
}

class Factory {
  createProduct(type) {
    switch (type) {
      case 'A':
        return new ConcreteProductA();
      case 'B':
        return new ConcreteProductB();
      default:
        throw new Error('未知产品类型');
    }
  }
}

const factory = new Factory();
const productA = factory.createProduct('A');
const productB = factory.createProduct('B');
```

### 7.5 手写代理模式

```javascript
class RealSubject {
  request() {
    return '真实对象的请求';
  }
}

class Proxy {
  constructor() {
    this.realSubject = new RealSubject();
  }

  request() {
    console.log('代理前置处理');
    const result = this.realSubject.request();
    console.log('代理后置处理');
    return result;
  }
}

const proxy = new Proxy();
console.log(proxy.request());
```

## 八、Vue 相关

### 8.1 手写简单的响应式系统

```javascript
class Dep {
  constructor() {
    this.subscribers = new Set();
  }

  depend() {
    if (activeEffect) {
      this.subscribers.add(activeEffect);
    }
  }

  notify() {
    this.subscribers.forEach(effect => effect());
  }
}

let activeEffect = null;

function watchEffect(effect) {
  activeEffect = effect;
  effect();
  activeEffect = null;
}

function reactive(obj) {
  const depsMap = new Map();

  return new Proxy(obj, {
    get(target, key) {
      let dep = depsMap.get(key);
      if (!dep) {
        dep = new Dep();
        depsMap.set(key, dep);
      }
      dep.depend();
      return target[key];
    },
    set(target, key, value) {
      target[key] = value;
      const dep = depsMap.get(key);
      if (dep) {
        dep.notify();
      }
      return true;
    }
  });
}

const state = reactive({ count: 0 });
watchEffect(() => {
  console.log('count:', state.count);
});
state.count++;
```

### 8.2 手写 computed

```javascript
function computed(getter) {
  let value;
  let dirty = true;
  const dep = new Dep();

  const effect = () => {
    dirty = true;
    dep.notify();
  };

  return {
    get value() {
      if (dirty) {
        activeEffect = effect;
        value = getter();
        activeEffect = null;
        dirty = false;
      }
      dep.depend();
      return value;
    }
  };
}

const state = reactive({ a: 1, b: 2 });
const sum = computed(() => state.a + state.b);
watchEffect(() => {
  console.log('sum:', sum.value);
});
state.a = 3;
```

### 8.3 手写 watch

```javascript
function watch(source, callback, options = {}) {
  let getter;
  
  if (typeof source === 'function') {
    getter = source;
  } else {
    getter = () => traverse(source);
  }

  let oldValue;
  let cleanup;

  const onCleanup = (fn) => {
    cleanup = fn;
  };

  const job = () => {
    const newValue = effectFn();
    if (cleanup) cleanup();
    callback(newValue, oldValue, onCleanup);
    oldValue = newValue;
  };

  const effectFn = () => {
    activeEffect = effectFn;
    const value = getter();
    activeEffect = null;
    return value;
  };

  if (options.immediate) {
    job();
  } else {
    oldValue = effectFn();
  }
}

function traverse(value, seen = new Set()) {
  if (value === null || typeof value !== 'object' || seen.has(value)) {
    return;
  }
  seen.add(value);
  for (const key in value) {
    traverse(value[key], seen);
  }
  return value;
}
```

### 8.4 手写 v-model

```javascript
function vModel(el, value, update) {
  el.value = value;
  el.addEventListener('input', (e) => {
    update(e.target.value);
  });
}

const input = document.querySelector('input');
const state = { value: '' };

vModel(input, state.value, (newValue) => {
  state.value = newValue;
  console.log('新值:', state.value);
});
```

## 九、React 相关

### 9.1 手写 useState

```javascript
let state = [];
let index = 0;

function useState(initialValue) {
  const currentIndex = index;
  state[currentIndex] = state[currentIndex] ?? initialValue;

  const setState = (newValue) => {
    state[currentIndex] = typeof newValue === 'function' 
      ? newValue(state[currentIndex]) 
      : newValue;
    render();
  };

  index++;
  return [state[currentIndex], setState];
}

function render() {
  index = 0;
  App();
}

function App() {
  const [count, setCount] = useState(0);
  console.log('count:', count);
  return { count, setCount };
}

const { setCount } = App();
setCount(1);
setCount(prev => prev + 1);
```

### 9.2 手写 useEffect

```javascript
let depsMap = [];
let effectIndex = 0;

function useEffect(callback, deps) {
  const oldDeps = depsMap[effectIndex];
  const hasChanged = !oldDeps || !deps || deps.some((dep, i) => dep !== oldDeps[i]);

  if (hasChanged) {
    if (oldDeps?.cleanup) {
      oldDeps.cleanup();
    }
    const cleanup = callback();
    depsMap[effectIndex] = { deps, cleanup };
  }

  effectIndex++;
}

function render2() {
  effectIndex = 0;
  App2();
}

function App2() {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    console.log('count变化:', count);
    return () => console.log('清理');
  }, [count]);

  return { count, setCount };
}
```

### 9.3 手写 useCallback

```javascript
let callbacksMap = [];
let callbackIndex = 0;

function useCallback(callback, deps) {
  const oldCallback = callbacksMap[callbackIndex];
  const hasChanged = !oldCallback || !deps || 
    deps.some((dep, i) => dep !== oldCallback.deps[i]);

  if (hasChanged) {
    callbacksMap[callbackIndex] = { callback, deps };
  }

  callbackIndex++;
  return callbacksMap[callbackIndex - 1].callback;
}
```

### 9.4 手写 useMemo

```javascript
let memosMap = [];
let memoIndex = 0;

function useMemo(factory, deps) {
  const oldMemo = memosMap[memoIndex];
  const hasChanged = !oldMemo || !deps || 
    deps.some((dep, i) => dep !== oldMemo.deps[i]);

  if (hasChanged) {
    memosMap[memoIndex] = { value: factory(), deps };
  }

  memoIndex++;
  return memosMap[memoIndex - 1].value;
}
```

### 9.5 手写 useRef

```javascript
let refsMap = [];
let refIndex = 0;

function useRef(initialValue) {
  if (!refsMap[refIndex]) {
    refsMap[refIndex] = { current: initialValue };
  }
  refIndex++;
  return refsMap[refIndex - 1];
}
```

## 十、DOM & 工具

### 10.1 手写事件委托

```javascript
function delegate(parent, eventType, selector, handler) {
  parent.addEventListener(eventType, function(e) {
    const target = e.target;
    const closest = target.closest(selector);
    
    if (closest && parent.contains(closest)) {
      handler.call(closest, e);
    }
  });
}

const list = document.querySelector('ul');
delegate(list, 'click', 'li', function(e) {
  console.log('点击了:', this.textContent);
});
```

### 10.2 手写懒加载

```javascript
class LazyLoad {
  constructor(options = {}) {
    this.options = {
      root: null,
      rootMargin: '0px',
      threshold: 0.1,
      ...options
    };
    this.init();
  }

  init() {
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          this.loadImage(entry.target);
          this.observer.unobserve(entry.target);
        }
      });
    }, this.options);

    document.querySelectorAll('img[data-src]').forEach(img => {
      this.observer.observe(img);
    });
  }

  loadImage(img) {
    const src = img.dataset.src;
    img.src = src;
    img.classList.add('loaded');
  }

  destroy() {
    this.observer.disconnect();
  }
}

new LazyLoad();
```

### 10.3 手写深比较

```javascript
function deepEqual(a, b) {
  if (a === b) return true;

  if (typeof a !== typeof b) return false;

  if (typeof a !== 'object' || a === null || b === null) return false;

  if (a instanceof Date && b instanceof Date) {
    return a.getTime() === b.getTime();
  }

  if (a instanceof RegExp && b instanceof RegExp) {
    return a.toString() === b.toString();
  }

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  return keysA.every(key => deepEqual(a[key], b[key]));
}
```

### 10.4 手写数据类型检测

```javascript
function getType(obj) {
  const type = typeof obj;

  if (type !== 'object') {
    return type;
  }

  return Object.prototype.toString.call(obj).slice(8, -1).toLowerCase();
}

console.log(getType(1));
console.log(getType('hello'));
console.log(getType(true));
console.log(getType(undefined));
console.log(getType(null));
console.log(getType({}));
console.log(getType([]));
console.log(getType(function(){}));
console.log(getType(new Date()));
console.log(getType(/regex/));
```

### 10.5 手写随机数生成

```javascript
function random(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomColor() {
  const hex = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += hex[random(0, 15)];
  }
  return color;
}

function randomString(length = 8) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars[random(0, chars.length - 1)];
  }
  return result;
}

function shuffle(arr) {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = random(0, i);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
```

### 10.6 手写并发控制

```javascript
class ConcurrencyControl {
  constructor(limit) {
    this.limit = limit;
    this.queue = [];
    this.activeCount = 0;
  }

  add(task) {
    return new Promise((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this.run();
    });
  }

  run() {
    if (this.activeCount >= this.limit || this.queue.length === 0) {
      return;
    }

    const { task, resolve, reject } = this.queue.shift();
    this.activeCount++;

    Promise.resolve(task())
      .then(resolve)
      .catch(reject)
      .finally(() => {
        this.activeCount--;
        this.run();
      });
  }
}

const concurrency = new ConcurrencyControl(3);

for (let i = 0; i < 10; i++) {
  concurrency.add(() => {
    return new Promise(resolve => {
      setTimeout(() => {
        console.log('任务完成:', i);
        resolve(i);
      }, 1000);
    });
  });
}
```

### 10.7 手写异步串行

```javascript
async function series(tasks) {
  const results = [];
  for (const task of tasks) {
    results.push(await task());
  }
  return results;
}

function series2(tasks) {
  return tasks.reduce((promise, task) => {
    return promise.then(results => {
      return task().then(result => [...results, result]);
    });
  }, Promise.resolve([]));
}
```

### 10.8 手写 JSON.stringify

```javascript
function jsonStringify(obj) {
  if (obj === null) return 'null';
  if (typeof obj === 'undefined') return undefined;
  if (typeof obj === 'boolean') return String(obj);
  if (typeof obj === 'number') return String(obj);
  if (typeof obj === 'string') return `"${obj}"`;

  if (typeof obj === 'object') {
    if (obj instanceof Date) {
      return `"${obj.toISOString()}"`;
    }
    if (obj instanceof RegExp) {
      return '{}';
    }
    if (Array.isArray(obj)) {
      const items = obj.map(item => jsonStringify(item) ?? 'null');
      return `[${items.join(',')}]`;
    }

    const pairs = Object.entries(obj)
      .filter(([, value]) => value !== undefined)
      .map(([key, value]) => `"${key}":${jsonStringify(value)}`);
    return `{${pairs.join(',')}}`;
  }
}
```

### 10.9 手写 JSON.parse

```javascript
function jsonParse(str) {
  return eval('(' + str + ')');
}

function jsonParse2(str) {
  let index = 0;

  function parseValue() {
    skipWhitespace();
    const char = str[index];

    if (char === '{') return parseObject();
    if (char === '[') return parseArray();
    if (char === '"') return parseString();
    if (char === 't') return parseTrue();
    if (char === 'f') return parseFalse();
    if (char === 'n') return parseNull();
    if (char === '-' || (char >= '0' && char <= '9')) return parseNumber();

    throw new Error('Unexpected character: ' + char);
  }

  function skipWhitespace() {
    while (index < str.length && /\s/.test(str[index])) {
      index++;
    }
  }

  function parseObject() {
    index++;
    const obj = {};
    skipWhitespace();

    if (str[index] === '}') {
      index++;
      return obj;
    }

    while (true) {
      const key = parseString();
      skipWhitespace();
      if (str[index] !== ':') throw new Error('Expected colon');
      index++;
      obj[key] = parseValue();
      skipWhitespace();
      
      if (str[index] === '}') {
        index++;
        break;
      }
      if (str[index] !== ',') throw new Error('Expected comma');
      index++;
    }

    return obj;
  }

  function parseArray() {
    index++;
    const arr = [];
    skipWhitespace();

    if (str[index] === ']') {
      index++;
      return arr;
    }

    while (true) {
      arr.push(parseValue());
      skipWhitespace();
      
      if (str[index] === ']') {
        index++;
        break;
      }
      if (str[index] !== ',') throw new Error('Expected comma');
      index++;
    }

    return arr;
  }

  function parseString() {
    index++;
    let result = '';
    while (index < str.length && str[index] !== '"') {
      if (str[index] === '\\') {
        index++;
        const escape = str[index];
        switch (escape) {
          case 'n': result += '\n'; break;
          case 't': result += '\t'; break;
          case 'r': result += '\r'; break;
          case 'b': result += '\b'; break;
          case 'f': result += '\f'; break;
          default: result += escape;
        }
      } else {
        result += str[index];
      }
      index++;
    }
    index++;
    return result;
  }

  function parseNumber() {
    let start = index;
    if (str[index] === '-') index++;
    while (index < str.length && str[index] >= '0' && str[index] <= '9') {
      index++;
    }
    if (str[index] === '.') {
      index++;
      while (index < str.length && str[index] >= '0' && str[index] <= '9') {
        index++;
      }
    }
    if (str[index] === 'e' || str[index] === 'E') {
      index++;
      if (str[index] === '+' || str[index] === '-') index++;
      while (index < str.length && str[index] >= '0' && str[index] <= '9') {
        index++;
      }
    }
    return Number(str.slice(start, index));
  }

  function parseTrue() {
    if (str.slice(index, index + 4) === 'true') {
      index += 4;
      return true;
    }
    throw new Error('Expected true');
  }

  function parseFalse() {
    if (str.slice(index, index + 5) === 'false') {
      index += 5;
      return false;
    }
    throw new Error('Expected false');
  }

  function parseNull() {
    if (str.slice(index, index + 4) === 'null') {
      index += 4;
      return null;
    }
    throw new Error('Expected null');
  }

  return parseValue();
}
```

### 10.10 手写 Cookie 操作

```javascript
const Cookie = {
  set(name, value, options = {}) {
    let cookie = `${encodeURIComponent(name)}=${encodeURIComponent(value)}`;

    if (options.expires) {
      const expires = options.expires instanceof Date 
        ? options.expires 
        : new Date(Date.now() + options.expires * 1000);
      cookie += `; expires=${expires.toUTCString()}`;
    }

    if (options.path) cookie += `; path=${options.path}`;
    if (options.domain) cookie += `; domain=${options.domain}`;
    if (options.secure) cookie += '; secure';
    if (options.sameSite) cookie += `; sameSite=${options.sameSite}`;

    document.cookie = cookie;
  },

  get(name) {
    const cookies = document.cookie.split('; ');
    for (const cookie of cookies) {
      const [key, value] = cookie.split('=');
      if (decodeURIComponent(key) === name) {
        return decodeURIComponent(value);
      }
    }
    return null;
  },

  remove(name, options = {}) {
    this.set(name, '', {
      ...options,
      expires: -1
    });
  },

  getAll() {
    const cookies = {};
    document.cookie.split('; ').forEach(cookie => {
      const [key, value] = cookie.split('=');
      cookies[decodeURIComponent(key)] = decodeURIComponent(value);
    });
    return cookies;
  }
};
```

## 总结

恭喜你坚持到了最后！这80+道手写题涵盖了前端开发的方方面面，从Promise到数组方法，从算法到设计模式，从Vue到React，应有尽有。

手写题不是为了刁难人，而是为了：
1. **考察你的基础功底** - 看你是不是真的懂原理
2. **考察你的编码能力** - 看你能不能把想法变成代码
3. **考察你的思维逻辑** - 看你解决问题的思路是否清晰

记住：**光看不练假把式**。找个安静的下午，把这些手写题都撸一遍，下次面试遇到手写题时，你就是全场最靓的仔！

下一篇，我们将进入大厂面试题专场，看看那些让无数前端er折腰的硬核面试题！
