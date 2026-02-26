---
title: JavaScript专题之函数柯里化
excerpt: 深入解析JavaScript函数柯里化
category: 前端开发
date: 2026-02-25
readTime: 20
tags: JavaScript, 底层系列
---

## 定义

维基百科中对柯里化 (Currying) 的定义为：

>In mathematics and computer science, currying is the technique of translating the evaluation of a function that takes multiple arguments (or a tuple of arguments) into evaluating a sequence of functions, each with a single argument.

翻译成中文：在数学和计算机科学中，柯里化是一种将使用多个参数的一个函数转换成一系列使用一个参数的函数的技术。

举个例子：

```javascript
function add(a, b) {
  return a + b;
}

// 执行 add 函数，一次传入两个参数即可
add(1, 2) // 3

// 假设有一个 curry 函数可以做到柯里化
var addCurry = curry(add);
addCurry(1)(2) // 3
```

## 用途

我们会讲到如何写出这个 curry 函数，并且会将这个 curry 函数写的很强大，但是在编写之前，我们需要知道柯里化到底有什么用？

举个例子：

```javascript
// 示意而已
function ajax(type, url, data) {
  var xhr = new XMLHttpRequest();
  xhr.open(type, url, true);
  xhr.send(data);
}

// 虽然 ajax 这个函数非常通用，但在重复调用的时候参数冗余
// 利用 curry
var ajaxCurry = curry(ajax);

// 以 POST 类型请求数据
var post = ajaxCurry('POST');
postFromTest("name=kevin");
```

想想 jQuery 虽然有.ajac这样通用的方法，但是也有.ajax这样通用的方法，但是也有.get 和 $.post 的语法糖。(当然 jQuery 底层是否是这样做的，我就没有研究了)。

curry 的这种用途可以理解为：参数复用。本质上是降低通用性，提高适用性。

可是即便如此，是不是依然感觉没什么用呢？如果我们仅仅是把参数一个一个传进去，意义可能不大，但是如果我们是把柯里化后的函数传给其他函数比如 map 呢？

举个例子：

比如我们有这样一段数据：

```javascript
var person = [{name: 'kevin'}, {name: 'daisy’}]
```

如果我们要获取所有的 name 值，我们可以这样做：

```javascript
var name = person.map(function (item) {
  return item.name;
})
```

不过如果我们有 curry 函数：

```javascript
var prop = curry(function (key, obj) {
  return obj[key]
});

var name = person.map(prop('name'))
```

我们为了获取 name 属性还要再编写一个 prop 函数，是不是又麻烦了些？但是要注意，prop 函数编写一次后，以后可以多次使用，实际上代码从原本的三行精简成了一行，而且你看代码是不是更加易懂了？

person.map(prop('name')) 就好像直白的告诉你：person 对象遍历(map)获取(prop) name 属性。是不是感觉有点意思了呢？

## 第一版

未来我们会接触到更多有关柯里化的应用，不过那是未来的事情了，现在我们该编写这个 curry 函数了。

一个经常会看到的 curry 函数的实现为：

```javascript
// 第一版
var curry = function (fn) {
  var args = [].slice.call(arguments, 1);
  return function() {
    var newArgs = args.concat([].slice.call(arguments));
    return fn.apply(this, newArgs);
  };
};
```

我们可以这样使用：

```javascript
function add(a, b) {
  return a + b;
}

var addCurry = curry(add, 1, 2);
addCurry() // 3

//或者
var addCurry = curry(add, 1);
addCurry(2) // 3

//或者
var addCurry = curry(add);
addCurry(1, 2) // 3
```

已经有柯里化的感觉了，但是还没有达到要求，不过我们可以把这个函数用作辅助函数，帮助我们写真正的 curry 函数。

## 第二版

```javascript
// 第二版
function sub_curry(fn) {
  var args = [].slice.call(arguments, 1);
  return function() {
    return fn.apply(this, args.concat([].slice.call(arguments)));
  };
}

function curry(fn, length) {
  length = length || fn.length;
  var slice = Array.prototype.slice;
  return function() {
    if (arguments.length < length) {
      var combined = [fn].concat(slice.call(arguments));
      return curry(sub_curry.apply(this, combined), length - arguments.length);
    } else {
      return fn.apply(this, arguments);
    }
  };
}
```

我们验证下这个函数：

```javascript
var fn = curry(function(a, b, c) {
  return [a, b, c];
});

fn("a", "b", "c") // ["a", "b", "c"]
fn("a", "b")("c") // ["a", "b", "c"]
fn("a")("b")("c") // ["a", "b", "c"]
fn("a")("b", "c") // ["a", "b", "c"]
```

效果已经达到我们的预期，然而这个 curry 函数的实现好难理解呐……

为了让大家更好的理解这个 curry 函数，我给大家写个极简版的代码：

```javascript
function sub_curry(fn){
  return function(){
    return fn()
  }
}

function curry(fn, length){
  length = length || 4;
  return function(){
    if (length > 1) {
      return curry(sub_curry(fn), --length)
    }
    else {
      return fn()
    }
  }
}

var fn0 = function(){
  console.log(1)
}

var fn1 = curry(fn0)
fn1()()()() // 1
```

大家先从理解这个 curry 函数开始。

当执行 fn1() 时，函数返回：

```javascript
curry(sub_curry(fn0))
// 相当于
curry(function(){
  return fn0()
})
```

当执行 fn1()() 时，函数返回：

```javascript
curry(sub_curry(function(){
  return fn0()
}))
// 相当于
curry(function(){
  return (function(){
    return fn0()
  })()
})
// 相当于
curry(function(){
  return fn0()
})
```

## 更多应用

柯里化的应用远不止这些，它还可以用于函数组合、惰性求值等。在函数式编程中，柯里化是一个非常重要的概念。

举个例子，我们可以使用柯里化来实现函数组合：

```javascript
function compose(f, g) {
  return function(x) {
    return f(g(x));
  };
}

function add1(x) {
  return x + 1;
}

function multiply2(x) {
  return x * 2;
}

var add1Multiply2 = compose(multiply2, add1);
add1Multiply2(5) // 12
```

这个例子中，我们使用 compose 函数将 add1 和 multiply2 组合成一个新的函数 add1Multiply2，它先执行 add1，再执行 multiply2。

## 总结

柯里化是一种将使用多个参数的一个函数转换成一系列使用一个参数的函数的技术。它可以用于参数复用、函数组合等场景。

希望通过本文的介绍，大家能够理解柯里化的概念和用途，并在实际开发中灵活运用。

如果你还有其他疑问，欢迎在评论区留言，我会尽量为大家解答。