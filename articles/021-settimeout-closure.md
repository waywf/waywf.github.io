---
title: 深入JavaScript系列-setTimeout与循环闭包面试题详解
excerpt: 深入解析setTimeout与循环闭包的面试题
category: 技术
date: 2026-02-25
readTime: 15
tags: JavaScript, 底层系列
---

## 前言

我在闭包一文中的结尾留下了一个关于setTimeout与循环闭包的思考题。利用闭包，修改下面的代码，让循环输出的结果依次为1， 2， 3， 4， 5

```javascript
for (var i = 1; i <= 5; i++) {
  setTimeout(function timer() {
    console.log(i);
  }, i * 1000);
}
```

值得高兴的是，很多朋友在阅读了我的文章之后确实对闭包有了更加深刻的了解，并准确的给出了好几种写法。大家能够认真的阅读我的文章并且一个例子一个例子的上手练习，这种认可对我而言真的非常感动。但是也有一些基础稍差的朋友在阅读了之后，对于这题的理解仍然感到困惑，因此应一些读者老爷的要求，借此文章专门对setTimeout进行一个相关的知识分享，希望大家读完之后都能够有新的收获。

## setTimeout基础

初学setTimeout，我们很容易知道setTimeout有两个参数，第一个参数为一个函数，我们通过该函数定义将要执行的操作。第二个参数为一个时间毫秒数，表示延迟执行的时间。

```javascript
setTimeout(function() {
  console.log('一秒钟之后我将被打印出来')
}, 1000)
```

执行结果如图

![setTimeout执行结果](image)

可能不少同学对于setTimeout的理解止步于此，但还是有不少人发现了一些其他的东西，并在评论里提出了疑问。比如上图中的这个数字7，是什么？每一个setTimeout在执行时，会返回一个唯一ID，上图中的数字7，就是这个唯一ID。我们在使用时，常常会使用一个变量将这个唯一ID保存起来，用以传入clearTimeout，清除定时器。

## setTimeout执行时机

接下来，我们还需要考虑另外一个重要的问题，那就是setTimeout中定义的操作，在什么时候执行？为了引起大家的重视，我们来看看下面的例子。

```javascript
var timer = setTimeout(function() {
  console.log('setTimeout actions.');
}, 0);
console.log('other actions.');
```

思考一下，当我将setTimeout的延迟时间设置为0时，上面的执行顺序会是什么？在浏览器中的console中运行试试看，很容易就能够知道答案，如果你没有猜中答案，那么我这篇文章就值得你点一个赞了，因为接下来我分享的小知识，可能会在笔试中救你一命。

在对于执行上下文的介绍中，我与大家分享了函数调用栈这种特殊数据结构的调用特性。在这里，将会介绍另外一个特殊的队列结构，页面中所有由setTimeout定义的操作，都将放在同一个队列中依次执行。我用下图跟大家展示一下队列数据结构的特点。

![队列数据结构](image)

队列：先进先出

而这个队列执行的时间，需要等待到函数调用栈清空之后才开始执行。即所有可执行代码执行完毕之后，才会开始执行由setTimeout定义的操作。而这些操作进入队列的顺序，则由设定的延迟时间来决定。更加详细的执行顺序，将会在事件循环的文中中描述

因此在上面这个例子中，即使我们将延迟时间设置为0，它定义的操作仍然需要等待所有代码执行完毕之后才开始执行。这里的延迟时间，并非相对于setTimeout执行这一刻，而是相对于其他代码执行完毕这一刻。所以上面的例子执行结果就非常容易理解了。

## 复杂例子

为了帮助大家理解，再来一个结合变量提升的更加复杂的例子。如果你能够正确看出执行顺序，那么你对于函数的执行就有了比较正确的认识了，如果还不能，就回过头去看看其他几篇文章。

```javascript
setTimeout(function () {
  console.log(a);
}, 0);
var a = 10;
console.log(b);
console.log(fn);
var b = 20;
function fn() {
  setTimeout(function () {
    console.log('setTImeout 10ms.');
  }, 10);
}
fn.toString = function () {
  return 30;
}
console.log(fn);
setTimeout(function () {
  console.log('setTimeout 20ms.');
}, 20);
fn();
```

执行结果如图所示。

![复杂例子执行结果](image)

OK，关于setTimeout就暂时先介绍到这里，我们回过头来看看那个循环闭包的思考题。

## 循环闭包问题

```javascript
for (var i = 1; i <= 5; i++) {
  setTimeout(function timer() {
    console.log(i);
  }, i * 1000);
}
```

如果我们直接这样写，根据setTimeout定义的操作在函数调用栈清空之后才会执行的特点，for循环里定义了5个setTimeout操作。而当这些操作开始执行时，for循环的i值，已经先一步变成了6。因此输出结果总为6。

而我们想要让输出结果依次执行，我们就必须借助闭包的特性，每次循环时，将i值保存在一个闭包中，当setTimeout中定义的操作执行时，则访问对应闭包保存的i值即可。

而我们知道在函数中闭包判定的准则，即执行时是否在内部定义的函数中访问了上层作用域的变量。我们需要包裹一层自执行函数为闭包的形成提供条件。因此，我们只需要2个操作就可以完成题目需求，一是使用自执行函数提供闭包条件，二是传入i值并保存在闭包中。

```javascript
for (var i = 1; i <= 5; i++) {
  (function (i) {
    setTimeout(function timer() {
      console.log(i);
    }, i * 1000);
  })(i)
}
```

![闭包执行结果](image)

利用断点调试，在chrome中查看执行顺序与每一个闭包中不同的i

## 其他解法

当然，除了使用自执行函数，我们还可以使用其他方法来解决这个问题。比如使用let关键字，因为let关键字会在每次循环时创建一个新的变量作用域。

```javascript
for (let i = 1; i <= 5; i++) {
  setTimeout(function timer() {
    console.log(i);
  }, i * 1000);
}
```

这种方法更加简洁，也是ES6中推荐的写法。

## 总结

通过本文的介绍，我们了解了setTimeout的执行时机和闭包的应用。希望大家能够掌握这些知识，并在面试中取得好成绩。

如果你还有其他疑问，欢迎在评论区留言，我会尽量为大家解答。