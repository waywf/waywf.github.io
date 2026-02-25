---
title: 一道js面试题引发的思考
excerpt: 深入解析一道js面试题
category: 技术
date: 2026-02-25
readTime: 30
tags: JavaScript, 底层系列
---

## 前言

前阵子帮部门面试一前端，看了下面试题(年轻的时候写后端java所以没做过前端试题)，其中有一道题是这样的

比较下面两段代码，试述两段代码的不同之处

```javascript
// A--------------------------
var scope = "global scope";
function checkscope(){
  var scope = "local scope";
  function f(){
    return scope;
  }
  return f();
}
checkscope();
```

```javascript
// B---------------------------
var scope = "global scope";
function checkscope(){
  var scope = "local scope";
  function f(){
    return scope;
  }
  return f;
}
checkscope()();
```

首先A、B两段代码输出返回的都是 "local scope"，如果对这一点还有疑问的同学请自觉回去温习一下js作用域的相关知识。。

那么既然输出一样那这两段代码具体的差异在哪呢？大部分人会说执行环境和作用域不一样，但根本上是哪里不一样就不是人人都能说清楚了。

前阵子就这个问题重新翻了下js基础跟ecmascript标准，如果我们想要刨根问底给出标准答案，那么我们需要先理解下面几个概念：

## 变量对象(variable object)

原文：Every execution context has associated with it a variable object. Variables and functions declared in the source text are added as properties of the variable object. For function code, parameters are added as properties of the variable object.

简言之就是：每一个执行上下文都会分配一个变量对象(variable object)，变量对象的属性由 变量(variable) 和 函数声明(function declaration) 构成。在函数上下文情况下，参数列表(parameter list)也会被加入到变量对象(variable object)中作为属性。

变量对象与当前作用域息息相关。不同作用域的变量对象互不相同，它保存了当前作用域的所有函数和变量。

这里有一点特殊就是只有 函数声明(function declaration) 会被加入到变量对象中，而 **函数表达式(function expression)**则不会。

看代码：

```javascript
// 函数声明
function a(){} 
console.log(typeof a); // "function"

// 函数表达式
var a = function _a(){};
console.log(typeof a); // "function"
console.log(typeof _a); // "undefined"
```

函数声明的方式下，a会被加入到变量对象中，故当前作用域能打印出 a。

函数表达式情况下，a作为变量会加入到变量对象中，_a作为函数表达式则不会加入，故 a 在当前作用域能被正确找到，_a则不会。

另外，关于变量如何初始化，看这里

## Global Object

当js编译器开始执行的时候会初始化一个Global Object用于关联全局的作用域。对于全局环境而言，global object就是变量对象(variable object)。变量对象对于程序而言是不可读的，只有编译器才有权访问变量对象。在浏览器端，global object被具象成window对象，也就是说 global object === window === 全局环境的variable object。因此global object对于程序而言也是唯一可读的variable object。

## 活动对象(activation object)

原文：When control enters an execution context for function code, an object called the activation object is created and associated with the execution context. The activation object is initialised with a property with name arguments and attributes { DontDelete }. The initial value of this property is the arguments object described below.

The activation object is then used as the variable object for the purposes of variable instantiation.

简言之：当函数被激活，那么一个活动对象(activation object)就会被创建并且分配给执行上下文。活动对象由特殊对象 arguments 初始化而成。随后，他被当做变量对象(variable object)用于变量初始化。

用代码来说明就是：

```javascript
function a(name, age){
  var gender = "male";
  function b(){}
}

a("k",10);
```

a被调用时，在a的执行上下文会创建一个活动对象AO，并且被初始化为 AO = [arguments]。随后AO又被当做变量对象(variable object)VO进行变量初始化,此时 VO = [arguments].concat([name,age,gender,b])。

## 执行环境和作用域链(execution context and scope chain)

### execution context

顾名思义 执行环境/执行上下文。在javascript中，执行环境可以抽象的理解为一个object，它由以下几个属性构成：

```javascript
executionContext：{
    variable object：vars,functions,arguments,
    scope chain: variable object + all parents scopes
    thisValue: context object
}
```

此外在js解释器运行阶段还会维护一个环境栈，当执行流进入一个函数时，函数的环境就会被压入环境栈，当函数执行完后会将其环境弹出，并将控制权返回前一个执行环境。环境栈的顶端始终是当前正在执行的环境。

### scope chain

作用域链，它在解释器进入到一个执行环境时初始化完成并将其分配给当前执行环境。每个执行环境的作用域链由当前环境的变量对象及父级环境的作用域链构成。

作用域链具体是如何构建起来的呢，先上代码：

```javascript
function test(num){
    var a = "2";
    return a+num;
}

test(1);
```

执行过程如下：

1. 全局执行上下文被创建，此时全局执行上下文的scope chain = [global.VO]

2. 全局执行上下文被压入环境栈

3. 执行test(1)，test函数执行上下文被创建，test函数执行上下文的scope chain = [test.AO, global.VO]

4. test函数执行上下文被压入环境栈

5. 执行test函数，返回结果

6. test函数执行上下文被弹出环境栈

7. 全局执行上下文被弹出环境栈

## 回到面试题

现在我们回到面试题，分析A、B两段代码的执行过程。

### 代码A

```javascript
var scope = "global scope";
function checkscope(){
  var scope = "local scope";
  function f(){
    return scope;
  }
  return f();
}
checkscope();
```

执行过程：

1. 全局执行上下文被创建，scope chain = [global.VO]

2. 全局执行上下文被压入环境栈

3. 执行checkscope()，checkscope函数执行上下文被创建，scope chain = [checkscope.AO, global.VO]

4. checkscope函数执行上下文被压入环境栈

5. 执行checkscope函数，创建f函数，f函数的[[scope]] = [checkscope.AO, global.VO]

6. 执行f()，f函数执行上下文被创建，scope chain = [f.AO, checkscope.AO, global.VO]

7. f函数执行上下文被压入环境栈

8. 执行f函数，返回scope值

9. f函数执行上下文被弹出环境栈

10. checkscope函数执行上下文被弹出环境栈

11. 全局执行上下文被弹出环境栈

### 代码B

```javascript
var scope = "global scope";
function checkscope(){
  var scope = "local scope";
  function f(){
    return scope;
  }
  return f;
}
checkscope()();
```

执行过程：

1. 全局执行上下文被创建，scope chain = [global.VO]

2. 全局执行上下文被压入环境栈

3. 执行checkscope()，checkscope函数执行上下文被创建，scope chain = [checkscope.AO, global.VO]

4. checkscope函数执行上下文被压入环境栈

5. 执行checkscope函数，创建f函数，f函数的[[scope]] = [checkscope.AO, global.VO]

6. 返回f函数，checkscope函数执行上下文被弹出环境栈

7. 执行f()，f函数执行上下文被创建，scope chain = [f.AO, checkscope.AO, global.VO]

8. f函数执行上下文被压入环境栈

9. 执行f函数，返回scope值

10. f函数执行上下文被弹出环境栈

11. 全局执行上下文被弹出环境栈

## 差异

通过上面的分析，我们可以看到两段代码的差异在于：

- 代码A中，f函数是在checkscope函数执行时被调用的，此时checkscope函数的执行上下文还在环境栈中
- 代码B中，f函数是在checkscope函数执行完毕后被调用的，此时checkscope函数的执行上下文已经被弹出环境栈

但是由于f函数的[[scope]]属性保存了checkscope函数的活动对象，所以即使checkscope函数的执行上下文被弹出环境栈，f函数依然可以访问到checkscope函数的活动对象，这就是闭包的原理。

## 总结

虽然两段代码的输出结果相同，但是它们的执行过程是不同的。代码A中f函数的执行上下文是在checkscope函数的执行上下文存在时创建的，而代码B中f函数的执行上下文是在checkscope函数的执行上下文被销毁后创建的。

这就是闭包的神奇之处，它可以让函数在其创建的上下文被销毁后依然可以访问到该上下文的变量对象。