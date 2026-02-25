---
title: JavaScript深入之从ECMAScript规范解读this
excerpt: 深入解析JavaScript的this关键字
category: 技术
date: 2026-02-25
readTime: 25
tags: JavaScript, 底层系列
---

## 前言

在《JavaScript深入之执行上下文栈》中讲到，当JavaScript代码执行一段可执行代码(executable code)时，会创建对应的执行上下文(execution context)。对于每个执行上下文，都有三个重要属性：

1. 变量对象(Variable object，VO)
2. 作用域链(Scope chain)
3. this

今天重点讲讲 this，然而不好讲。……因为我们要从 ECMASciript5 规范开始讲起。

先奉上 ECMAScript 5.1 规范地址：

## Types

首先是第 8 章 Types：

Types are further subclassified into ECMAScript language types and specification types.

An ECMAScript language type corresponds to values that are directly manipulated by an ECMAScript programmer using the ECMAScript language. The ECMAScript language types are Undefined, Null, Boolean, String, Number, and Object.

A specification type corresponds to meta-values that are used within algorithms to describe the semantics of ECMAScript language constructs and ECMAScript language types. The specification types are Reference, List, Completion, Property Descriptor, Property Identifier, Lexical Environment, and Environment Record.

我们简单的翻译一下：

ECMAScript 的类型分为语言类型和规范类型。

ECMAScript 语言类型是开发者直接使用 ECMAScript 可以操作的。其实就是我们常说的Undefined, Null, Boolean, String, Number, 和 Object。

而规范类型相当于 meta-values，是用来用算法描述 ECMAScript 语言结构和 ECMAScript 语言类型的。规范类型包括：Reference, List, Completion, Property Descriptor, Property Identifier, Lexical Environment, 和 Environment Record。

没懂？没关系，我们只要知道在 ECMAScript 规范中还有一种只存在于规范中的类型，它们的作用是用来描述语言底层行为逻辑。

今天我们要讲的重点是便是其中的 Reference 类型。它与 this 的指向有着密切的关联。

## Reference

那什么又是 Reference ？让我们看 8.7 章 The Reference Specification Type：

The Reference type is used to explain the behaviour of such operators as delete, typeof, and the assignment operators.

所以 Reference 类型就是用来解释诸如 delete、typeof 以及赋值等操作行为的。

抄袭尤雨溪大大的话，就是：

这里的 Reference 是一个 Specification Type，也就是 “只存在于规范里的抽象类型”。它们是为了更好地描述语言的底层行为逻辑才存在的，但并不存在于实际的 js 代码中。

再看接下来的这段具体介绍 Reference 的内容：

A Reference is a resolved name binding.

A Reference consists of three components, the base value, the referenced name and the Boolean valued strict reference flag.

The base value is either undefined, an Object, a Boolean, a String, a Number, or an environment record (10.2.1).

A base value of undefined indicates that the reference could not be resolved to a binding. The referenced name is a String.

这段讲述了 Reference 的构成，由三个组成部分，分别是：

1. base value
2. referenced name
3. strict reference

可是这些到底是什么呢？我们简单的理解的话：

base value 就是属性所在的对象或者就是 EnvironmentRecord，它的值只可能是 undefined, an Object, a Boolean, a String, a Number, or an environment record 其中的一种。

referenced name 就是属性的名称。

举个例子：

```javascript
var foo = 1;

// 对应的Reference是：
var fooReference = {
  base: EnvironmentRecord,
  name: 'foo',
  strict: false
};
```

再举个例子：

```javascript
var foo = {
  bar: function () {
    return this;
  }
};

foo.bar(); // foo

// bar对应的Reference是：
var BarReference = {
  base: foo,
  propertyName: 'bar',
  strict: false
};
```

而且规范中还提供了获取 Reference 组成部分的方法，比如 GetBase 和 IsPropertyReference。这两个方法很简单，简单看一看：

### GetBase

GetBase(V). Returns the base value component of the reference V.

返回 reference 的 base value。

### IsPropertyReference

IsPropertyReference(V). Returns true if either the base value is an object or HasPrimitiveBase(V) is true; otherwise returns false.

简单的理解：如果 base value 是一个对象，就返回true。

## GetValue

除此之外，紧接着在 8.7.1 章规范中就讲了一个用于从 Reference 类型获取对应值的方法： GetValue。

简单模拟 GetValue 的使用：

```javascript
var foo = 1;

var fooReference = {
  base: EnvironmentRecord,
  name: 'foo',
  strict: false
};

GetValue(fooReference) // 1;
```

GetValue 返回对象属性真正的值，但是要注意：调用 GetValue，返回的将是具体的值，而不再是一个 Reference

这个很重要，这个很重要，这个很重要。

关于 Reference 讲了那么多，为什么要讲 Reference 呢？到底 Reference 跟本文的主题 this 有哪些关联呢？

如果你能耐心看完之前的内容，以下开始进入高能阶段：

看规范 11.2.3 Function Calls：

这里讲了当函数调用的时候，如何确定 this 的取值。只看第一步、第六步、第七步：

1. Let ref be the result of evaluating MemberExpression.

6. If Type(ref) is Reference, then  a.If IsPropertyReference(ref) is true, then  i.Let thisValue be GetBase(ref). b.Else, the base value is an Environment Record i.Let thisValue be the result of calling the ImplicitThisValue concrete method of base value.

7. Else, Type(ref) is not Reference. a.Let thisValue be undefined.

8. Let thisBinding be the result of calling BindThisValue(function, thisValue).

我们来分析一下：

1. 计算 MemberExpression 的结果赋值给 ref

2. 判断 ref 是不是一个 Reference 类型

   a. 如果是 Reference 类型，并且 IsPropertyReference(ref) 是 true，那么 this 的值为 GetBase(ref)

   b. 如果是 Reference 类型，并且 IsPropertyReference(ref) 是 false，那么 this 的值为 ImplicitThisValue(ref)，而 ImplicitThisValue 总是返回 undefined

3. 如果不是 Reference 类型，那么 this 的值为 undefined

所以关键就在于：

MemberExpression 计算的结果是不是一个 Reference 类型？

## 什么是 MemberExpression？

看规范 11.2 Left-Hand-Side Expressions：

MemberExpression :

- PrimaryExpression
- FunctionExpression
- MemberExpression [ Expression ]
- MemberExpression . IdentifierName
- new MemberExpression Arguments

简单理解 MemberExpression 就是()左边的部分。

举个例子：

```javascript
foo(); // MemberExpression 是 foo

foo.bar(); // MemberExpression 是 foo.bar

foo['bar'](); // MemberExpression 是 foo['bar']

new foo(); // MemberExpression 是 new foo
```

## 举个例子

### 例子1：

```javascript
function foo() {
  console.log(this);
}

foo(); // undefined
```

MemberExpression 是 foo，foo 是一个标识符，标识符的计算结果是一个 Reference 类型。

```javascript
var fooReference = {
  base: EnvironmentRecord,
  name: 'foo',
  strict: false
};
```

IsPropertyReference(fooReference) 是 false，因为 base value 是 EnvironmentRecord，不是一个对象。

所以 this 的值为 ImplicitThisValue(fooReference)，而 ImplicitThisValue 总是返回 undefined。

所以 this 的值是 undefined。

### 例子2：

```javascript
var foo = {
  bar: function () {
    console.log(this);
  }
};

foo.bar(); // foo
```

MemberExpression 是 foo.bar，foo.bar 是一个属性访问表达式，属性访问表达式的计算结果是一个 Reference 类型。

```javascript
var barReference = {
  base: foo,
  name: 'bar',
  strict: false
};
```

IsPropertyReference(barReference) 是 true，因为 base value 是 foo，是一个对象。

所以 this 的值为 GetBase(barReference)，也就是 foo。

所以 this 的值是 foo。

### 例子3：

```javascript
var foo = {
  bar: function () {
    console.log(this);
  }
};

var bar = foo.bar;

bar(); // undefined
```

MemberExpression 是 bar，bar 是一个标识符，标识符的计算结果是一个 Reference 类型。

```javascript
var barReference = {
  base: EnvironmentRecord,
  name: 'bar',
  strict: false
};
```

IsPropertyReference(barReference) 是 false，因为 base value 是 EnvironmentRecord，不是一个对象。

所以 this 的值为 ImplicitThisValue(barReference)，而 ImplicitThisValue 总是返回 undefined。

所以 this 的值是 undefined。

### 例子4：

```javascript
function foo() {
  console.log(this);
}

var foo = {
  bar: foo
};

foo.bar(); // foo
```

MemberExpression 是 foo.bar，foo.bar 是一个属性访问表达式，属性访问表达式的计算结果是一个 Reference 类型。

```javascript
var barReference = {
  base: foo,
  name: 'bar',
  strict: false
};
```

IsPropertyReference(barReference) 是 true，因为 base value 是 foo，是一个对象。

所以 this 的值为 GetBase(barReference)，也就是 foo。

所以 this 的值是 foo。

### 例子5：

```javascript
function foo() {
  console.log(this);
}

var foo = {
  bar: function () {
    return foo;
  }
};

foo.bar()(); // foo
```

第一个 () 的 MemberExpression 是 foo.bar，this 的值是 foo。

第二个 () 的 MemberExpression 是 foo.bar()，foo.bar() 的结果是 foo，foo 是一个标识符，标识符的计算结果是一个 Reference 类型。

```javascript
var fooReference = {
  base: EnvironmentRecord,
  name: 'foo',
  strict: false
};
```

IsPropertyReference(fooReference) 是 false，因为 base value 是 EnvironmentRecord，不是一个对象。

所以 this 的值为 ImplicitThisValue(fooReference)，而 ImplicitThisValue 总是返回 undefined。

所以 this 的值是 undefined。

### 例子6：

```javascript
function foo() {
  console.log(this);
}

var foo = {
  bar: function () {
    return function () {
      return this;
    };
  }
};

foo.bar()(); // undefined
```

第一个 () 的 MemberExpression 是 foo.bar，this 的值是 foo。

第二个 () 的 MemberExpression 是 foo.bar()，foo.bar() 的结果是 function () { return this; }，这个函数不是一个 Reference 类型，因为它是一个函数表达式，不是一个标识符，也不是一个属性访问表达式。

所以 this 的值是 undefined。

## 总结

通过规范我们可以得出以下结论：

1. 如果 MemberExpression 是一个标识符，那么 this 的值是 undefined

2. 如果 MemberExpression 是一个属性访问表达式，那么 this 的值是属性所在的对象

3. 如果 MemberExpression 是一个函数表达式，那么 this 的值是 undefined

4. 如果 MemberExpression 是一个 new 表达式，那么 this 的值是新创建的对象

5. 如果 MemberExpression 是一个 call 或 apply 表达式，那么 this 的值是指定的对象

这些结论可以帮助我们快速判断 this 的值。

## 思考题

```javascript
var a = 1;

function foo() {
  console.log(this.a);
}

foo(); // undefined

var obj = {
  a: 2,
  foo: foo
};

obj.foo(); // 2

var bar = obj.foo;

bar(); // undefined

foo.call(obj); // 2

foo.apply(obj); // 2

foo.bind(obj)(); // 2

new foo(); // undefined
```

这些例子的 this 值分别是多少呢？请根据我们上面的结论进行判断。

## 总结

this 是 JavaScript 中一个非常重要的概念，也是一个容易混淆的概念。通过从 ECMAScript 规范的角度来解读 this，可以帮助我们更好地理解 this 的指向。

希望通过本文的讲解，大家能够对 this 有一个更深入的理解。