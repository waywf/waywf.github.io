---
title: 08.前端大厂面试题深度解析：阿里腾讯字节美团面试真题一网打尽
category: 前端开发
excerpt: 前端面试题第八弹！100+大厂面试真题，从JavaScript到系统设计，从Vue/React到性能优化。看完这篇，大厂Offer拿到手软！
tags: 前端面试, 大厂面试, JavaScript, Vue, React, 性能优化, 系统设计
date: 2025-07-25
readTime: 70
---

各位前端er，欢迎来到大厂面试专场！这里汇聚了阿里、腾讯、字节、美团等一线大厂的真实面试题，每一道都是经过千锤百炼的经典之作。准备好了吗？让我们一起挑战这些大厂面试题！

## 一、JavaScript 基础篇

### 1.1 数据类型相关

**面试题1**：JavaScript 有哪些数据类型？如何准确判断？

**答案**：
- 基本类型：String、Number、Boolean、Null、Undefined、Symbol、BigInt
- 引用类型：Object、Array、Function、Date、RegExp等
- 判断方法：
  - typeof：判断基本类型（null返回object，function返回function）
  - instanceof：判断引用类型
  - Object.prototype.toString.call()：最准确的判断方法

**面试题2**：null 和 undefined 的区别？

**答案**：
- null：表示"没有对象"，即此处不应该有值
- undefined：表示"缺少值"，即此处应该有值但还未定义
- 用途不同：null常用于清空对象引用，undefined是变量未赋值的默认值

**面试题3**：== 和 === 的区别？

**答案**：
- ==：抽象相等，会进行类型转换
- ===：严格相等，不进行类型转换
- 建议：优先使用 ===，避免类型转换带来的意外

### 1.2 原型与原型链

**面试题4**：什么是原型链？

**答案**：
- 每个对象都有__proto__属性指向它的原型对象
- 每个构造函数都有prototype属性指向原型对象
- 原型对象也有自己的原型，层层向上直到null，形成原型链
- 属性查找时会沿着原型链向上查找

**面试题5**：如何实现继承？

**答案**：
- 原型链继承：Child.prototype = new Parent()
- 构造函数继承：Parent.call(this, ...args)
- 组合继承：原型链+构造函数
- 原型式继承：Object.create()
- 寄生式继承
- 寄生组合式继承（最完美）

**面试题6**：new 操作符做了什么？

**答案**：
1. 创建一个新对象
2. 将新对象的__proto__指向构造函数的prototype
3. 将构造函数的this指向新对象
4. 执行构造函数代码
5. 如果构造函数返回对象则返回该对象，否则返回新对象

### 1.3 作用域与闭包

**面试题7**：什么是闭包？闭包的用途？

**答案**：
- 闭包：能够访问自由变量的函数
- 用途：
  - 数据私有化
  - 函数柯里化
  - 模块封装
  - 保持状态

**面试题8**：什么是作用域链？

**答案**：
- 每个函数执行时会创建执行上下文
- 执行上下文包含变量对象、作用域链、this
- 作用域链由当前函数变量对象和父级作用域链组成
- 变量查找时沿着作用域链向上查找

**面试题9**：var、let、const 的区别？

**答案**：
- var：函数作用域，变量提升，可重复声明
- let：块级作用域，无变量提升，不可重复声明
- const：块级作用域，无变量提升，不可重复声明，声明后不能重新赋值

### 1.4 this 指向

**面试题10**：this 的指向规则？

**答案**：
1. 默认绑定：函数独立调用，this指向全局对象（严格模式undefined）
2. 隐式绑定：作为对象方法调用，this指向该对象
3. 显式绑定：call/apply/bind，this指向指定对象
4. new绑定：new构造函数，this指向新创建的对象
5. 箭头函数：继承外层作用域的this

**面试题11**：箭头函数和普通函数的区别？

**答案**：
- 语法更简洁
- 没有自己的this，继承外层this
- 不能作为构造函数
- 没有arguments对象
- 没有prototype属性

### 1.5 异步编程

**面试题12**：Event Loop 的理解？

**答案**：
- 调用栈：同步任务执行
- 任务队列：异步任务等待
- 宏任务：setTimeout、setInterval、I/O、DOM事件
- 微任务：Promise.then/catch/finally、async/await、MutationObserver
- 执行顺序：同步任务 → 微任务 → 宏任务

**面试题13**：Promise 的理解？

**答案**：
- Promise是异步编程的解决方案
- 三种状态：pending、fulfilled、rejected
- 状态一旦改变就不可逆转
- 链式调用：then/catch/finally
- 静态方法：all、race、allSettled、any、resolve、reject

**面试题14**：async/await 的理解？

**答案**：
- async函数返回Promise
- await只能在async函数中使用
- await后面跟Promise，会暂停执行直到Promise解决
- async/await让异步代码看起来像同步代码
- 错误处理用try/catch

### 1.6 ES6+ 新特性

**面试题15**：let/const 和 var 的区别？（前面已问过，换角度）

**答案**：
- 块级作用域 vs 函数/全局作用域
- 暂时性死区 vs 变量提升
- 不可重复声明 vs 可重复声明
- const声明常量（引用类型可修改属性）

**面试题16**：解构赋值的使用场景？

**答案**：
- 对象解构：从对象中提取属性
- 数组解构：从数组中提取元素
- 函数参数解构
- 默认值设置
- 交换变量值

**面试题17**：扩展运算符的用途？

**答案**：
- 数组展开：[...arr1, ...arr2]
- 对象展开：{...obj1, ...obj2}
- 函数参数：fn(...args)
- 数组浅拷贝
- 对象浅拷贝
- 将类数组转为数组

**面试题18**：Set 和 Map 的区别？

**答案**：
- Set：类似数组，成员唯一，无重复值
- Map：键值对集合，键可以是任意类型
- Set用于去重、交集、并集等
- Map用于字典、缓存等场景

**面试题19**：Symbol 的用途？

**答案**：
- 表示独一无二的值
- 作为对象属性名，避免属性名冲突
- 定义私有属性
- 定义迭代器接口（Symbol.iterator）

**面试题20**：Proxy 的理解？

**答案**：
- 用于定义对象的自定义行为
- 可以拦截对象的各种操作
- 常用于数据劫持、验证、日志等
- Vue3响应式系统的核心

### 1.7 数组与对象

**面试题21**：数组去重的方法？

**答案**：
- [...new Set(arr)]
- arr.filter((item, index) => arr.indexOf(item) === index)
- 利用Map或Object
- 遍历判断

**面试题22**：数组扁平化的方法？

**答案**：
- arr.flat(Infinity)
- 递归+concat
- reduce+递归
- toString+split
- 正则替换

**面试题23**：深拷贝和浅拷贝的区别？

**答案**：
- 浅拷贝：只拷贝第一层，嵌套对象仍共享引用
- 深拷贝：完全拷贝，嵌套对象也独立
- 浅拷贝：Object.assign、展开运算符、slice、concat
- 深拷贝：JSON.parse(JSON.stringify())、递归、lodash.cloneDeep

**面试题24**：Object.keys()、Object.values()、Object.entries()？

**答案**：
- Object.keys()：返回键数组
- Object.values()：返回值数组
- Object.entries()：返回键值对数组
- 都只返回可枚举属性
- 不包括Symbol属性

### 1.8 函数相关

**面试题25**：函数柯里化的理解？

**答案**：
- 将多参数函数转为单参数函数
- 可以延迟执行
- 参数复用
- 函数式编程常用技巧

**面试题26**：函数防抖和节流的区别？

**答案**：
- 防抖：n秒内只执行一次，重新触发则重新计时
- 节流：n秒内只执行一次，不管触发多少次
- 防抖适用于搜索框输入
- 节流适用于滚动、resize等

**面试题27**：高阶函数的理解？

**答案**：
- 接受函数作为参数
- 或返回函数的函数
- 常见：map、filter、reduce、Promise.then
- 函数式编程核心

### 1.9 正则表达式

**面试题28**：正则表达式的常用方法？

**答案**：
- RegExp：test、exec
- String：match、matchAll、replace、search、split
- test：测试是否匹配，返回布尔值
- exec：返回匹配结果数组

**面试题29**：如何验证邮箱/手机号？

**答案**：
- 邮箱：/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
- 手机号：/^1[3-9]\d{9}$/
- 注意：正则不是万能的，根据需求调整

**面试题30**：正则的贪婪和非贪婪匹配？

**答案**：
- 贪婪：尽可能多匹配（*、+、?、{n,}）
- 非贪婪：尽可能少匹配（*?、+?、??、{n,}?）
- 加?变成非贪婪

## 二、浏览器与DOM篇

### 2.1 浏览器原理

**面试题31**：从输入URL到页面显示发生了什么？

**答案**：
1. DNS解析：域名转IP
2. TCP连接：三次握手
3. 发送HTTP请求
4. 服务器处理并返回响应
5. 浏览器解析HTML构建DOM树
6. 解析CSS构建CSSOM树
7. 合并生成渲染树
8. 布局（Layout）
9. 绘制（Paint）
10. 合成（Composite）

**面试题32**：浏览器的渲染流程？

**答案**：
- HTML → DOM树
- CSS → CSSOM树
- DOM + CSSOM → 渲染树
- 布局：计算元素位置和大小
- 绘制：将元素绘制到屏幕
- 合成：将多层合并显示

**面试题33**：重排（Reflow）和重绘（Repaint）的区别？

**答案**：
- 重排：元素几何属性改变，需要重新计算布局
- 重绘：元素外观改变但几何属性不变
- 重排一定触发重绘，重绘不一定触发重排
- 重排开销更大

**面试题34**：如何减少重排和重绘？

**答案**：
- 避免频繁操作DOM
- 使用DocumentFragment
- 批量修改样式
- 使用transform代替top/left
- 使用will-change
- 避免读取offset等属性后立即修改

**面试题35**：浏览器缓存机制？

**答案**：
- 强缓存：Cache-Control、Expires
- 协商缓存：Last-Modified/If-Modified-Since、ETag/If-None-Match
- 缓存优先级：强缓存 > 协商缓存
- Service Worker缓存

**面试题36**：Cookie、SessionStorage、LocalStorage的区别？

**答案**：
- Cookie：大小4KB，随请求发送，可设置过期时间
- SessionStorage：大小5-10MB，会话级，页面关闭清除
- LocalStorage：大小5-10MB，持久化，需手动清除
- Web Storage不随请求发送

### 2.2 DOM操作

**面试题37**：DOM事件流？

**答案**：
- 捕获阶段：从window到目标元素
- 目标阶段：到达目标元素
- 冒泡阶段：从目标元素到window
- addEventListener第三个参数控制捕获/冒泡

**面试题38**：事件委托的理解？

**答案**：
- 将事件绑定在父元素上
- 利用事件冒泡机制
- 减少事件绑定数量
- 可以处理动态添加的元素

**面试题39**：如何阻止事件冒泡和默认行为？

**答案**：
- 阻止冒泡：e.stopPropagation()、e.cancelBubble = true
- 阻止默认：e.preventDefault()、return false
- return false在jQuery中同时阻止两者

**面试题40**：DOM查询方法？

**答案**：
- getElementById
- getElementsByTagName
- getElementsByClassName
- querySelector
- querySelectorAll
- closest（向上查找）

### 2.3 性能优化

**面试题41**：前端性能优化有哪些方法？

**答案**：
- 加载优化：减少HTTP请求、使用CDN、缓存
- 资源优化：压缩、图片优化、Tree Shaking
- 渲染优化：减少重排重绘、虚拟列表、懒加载
- 代码优化：避免长任务、使用WebWorker、requestIdleCallback

**面试题42**：图片优化有哪些方法？

**答案**：
- 使用WebP/AVIF等现代格式
- 懒加载
- 响应式图片（srcset）
- 图片压缩
- 雪碧图
- 占位图
- 渐进式图片

**面试题43**：首屏加载优化？

**答案**：
- 代码分割
- 路由懒加载
- 预加载重要资源
- SSR/SSG
- 减少关键路径资源
- 使用骨架屏

**面试题44**：WebWorker的理解？

**答案**：
- 浏览器提供的多线程方案
- 可以执行耗时计算不阻塞主线程
- 不能操作DOM
- 通过postMessage通信
- 适用于大数据计算、复杂运算

### 2.4 安全相关

**面试题45**：XSS攻击的理解和防范？

**答案**：
- 跨站脚本攻击
- 攻击者注入恶意脚本
- 防范：
  - 转义输出
  - Content Security Policy
  - HttpOnly Cookie
  - 输入验证

**面试题46**：CSRF攻击的理解和防范？

**答案**：
- 跨站请求伪造
- 利用用户登录状态发起请求
- 防范：
  - CSRF Token
  - SameSite Cookie
  - 验证Referer
  - 双重Cookie验证

**面试题47**：CORS的理解？

**答案**：
- 跨域资源共享
- 浏览器的同源策略
- 简单请求 vs 预检请求
- Access-Control-Allow-Origin等响应头
- withCredentials携带Cookie

## 三、网络篇

### 3.1 HTTP/HTTPS

**面试题48**：HTTP和HTTPS的区别？

**答案**：
- HTTP：明文传输，80端口，不安全
- HTTPS：HTTP+SSL/TLS，443端口，加密传输
- HTTPS需要证书
- HTTPS性能略低但更安全

**面试题49**：HTTP1.1 vs HTTP2 vs HTTP3？

**答案**：
- HTTP1.1：持久连接、管道化（有问题）、队头阻塞
- HTTP2：二进制分帧、多路复用、头部压缩、服务器推送
- HTTP3：基于QUIC、无队头阻塞、更快握手、连接迁移

**面试题50**：常见的HTTP状态码？

**答案**：
- 2xx：成功（200 OK、204 No Content、206 Partial Content）
- 3xx：重定向（301永久、302临时、304协商缓存）
- 4xx：客户端错误（400请求错误、401未授权、403禁止、404未找到）
- 5xx：服务器错误（500服务器错误、502网关错误、503服务不可用）

**面试题51**：HTTP请求方法有哪些？

**答案**：
- GET：获取资源
- POST：提交资源
- PUT：更新资源（完整替换）
- PATCH：更新资源（部分修改）
- DELETE：删除资源
- HEAD：获取响应头
- OPTIONS：获取允许的方法

**面试题52**：GET和POST的区别？

**答案**：
- GET：参数在URL，有长度限制，可缓存，幂等
- POST：参数在请求体，无长度限制，不可缓存，不幂等
- GET用于查询，POST用于提交
- 安全性都是相对的

### 3.2 TCP/IP

**面试题53**：TCP三次握手？

**答案**：
1. 客户端发送SYN，序列号x
2. 服务器返回SYN+ACK，序列号y，确认号x+1
3. 客户端发送ACK，确认号y+1
- 原因：防止已失效的连接请求到达服务器

**面试题54**：TCP四次挥手？

**答案**：
1. 客户端发送FIN，序列号x
2. 服务器返回ACK，确认号x+1
3. 服务器发送FIN，序列号y
4. 客户端返回ACK，确认号y+1
- 原因：TCP是全双工，双方都需要确认关闭

**面试题55**：TCP和UDP的区别？

**答案**：
- TCP：面向连接、可靠传输、流量控制、拥塞控制
- UDP：无连接、不可靠、速度快
- TCP用于HTTP、FTP等
- UDP用于视频、直播、DNS等

### 3.3 其他网络知识

**面试题56**：DNS解析过程？

**答案**：
1. 浏览器缓存
2. 操作系统缓存
3. hosts文件
4. 本地DNS服务器
5. 根域名服务器
6. 顶级域名服务器
7. 权威域名服务器

**面试题57**：WebSocket的理解？

**答案**：
- 全双工通信
- 基于TCP，HTTP握手升级
- 实时通信
- 数据格式轻量
- 适用场景：聊天、实时数据、协作编辑

**面试题58**：从浏览器角度看输入URL的全过程？（前面已问，换更详细）

**答案**：
1. URL解析
2. DNS查询
3. TCP连接
4. TLS握手（HTTPS）
5. 发送HTTP请求
6. 服务器处理
7. 返回响应
8. 浏览器解析
9. 渲染页面
10. 执行JavaScript

## 四、Vue 篇

### 4.1 Vue2 vs Vue3

**面试题59**：Vue2和Vue3的区别？

**答案**：
- 响应式：Object.defineProperty → Proxy
- API：Options API → Composition API
- 生命周期：beforeDestroy/updated → beforeUnmount/updated
- 性能：更快、更小
- TypeScript：更好的支持
- 新特性：Teleport、Suspense、Fragment

**面试题60**：Vue3的Composition API？

**答案**：
- setup函数
- ref/reactive定义响应式数据
- computed/watch/watchEffect
- 更好的逻辑复用
- 更好的TypeScript支持
- 更灵活的代码组织

### 4.2 响应式原理

**面试题61**：Vue2响应式原理？

**答案**：
- Object.defineProperty
- 拦截get/set
- 依赖收集（get）
- 派发更新（set）
- 数组通过重写方法实现
- 无法监听新增/删除属性

**面试题62**：Vue3响应式原理？

**答案**：
- Proxy代理对象
- Reflect操作对象
- 可以监听数组
- 可以监听新增/删除属性
- 懒代理，性能更好
- 深层响应式

**面试题63**：nextTick的理解？

**答案**：
- 在下次DOM更新循环结束后执行回调
- 利用微任务
- Promise优先，降级到MutationObserver、setImmediate、setTimeout
- 用于获取更新后的DOM

### 4.3 组件相关

**面试题64**：Vue组件通信方式？

**答案**：
- 父子：props/$emit
- 父子：v-model
- 父子：provide/inject
- 兄弟：EventBus
- 跨级：Vuex/Pinia
- 跨级：$attrs/$listeners
- ref/$parent/$children

**面试题65**：v-model的原理？

**答案**：
- Vue2：value + input事件
- Vue3：modelValue + update:modelValue
- 语法糖
- 可以自定义

**面试题66**：computed和watch的区别？

**答案**：
- computed：计算属性，有缓存，依赖不变不重新计算
- watch：监听数据变化，执行回调
- computed用于数据转换
- watch用于异步或开销大的操作

**面试题67**：Vue生命周期？

**答案**：
- Vue2：beforeCreate、created、beforeMount、mounted、beforeUpdate、updated、beforeDestroy、destroyed
- Vue3：beforeCreate、created、beforeMount、mounted、beforeUpdate、updated、beforeUnmount、unmounted
- setup替代beforeCreate和created
- onMounted等组合式API

### 4.4 路由相关

**面试题68**：Vue Router的模式？

**答案**：
- hash模式：#后面的内容，不刷新页面
- history模式：利用history.pushState，需要服务器配置
- abstract模式：不依赖浏览器，用于Node等环境

**面试题69**：路由守卫有哪些？

**答案**：
- 全局守卫：beforeEach、beforeResolve、afterEach
- 路由独享守卫：beforeEnter
- 组件内守卫：beforeRouteEnter、beforeRouteUpdate、beforeRouteLeave

**面试题70**：$route和$router的区别？

**答案**：
- $route：路由信息对象，包含path、params、query等
- $router：路由实例，包含push、replace、go等方法
- 一个是信息，一个是实例

### 4.5 状态管理

**面试题71**：Vuex的核心概念？

**答案**：
- State：状态
- Getters：计算状态
- Mutations：同步修改状态
- Actions：异步操作，提交mutation
- Modules：模块化

**面试题72**：Pinia和Vuex的区别？

**答案**：
- Pinia：更简单，无mutations，更好的TypeScript支持，更轻量
- Vuex：更复杂，有mutations，更成熟
- Pinia是Vue3推荐的状态管理
- 都可以用于Vue2/Vue3

### 4.6 性能优化

**面试题73**：Vue性能优化？

**答案**：
- v-for加key
- v-if和v-for不要一起用
- 计算属性缓存
- v-once
- 组件懒加载
- 虚拟列表
- 防抖节流
- keep-alive

**面试题74**：key的作用？

**答案**：
- 标识节点，帮助Diff算法
- 提高Diff效率
- 避免节点复用导致的问题
- 不要用index作为key

### 4.7 虚拟DOM与Diff

**面试题75**：虚拟DOM的理解？

**答案**：
- 用JavaScript对象描述DOM
- 减少真实DOM操作
- 跨平台
- Diff算法比较差异
- 批量更新

**面试题76**：Vue的Diff算法？

**答案**：
- 同层比较
- 双端对比
- key优化
- 时间复杂度O(n)
- Vue2和Vue3有所不同

## 五、React 篇

### 5.1 React基础

**面试题77**：React的核心思想？

**答案**：
- 组件化
- 虚拟DOM
- 单向数据流
- 声明式编程

**面试题78**：JSX的理解？

**答案**：
- JavaScript XML
- 语法糖
- 编译后是React.createElement
- 可以嵌入表达式
- 必须有一个根元素

**面试题79**：React组件的类型？

**答案**：
- 类组件
- 函数组件
- 无状态组件
- 有状态组件
- 高阶组件
- 渲染属性组件

### 5.2 Hooks

**面试题80**：useState的理解？

**答案**：
- 在函数组件中添加state
- 返回state和setter
- setter可以接受函数
- 每次渲染都是独立的闭包

**面试题81**：useEffect的理解？

**答案**：
- 处理副作用
- 替代componentDidMount等生命周期
- 第二个参数是依赖数组
- 返回清理函数
- 多个useEffect可以分离关注点

**面试题82**：useCallback和useMemo的区别？

**答案**：
- useCallback：缓存函数
- useMemo：缓存计算结果
- 都用于性能优化
- 避免不必要的重新渲染

**面试题83**：useRef的理解？

**答案**：
- 保存可变值
- 不会触发重新渲染
- 用于DOM引用
- 用于保存前一个值

**面试题84**：useContext的理解？

**答案**：
- 消费Context
- 避免props层层传递
- 配合createContext使用
- Provider提供值
- useContext消费值

**面试题85**：自定义Hook的理解？

**答案**：
- 复用状态逻辑
- 以use开头
- 可以调用其他Hook
- 不共享状态
- 逻辑复用的最佳方式

### 5.3 组件通信

**面试题86**：React组件通信方式？

**答案**：
- 父子：props/callback
- 跨级：Context
- 状态管理：Redux、MobX、Zustand
- 事件订阅
- ref

**面试题87**：props和state的区别？

**答案**：
- props：父组件传入，只读
- state：组件内部状态，可修改
- props变化触发更新
- state变化触发更新
- props用于外部数据，state用于内部数据

### 5.4 性能优化

**面试题88**：React性能优化？

**答案**：
- useCallback/useMemo
- React.memo
- useTransition
- useDeferredValue
- 虚拟列表
- 代码分割
- 避免不必要的渲染
- 减少re-render

**面试题89**：React.memo的理解？

**答案**：
- 高阶组件
- 浅比较props
- 避免不必要的重新渲染
- 类组件用PureComponent
- 配合useCallback/useMemo使用

**面试题90**：key的作用？（和Vue类似）

**答案**：
- 标识节点
- 帮助Diff算法
- 提高更新效率
- 不要用index

### 5.5 虚拟DOM与Diff

**面试题91**：React的Diff算法？

**答案**：
- 同层比较
- 类型不同直接替换
- 类型相同比较属性
- 列表用key优化
- 时间复杂度O(n)

**面试题92**：Fiber架构的理解？

**答案**：
- React16引入
- 可中断的渲染
- 时间切片
- 优先级调度
- 更好的用户体验

### 5.6 状态管理

**面试题93**：Redux的核心概念？

**答案**：
- Store：存储状态
- Action：描述发生了什么
- Reducer：纯函数，更新状态
- Dispatch：派发action
- Middleware：处理异步

**面试题94**：Redux和MobX的区别？

**答案**：
- Redux：单向数据流，显式更新，函数式，可预测
- MobX：响应式，隐式更新，面向对象，简单
- Redux更适合大型应用
- MobX更适合中小型应用

**面试题95**：Zustand的理解？

**答案**：
- 轻量级状态管理
- 简单易用
- 无Provider
- TypeScript友好
- React推荐的轻量级方案

### 5.7 React Router

**面试题96**：React Router v6的变化？

**答案**：
- 组件变化：Switch → Routes
- 路由写法变化
- useNavigate替代useHistory
- 嵌套路由简化
- 相对路径
- 更好的TypeScript支持

**面试题97**：路由懒加载？

**答案**：
- React.lazy
- Suspense
- 动态import
- 减少首屏加载
- 提高性能

## 六、TypeScript 篇

### 6.1 基础类型

**面试题98**：TypeScript的基本类型？

**答案**：
- string、number、boolean
- null、undefined
- array、object、tuple
- enum、any、void、never
- unknown、symbol、bigint

**面试题99**：any和unknown的区别？

**答案**：
- any：任意类型，绕过类型检查
- unknown：未知类型，需要类型断言或缩小
- unknown更安全
- 推荐用unknown代替any

**面试题100**：interface和type的区别？

**答案**：
- interface：接口，可声明合并，extends继承
- type：类型别名，不能声明合并，&交叉
- interface更适合定义对象
- type更适合定义联合、交叉等

### 6.2 高级类型

**面试题101**：泛型的理解？

**答案**：
- 类型参数
- 增加复用性
- 保持类型安全
- T、U、V常用命名
- 函数、接口、类都可用

**面试题102**：联合类型和交叉类型？

**答案**：
- 联合类型：A | B，或的关系
- 交叉类型：A & B，且的关系
- 联合类型需要类型缩小
- 交叉类型合并所有属性

**面试题103**：类型守卫？

**答案**：
- 缩小类型范围
- typeof
- instanceof
- in
- 自定义类型守卫
- is关键字

**面试题104**：工具类型？

**答案**：
- Partial：所有属性可选
- Required：所有属性必需
- Readonly：所有属性只读
- Pick：选取部分属性
- Omit：排除部分属性
- Record：构造对象类型
- Exclude：从联合类型排除
- Extract：从联合类型提取

### 6.3 装饰器

**面试题105**：装饰器的理解？

**答案**：
- 注解/装饰类、方法、属性
- 实验性特性
- @语法
- NestJS、Angular常用
- 需要配置experimentalDecorators

## 七、工程化篇

### 7.1 构建工具

**面试题106**：Webpack的理解？

**答案**：
- 模块打包器
- 入口、出口、Loader、Plugin
- 模块热替换
- 代码分割
- Tree Shaking
- 插件生态丰富

**面试题107**：Vite的理解？

**答案**：
- 新一代构建工具
- 开发服务器快
- 基于ESBuild
- 预构建依赖
- 生产用Rollup
- 原生ESM支持

**面试题108**：Loader和Plugin的区别？

**答案**：
- Loader：转换模块，处理文件
- Plugin：扩展功能，处理打包过程
- Loader在module.rules配置
- Plugin在plugins数组配置

**面试题109**：Tree Shaking的理解？

**答案**：
- 删除未使用的代码
- 基于ES模块
- 依赖静态分析
- production模式启用
- 配合sideEffects

### 7.2 模块化

**面试题110**：CommonJS和ES Module的区别？

**答案**：
- CommonJS：require/module.exports，动态，运行时
- ES Module：import/export，静态，编译时
- CommonJS在Node.js使用
- ES Module在浏览器和Node.js使用
- ES Module支持Tree Shaking

**面试题111**：AMD和CMD的区别？

**答案**：
- AMD：RequireJS，依赖前置，提前执行
- CMD：SeaJS，依赖就近，延迟执行
- 都是浏览器端模块化方案
- 现在基本不用了

### 7.3 代码规范

**面试题112**：ESLint的理解？

**答案**：
- 代码检查工具
- 规则可配置
- 自动修复
- 统一代码风格
- 配合Prettier

**面试题113**：Prettier的理解？

**答案**：
- 代码格式化工具
-  Opinionated
- 支持多种语言
- 自动格式化
- 配合ESLint

### 7.4 测试

**面试题114**：前端测试的类型？

**答案**：
- 单元测试：测试函数、组件
- 集成测试：测试多个模块
- E2E测试：测试完整流程
- 快照测试：UI对比

**面试题115**：Jest的理解？

**答案**：
- Facebook出品的测试框架
- 内置断言库
- 快照测试
- Mock功能
- 零配置

**面试题116**：Testing Library的理解？

**答案**：
- React Testing Library
- 从用户角度测试
- 不测试实现细节
- 更可靠的测试
- 最佳实践

## 八、系统设计篇

### 8.1 前端架构

**面试题117**：如何设计一个可扩展的前端架构？

**答案**：
- 模块化设计
- 组件库建设
- 状态管理
- 路由设计
- API层封装
- 工具函数
- 代码规范
- 自动化流程

**面试题118**：微前端的理解？

**答案**：
- 将大型应用拆分为多个小应用
- 独立开发、独立部署
- 技术栈无关
- 框架：Single SPA、qiankun、MicroApp
- 适用大型团队、遗留系统

**面试题119**：组件库设计原则？

**答案**：
- 单一职责
- 可复用
- 可配置
- 文档完善
- TypeScript支持
- 测试覆盖
- 性能优化

### 8.2 性能优化

**面试题120**：如何做性能监控？

**答案**：
- 首屏时间
- 白屏时间
- FP、FCP、LCP
- FID、TTI、CLS
- 错误监控
- 用户行为监控
- 工具：Performance API、Lighthouse、Web Vitals

**面试题121**：如何做错误监控？

**答案**：
- try/catch
- window.onerror
- unhandledrejection
- 错误上报
- 错误分析
- 工具：Sentry、Fundebug

### 8.3 其他

**面试题122**：如何设计一个图片上传组件？

**答案**：
- 支持多选
- 拖拽上传
- 预览
- 裁剪
- 压缩
- 分片上传
- 断点续传
- 上传进度
- 错误处理

**面试题123**：如何设计一个表格组件？

**答案**：
- 固定表头
- 虚拟滚动
- 排序
- 筛选
- 分页
- 列宽调整
- 行选择
- 编辑功能
- 导出功能

**面试题124**：如何设计一个表单组件？

**答案**：
- 双向绑定
- 表单验证
- 联动
- 动态字段
- 文件上传
- 提交
- 重置
- 错误提示
- 布局

## 九、算法与数据结构篇

### 9.1 数组

**面试题125**：数组去重？

**面试题126**：数组扁平化？

**面试题127**：数组交集、并集、差集？

**面试题128**：数组旋转？

**面试题129**：两数之和？

### 9.2 字符串

**面试题130**：反转字符串？

**面试题131**：判断回文？

**面试题132**：最长公共前缀？

**面试题133**：无重复字符的最长子串？

### 9.3 链表

**面试题134**：反转链表？

**面试题135**：判断环形链表？

**面试题136**：合并两个有序链表？

**面试题137**：删除链表倒数第n个节点？

### 9.4 树

**面试题138**：二叉树前中后序遍历？

**面试题139**：二叉树层序遍历？

**面试题140**：二叉树最大深度？

**面试题141**：对称二叉树？

### 9.5 排序

**面试题142**：快速排序？

**面试题143**：归并排序？

**面试题144**：冒泡排序？

**面试题145**：插入排序？

### 9.6 其他

**面试题146**：LRU缓存？

**面试题147**：实现Promise？

**面试题148**：实现防抖节流？

**面试题149**：实现深拷贝？

**面试题150**：实现发布订阅？

## 十、HR/软技能篇

### 10.1 自我介绍

**面试题151**：自我介绍一下？

**答案**：
- 姓名、工作年限
- 技术栈
- 项目经验
- 亮点/优势
- 职业规划
- 简洁有力，3-5分钟

### 10.2 项目相关

**面试题152**：介绍一下你做过的项目？

**答案**：
- 项目背景
- 你的职责
- 技术难点
- 解决方案
- 项目成果
- STAR法则

**面试题153**：项目中遇到的最大困难是什么？

**答案**：
- 真实困难
- 你如何解决
- 学到了什么
- 不要抱怨，积极向上

### 10.3 职业规划

**面试题154**：你的职业规划是什么？

**答案**：
- 短期：1-2年，技术深耕
- 中期：3-5年，技术专家/架构师
- 长期：5年以上，技术负责人
- 结合公司发展

**面试题155**：为什么想加入我们公司？

**答案**：
- 公司文化认可
- 技术氛围
- 发展机会
- 产品/业务
- 真诚，不要太虚伪

### 10.4 其他

**面试题156**：你的优点/缺点是什么？

**答案**：
- 优点：举例说明
- 缺点：不要致命缺点，说可以改进的
- 诚恳，不夸大

**面试题157**：你有什么想问我的？

**答案**：
- 团队情况
- 技术栈
- 开发流程
- 晋升机制
- 不要问薪资福利（最后问）
- 不要问百度能查到的

## 总结

恭喜你！这150+道大厂面试题已经全部看完了！这些题目覆盖了前端开发的方方面面，从基础到高级，从技术到软技能，应有尽有。

记住：
1. **基础要牢**：JavaScript、浏览器、网络这些基础一定要扎实
2. **原理要懂**：不要只知其然，要知其所以然
3. **项目要熟**：准备2-3个亮点项目，能用STAR法则讲清楚
4. **算法要练**：LeetCode Hot 100刷起来
5. **心态要好**：面试是双向选择，不要紧张

最后，祝大家都能拿到心仪的大厂Offer！我们下一篇再见！
