---
title: Vue 3 Composition API 入门指南11
excerpt: 深入了解 Vue 3 的 Composition API，学习如何编写更灵活、可复用的组件逻辑。
category: 技术
date: 2026-02-23
readTime: 8
tags: [Vue3, JavaScript, 前端开发]
---

# Vue 3 Composition API 入门指南

Vue 3 引入了 Composition API，这是一种全新的组织组件逻辑的方式。相比 Options API，Composition API 提供了更好的代码复用性和灵活性。

## 什么是 Composition API？

Composition API 是一组低级别的 API，允许我们使用函数来组织组件逻辑。它提供了一种更灵活的方式来组织和复用组件中的逻辑。

### 核心函数

#### 1. `setup()` 函数

```javascript
import { ref, computed } from 'vue'

export default {
  setup() {
    const count = ref(0)
    const doubled = computed(() => count.value * 2)
    
    const increment = () => {
      count.value++
    }
    
    return {
      count,
      doubled,
      increment
    }
  }
}
```

#### 2. `ref()` - 创建响应式引用

```javascript
import { ref } from 'vue'

const count = ref(0)
console.log(count.value) // 0
count.value++
console.log(count.value) // 1
```

#### 3. `reactive()` - 创建响应式对象

```javascript
import { reactive } from 'vue'

const state = reactive({
  count: 0,
  name: 'Vue'
})

state.count++
```

## 生命周期钩子

在 Composition API 中，生命周期钩子使用 `on` 前缀：

```javascript
import { onMounted, onUpdated, onUnmounted } from 'vue'

export default {
  setup() {
    onMounted(() => {
      console.log('组件已挂载')
    })
    
    onUpdated(() => {
      console.log('组件已更新')
    })
    
    onUnmounted(() => {
      console.log('组件已卸载')
    })
  }
}
```

## 提取可复用的逻辑

Composition API 最强大的特性是能够轻松提取和复用逻辑：

```javascript
// useCounter.js
import { ref, computed } from 'vue'

export function useCounter(initialValue = 0) {
  const count = ref(initialValue)
  const doubled = computed(() => count.value * 2)
  
  const increment = () => count.value++
  const decrement = () => count.value--
  
  return {
    count,
    doubled,
    increment,
    decrement
  }
}

// 在组件中使用
import { useCounter } from './useCounter'

export default {
  setup() {
    return useCounter(10)
  }
}
```

## 总结

Vue 3 的 Composition API 提供了一种更灵活、更强大的方式来组织组件逻辑。它特别适合处理复杂的组件和逻辑复用场景。

通过学习和掌握 Composition API，你将能够编写更加模块化、可维护的 Vue 应用程序。
