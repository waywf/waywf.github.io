---
title: Vue3组合式API深度剖析：从Options API到Composition API的进化
date: 2025-10-20
category: 前端开发
tags: Vue3, Composition API, 响应式系统, 前端开发, 组合式函数
excerpt: 深入理解Vue3组合式API的设计理念，探索ref、reactive、computed等核心API的实现原理，学习如何组织和复用逻辑，掌握从Options API迁移的最佳实践。
readTime: 23
---

> 还记得Vue2时代吗？data、methods、computed、watch——逻辑被分散在不同的选项中，一个功能的相关代码可能相隔数百行。Vue3的Composition API就像给Vue开发者发了一把瑞士军刀，让逻辑组织变得前所未有的灵活。但这把刀该怎么用？今天，让我们深入探索Composition API的奥秘。

## 一、为什么需要Composition API？

### 1.1 Options API的局限性

在Vue2中，我们使用Options API组织代码：

```javascript
export default {
  data() {
    return {
      searchQuery: '',
      searchResults: [],
      loading: false,
      error: null,
    };
  },
  
  computed: {
    filteredResults() {
      return this.searchResults.filter(item => 
        item.title.includes(this.searchQuery)
      );
    },
  },
  
  watch: {
    searchQuery(newQuery) {
      this.fetchResults(newQuery);
    },
  },
  
  methods: {
    async fetchResults(query) {
      this.loading = true;
      try {
        const results = await api.search(query);
        this.searchResults = results;
      } catch (err) {
        this.error = err.message;
      } finally {
        this.loading = false;
      }
    },
  },
  
  mounted() {
    this.fetchResults(this.searchQuery);
  },
};
```

**问题在哪里？**

1. **逻辑分散**：搜索功能的代码分散在data、computed、watch、methods中
2. **代码复用困难**：mixin有命名冲突、来源不明等问题
3. **TypeScript支持差**：this的类型推断困难
4. **大型组件难以维护**：几百行的组件找逻辑像寻宝

### 1.2 Composition API的解决方案

Vue3的Composition API让我们按功能组织代码：

```vue
<script setup>
import { ref, computed, watch, onMounted } from 'vue';

// 搜索功能的所有逻辑在一起
const searchQuery = ref('');
const searchResults = ref([]);
const loading = ref(false);
const error = ref(null);

const filteredResults = computed(() => {
  return searchResults.value.filter(item => 
    item.title.includes(searchQuery.value)
  );
});

async function fetchResults(query) {
  loading.value = true;
  try {
    const results = await api.search(query);
    searchResults.value = results;
  } catch (err) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
}

watch(searchQuery, (newQuery) => {
  fetchResults(newQuery);
});

onMounted(() => {
  fetchResults(searchQuery.value);
});
</script>
```

**优势**：
- ✅ 逻辑聚合：相关代码放在一起
- ✅ 易于复用：组合式函数轻松共享逻辑
- ✅ 更好的TypeScript支持
- ✅ 更灵活的组织方式

## 二、响应式系统核心：ref vs reactive

### 2.1 ref：基本类型的响应式

`ref`接受一个值并返回一个响应式引用：

```javascript
import { ref } from 'vue';

const count = ref(0);

console.log(count);     // RefImpl { ... }
console.log(count.value); // 0

count.value++;          // 修改需要使用.value
console.log(count.value); // 1
```

**为什么需要.value？**

```javascript
// JavaScript的限制：基本类型按值传递
let a = 0;
let b = a;  // b是a的副本
b++;
console.log(a); // 0，a不变

// ref使用对象包装，保持引用
const count = ref(0);
// 实际上count是一个对象：{ value: 0 }
// 修改count.value会触发响应式更新
```

**ref在模板中的自动解包**：

```vue
<template>
  <div>
    <!-- 模板中自动解包，不需要.value -->
    <p>{{ count }}</p>
    <button @click="count++">+1</button>
  </div>
</template>

<script setup>
const count = ref(0);
</script>
```

### 2.2 reactive：对象的响应式

`reactive`用于创建响应式对象：

```javascript
import { reactive } from 'vue';

const state = reactive({
  count: 0,
  user: {
    name: 'Alice',
    age: 25,
  },
});

// 直接访问属性，不需要.value
console.log(state.count); // 0
state.count++;
console.log(state.count); // 1

// 嵌套对象也是响应式的
state.user.age++;
```

**reactive的限制**：

```javascript
// ❌ 不能替换整个对象
let state = reactive({ count: 0 });
state = { count: 1 }; // 失去响应式连接

// ✅ 使用Object.assign
Object.assign(state, { count: 1 });

// ❌ 对数组的某些操作不友好
const arr = reactive([1, 2, 3]);
const newArr = arr.filter(n => n > 1); // 返回普通数组

// ✅ 使用ref处理数组
const arr = ref([1, 2, 3]);
arr.value = arr.value.filter(n => n > 1); // 保持响应式
```

### 2.3 选择ref还是reactive？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 基本类型 | ref | reactive不能包装基本类型 |
| 对象 | 都可以 | ref需要.value，reactive更直接 |
| 数组 | ref | 避免reactive的数组限制 |
| 需要替换整个值 | ref | reactive替换会失去响应式 |
| 解构 | ref | reactive解构会失去响应式 |

**最佳实践建议**：

```javascript
// 统一使用ref，减少心智负担
const count = ref(0);
const user = ref({ name: 'Alice', age: 25 });
const items = ref([1, 2, 3]);

// 在组合式函数中返回ref
function useFeature() {
  const data = ref(null);
  const loading = ref(false);
  
  // ...逻辑
  
  return { data, loading };
}
```

## 三、计算属性与侦听器

### 3.1 computed：缓存的计算值

```javascript
import { ref, computed } from 'vue';

const firstName = ref('John');
const lastName = ref('Doe');

// 只读计算属性
const fullName = computed(() => {
  return `${firstName.value} ${lastName.value}`;
});

// 可写计算属性
const fullName = computed({
  get() {
    return `${firstName.value} ${lastName.value}`;
  },
  set(newValue) {
    [firstName.value, lastName.value] = newValue.split(' ');
  },
});

fullName.value = 'Jane Smith'; // firstName = 'Jane', lastName = 'Smith'
```

**computed vs 普通函数**：

```javascript
// 普通函数：每次渲染都执行
function getFullName() {
  console.log('executing');
  return `${firstName.value} ${lastName.value}`;
}

// computed：只有依赖变化时才重新计算
const fullName = computed(() => {
  console.log('computing');
  return `${firstName.value} ${lastName.value}`;
});
```

### 3.2 watch：副作用侦听

```javascript
import { ref, watch } from 'vue';

const searchQuery = ref('');
const searchResults = ref([]);

// 侦听单个ref
watch(searchQuery, (newValue, oldValue) => {
  console.log(`Query changed from "${oldValue}" to "${newValue}"`);
  fetchResults(newValue);
});

// 立即执行
watch(searchQuery, (newValue) => {
  fetchResults(newValue);
}, { immediate: true });

// 深度侦听
const user = ref({
  profile: {
    name: 'Alice',
  },
});

watch(user, (newValue) => {
  console.log('User changed:', newValue);
}, { deep: true });

// 侦听多个源
watch([searchQuery, pageNumber], ([newQuery, newPage], [oldQuery, oldPage]) => {
  console.log('Something changed');
});
```

### 3.3 watchEffect：自动追踪依赖

```javascript
import { ref, watchEffect } from 'vue';

const count = ref(0);
const user = ref({ name: 'Alice' });

// 自动追踪使用的响应式依赖
watchEffect(() => {
  console.log(count.value); // 追踪count
  console.log(user.value.name); // 追踪user.name
});

// 组件卸载时自动停止
// 也可以手动停止
const stop = watchEffect(() => {
  console.log(count.value);
});

// 之后调用stop()停止侦听
```

**watch vs watchEffect**：

| 特性 | watch | watchEffect |
|------|-------|-------------|
| 懒执行 | ✅ 默认懒执行 | ❌ 立即执行 |
| 依赖追踪 | 手动指定 | 自动追踪 |
| 访问旧值 | ✅ 可以 | ❌ 不可以 |
| 使用场景 | 特定变化时执行 | 自动追踪依赖 |

## 四、生命周期钩子

### 4.1 Composition API生命周期

```javascript
import {
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted,
  onErrorCaptured,
  onRenderTracked,
  onRenderTriggered,
} from 'vue';

// 对应Options API的生命周期
onBeforeMount(() => {
  // beforeMount
});

onMounted(() => {
  // mounted
});

onBeforeUpdate(() => {
  // beforeUpdate
});

onUpdated(() => {
  // updated
});

onBeforeUnmount(() => {
  // beforeDestroy
});

onUnmounted(() => {
  // destroyed
});

// 调试钩子
onRenderTracked((event) => {
  // 响应式依赖被追踪时调用
  console.log('Tracked:', event);
});

onRenderTriggered((event) => {
  // 响应式依赖触发重新渲染时调用
  console.log('Triggered:', event);
});
```

### 4.2 在setup中使用生命周期

```vue
<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const el = ref(null);

onMounted(() => {
  // DOM已挂载
  console.log(el.value);
  
  // 添加事件监听
  window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
  // 清理工作
  window.removeEventListener('resize', handleResize);
});
</script>

<template>
  <div ref="el">Content</div>
</template>
```

## 五、组合式函数：逻辑复用的艺术

### 5.1 什么是组合式函数？

组合式函数是以`use`开头的函数，封装可复用的逻辑：

```javascript
// useMouse.js
import { ref, onMounted, onUnmounted } from 'vue';

export function useMouse() {
  const x = ref(0);
  const y = ref(0);

  function update(event) {
    x.value = event.pageX;
    y.value = event.pageY;
  }

  onMounted(() => {
    window.addEventListener('mousemove', update);
  });

  onUnmounted(() => {
    window.removeEventListener('mousemove', update);
  });

  return { x, y };
}
```

```vue
<script setup>
import { useMouse } from './useMouse';

const { x, y } = useMouse();
</script>

<template>
  <div>Mouse: {{ x }}, {{ y }}</div>
</template>
```

### 5.2 实战：常用组合式函数

**useLocalStorage**：

```javascript
import { ref, watch } from 'vue';

export function useLocalStorage(key, defaultValue) {
  const stored = localStorage.getItem(key);
  const data = ref(stored ? JSON.parse(stored) : defaultValue);

  watch(data, (newValue) => {
    localStorage.setItem(key, JSON.stringify(newValue));
  }, { deep: true });

  return data;
}

// 使用
const user = useLocalStorage('user', { name: '', email: '' });
```

**useFetch**：

```javascript
import { ref, watchEffect, toValue } from 'vue';

export function useFetch(url) {
  const data = ref(null);
  const error = ref(null);
  const loading = ref(false);

  const fetchData = async () => {
    loading.value = true;
    error.value = null;
    
    try {
      const response = await fetch(toValue(url));
      if (!response.ok) throw new Error(response.statusText);
      data.value = await response.json();
    } catch (err) {
      error.value = err.message;
    } finally {
      loading.value = false;
    }
  };

  watchEffect(() => {
    fetchData();
  });

  return { data, error, loading, refresh: fetchData };
}

// 使用
const { data: user, loading, error } = useFetch(() => `/api/users/${userId.value}`);
```

**useDebounce**：

```javascript
import { ref, watch } from 'vue';

export function useDebounce(value, delay = 300) {
  const debouncedValue = ref(value.value);

  let timeout;
  watch(value, (newValue) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      debouncedValue.value = newValue;
    }, delay);
  });

  return debouncedValue;
}

// 使用
const searchQuery = ref('');
const debouncedQuery = useDebounce(searchQuery, 500);

watch(debouncedQuery, (query) => {
  performSearch(query);
});
```

## 六、从Options API迁移

### 6.1 迁移策略

```javascript
// Options API
export default {
  data() {
    return {
      count: 0,
    };
  },
  computed: {
    double() {
      return this.count * 2;
    },
  },
  methods: {
    increment() {
      this.count++;
    },
  },
};

// Composition API
<script setup>
import { ref, computed } from 'vue';

const count = ref(0);
const double = computed(() => count.value * 2);

function increment() {
  count.value++;
}
</script>
```

### 6.2 混合使用

Vue3支持在同一个组件中混合使用：

```vue
<script>
// Options API部分
export default {
  inheritAttrs: false,
  
  data() {
    return {
      legacyData: 'old',
    };
  },
};
</script>

<script setup>
// Composition API部分
import { ref } from 'vue';

const newData = ref('new');
</script>
```

## 七、响应式原理进阶

### 7.1 Proxy-based响应式

Vue3使用ES6 Proxy替代Vue2的Object.defineProperty：

```javascript
// 简化的响应式实现
function reactive(target) {
  return new Proxy(target, {
    get(target, key, receiver) {
      const result = Reflect.get(target, key, receiver);
      // 收集依赖
      track(target, key);
      return result;
    },
    set(target, key, value, receiver) {
      const oldValue = target[key];
      const result = Reflect.set(target, key, value, receiver);
      // 触发更新
      if (oldValue !== value) {
        trigger(target, key);
      }
      return result;
    },
  });
}
```

**Proxy的优势**：
- 可以监听新增和删除的属性
- 可以监听数组索引和长度的变化
- 更好的性能

### 7.2  effect调度

```javascript
import { ref, effect } from 'vue';

const count = ref(0);

// effect会立即执行，并追踪依赖
effect(() => {
  console.log(count.value);
});

count.value++; // 自动重新执行effect
```

## 八、总结与最佳实践

### 8.1 何时使用Composition API？

- ✅ 大型组件，逻辑复杂
- ✅ 需要复用逻辑
- ✅ 更好的TypeScript支持
- ✅ 团队熟悉函数式编程

### 8.2 最佳实践

1. **使用<script setup>**：更简洁的语法
2. **组合式函数命名**：以`use`开头
3. **逻辑组织**：按功能而非选项组织代码
4. **响应式选择**：优先使用ref
5. **类型安全**：使用TypeScript

### 8.3 学习资源

- [Vue3官方文档](https://vuejs.org/)
- [Vue Mastery](https://www.vuemastery.com/)
- [Vue3源码](https://github.com/vuejs/core)

Composition API是Vue3最重要的特性之一，它让Vue的开发体验提升到了新的高度。掌握它，就是掌握了现代Vue开发的核心。
