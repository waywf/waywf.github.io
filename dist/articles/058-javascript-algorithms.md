---
title: JavaScript算法详解：从实现到优化
category: 算法
excerpt: 本文详细介绍了常见算法的JavaScript实现，包括排序算法、搜索算法、动态规划等，配有详细注释和动图展示，帮助你彻底掌握算法的实现原理和应用场景。
tags: JavaScript, 算法, 排序算法, 搜索算法, 动态规划
date: 2025-04-12
---

## 引言

算法是编程的核心，掌握算法不仅能帮助我们写出更高效的代码，更能让我们在面对复杂问题时，选择最合适的解决方案。JavaScript作为一门广泛应用的编程语言，虽然在性能上不如C++、Java等编译型语言，但它的灵活性和易用性使得它成为学习算法的绝佳选择。

本文将详细介绍常见算法的JavaScript实现，包括排序算法、搜索算法、动态规划等，配有详细注释和动图展示，帮助你彻底掌握算法的实现原理和应用场景。

---

## 一、排序算法

排序算法是最基础也是最重要的算法之一。它的作用是将一组数据按照特定的顺序排列。常见的排序算法有冒泡排序、选择排序、插入排序、快速排序、归并排序等。

### 1.1 冒泡排序

**原理**：重复地走访过要排序的数列，一次比较两个元素，如果它们的顺序错误就把它们交换过来。走访数列的工作是重复地进行直到没有再需要交换，也就是说该数列已经排序完成。

**动图展示**：
![冒泡排序动图](https://upload.wikimedia.org/wikipedia/commons/0/06/Bubble_sort_animation.gif)

**JavaScript实现**：
```javascript
/**
 * 冒泡排序
 * @param {Array} arr - 要排序的数组
 * @returns {Array} 排序后的数组
 */
function bubbleSort(arr) {
    // 复制原数组，避免修改原数组
    const arrCopy = [...arr];
    const n = arrCopy.length;
    
    // 外层循环：控制排序的轮数，最多需要n-1轮
    for (let i = 0; i < n - 1; i++) {
        // 标记本轮是否发生了交换
        let swapped = false;
        
        // 内层循环：比较相邻元素，将较大的元素逐步"冒泡"到末尾
        // 每完成一轮排序，末尾的i个元素已经是有序的，不需要再比较
        for (let j = 0; j < n - i - 1; j++) {
            // 如果当前元素大于下一个元素，交换它们
            if (arrCopy[j] > arrCopy[j + 1]) {
                [arrCopy[j], arrCopy[j + 1]] = [arrCopy[j + 1], arrCopy[j]];
                swapped = true;
            }
        }
        
        // 如果本轮没有发生交换，说明数组已经有序，可以提前结束排序
        if (!swapped) {
            break;
        }
    }
    
    return arrCopy;
}

// 示例
const arr = [64, 34, 25, 12, 22, 11, 90];
console.log(bubbleSort(arr)); // [11, 12, 22, 25, 34, 64, 90]
```

**优缺点**：
- **优点**：实现简单，稳定性好（相等元素的相对位置保持不变）。
- **缺点**：时间复杂度高（O(n²)），效率低，不适合大规模数据排序。

**适用场景**：小规模数据排序，或者作为教学示例。

---

### 1.2 选择排序

**原理**：首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

**动图展示**：
![选择排序动图](https://upload.wikimedia.org/wikipedia/commons/b/b0/Selection_sort_animation.gif)

**JavaScript实现**：
```javascript
/**
 * 选择排序
 * @param {Array} arr - 要排序的数组
 * @returns {Array} 排序后的数组
 */
function selectionSort(arr) {
    const arrCopy = [...arr];
    const n = arrCopy.length;
    
    // 外层循环：控制已排序序列的长度
    for (let i = 0; i < n - 1; i++) {
        // 找到未排序序列中的最小元素的索引
        let minIndex = i;
        for (let j = i + 1; j < n; j++) {
            if (arrCopy[j] < arrCopy[minIndex]) {
                minIndex = j;
            }
        }
        
        // 如果最小元素不是当前元素，交换它们
        if (minIndex !== i) {
            [arrCopy[i], arrCopy[minIndex]] = [arrCopy[minIndex], arrCopy[i]];
        }
    }
    
    return arrCopy;
}

// 示例
const arr = [64, 34, 25, 12, 22, 11, 90];
console.log(selectionSort(arr)); // [11, 12, 22, 25, 34, 64, 90]
```

**优缺点**：
- **优点**：实现简单，交换次数少（最多n-1次交换）。
- **缺点**：时间复杂度高（O(n²)），不稳定（相等元素的相对位置可能改变）。

**适用场景**：小规模数据排序，或者对交换次数有严格限制的场景。

---

### 1.3 插入排序

**原理**：通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，通常采用in-place排序（即只需用到O(1)的额外空间的排序），因而在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。

**动图展示**：
![插入排序动图](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)

**JavaScript实现**：
```javascript
/**
 * 插入排序
 * @param {Array} arr - 要排序的数组
 * @returns {Array} 排序后的数组
 */
function insertionSort(arr) {
    const arrCopy = [...arr];
    const n = arrCopy.length;
    
    // 外层循环：从第二个元素开始，将每个元素插入到已排序序列中
    for (let i = 1; i < n; i++) {
        // 当前要插入的元素
        const current = arrCopy[i];
        // 已排序序列的最后一个元素的索引
        let j = i - 1;
        
        // 内层循环：从后向前扫描已排序序列，找到插入位置
        while (j >= 0 && arrCopy[j] > current) {
            // 将比current大的元素向后移动一位
            arrCopy[j + 1] = arrCopy[j];
            j--;
        }
        
        // 将current插入到正确的位置
        arrCopy[j + 1] = current;
    }
    
    return arrCopy;
}

// 示例
const arr = [64, 34, 25, 12, 22, 11, 90];
console.log(insertionSort(arr)); // [11, 12, 22, 25, 34, 64, 90]
```

**优缺点**：
- **优点**：实现简单，稳定性好，对接近有序的数据排序效率高（O(n)时间复杂度）。
- **缺点**：时间复杂度高（O(n²)），不适合大规模数据排序。

**适用场景**：小规模数据排序，或者数据接近有序的场景。

---

### 1.4 快速排序

**原理**：通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。

**动图展示**：
![快速排序动图](https://upload.wikimedia.org/wikipedia/commons/6/6a/Sorting_quicksort_anim.gif)

**JavaScript实现**：
```javascript
/**
 * 快速排序
 * @param {Array} arr - 要排序的数组
 * @returns {Array} 排序后的数组
 */
function quickSort(arr) {
    // 递归终止条件：如果数组长度小于等于1，直接返回数组
    if (arr.length <= 1) {
        return arr;
    }
    
    // 选择基准元素：这里选择数组的第一个元素作为基准
    const pivot = arr[0];
    // 小于等于基准的元素
    const left = [];
    // 大于基准的元素
    const right = [];
    
    // 遍历数组，将元素分到left或right中
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] <= pivot) {
            left.push(arr[i]);
        } else {
            right.push(arr[i]);
        }
    }
    
    // 递归排序left和right，并将结果与基准元素合并
    return [...quickSort(left), pivot, ...quickSort(right)];
}

// 优化版本：原地快速排序（减少空间使用）
function quickSortInPlace(arr, left = 0, right = arr.length - 1) {
    // 递归终止条件
    if (left >= right) {
        return;
    }
    
    // 分区函数：将数组分成两部分，返回基准元素的最终位置
    function partition(arr, left, right) {
        // 选择基准元素：这里选择数组的最后一个元素作为基准
        const pivot = arr[right];
        // 基准元素的最终位置
        let i = left - 1;
        
        // 遍历数组，将小于等于基准的元素放到左边
        for (let j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                i++;
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
        }
        
        // 将基准元素放到正确的位置
        [arr[i + 1], arr[right]] = [arr[right], arr[i + 1]];
        return i + 1;
    }
    
    // 获取基准元素的最终位置
    const pivotIndex = partition(arr, left, right);
    // 递归排序左半部分
    quickSortInPlace(arr, left, pivotIndex - 1);
    // 递归排序右半部分
    quickSortInPlace(arr, pivotIndex + 1, right);
    
    return arr;
}

// 示例
const arr = [64, 34, 25, 12, 22, 11, 90];
console.log(quickSort(arr)); // [11, 12, 22, 25, 34, 64, 90]
console.log(quickSortInPlace([...arr])); // [11, 12, 22, 25, 34, 64, 90]
```

**优缺点**：
- **优点**：平均时间复杂度低（O(n log n)），效率高，是实际应用中最常用的排序算法之一。
- **缺点**：不稳定（相等元素的相对位置可能改变），最坏情况下时间复杂度为O(n²)（当数组已经有序时）。

**适用场景**：大规模数据排序，是实际应用中最常用的排序算法之一。

---

### 1.5 归并排序

**原理**：将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。若将两个有序表合并成一个有序表，称为二路归并。

**动图展示**：
![归并排序动图](https://upload.wikimedia.org/wikipedia/commons/c/cc/Merge-sort-example-300px.gif)

**JavaScript实现**：
```javascript
/**
 * 归并排序
 * @param {Array} arr - 要排序的数组
 * @returns {Array} 排序后的数组
 */
function mergeSort(arr) {
    // 递归终止条件：如果数组长度小于等于1，直接返回数组
    if (arr.length <= 1) {
        return arr;
    }
    
    // 将数组分成两部分
    const mid = Math.floor(arr.length / 2);
    const left = arr.slice(0, mid);
    const right = arr.slice(mid);
    
    // 递归排序左右两部分
    return merge(mergeSort(left), mergeSort(right));
}

/**
 * 合并两个有序数组
 * @param {Array} left - 第一个有序数组
 * @param {Array} right - 第二个有序数组
 * @returns {Array} 合并后的有序数组
 */
function merge(left, right) {
    const result = [];
    let i = 0; // left数组的指针
    let j = 0; // right数组的指针
    
    // 比较两个数组的元素，将较小的元素加入结果数组
    while (i < left.length && j < right.length) {
        if (left[i] <= right[j]) {
            result.push(left[i]);
            i++;
        } else {
            result.push(right[j]);
            j++;
        }
    }
    
    // 将剩余的元素加入结果数组
    return result.concat(left.slice(i)).concat(right.slice(j));
}

// 示例
const arr = [64, 34, 25, 12, 22, 11, 90];
console.log(mergeSort(arr)); // [11, 12, 22, 25, 34, 64, 90]
```

**优缺点**：
- **优点**：时间复杂度低（O(n log n)），稳定性好，对大规模数据排序效率高。
- **缺点**：空间复杂度高（O(n)），需要额外的空间存储临时数组。

**适用场景**：大规模数据排序，特别是对稳定性有要求的场景。

---

## 二、搜索算法

搜索算法是用来在一组数据中查找特定元素的算法。常见的搜索算法有线性搜索、二分搜索、深度优先搜索、广度优先搜索等。

### 2.1 线性搜索

**原理**：逐个检查数组中的每个元素，直到找到目标元素或者遍历完整个数组。

**动图展示**：
![线性搜索动图](https://upload.wikimedia.org/wikipedia/commons/b/bc/Linear_search.gif)

**JavaScript实现**：
```javascript
/**
 * 线性搜索
 * @param {Array} arr - 要搜索的数组
 * @param {*} target - 要搜索的目标元素
 * @returns {number} 目标元素在数组中的索引，如果未找到则返回-1
 */
function linearSearch(arr, target) {
    // 遍历数组中的每个元素
    for (let i = 0; i < arr.length; i++) {
        // 如果找到目标元素，返回其索引
        if (arr[i] === target) {
            return i;
        }
    }
    // 如果遍历完整个数组都没有找到目标元素，返回-1
    return -1;
}

// 示例
const arr = [64, 34, 25, 12, 22, 11, 90];
console.log(linearSearch(arr, 22)); // 4
console.log(linearSearch(arr, 100)); // -1
```

**优缺点**：
- **优点**：实现简单，不需要数组有序。
- **缺点**：时间复杂度高（O(n)），效率低，不适合大规模数据搜索。

**适用场景**：小规模数据搜索，或者数组无序的场景。

---

### 2.2 二分搜索

**原理**：在有序数组中，通过不断将搜索范围减半来查找目标元素。首先比较中间元素与目标元素，如果中间元素等于目标元素，则找到目标元素；如果中间元素大于目标元素，则在左半部分继续搜索；如果中间元素小于目标元素，则在右半部分继续搜索。

**动图展示**：
![二分搜索动图](https://upload.wikimedia.org/wikipedia/commons/8/83/Binary_Search_Depiction.svg)

**JavaScript实现**：
```javascript
/**
 * 二分搜索（迭代版本）
 * @param {Array} arr - 要搜索的有序数组
 * @param {*} target - 要搜索的目标元素
 * @returns {number} 目标元素在数组中的索引，如果未找到则返回-1
 */
function binarySearch(arr, target) {
    let left = 0; // 搜索范围的左边界
    let right = arr.length - 1; // 搜索范围的右边界
    
    // 当左边界小于等于右边界时，继续搜索
    while (left <= right) {
        // 计算中间位置
        const mid = Math.floor((left + right) / 2);
        
        // 如果中间元素等于目标元素，返回其索引
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            // 如果中间元素小于目标元素，在右半部分继续搜索
            left = mid + 1;
        } else {
            // 如果中间元素大于目标元素，在左半部分继续搜索
            right = mid - 1;
        }
    }
    
    // 如果遍历完整个搜索范围都没有找到目标元素，返回-1
    return -1;
}

// 递归版本
function binarySearchRecursive(arr, target, left = 0, right = arr.length - 1) {
    // 如果左边界大于右边界，说明没有找到目标元素
    if (left > right) {
        return -1;
    }
    
    // 计算中间位置
    const mid = Math.floor((left + right) / 2);
    
    // 如果中间元素等于目标元素，返回其索引
    if (arr[mid] === target) {
        return mid;
    } else if (arr[mid] < target) {
        // 如果中间元素小于目标元素，在右半部分继续搜索
        return binarySearchRecursive(arr, target, mid + 1, right);
    } else {
        // 如果中间元素大于目标元素，在左半部分继续搜索
        return binarySearchRecursive(arr, target, left, mid - 1);
    }
}

// 示例
const arr = [11, 12, 22, 25, 34, 64, 90];
console.log(binarySearch(arr, 22)); // 2
console.log(binarySearch(arr, 100)); // -1
```

**优缺点**：
- **优点**：时间复杂度低（O(log n)），效率高，适合大规模数据搜索。
- **缺点**：需要数组有序，不适合动态变化的数组。

**适用场景**：大规模有序数据搜索，是实际应用中最常用的搜索算法之一。

---

### 2.3 深度优先搜索（DFS）

**原理**：从起始节点开始，沿着一条路径尽可能深地探索，直到无法继续或者找到目标节点，然后回溯到上一个节点，继续探索其他路径。

**动图展示**：
![深度优先搜索动图](https://upload.wikimedia.org/wikipedia/commons/7/7f/Depth-First-Search.gif)

**JavaScript实现**：
```javascript
/**
 * 深度优先搜索（递归版本）
 * @param {Object} graph - 图的邻接表表示
 * @param {string} start - 起始节点
 * @param {string} target - 目标节点
 * @param {Set} visited - 已访问的节点
 * @returns {boolean} 是否找到目标节点
 */
function dfs(graph, start, target, visited = new Set()) {
    // 如果起始节点就是目标节点，返回true
    if (start === target) {
        return true;
    }
    
    // 如果起始节点已经被访问过，返回false
    if (visited.has(start)) {
        return false;
    }
    
    // 标记起始节点为已访问
    visited.add(start);
    
    // 遍历起始节点的所有邻居
    for (const neighbor of graph[start]) {
        // 如果邻居未被访问过，递归搜索
        if (!visited.has(neighbor)) {
            if (dfs(graph, neighbor, target, visited)) {
                return true;
            }
        }
    }
    
    // 如果遍历完所有邻居都没有找到目标节点，返回false
    return false;
}

// 迭代版本
function dfsIterative(graph, start, target) {
    const stack = [start]; // 使用栈存储待访问的节点
    const visited = new Set(); // 存储已访问的节点
    
    while (stack.length > 0) {
        // 弹出栈顶节点
        const node = stack.pop();
        
        // 如果节点就是目标节点，返回true
        if (node === target) {
            return true;
        }
        
        // 如果节点未被访问过
        if (!visited.has(node)) {
            // 标记节点为已访问
            visited.add(node);
            // 将节点的邻居加入栈中（注意：为了保持顺序，需要反转邻居数组）
            for (const neighbor of graph[node].reverse()) {
                if (!visited.has(neighbor)) {
                    stack.push(neighbor);
                }
            }
        }
    }
    
    // 如果遍历完所有节点都没有找到目标节点，返回false
    return false;
}

// 示例
const graph = {
    A: ['B', 'C'],
    B: ['A', 'D', 'E'],
    C: ['A', 'F'],
    D: ['B'],
    E: ['B', 'F'],
    F: ['C', 'E']
};
console.log(dfs(graph, 'A', 'F')); // true
console.log(dfsIterative(graph, 'A', 'G')); // false
```

**优缺点**：
- **优点**：实现简单，空间复杂度低（O(n)），适合搜索深度优先的问题。
- **缺点**：可能陷入无限循环（需要标记已访问的节点），不适合搜索最短路径。

**适用场景**：图的遍历、拓扑排序、连通性检测等问题。

---

### 2.4 广度优先搜索（BFS）

**原理**：从起始节点开始，逐层探索所有邻居节点，直到找到目标节点或者遍历完所有节点。广度优先搜索可以保证找到最短路径。

**动图展示**：
![广度优先搜索动图](https://upload.wikimedia.org/wikipedia/commons/5/5d/Breadth-First-Search-Algorithm.gif)

**JavaScript实现**：
```javascript
/**
 * 广度优先搜索
 * @param {Object} graph - 图的邻接表表示
 * @param {string} start - 起始节点
 * @param {string} target - 目标节点
 * @returns {boolean} 是否找到目标节点
 */
function bfs(graph, start, target) {
    const queue = [start]; // 使用队列存储待访问的节点
    const visited = new Set(); // 存储已访问的节点
    
    // 标记起始节点为已访问
    visited.add(start);
    
    while (queue.length > 0) {
        // 出队队列的第一个节点
        const node = queue.shift();
        
        // 如果节点就是目标节点，返回true
        if (node === target) {
            return true;
        }
        
        // 遍历节点的所有邻居
        for (const neighbor of graph[node]) {
            // 如果邻居未被访问过
            if (!visited.has(neighbor)) {
                // 标记邻居为已访问
                visited.add(neighbor);
                // 将邻居加入队列
                queue.push(neighbor);
            }
        }
    }
    
    // 如果遍历完所有节点都没有找到目标节点，返回false
    return false;
}

// 扩展版本：找到最短路径
function bfsShortestPath(graph, start, target) {
    const queue = [[start]]; // 使用队列存储路径
    const visited = new Set(); // 存储已访问的节点
    
    // 标记起始节点为已访问
    visited.add(start);
    
    while (queue.length > 0) {
        // 出队队列的第一个路径
        const path = queue.shift();
        // 获取路径的最后一个节点
        const node = path[path.length - 1];
        
        // 如果节点就是目标节点，返回路径
        if (node === target) {
            return path;
        }
        
        // 遍历节点的所有邻居
        for (const neighbor of graph[node]) {
            // 如果邻居未被访问过
            if (!visited.has(neighbor)) {
                // 标记邻居为已访问
                visited.add(neighbor);
                // 将新路径加入队列
                queue.push([...path, neighbor]);
            }
        }
    }
    
    // 如果遍历完所有节点都没有找到目标节点，返回null
    return null;
}

// 示例
const graph = {
    A: ['B', 'C'],
    B: ['A', 'D', 'E'],
    C: ['A', 'F'],
    D: ['B'],
    E: ['B', 'F'],
    F: ['C', 'E']
};
console.log(bfs(graph, 'A', 'F')); // true
console.log(bfsShortestPath(graph, 'A', 'F')); // ['A', 'C', 'F']
```

**优缺点**：
- **优点**：可以找到最短路径，适合搜索最短路径的问题。
- **缺点**：空间复杂度高（O(n)），需要存储所有待访问的节点。

**适用场景**：图的遍历、最短路径搜索、连通性检测等问题。

---

## 三、动态规划

动态规划是一种通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。动态规划的核心思想是将问题分解为子问题，保存子问题的解，避免重复计算。

### 3.1 斐波那契数列

**问题**：计算斐波那契数列的第n项。斐波那契数列的定义是：F(0)=0，F(1)=1，F(n)=F(n-1)+F(n-2)（n≥2）。

**JavaScript实现**：
```javascript
/**
 * 动态规划求解斐波那契数列
 * @param {number} n - 要计算的斐波那契数列的项数
 * @returns {number} 斐波那契数列的第n项
 */
function fibonacci(n) {
    // 如果n小于等于1，直接返回n
    if (n <= 1) {
        return n;
    }
    
    // 创建一个数组存储斐波那契数列的前n+1项
    const dp = new Array(n + 1);
    // 初始化前两项
    dp[0] = 0;
    dp[1] = 1;
    
    // 计算从第2项到第n项
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    // 返回第n项
    return dp[n];
}

// 优化版本：减少空间使用
function fibonacciOptimized(n) {
    // 如果n小于等于1，直接返回n
    if (n <= 1) {
        return n;
    }
    
    // 只保存前两项
    let a = 0; // F(0)
    let b = 1; // F(1)
    
    // 计算从第2项到第n项
    for (let i = 2; i <= n; i++) {
        const c = a + b; // F(i) = F(i-1) + F(i-2)
        a = b; // 更新F(i-1)为F(i-2)
        b = c; // 更新F(i-2)为F(i)
    }
    
    // 返回第n项
    return b;
}

// 示例
console.log(fibonacci(10)); // 55
console.log(fibonacciOptimized(10)); // 55
```

**优缺点**：
- **优点**：时间复杂度低（O(n)），避免了递归版本的重复计算。
- **缺点**：需要额外的空间存储子问题的解（优化版本可以减少空间使用）。

**适用场景**：具有重叠子问题和最优子结构性质的问题。

---

### 3.2 最长公共子序列

**问题**：给定两个字符串，找到它们的最长公共子序列（LCS）。最长公共子序列是指两个字符串中都出现的最长的子序列，子序列中的字符不需要连续，但顺序必须一致。

**JavaScript实现**：
```javascript
/**
 * 动态规划求解最长公共子序列
 * @param {string} str1 - 第一个字符串
 * @param {string} str2 - 第二个字符串
 * @returns {string} 最长公共子序列
 */
function longestCommonSubsequence(str1, str2) {
    const m = str1.length;
    const n = str2.length;
    
    // 创建一个二维数组存储子问题的解
    const dp = new Array(m + 1).fill(0).map(() => new Array(n + 1).fill(0));
    
    // 填充dp数组
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            // 如果当前字符相等，LCS长度为左上角的值加1
            if (str1[i - 1] === str2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                // 如果当前字符不相等，LCS长度为上方或左方的最大值
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    
    // 回溯dp数组，找到最长公共子序列
    let i = m;
    let j = n;
    let lcs = '';
    
    while (i > 0 && j > 0) {
        // 如果当前字符相等，将其加入LCS，并向左上方移动
        if (str1[i - 1] === str2[j - 1]) {
            lcs = str1[i - 1] + lcs;
            i--;
            j--;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            // 如果上方的值大于左方的值，向上移动
            i--;
        } else {
            // 如果左方的值大于等于上方的值，向左移动
            j--;
        }
    }
    
    return lcs;
}

// 示例
const str1 = 'ABCBDAB';
const str2 = 'BDCAB';
console.log(longestCommonSubsequence(str1, str2)); // 'BCAB'
```

**优缺点**：
- **优点**：可以有效地解决具有重叠子问题和最优子结构性质的问题。
- **缺点**：空间复杂度高（O(mn)），需要存储二维数组。

**适用场景**：字符串匹配、DNA序列比对、文件差异比较等问题。

---

## 四、算法对比与选择

### 4.1 排序算法对比

| 算法 | 平均时间复杂度 | 最坏时间复杂度 | 空间复杂度 | 稳定性 | 适用场景 |
|-----|----------------|----------------|----------|------|--------|
| 冒泡排序 | O(n²) | O(n²) | O(1) | 稳定 | 小规模数据排序 |
| 选择排序 | O(n²) | O(n²) | O(1) | 不稳定 | 小规模数据排序 |
| 插入排序 | O(n²) | O(n²) | O(1) | 稳定 | 小规模数据排序，数据接近有序 |
| 快速排序 | O(n log n) | O(n²) | O(log n) | 不稳定 | 大规模数据排序 |
| 归并排序 | O(n log n) | O(n log n) | O(n) | 稳定 | 大规模数据排序，对稳定性有要求 |
| 堆排序 | O(n log n) | O(n log n) | O(1) | 不稳定 | 大规模数据排序，对空间有严格限制 |

### 4.2 搜索算法对比

| 算法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|-----|----------|----------|--------|
| 线性搜索 | O(n) | O(1) | 小规模数据搜索，数组无序 |
| 二分搜索 | O(log n) | O(1) | 大规模有序数据搜索 |
| 深度优先搜索 | O(n + e) | O(n) | 图的遍历，拓扑排序 |
| 广度优先搜索 | O(n + e) | O(n) | 图的遍历，最短路径搜索 |

### 4.3 如何选择合适的算法

选择算法时需要考虑以下因素：

1. **数据规模**：小规模数据可以选择简单的算法（如冒泡排序、线性搜索），大规模数据需要选择高效的算法（如快速排序、二分搜索）。
2. **数据特性**：如果数据接近有序，可以选择插入排序；如果数据需要稳定排序，可以选择归并排序或插入排序。
3. **空间限制**：如果内存有限，需要选择空间复杂度低的算法（如冒泡排序、选择排序、插入排序、堆排序）。
4. **时间限制**：如果对时间要求很高，需要选择时间复杂度低的算法（如快速排序、归并排序、二分搜索）。
5. **算法特性**：如果需要找到最短路径，可以选择广度优先搜索；如果需要解决具有重叠子问题和最优子结构性质的问题，可以选择动态规划。

---

## 五、总结

算法是编程的核心，掌握算法不仅能帮助我们写出更高效的代码，更能让我们在面对复杂问题时，选择最合适的解决方案。本文详细介绍了常见算法的JavaScript实现，包括排序算法、搜索算法、动态规划等，配有详细注释和动图展示，帮助你彻底掌握算法的实现原理和应用场景。

### 重点回顾

1. **排序算法**：冒泡排序、选择排序、插入排序、快速排序、归并排序等。
2. **搜索算法**：线性搜索、二分搜索、深度优先搜索、广度优先搜索等。
3. **动态规划**：斐波那契数列、最长公共子序列等。
4. **算法选择**：根据数据规模、数据特性、空间限制、时间限制等因素选择合适的算法。

### 学习建议

1. **理解原理**：不要死记硬背算法代码，要理解算法的原理和思想。
2. **多写代码**：通过编写代码来加深对算法的理解。
3. **分析复杂度**：分析算法的时间复杂度和空间复杂度，了解算法的效率。
4. **对比选择**：对比不同算法的优缺点，学会选择合适的算法解决问题。
5. **实践应用**：将算法应用到实际项目中，解决实际问题。

掌握算法需要不断学习和实践，希望本文能帮助你在算法学习的道路上更进一步。

---

## 参考资料

1. 《算法导论》（Introduction to Algorithms）- Thomas H. Cormen等
2. 《数据结构与算法分析》（Data Structures and Algorithm Analysis in C++）- Mark Allen Weiss
3. 《算法图解》（Grokking Algorithms）- Aditya Bhargava
4. 《JavaScript数据结构与算法》- 周俊鹏
5. Wikipedia：https://en.wikipedia.org/wiki/Sorting_algorithm
6. Visualgo：https://visualgo.net/en
