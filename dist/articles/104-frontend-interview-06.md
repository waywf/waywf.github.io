---
title: 06.前端高频业务场景面试题深度解析：从表单到支付的实战指南
category: 前端开发
excerpt: 前端面试题第六弹！深入解析表单处理、图片加载、购物车、搜索、支付、聊天等10大高频业务场景。结合真实项目经验，带你掌握业务开发的核心技能！
tags: 前端面试, 业务场景, 表单处理, 购物车, 搜索, 支付, 图片优化
date: 2025-07-12
readTime: 60
---

> 面试官："实现一个购物车，商品加购、减购、删除、全选、小计、总计..."
> 
> 你："用 state 存数据，然后计算..."
> 
> 面试官："那如何处理并发加购？如何持久化购物车数据？如何优化购物车的性能？"
> 
> 你："..."
> 
> 欢迎来到前端业务场景面试战场！这次我们深入10大高频业务场景，从表单处理到支付，从图片加载到聊天，都是真实项目中天天在用的！准备好了吗？

## 一、表单处理场景

### 1.1 复杂表单验证

**面试题：请设计一个复杂表单的验证方案，包括实时验证、异步验证、联动验证等。**

#### 表单验证方案设计

```javascript
// 验证规则配置
const validationRules = {
  username: [
    { type: 'required', message: '用户名不能为空' },
    { type: 'minLength', value: 3, message: '用户名至少3个字符' },
    { type: 'maxLength', value: 20, message: '用户名最多20个字符' },
    { type: 'pattern', pattern: /^[a-zA-Z0-9_]+$/, message: '用户名只能包含字母、数字和下划线' },
    { type: 'async', validator: checkUsernameExists, message: '用户名已存在' }
  ],
  email: [
    { type: 'required', message: '邮箱不能为空' },
    { type: 'email', message: '邮箱格式不正确' },
    { type: 'async', validator: checkEmailExists, message: '邮箱已被注册' }
  ],
  password: [
    { type: 'required', message: '密码不能为空' },
    { type: 'minLength', value: 6, message: '密码至少6个字符' },
    { type: 'pattern', pattern: /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)/, message: '密码必须包含大小写字母和数字' }
  ],
  confirmPassword: [
    { type: 'required', message: '请确认密码' },
    { type: 'same', field: 'password', message: '两次密码输入不一致' }
  ],
  phone: [
    { type: 'required', message: '手机号不能为空' },
    { type: 'pattern', pattern: /^1[3-9]\d{9}$/, message: '手机号格式不正确' }
  ]
};

// 验证器函数
const validators = {
  required: (value) => value !== null && value !== undefined && value !== '',
  
  minLength: (value, min) => value.length >= min,
  
  maxLength: (value, max) => value.length <= max,
  
  pattern: (value, pattern) => pattern.test(value),
  
  email: (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
  
  same: (value, _, field, form) => value === form[field],
  
  async: async (value, _, __, ___, validator) => {
    try {
      const result = await validator(value);
      return result;
    } catch (error) {
      return false;
    }
  }
};

// 表单验证类
class FormValidator {
  constructor(rules) {
    this.rules = rules;
    this.errors = {};
    this.touched = {};
  }
  
  // 验证单个字段
  async validateField(fieldName, value, form) {
    const fieldRules = this.rules[fieldName];
    if (!fieldRules) return true;
    
    for (const rule of fieldRules) {
      const validator = validators[rule.type];
      
      if (rule.type === 'async') {
        const isValid = await validator(value, rule.value, rule.field, form, rule.validator);
        if (!isValid) {
          this.errors[fieldName] = rule.message;
          return false;
        }
      } else {
        const isValid = validator(value, rule.value, rule.field, form);
        if (!isValid) {
          this.errors[fieldName] = rule.message;
          return false;
        }
      }
    }
    
    delete this.errors[fieldName];
    return true;
  }
  
  // 验证整个表单
  async validateForm(form) {
    let isValid = true;
    
    for (const fieldName in this.rules) {
      const fieldValid = await this.validateField(fieldName, form[fieldName], form);
      this.touched[fieldName] = true;
      
      if (!fieldValid) {
        isValid = false;
      }
    }
    
    return isValid;
  }
  
  // 标记字段为已触摸
  touchField(fieldName) {
    this.touched[fieldName] = true;
  }
  
  // 获取字段错误
  getError(fieldName) {
    return this.touched[fieldName] ? this.errors[fieldName] : null;
  }
  
  // 表单是否有错误
  hasErrors() {
    return Object.keys(this.errors).length > 0;
  }
}

// 使用示例
const validator = new FormValidator(validationRules);

// 实时验证
function handleFieldChange(fieldName, value, form) {
  validator.touchField(fieldName);
  validator.validateField(fieldName, value, form);
}

// 表单提交
async function handleSubmit(form) {
  const isValid = await validator.validateForm(form);
  
  if (isValid) {
    await submitForm(form);
  }
}
```

### 1.2 表单联动与数据联动

```javascript
// 省市区三级联动
class Cascader {
  constructor(options) {
    this.onChange = options.onChange;
    this.data = options.data || [];
    this.value = [];
  }
  
  // 获取选项
  getOptions(level) {
    if (level === 0) {
      return this.data;
    }
    
    const parentValue = this.value[level - 1];
    const parentOption = this.findOption(this.data, level - 1, parentValue);
    
    return parentOption?.children || [];
  }
  
  // 查找选项
  findOption(options, targetLevel, value, currentLevel = 0) {
    for (const option of options) {
      if (currentLevel === targetLevel && option.value === value) {
        return option;
      }
      
      if (option.children) {
        const found = this.findOption(option.children, targetLevel, value, currentLevel + 1);
        if (found) return found;
      }
    }
    
    return null;
  }
  
  // 设置值
  setValue(value) {
    this.value = value;
    this.onChange?.(value);
  }
  
  // 选择选项
  selectOption(level, option) {
    this.value = this.value.slice(0, level);
    this.value[level] = option.value;
    
    // 清空子级
    for (let i = level + 1; i < this.value.length; i++) {
      delete this.value[i];
    }
    
    this.onChange?.(this.value);
  }
}

// 使用
const cascader = new Cascader({
  data: [
    {
      value: 'beijing',
      label: '北京',
      children: [
        {
          value: 'chaoyang',
          label: '朝阳区',
          children: [
            { value: 'sanlitun', label: '三里屯' }
          ]
        }
      ]
    }
  ],
  onChange: (value) => {
    console.log('选中的值:', value);
  }
});
```

## 二、图片加载场景

### 2.1 图片懒加载与预加载

**面试题：请实现一个图片懒加载方案，包括占位图、错误处理、渐进式加载等。**

#### 完整的图片加载解决方案

```javascript
class SmartImage {
  constructor(options) {
    this.placeholder = options.placeholder || this.generatePlaceholder();
    this.errorPlaceholder = options.errorPlaceholder;
    this.lazyLoad = options.lazyLoad !== false;
    this.loading = new Map();
    this.cache = new Map();
  }
  
  // 生成占位图
  generatePlaceholder(width = 100, height = 100, color = '#f0f0f0') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, width, height);
    return canvas.toDataURL();
  }
  
  // 创建图片元素
  createImage(src, options = {}) {
    const img = document.createElement('img');
    
    // 设置占位图
    img.src = this.placeholder;
    img.dataset.src = src;
    
    // 设置样式
    img.style.transition = 'opacity 0.3s';
    img.style.opacity = '0.3';
    
    // 懒加载
    if (this.lazyLoad) {
      this.setupLazyLoad(img, options);
    } else {
      this.loadImage(img, options);
    }
    
    return img;
  }
  
  // 设置懒加载
  setupLazyLoad(img, options) {
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            this.loadImage(img, options);
            observer.unobserve(img);
          }
        });
      }, {
        rootMargin: '50px',
        threshold: 0.01
      });
      
      observer.observe(img);
    } else {
      window.addEventListener('scroll', this.debounce(() => {
        if (this.isInViewport(img)) {
          this.loadImage(img, options);
        }
      }, 100));
    }
  }
  
  // 加载图片
  loadImage(img, options) {
    const src = img.dataset.src;
    
    if (this.cache.has(src)) {
      this.showImage(img, src, options);
      return;
    }
    
    if (this.loading.has(src)) {
      this.loading.get(src).push(() => this.showImage(img, src, options));
      return;
    }
    
    this.loading.set(src, [() => this.showImage(img, src, options)]);
    
    const tempImg = new Image();
    
    tempImg.onload = () => {
      this.cache.set(src, true);
      const callbacks = this.loading.get(src) || [];
      callbacks.forEach(cb => cb());
      this.loading.delete(src);
    };
    
    tempImg.onerror = () => {
      if (this.errorPlaceholder) {
        img.src = this.errorPlaceholder;
      }
      img.style.opacity = '1';
      this.loading.delete(src);
      options?.onError?.();
    };
    
    // 渐进式加载（支持 webp）
    if (this.supportsWebP() && options?.webp) {
      tempImg.src = options.webp;
    } else if (options?.srcset) {
      tempImg.srcset = options.srcset;
    } else {
      tempImg.src = src;
    }
  }
  
  // 显示图片
  showImage(img, src, options) {
    img.src = src;
    img.style.opacity = '1';
    options?.onLoad?.();
  }
  
  // 检查是否在视口内
  isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
      rect.top < window.innerHeight + 100 &&
      rect.bottom > -100
    );
  }
  
  // 检查 WebP 支持
  supportsWebP() {
    if (this._supportsWebP !== undefined) {
      return this._supportsWebP;
    }
    
    const canvas = document.createElement('canvas');
    this._supportsWebP = canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
    
    return this._supportsWebP;
  }
  
  // 防抖
  debounce(fn, delay) {
    let timer = null;
    return function(...args) {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  }
  
  // 预加载图片
  preloadImages(sources) {
    sources.forEach(src => {
      const img = new Image();
      img.src = src;
    });
  }
}

// 使用示例
const smartImage = new SmartImage({
  placeholder: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
});

const img = smartImage.createImage('/path/to/image.jpg', {
  webp: '/path/to/image.webp',
  onLoad: () => console.log('图片加载成功'),
  onError: () => console.log('图片加载失败')
});

document.body.appendChild(img);

// 预加载
smartImage.preloadImages([
  '/image1.jpg',
  '/image2.jpg'
]);
```

### 2.2 图片裁剪与压缩

```javascript
// 图片裁剪工具
class ImageCropper {
  constructor(options) {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.options = options || {};
  }
  
  // 裁剪图片
  async crop(image, cropData) {
    const img = await this.loadImage(image);
    
    // 计算裁剪区域
    const { x, y, width, height } = cropData;
    
    this.canvas.width = width;
    this.canvas.height = height;
    
    // 绘制裁剪区域
    this.ctx.drawImage(
      img,
      x, y, width, height,
      0, 0, width, height
    );
    
    return this.canvas.toDataURL('image/jpeg', this.options.quality || 0.8);
  }
  
  // 压缩图片
  async compress(image, options = {}) {
    const img = await this.loadImage(image);
    const { maxWidth, maxHeight, quality = 0.8 } = options;
    
    let { width, height } = img;
    
    // 计算缩放比例
    if (maxWidth && width > maxWidth) {
      height = (maxWidth / width) * height;
      width = maxWidth;
    }
    
    if (maxHeight && height > maxHeight) {
      width = (maxHeight / height) * width;
      height = maxHeight;
    }
    
    this.canvas.width = width;
    this.canvas.height = height;
    
    this.ctx.drawImage(img, 0, 0, width, height);
    
    return this.canvas.toDataURL('image/jpeg', quality);
  }
  
  // 旋转图片
  async rotate(image, degrees) {
    const img = await this.loadImage(image);
    
    const radians = (degrees * Math.PI) / 180;
    
    // 计算旋转后的尺寸
    const sin = Math.abs(Math.sin(radians));
    const cos = Math.abs(Math.cos(radians));
    const newWidth = img.width * cos + img.height * sin;
    const newHeight = img.width * sin + img.height * cos;
    
    this.canvas.width = newWidth;
    this.canvas.height = newHeight;
    
    this.ctx.translate(newWidth / 2, newHeight / 2);
    this.ctx.rotate(radians);
    this.ctx.drawImage(img, -img.width / 2, -img.height / 2);
    
    return this.canvas.toDataURL('image/jpeg', 0.8);
  }
  
  // 加载图片
  loadImage(image) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      
      if (typeof image === 'string') {
        img.src = image;
      } else if (image instanceof File) {
        img.src = URL.createObjectURL(image);
      }
    });
  }
}

// 使用
const cropper = new ImageCropper();

// 裁剪
const cropped = await cropper.crop('/image.jpg', {
  x: 100, y: 100, width: 200, height: 200
});

// 压缩
const compressed = await cropper.compress('/image.jpg', {
  maxWidth: 1920,
  maxHeight: 1080,
  quality: 0.7
});
```

## 三、购物车场景

### 3.1 购物车核心功能

**面试题：请实现一个完整的购物车，包括加购、减购、删除、全选、小计、总计、数据持久化等。**

#### 购物车完整实现

```javascript
class ShoppingCart {
  constructor(options = {}) {
    this.storageKey = options.storageKey || 'shopping_cart';
    this.onChange = options.onChange;
    this.items = this.load();
    this.listeners = new Set();
  }
  
  // 加载购物车
  load() {
    try {
      const data = localStorage.getItem(this.storageKey);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('加载购物车失败:', error);
      return [];
    }
  }
  
  // 保存购物车
  save() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.items));
      this.notify();
    } catch (error) {
      console.error('保存购物车失败:', error);
    }
  }
  
  // 添加商品
  addItem(product, quantity = 1) {
    const existingIndex = this.items.findIndex(
      item => item.productId === product.id && item.skuId === product.skuId
    );
    
    if (existingIndex > -1) {
      this.items[existingIndex].quantity += quantity;
    } else {
      this.items.push({
        id: Date.now().toString(),
        productId: product.id,
        skuId: product.skuId,
        name: product.name,
        price: product.price,
        image: product.image,
        quantity: quantity,
        maxQuantity: product.maxQuantity || 99,
        selected: true,
        specs: product.specs || []
      });
    }
    
    this.save();
  }
  
  // 更新数量
  updateQuantity(itemId, quantity) {
    const item = this.items.find(item => item.id === itemId);
    
    if (item) {
      item.quantity = Math.min(Math.max(1, quantity), item.maxQuantity);
      this.save();
    }
  }
  
  // 删除商品
  removeItem(itemId) {
    this.items = this.items.filter(item => item.id !== itemId);
    this.save();
  }
  
  // 清空购物车
  clear() {
    this.items = [];
    this.save();
  }
  
  // 切换选中状态
  toggleSelection(itemId) {
    const item = this.items.find(item => item.id === itemId);
    if (item) {
      item.selected = !item.selected;
      this.save();
    }
  }
  
  // 全选/取消全选
  toggleSelectAll(selected) {
    this.items.forEach(item => {
      item.selected = selected;
    });
    this.save();
  }
  
  // 获取商品小计
  getItemSubtotal(itemId) {
    const item = this.items.find(item => item.id === itemId);
    return item ? item.price * item.quantity : 0;
  }
  
  // 获取选中商品数量
  getSelectedCount() {
    return this.items
      .filter(item => item.selected)
      .reduce((sum, item) => sum + item.quantity, 0);
  }
  
  // 获取选中商品总价
  getSelectedTotal() {
    return this.items
      .filter(item => item.selected)
      .reduce((sum, item) => sum + item.price * item.quantity, 0);
  }
  
  // 检查是否全选
  isAllSelected() {
    return this.items.length > 0 && 
           this.items.every(item => item.selected);
  }
  
  // 获取购物车商品数量
  getTotalCount() {
    return this.items.reduce((sum, item) => sum + item.quantity, 0);
  }
  
  // 订阅变化
  subscribe(listener) {
    this.listeners.add(listener);
  }
  
  // 取消订阅
  unsubscribe(listener) {
    this.listeners.delete(listener);
  }
  
  // 通知变化
  notify() {
    this.listeners.forEach(listener => listener(this.items));
    this.onChange?.(this.items);
  }
  
  // 同步到服务器
  async syncToServer() {
    try {
      const response = await fetch('/api/cart/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: this.items })
      });
      
      if (response.ok) {
        const data = await response.json();
        this.items = data.items;
        this.save();
      }
    } catch (error) {
      console.error('同步购物车失败:', error);
    }
  }
}

// 使用示例
const cart = new ShoppingCart({
  storageKey: 'my_cart',
  onChange: (items) => {
    console.log('购物车变化:', items);
    updateCartUI(items);
  }
});

// 添加商品
cart.addItem({
  id: '1',
  skuId: 'sku-1',
  name: 'Apple iPhone 15',
  price: 5999,
  image: '/iphone15.jpg',
  maxQuantity: 10
}, 1);

// 更新数量
cart.updateQuantity('1', 2);

// 删除商品
cart.removeItem('1');

// 全选
cart.toggleSelectAll(true);

// 获取总价
const total = cart.getSelectedTotal();
console.log('总价:', total);

// 订阅变化
cart.subscribe((items) => {
  updateBadge(cart.getTotalCount());
});
```

### 3.2 购物车优化

```javascript
// 购物车优化版本
class OptimizedShoppingCart extends ShoppingCart {
  constructor(options) {
    super(options);
    this.snapshot = this.createSnapshot();
    this.debounceTimer = null;
  }
  
  // 创建快照（用于撤销）
  createSnapshot() {
    return JSON.parse(JSON.stringify(this.items));
  }
  
  // 撤销
  undo() {
    this.items = this.snapshot;
    this.save();
  }
  
  // 批量操作
  batchUpdate(updates) {
    this.snapshot = this.createSnapshot();
    
    updates.forEach(update => {
      switch (update.type) {
        case 'add':
          this.addItem(update.product, update.quantity);
          break;
        case 'update':
          this.updateQuantity(update.itemId, update.quantity);
          break;
        case 'remove':
          this.removeItem(update.itemId);
          break;
      }
    });
  }
  
  // 防抖保存
  debouncedSave() {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    
    this.debounceTimer = setTimeout(() => {
      this.save();
      this.syncToServer();
    }, 500);
  }
  
  // 本地持久化 + 服务端同步
  async syncStrategy() {
    try {
      const response = await fetch('/api/cart');
      const serverCart = await response.json();
      
      if (serverCart.items.length > this.items.length) {
        this.items = serverCart.items;
        this.save();
      } else if (this.items.length > serverCart.items.length) {
        await this.syncToServer();
      }
    } catch (error) {
      console.warn('使用本地购物车');
    }
  }
}
```

## 四、搜索场景

### 4.1 实时搜索与防抖

**面试题：请实现一个实时搜索功能，包括防抖、历史记录、高亮关键词等。**

#### 实时搜索完整实现

```javascript
class SearchEngine {
  constructor(options = {}) {
    this.debounceTime = options.debounceTime || 300;
    this.historyKey = options.historyKey || 'search_history';
    this.historyLimit = options.historyLimit || 10;
    this.highlightClass = options.highlightClass || 'search-highlight';
    this.debounceTimer = null;
  }
  
  // 搜索
  search(query, data) {
    return new Promise((resolve) => {
      if (this.debounceTimer) {
        clearTimeout(this.debounceTimer);
      }
      
      this.debounceTimer = setTimeout(() => {
        const results = this.performSearch(query, data);
        this.saveHistory(query);
        resolve(results);
      }, this.debounceTime);
    });
  }
  
  // 执行搜索
  performSearch(query, data) {
    if (!query) return [];
    
    const lowerQuery = query.toLowerCase();
    
    return data
      .filter(item => {
        return Object.values(item).some(value => 
          String(value).toLowerCase().includes(lowerQuery)
        );
      })
      .map(item => ({
        ...item,
        score: this.calculateScore(item, query)
      }))
      .sort((a, b) => b.score - a.score);
  }
  
  // 计算匹配分数
  calculateScore(item, query) {
    let score = 0;
    const lowerQuery = query.toLowerCase();
    
    Object.values(item).forEach(value => {
      const strValue = String(value).toLowerCase();
      
      if (strValue === lowerQuery) {
        score += 100;
      } else if (strValue.startsWith(lowerQuery)) {
        score += 50;
      } else if (strValue.includes(lowerQuery)) {
        score += 10;
      }
    });
    
    return score;
  }
  
  // 高亮关键词
  highlight(text, query) {
    if (!query) return text;
    
    const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
    
    return text.replace(regex, (match) => {
      return `<span class="${this.highlightClass}">${match}</span>`;
    });
  }
  
  // 转义正则特殊字符
  escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  
  // 保存搜索历史
  saveHistory(query) {
    if (!query.trim()) return;
    
    try {
      const history = this.getHistory();
      
      // 移除已存在的
      const filteredHistory = history.filter(
        item => item.toLowerCase() !== query.toLowerCase()
      );
      
      // 添加到开头
      filteredHistory.unshift(query);
      
      // 限制数量
      const limitedHistory = filteredHistory.slice(0, this.historyLimit);
      
      localStorage.setItem(this.historyKey, JSON.stringify(limitedHistory));
    } catch (error) {
      console.error('保存搜索历史失败:', error);
    }
  }
  
  // 获取搜索历史
  getHistory() {
    try {
      const history = localStorage.getItem(this.historyKey);
      return history ? JSON.parse(history) : [];
    } catch (error) {
      console.error('获取搜索历史失败:', error);
      return [];
    }
  }
  
  // 清除搜索历史
  clearHistory() {
    localStorage.removeItem(this.historyKey);
  }
}

// 使用示例
const searchEngine = new SearchEngine({
  debounceTime: 300,
  historyLimit: 15
});

const products = [
  { id: 1, name: 'Apple iPhone 15', category: '手机' },
  { id: 2, name: 'Samsung Galaxy S23', category: '手机' },
  { id: 3, name: 'MacBook Pro 14', category: '电脑' }
];

// 搜索
const handleSearch = async (query) => {
  const results = await searchEngine.search(query, products);
  renderResults(results, query);
};

// 渲染结果
const renderResults = (results, query) => {
  const container = document.getElementById('search-results');
  container.innerHTML = results.map(item => `
    <div class="search-result">
      <span>${searchEngine.highlight(item.name, query)}</span>
      <small>${item.category}</small>
    </div>
  `).join('');
};

// 显示历史记录
const renderHistory = () => {
  const history = searchEngine.getHistory();
  const container = document.getElementById('search-history');
  
  container.innerHTML = history.map(item => `
    <div class="history-item" onclick="handleSearch('${item}')">
      ${item}
    </div>
  `).join('');
};
```

### 4.2 搜索优化

```javascript
// 搜索优化版本
class OptimizedSearchEngine extends SearchEngine {
  constructor(options) {
    super(options);
    this.invertedIndex = new Map();
  }
  
  // 构建倒排索引
  buildIndex(data) {
    this.invertedIndex.clear();
    
    data.forEach((item, index) => {
      Object.values(item).forEach(value => {
        const tokens = this.tokenize(String(value));
        tokens.forEach(token => {
          if (!this.invertedIndex.has(token)) {
            this.invertedIndex.set(token, new Set());
          }
          this.invertedIndex.get(token).add(index);
        });
      });
    });
  }
  
  // 分词
  tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 0);
  }
  
  // 优化搜索
  optimizedSearch(query, data) {
    const tokens = this.tokenize(query);
    
    if (tokens.length === 0) return [];
    
    // 获取匹配的索引
    const resultSets = tokens.map(token => 
      this.invertedIndex.get(token) || new Set()
    );
    
    // 取交集
    const intersection = resultSets.reduce((acc, set) => {
      if (acc.size === 0) return set;
      return new Set([...acc].filter(x => set.has(x)));
    }, new Set());
    
    // 返回结果
    return [...intersection].map(index => data[index]);
  }
  
  // 拼音搜索（使用拼音库）
  pinyinSearch(query, data, pinyinMap) {
    const pinyinQuery = this.toPinyin(query);
    
    return data.filter(item => {
      const itemPinyin = pinyinMap[item.id] || this.toPinyin(item.name);
      return itemPinyin.includes(pinyinQuery);
    });
  }
  
  // 简单的拼音转换（实际使用拼音库如 pinyin-pro）
  toPinyin(text) {
    return text.toLowerCase();
  }
}
```

## 五、支付场景

### 5.1 支付流程设计

**面试题：请设计一个完整的支付流程，包括订单创建、支付状态轮询、支付结果处理等。**

#### 支付流程完整实现

```javascript
class PaymentManager {
  constructor(options = {}) {
    this.pollingInterval = options.pollingInterval || 2000;
    this.maxPollingTime = options.maxPollingTime || 60000;
    this.onSuccess = options.onSuccess;
    this.onFailure = options.onFailure;
    this.onPending = options.onPending;
    this.pollingTimer = null;
    this.pollingStartTime = null;
  }
  
  // 发起支付
  async initiatePayment(paymentData) {
    try {
      const response = await fetch('/api/payment/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(paymentData)
      });
      
      const data = await response.json();
      
      if (data.success) {
        this.startPolling(data.orderId);
        this.openPaymentMethod(data);
        return data;
      } else {
        this.onFailure?.(data.error);
        throw new Error(data.error);
      }
    } catch (error) {
      console.error('发起支付失败:', error);
      this.onFailure?.(error.message);
      throw error;
    }
  }
  
  // 打开支付方式
  openPaymentMethod(data) {
    switch (data.paymentMethod) {
      case 'alipay':
        this.openAlipay(data);
        break;
      case 'wechat':
        this.openWechatPay(data);
        break;
      case 'qr':
        this.showQRCode(data);
        break;
    }
  }
  
  // 支付宝
  openAlipay(data) {
    if (window.AlipayJSBridge) {
      window.AlipayJSBridge.call('tradePay', {
        tradeNO: data.tradeNo
      }, (result) => {
        this.handlePaymentResult(result);
      });
    } else {
      window.location.href = data.payUrl;
    }
  }
  
  // 微信支付
  openWechatPay(data) {
    if (typeof WeixinJSBridge !== 'undefined') {
      WeixinJSBridge.invoke('getBrandWCPayRequest', {
        appId: data.appId,
        timeStamp: data.timeStamp,
        nonceStr: data.nonceStr,
        package: data.package,
        signType: data.signType,
        paySign: data.paySign
      }, (result) => {
        this.handlePaymentResult(result);
      });
    } else {
      this.showQRCode(data);
    }
  }
  
  // 显示二维码
  showQRCode(data) {
    const qrContainer = document.getElementById('qr-container');
    qrContainer.innerHTML = `
      <div class="qr-code">
        <img src="${data.qrCodeUrl}" alt="支付二维码">
        <p>请使用 ${data.paymentMethod === 'wechat' ? '微信' : '支付宝'} 扫码支付</p>
      </div>
    `;
  }
  
  // 开始轮询
  startPolling(orderId) {
    this.pollingStartTime = Date.now();
    this.poll(orderId);
  }
  
  // 轮询支付状态
  poll(orderId) {
    const elapsed = Date.now() - this.pollingStartTime;
    
    if (elapsed > this.maxPollingTime) {
      this.stopPolling();
      this.onFailure?.('支付超时，请稍后查询');
      return;
    }
    
    this.pollingTimer = setTimeout(async () => {
      try {
        const response = await fetch(`/api/payment/status?orderId=${orderId}`);
        const data = await response.json();
        
        switch (data.status) {
          case 'success':
            this.stopPolling();
            this.onSuccess?.(data);
            break;
          case 'pending':
            this.onPending?.(data);
            this.poll(orderId);
            break;
          case 'failed':
            this.stopPolling();
            this.onFailure?.(data.error);
            break;
          default:
            this.poll(orderId);
        }
      } catch (error) {
        console.error('轮询支付状态失败:', error);
        this.poll(orderId);
      }
    }, this.pollingInterval);
  }
  
  // 停止轮询
  stopPolling() {
    if (this.pollingTimer) {
      clearTimeout(this.pollingTimer);
      this.pollingTimer = null;
    }
  }
  
  // 处理支付结果
  handlePaymentResult(result) {
    if (result.resultCode === '9000') {
      this.onSuccess?.({ orderId: result.tradeNo });
    } else {
      this.onFailure?.(result.resultMsg || '支付失败');
    }
  }
  
  // 查询订单
  async queryOrder(orderId) {
    try {
      const response = await fetch(`/api/order/${orderId}`);
      return await response.json();
    } catch (error) {
      console.error('查询订单失败:', error);
      throw error;
    }
  }
  
  // 取消订单
  async cancelOrder(orderId) {
    this.stopPolling();
    
    try {
      const response = await fetch(`/api/order/${orderId}/cancel`, {
        method: 'POST'
      });
      return await response.json();
    } catch (error) {
      console.error('取消订单失败:', error);
      throw error;
    }
  }
}

// 使用示例
const paymentManager = new PaymentManager({
  pollingInterval: 3000,
  maxPollingTime: 120000,
  onSuccess: (data) => {
    showPaymentSuccess(data.orderId);
  },
  onFailure: (error) => {
    showPaymentError(error);
  },
  onPending: (data) => {
    updatePaymentStatus('支付中...');
  }
});

// 发起支付
const handlePayment = async () => {
  await paymentManager.initiatePayment({
    orderId: 'order-123',
    amount: 99.00,
    paymentMethod: 'wechat',
    goodsName: 'iPhone 15'
  });
};

// 取消支付
const handleCancel = async () => {
  await paymentManager.cancelOrder('order-123');
};
```

### 5.2 支付安全

```javascript
// 支付安全增强
class SecurePaymentManager extends PaymentManager {
  constructor(options) {
    super(options);
    this.encryptedFields = ['cardNumber', 'cvv', 'expiryDate'];
  }
  
  // 加密敏感数据
  encryptSensitiveData(data) {
    const encryptedData = { ...data };
    
    this.encryptedFields.forEach(field => {
      if (encryptedData[field]) {
        encryptedData[field] = this.encrypt(encryptedData[field]);
      }
    });
    
    return encryptedData;
  }
  
  // 简单加密（实际使用 crypto-js）
  encrypt(text) {
    return btoa(text);  // base64，实际项目用 AES
  }
  
  // 验证支付环境
  validateEnvironment() {
    const checks = {
      https: window.location.protocol === 'https:',
      referer: document.referrer.includes(window.location.origin),
      userAgent: navigator.userAgent.includes('Mozilla')
    };
    
    return Object.values(checks).every(Boolean);
  }
  
  // 防止重复支付
  preventDuplicatePayment() {
    let isPaying = false;
    
    return {
      start: () => {
        if (isPaying) {
          throw new Error('支付正在进行中，请稍候');
        }
        isPaying = true;
      },
      end: () => {
        isPaying = false;
      }
    };
  }
  
  // 记录支付日志
  logPaymentEvent(event, data) {
    const logData = {
      event,
      timestamp: Date.now(),
      orderId: data.orderId,
      amount: data.amount,
      userAgent: navigator.userAgent,
      ip: '127.0.0.1'  // 从服务端获取
    };
    
    console.log('支付日志:', logData);
    // 发送到日志服务
  }
}
```

## 六、聊天场景

### 6.1 聊天功能实现

**面试题：请实现一个聊天功能，包括消息发送、消息接收、消息列表优化等。**

#### 聊天功能完整实现

```javascript
class ChatManager {
  constructor(options = {}) {
    this.wsUrl = options.wsUrl;
    this.roomId = options.roomId;
    this.userId = options.userId;
    this.messages = [];
    this.maxMessages = options.maxMessages || 100;
    this.messageCache = new Map();
    this.onMessage = options.onMessage;
    this.onStatusChange = options.onStatusChange;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }
  
  // 连接
  connect() {
    this.ws = new WebSocket(this.wsUrl);
    
    this.ws.onopen = () => {
      console.log('聊天连接成功');
      this.reconnectAttempts = 0;
      this.onStatusChange?.('connected');
      this.joinRoom();
    };
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
    
    this.ws.onclose = () => {
      console.log('聊天连接关闭');
      this.onStatusChange?.('disconnected');
      this.reconnect();
    };
    
    this.ws.onerror = (error) => {
      console.error('聊天连接错误:', error);
      this.onStatusChange?.('error');
    };
  }
  
  // 加入房间
  joinRoom() {
    this.send({
      type: 'join',
      roomId: this.roomId,
      userId: this.userId
    });
  }
  
  // 发送消息
  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      this.messageCache.set(Date.now().toString(), message);
    }
  }
  
  // 发送文本消息
  sendText(text, options = {}) {
    const message = {
      id: Date.now().toString(),
      type: 'text',
      roomId: this.roomId,
      userId: this.userId,
      content: text,
      timestamp: Date.now(),
      ...options
    };
    
    this.addMessage(message);
    this.send({ type: 'message', data: message });
  }
  
  // 发送图片
  sendImage(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      const message = {
        id: Date.now().toString(),
        type: 'image',
        roomId: this.roomId,
        userId: this.userId,
        content: event.target.result,
        timestamp: Date.now()
      };
      
      this.addMessage(message);
      this.send({ type: 'message', data: message });
    };
    reader.readAsDataURL(file);
  }
  
  // 处理消息
  handleMessage(data) {
    switch (data.type) {
      case 'message':
        this.addMessage(data.data);
        this.onMessage?.(data.data);
        break;
      case 'history':
        this.messages = data.messages;
        break;
      case 'typing':
        this.handleTyping(data);
        break;
    }
  }
  
  // 添加消息
  addMessage(message) {
    this.messages.push(message);
    
    if (this.messages.length > this.maxMessages) {
      this.messages.shift();
    }
  }
  
  // 获取消息
  getMessages() {
    return [...this.messages];
  }
  
  // 发送正在输入状态
  sendTyping(isTyping) {
    this.send({
      type: 'typing',
      roomId: this.roomId,
      userId: this.userId,
      isTyping
    });
  }
  
  // 处理正在输入
  handleTyping(data) {
    if (data.userId !== this.userId) {
      console.log(`${data.userId} 正在输入...`);
    }
  }
  
  // 加载历史消息
  loadHistory(before = null) {
    this.send({
      type: 'history',
      roomId: this.roomId,
      before,
      limit: 50
    });
  }
  
  // 断开连接
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
  
  // 重连
  reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      
      setTimeout(() => {
        console.log(`第 ${this.reconnectAttempts} 次重连...`);
        this.connect();
      }, delay);
    }
  }
}

// 使用示例
const chatManager = new ChatManager({
  wsUrl: 'ws://localhost:8080/chat',
  roomId: 'room-1',
  userId: 'user-1',
  onMessage: (message) => {
    renderMessage(message);
    scrollToBottom();
  },
  onStatusChange: (status) => {
    updateConnectionStatus(status);
  }
});

// 连接
chatManager.connect();

// 发送消息
const sendMessage = (text) => {
  chatManager.sendText(text);
  clearInput();
};

// 发送图片
const handleImageUpload = (file) => {
  chatManager.sendImage(file);
};

// 加载更多历史消息
const loadMoreMessages = () => {
  const firstMessage = chatManager.getMessages()[0];
  chatManager.loadHistory(firstMessage?.timestamp);
};

// 正在输入
let typingTimer = null;
const handleInput = () => {
  chatManager.sendTyping(true);
  
  if (typingTimer) {
    clearTimeout(typingTimer);
  }
  
  typingTimer = setTimeout(() => {
    chatManager.sendTyping(false);
  }, 1000);
};
```

### 6.2 消息列表优化

```javascript
// 虚拟滚动消息列表
class VirtualMessageList {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    this.itemHeight = options.itemHeight || 80;
    this.bufferSize = options.bufferSize || 5;
    
    this.messages = [];
    this.init();
  }
  
  init() {
    this.container.style.overflow = 'auto';
    this.container.style.position = 'relative';
    
    this.placeholder = document.createElement('div');
    this.placeholder.style.position = 'absolute';
    this.placeholder.style.top = 0;
    this.placeholder.style.left = 0;
    this.placeholder.style.right = 0;
    this.container.appendChild(this.placeholder);
    
    this.viewport = document.createElement('div');
    this.viewport.style.position = 'absolute';
    this.viewport.style.top = 0;
    this.viewport.style.left = 0;
    this.viewport.style.right = 0;
    this.container.appendChild(this.viewport);
    
    this.container.addEventListener('scroll', () => this.render());
  }
  
  setMessages(messages) {
    this.messages = messages;
    this.updatePlaceholder();
    this.render();
  }
  
  updatePlaceholder() {
    const totalHeight = this.messages.length * this.itemHeight;
    this.placeholder.style.height = `${totalHeight}px`;
  }
  
  render() {
    const scrollTop = this.container.scrollTop;
    const viewportHeight = this.container.clientHeight;
    const totalHeight = this.messages.length * this.itemHeight;
    
    const startIndex = Math.max(0, Math.floor(scrollTop / this.itemHeight) - this.bufferSize);
    const endIndex = Math.min(
      this.messages.length,
      Math.ceil((scrollTop + viewportHeight) / this.itemHeight) + this.bufferSize
    );
    
    const visibleMessages = this.messages.slice(startIndex, endIndex);
    
    this.viewport.innerHTML = '';
    this.viewport.style.transform = `translateY(${startIndex * this.itemHeight}px)`;
    
    visibleMessages.forEach((message, index) => {
      const el = this.renderMessage(message, startIndex + index);
      this.viewport.appendChild(el);
    });
  }
  
  renderMessage(message, index) {
    const el = document.createElement('div');
    el.style.minHeight = `${this.itemHeight}px`;
    el.style.padding = '10px';
    el.style.borderBottom = '1px solid #eee';
    
    el.innerHTML = `
      <div class="message">
        <div class="user">${message.userId}</div>
        <div class="content">${message.content}</div>
        <div class="time">${new Date(message.timestamp).toLocaleTimeString()}</div>
      </div>
    `;
    
    return el;
  }
  
  scrollToBottom() {
    this.container.scrollTop = this.placeholder.offsetHeight;
  }
}

// 使用
const virtualList = new VirtualMessageList('#message-list', {
  itemHeight: 100,
  bufferSize: 10
});

virtualList.setMessages(messages);
virtualList.scrollToBottom();
```

## 七、上传下载场景

### 7.1 大文件上传

**面试题：请实现一个大文件上传功能，包括分片上传、断点续传、上传进度等。**

#### 大文件上传完整实现

```javascript
class ChunkedUploader {
  constructor(options = {}) {
    this.chunkSize = options.chunkSize || 1024 * 1024; // 1MB
    this.concurrency = options.concurrency || 3;
    this.uploadUrl = options.uploadUrl;
    this.onProgress = options.onProgress;
    this.onComplete = options.onComplete;
    this.onError = options.onError;
    this.uploadingFiles = new Map();
  }
  
  // 上传文件
  async upload(file) {
    const fileId = this.generateFileId(file);
    const chunks = this.createChunks(file);
    
    this.uploadingFiles.set(fileId, {
      file,
      chunks,
      uploadedChunks: new Set(),
      totalChunks: chunks.length
    });
    
    try {
      const uploadedChunks = await this.checkUploadedChunks(fileId);
      this.uploadingFiles.get(fileId).uploadedChunks = new Set(uploadedChunks);
      
      await this.uploadChunks(fileId, chunks);
      await this.mergeChunks(fileId, file.name);
      
      this.onComplete?.(fileId);
      this.uploadingFiles.delete(fileId);
    } catch (error) {
      this.onError?.(error);
      throw error;
    }
  }
  
  // 生成文件 ID
  generateFileId(file) {
    return `${file.name}-${file.size}-${file.lastModified}`;
  }
  
  // 创建分片
  createChunks(file) {
    const chunks = [];
    let start = 0;
    
    while (start < file.size) {
      const end = Math.min(start + this.chunkSize, file.size);
      chunks.push({
        index: chunks.length,
        start,
        end,
        blob: file.slice(start, end)
      });
      start = end;
    }
    
    return chunks;
  }
  
  // 检查已上传的分片
  async checkUploadedChunks(fileId) {
    try {
      const response = await fetch(`${this.uploadUrl}/check?fileId=${fileId}`);
      const data = await response.json();
      return data.uploadedChunks || [];
    } catch (error) {
      console.error('检查上传状态失败:', error);
      return [];
    }
  }
  
  // 上传分片
  async uploadChunks(fileId, chunks) {
    const uploadInfo = this.uploadingFiles.get(fileId);
    const pendingChunks = chunks.filter(
      chunk => !uploadInfo.uploadedChunks.has(chunk.index)
    );
    
    let completed = uploadInfo.uploadedChunks.size;
    const total = chunks.length;
    
    const uploadChunk = async (chunk) => {
      const formData = new FormData();
      formData.append('fileId', fileId);
      formData.append('chunkIndex', chunk.index);
      formData.append('chunk', chunk.blob);
      
      try {
        const response = await fetch(`${this.uploadUrl}/chunk`, {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          uploadInfo.uploadedChunks.add(chunk.index);
          completed++;
          
          this.onProgress?.({
            fileId,
            progress: (completed / total) * 100,
            uploaded: completed,
            total
          });
        }
      } catch (error) {
        console.error(`分片 ${chunk.index} 上传失败:`, error);
        throw error;
      }
    };
    
    const pool = [];
    for (let i = 0; i < pendingChunks.length; i++) {
      const chunk = pendingChunks[i];
      const promise = uploadChunk(chunk);
      
      pool.push(promise);
      
      if (pool.length >= this.concurrency) {
        await Promise.race(pool);
        pool.splice(0, pool.length);
      }
    }
    
    await Promise.all(pool);
  }
  
  // 合并分片
  async mergeChunks(fileId, fileName) {
    const response = await fetch(`${this.uploadUrl}/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fileId, fileName })
    });
    
    if (!response.ok) {
      throw new Error('合并分片失败');
    }
    
    return await response.json();
  }
  
  // 暂停上传
  pause(fileId) {
    const uploadInfo = this.uploadingFiles.get(fileId);
    if (uploadInfo) {
      uploadInfo.paused = true;
    }
  }
  
  // 恢复上传
  resume(fileId) {
    const uploadInfo = this.uploadingFiles.get(fileId);
    if (uploadInfo) {
      uploadInfo.paused = false;
      this.upload(fileId);
    }
  }
  
  // 取消上传
  cancel(fileId) {
    this.uploadingFiles.delete(fileId);
  }
}

// 使用示例
const uploader = new ChunkedUploader({
  chunkSize: 2 * 1024 * 1024, // 2MB
  concurrency: 5,
  uploadUrl: '/api/upload',
  onProgress: (data) => {
    updateProgressBar(data.progress);
  },
  onComplete: (fileId) => {
    showUploadSuccess(fileId);
  },
  onError: (error) => {
    showUploadError(error);
  }
});

// 选择文件
const handleFileSelect = async (file) => {
  await uploader.upload(file);
};
```

### 7.2 下载功能

```javascript
class DownloadManager {
  constructor(options = {}) {
    this.downloads = new Map();
    this.onProgress = options.onProgress;
    this.onComplete = options.onComplete;
  }
  
  // 下载文件
  async download(url, filename) {
    const downloadId = Date.now().toString();
    
    try {
      const response = await fetch(url);
      const total = parseInt(response.headers.get('content-length'));
      const reader = response.body.getReader();
      
      let received = 0;
      const chunks = [];
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        chunks.push(value);
        received += value.length;
        
        const progress = total ? (received / total) * 100 : 0;
        this.onProgress?.({ downloadId, progress, received, total });
      }
      
      const blob = new Blob(chunks);
      this.saveFile(blob, filename);
      
      this.onComplete?.(downloadId);
      return blob;
    } catch (error) {
      console.error('下载失败:', error);
      throw error;
    }
  }
  
  // 保存文件
  saveFile(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
  
  // 分块下载大文件
  async downloadInChunks(url, filename, chunkSize = 10 * 1024 * 1024) {
    try {
      const headResponse = await fetch(url, { method: 'HEAD' });
      const totalSize = parseInt(headResponse.headers.get('content-length'));
      
      const chunks = [];
      let start = 0;
      
      while (start < totalSize) {
        const end = Math.min(start + chunkSize, totalSize);
        const response = await fetch(url, {
          headers: {
            Range: `bytes=${start}-${end - 1}`
          }
        });
        
        const blob = await response.blob();
        chunks.push(blob);
        
        start = end;
        
        const progress = (start / totalSize) * 100;
        this.onProgress?.({ progress, received: start, total: totalSize });
      }
      
      const finalBlob = new Blob(chunks);
      this.saveFile(finalBlob, filename);
      
      return finalBlob;
    } catch (error) {
      console.error('分块下载失败:', error);
      throw error;
    }
  }
}

// 使用
const downloadManager = new DownloadManager({
  onProgress: (data) => {
    updateDownloadProgress(data.progress);
  },
  onComplete: () => {
    showDownloadComplete();
  }
});

const handleDownload = async () => {
  await downloadManager.download('/file.pdf', 'document.pdf');
};
```

## 八、图表展示场景

### 8.1 图表封装

**面试题：请封装一个图表组件，支持多种图表类型、数据更新、事件监听等。**

#### 图表组件封装（基于 ECharts）

```javascript
class ChartComponent {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    this.options = options;
    this.data = [];
    this.theme = options.theme || 'light';
    this.listeners = new Map();
    
    this.init();
  }
  
  init() {
    this.chart = echarts.init(this.container, this.theme);
    
    window.addEventListener('resize', () => {
      this.chart.resize();
    });
    
    this.chart.on('click', (params) => {
      this.emit('click', params);
    });
    
    this.chart.on('legendselectchanged', (params) => {
      this.emit('legendselectchanged', params);
    });
  }
  
  setData(data) {
    this.data = data;
    this.render();
  }
  
  updateData(data) {
    this.data = { ...this.data, ...data };
    this.render();
  }
  
  render() {
    const option = this.getOption();
    this.chart.setOption(option, true);
  }
  
  getOption() {
    return {
      title: this.options.title,
      tooltip: {
        trigger: this.options.tooltipTrigger || 'axis',
        formatter: this.options.tooltipFormatter
      },
      legend: {
        data: this.data.legend || []
      },
      xAxis: {
        type: 'category',
        data: this.data.xAxis || []
      },
      yAxis: {
        type: 'value'
      },
      series: this.data.series || []
    };
  }
  
  setOption(option) {
    this.chart.setOption(option);
  }
  
  on(event, handler) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(handler);
  }
  
  off(event, handler) {
    if (this.listeners.has(event)) {
      const handlers = this.listeners.get(event);
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }
  
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(handler => {
        handler(data);
      });
    }
  }
  
  resize() {
    this.chart.resize();
  }
  
  getInstance() {
    return this.chart;
  }
  
  destroy() {
    this.chart.dispose();
    this.listeners.clear();
  }
}

// 折线图
class LineChart extends ChartComponent {
  getOption() {
    return {
      ...super.getOption(),
      series: this.data.series?.map(s => ({
        ...s,
        type: 'line',
        smooth: s.smooth !== false
      }))
    };
  }
}

// 柱状图
class BarChart extends ChartComponent {
  getOption() {
    return {
      ...super.getOption(),
      series: this.data.series?.map(s => ({
        ...s,
        type: 'bar'
      }))
    };
  }
}

// 饼图
class PieChart extends ChartComponent {
  getOption() {
    return {
      tooltip: {
        trigger: 'item'
      },
      series: [
        {
          type: 'pie',
          radius: this.options.radius || '60%',
          data: this.data
        }
      ]
    };
  }
}

// 使用示例
const lineChart = new LineChart('#line-chart', {
  title: { text: '销售趋势' }
});

lineChart.setData({
  xAxis: ['一月', '二月', '三月', '四月', '五月'],
  series: [
    {
      name: '销售额',
      data: [820, 932, 901, 934, 1290]
    }
  ]
});

lineChart.on('click', (params) => {
  console.log('点击了:', params);
});

// 响应式
window.addEventListener('resize', () => {
  lineChart.resize();
});
```

### 8.2 图表数据处理

```javascript
class ChartDataProcessor {
  // 数据聚合
  aggregate(data, groupBy, aggregateBy) {
    const grouped = {};
    
    data.forEach(item => {
      const key = item[groupBy];
      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(item);
    });
    
    return Object.entries(grouped).map(([key, items]) => ({
      [groupBy]: key,
      [aggregateBy]: items.reduce((sum, item) => sum + item[aggregateBy], 0)
    }));
  }
  
  // 时间序列数据
  timeSeries(data, timeField, valueField, interval = 'day') {
    const sorted = [...data].sort((a, b) => 
      new Date(a[timeField]) - new Date(b[timeField])
    );
    
    return sorted.map(item => ({
      time: new Date(item[timeField]).toLocaleDateString(),
      value: item[valueField]
    }));
  }
  
  // 数据过滤
  filter(data, condition) {
    return data.filter(condition);
  }
  
  // 数据排序
  sort(data, field, order = 'asc') {
    return [...data].sort((a, b) => {
      const aVal = a[field];
      const bVal = b[field];
      return order === 'asc' 
        ? aVal - bVal 
        : bVal - aVal;
    });
  }
  
  // 数据格式化
  format(data, formatter) {
    return data.map(formatter);
  }
  
  // 计算移动平均
  movingAverage(data, windowSize = 5) {
    const result = [];
    
    for (let i = 0; i < data.length; i++) {
      const windowStart = Math.max(0, i - windowSize + 1);
      const windowData = data.slice(windowStart, i + 1);
      const avg = windowData.reduce((sum, item) => sum + item.value, 0) / windowData.length;
      
      result.push({
        ...data[i],
        average: avg
      });
    }
    
    return result;
  }
}

// 使用
const processor = new ChartDataProcessor();

const rawData = [
  { date: '2024-01-01', category: 'A', value: 100 },
  { date: '2024-01-02', category: 'A', value: 150 },
  { date: '2024-01-01', category: 'B', value: 200 }
];

const aggregated = processor.aggregate(rawData, 'category', 'value');
const timeSeries = processor.timeSeries(rawData, 'date', 'value');
const withMA = processor.movingAverage(timeSeries, 3);
```

## 九、地图展示场景

### 9.1 地图组件封装

**面试题：请封装一个地图组件，支持标记点、区域、路径、交互等功能。**

#### 地图组件封装（基于百度地图）

```javascript
class MapComponent {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    this.options = options;
    this.markers = [];
    this.polygons = [];
    this.listeners = new Map();
    
    this.init();
  }
  
  init() {
    this.map = new BMapGL.Map(this.container);
    
    const center = this.options.center || new BMapGL.Point(116.404, 39.915);
    const zoom = this.options.zoom || 12;
    
    this.map.centerAndZoom(center, zoom);
    this.map.enableScrollWheelZoom(true);
    
    if (this.options.mapStyle) {
      this.map.setMapStyleV2({ styleId: this.options.mapStyle });
    }
  }
  
  addMarker(point, options = {}) {
    const marker = new BMapGL.Marker(point, options);
    
    if (options.onClick) {
      marker.addEventListener('click', options.onClick);
    }
    
    this.map.addOverlay(marker);
    this.markers.push(marker);
    return marker;
  }
  
  destroy() {
    this.markers.forEach(m => this.map.removeOverlay(m));
    this.markers = [];
  }
}
```

## 十、总结

以上十大高频业务场景，涵盖了前端开发中最常遇到的问题：

1. **表单处理**：复杂验证、异步验证、联动
2. **图片加载**：懒加载、预加载、裁剪压缩
3. **购物车**：核心功能、持久化、并发处理
4. **搜索**：实时搜索、历史记录、高亮、倒排索引
5. **支付**：流程设计、状态轮询、安全防护
6. **聊天**：WebSocket、消息优化、虚拟滚动
7. **上传下载**：分片上传、断点续传、进度显示
8. **图表**：组件封装、数据处理、响应式
9. **地图**：基础组件、标记点、路径绘制

掌握这些业务场景，面试时遇到业务题就稳了！

---

**相关阅读**：
- [第一篇：JavaScript 核心与框架原理](099-frontend-interview.md)
- [第二篇：网络协议与工程化](100-frontend-interview-02.md)
    }
    
    if (options.label) {
      const label = new BMap