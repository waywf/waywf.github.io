export interface Article {
  id: string;
  title: string;
  excerpt: string;
  content: string;
  category: string;
  date: string;
  readTime: number;
  tags: string[];
}

export const articles: Article[] = [
  {
    id: '1',
    title: '从零开始学习 Web 开发',
    excerpt: '探索现代 Web 开发的基础知识，从 HTML、CSS 到 JavaScript 的完整学习路径。',
    content: `# 从零开始学习 Web 开发

## 前言
Web 开发是当今最热门的技术领域之一。无论你是初学者还是有经验的开发者，都能在这个领域找到自己的位置。

## 基础知识
### HTML
HTML 是网页的骨架，提供了页面的结构和语义。

### CSS
CSS 负责页面的样式和布局，让网页看起来更加美观。

### JavaScript
JavaScript 是网页的大脑，提供了交互和动态功能。

## 学习路径
1. 掌握 HTML 基础
2. 学习 CSS 布局和设计
3. 深入学习 JavaScript
4. 学习前端框架（React、Vue 等）
5. 了解后端基础
6. 实践项目开发

## 总结
Web 开发是一个不断学习和进步的过程。坚持练习，不断挑战自己，你一定能成为优秀的开发者。`,
    category: '技术',
    date: '2026-01-25',
    readTime: 8,
    tags: ['Web', 'HTML', 'CSS', 'JavaScript'],
  },
  {
    id: '2',
    title: 'React Hooks 深度解析',
    excerpt: '深入理解 React Hooks 的原理和最佳实践，提升你的 React 开发能力。',
    content: `# React Hooks 深度解析

## 什么是 Hooks？
React Hooks 是 React 16.8 引入的新特性，允许你在函数组件中使用状态和其他 React 特性。

## 常用 Hooks
### useState
用于在函数组件中添加状态。

### useEffect
用于处理副作用，如数据获取、订阅等。

### useContext
用于在组件树中传递数据。

### useReducer
用于管理复杂的状态逻辑。

## 最佳实践
1. 只在顶层调用 Hooks
2. 只在 React 函数中调用 Hooks
3. 使用 ESLint 插件来检查 Hooks 的使用

## 总结
掌握 Hooks 是现代 React 开发的必备技能。`,
    category: '技术',
    date: '2026-01-20',
    readTime: 10,
    tags: ['React', 'Hooks', 'JavaScript'],
  },
  {
    id: '3',
    title: '生活的意义在于体验',
    excerpt: '思考生活的本质，探索如何在日常中找到意义和快乐。',
    content: `# 生活的意义在于体验

## 开篇
我们常常在忙碌中迷失自己，忘记了生活的本质。

## 体验的力量
每一个瞬间都是独特的，每一次体验都能让我们成长。

### 旅行
旅行让我们看到不同的世界，拓展我们的视野。

### 阅读
阅读让我们进入他人的世界，获得新的思想。

### 创作
创作让我们表达自己，与世界对话。

## 反思
生活不是为了到达某个终点，而是享受沿途的风景。

## 结语
珍惜每一个瞬间，体验生活的美好。`,
    category: '生活',
    date: '2026-01-15',
    readTime: 6,
    tags: ['生活', '思考', '体验'],
  },
  {
    id: '4',
    title: '2026 年前端技术展望',
    excerpt: '预测 2026 年前端技术的发展方向，包括 AI 集成、性能优化等。',
    content: `# 2026 年前端技术展望

## 人工智能集成
AI 将深入集成到前端应用中，提供更智能的用户体验。

## 性能优化
- 更快的加载速度
- 更优的用户体验
- 更低的能耗

## 新框架和工具
新的框架和工具将继续涌现，提高开发效率。

## Web 3.0
去中心化应用将逐渐成为主流。

## 总结
前端技术的发展永不停止，我们需要不断学习和适应。`,
    category: '技术',
    date: '2026-01-10',
    readTime: 7,
    tags: ['前端', 'AI', '技术展望'],
  },
];

export const categories = ['全部', '技术', '生活', '思考'];

export function getArticleById(id: string): Article | undefined {
  return articles.find(article => article.id === id);
}

export function getArticlesByCategory(category: string): Article[] {
  if (category === '全部') {
    return articles;
  }
  return articles.filter(article => article.category === category);
}

export function getLatestArticles(limit: number = 3): Article[] {
  return articles.slice(0, limit);
}
