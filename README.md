# 70KG的个人空间

一个现代化的个人博客网站，采用 Vue 3 + Vite 构建，支持 Markdown 文章发布。

## 🎨 设计特色

- **赛博朋克风格**：深灰蓝背景配荧光绿和紫色，充满科技感
- **完全静态**：无需服务器，可直接部署到 GitHub Pages
- **Markdown 驱动**：通过编辑 Markdown 文件来发布文章
- **响应式设计**：完美适配所有设备

## 🚀 快速开始

### 安装依赖

```bash
pnpm install
```

### 开发模式

```bash
pnpm dev
```

访问 `http://localhost:3000` 查看网站。

### 构建生产版本

```bash
pnpm build
```

生成的静态文件在 `dist` 目录中。

## 📝 添加文章

### 1. 创建 Markdown 文件

在 `client/public/articles/` 目录下创建新的 Markdown 文件，例如 `my-article.md`：

```markdown
---
title: 我的第一篇文章
date: 2026-02-24
category: 技术
tags: Vue, 博客, 静态网站
excerpt: 这是文章摘要
---

# 文章内容

这里是你的文章内容...
```

### 2. 更新文章清单

编辑 `client/public/articles/manifest.json`，添加新文章：

```json
{
  "articles": [
    "001-vue3-intro.md",
    "002-tailwind-css.md",
    "my-article.md"
  ]
}
```

### 3. 重新构建

```bash
pnpm build
```

## 📦 部署到 GitHub Pages

### 1. 创建 GitHub 仓库

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/waywf/waywf.github.io.git
git branch -M main
git push -u origin main
```

### 2. 配置 GitHub Pages

1. 进入仓库的 Settings
2. 在 Pages 部分，选择 "GitHub Actions" 作为部署源
3. 推送代码后，GitHub Actions 会自动构建并部署

### 3. 访问网站

网站将在 `https://waywf.github.io` 上线。

## 📁 项目结构

```
ancheng_blog/
├── client/
│   ├── src/
│   │   ├── pages/          # Vue 页面组件
│   │   ├── components/     # 可复用组件
│   │   ├── lib/            # 工具函数
│   │   ├── App.vue         # 根组件
│   │   ├── main.ts         # 入口文件
│   │   └── index.css       # 全局样式
│   ├── public/
│   │   ├── articles/       # Markdown 文章
│   │   └── images/         # 图片资源
│   └── index.html          # HTML 模板
├── vite.config.ts          # Vite 配置
├── tsconfig.json           # TypeScript 配置
├── package.json            # 项目依赖
└── .github/workflows/      # GitHub Actions 工作流
```

## 🛠️ 技术栈

- **框架**：Vue 3 + Vue Router 4
- **构建工具**：Vite 7
- **样式**：Tailwind CSS 4
- **Markdown 渲染**：Streamdown
- **UI 组件**：shadcn/ui

## 📄 许可证

MIT

## 🤝 联系方式

- GitHub: [@waywf](https://github.com/waywf)
- Email: your-email@example.com
- Twitter: [@waywf](https://twitter.com/waywf)
