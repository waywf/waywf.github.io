---
title: 博客系统开发实战 - 从设计到部署
excerpt: 记录使用 Vue 3 + Vite + Tailwind CSS 开发个人博客系统的完整过程，包括技术选型、架构设计和部署上线。
category: 前端开发
date: 2023-02-28
# readTime: 12
tags: Vue3, Vite, Tailwind, 部署
---

# 博客系统开发实战 - 从设计到部署

在这篇文章中，我将分享如何使用现代前端技术栈构建一个高性能、美观且易于维护的个人博客系统。

## 技术选型

### 前端框架：Vue 3

选择 Vue 3 是因为其强大的 Composition API 和出色的性能。Vue 3 的响应式系统得到了全面升级，组件渲染效率更高。

### 构建工具：Vite

Vite 是由 Vue 团队打造的下一代构建工具，它利用浏览器原生 ES 模块实现极速的热更新（HMR）。

### 样式方案：Tailwind CSS

Tailwind CSS 是一个实用优先的 CSS 框架，允许我们快速构建现代化的用户界面。

## 项目架构

```
ancheng_blog/
├── client/                 # 前端源代码
│   ├── src/
│   │   ├── components/    # 可复用组件
│   │   ├── pages/         # 页面组件
│   │   ├── lib/           # 工具函数
│   │   └── router/        # 路由配置
│   └── public/            # 静态资源
├── server/                # 后端服务
├── shared/                # 共享代码
└── dist/                  # 构建输出
```

## 核心功能实现

### 1. 文章管理系统

使用 Markdown 文件存储文章内容，通过解析 frontmatter 获取元数据。

### 2. 响应式设计

确保博客在各种设备上都能完美展示。

### 3. 主题切换

支持深色/浅色主题切换，提升用户体验。

## 部署上线

使用 GitHub Pages 作为静态托管，配合 GitHub Actions 实现自动化部署。

```bash
pnpm deploy
```

一行命令即可将博客部署到 GitHub Pages。

## 总结

通过这个项目，我深刻体会到了现代前端工具链的强大之处。Vue 3 + Vite + Tailwind CSS 的组合提供了极佳的开发体验。
