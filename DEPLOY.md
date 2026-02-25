# GitHub Pages 部署指南

本指南将帮助您将"70KG的个人空间"博客网站部署到 GitHub Pages。

## 前置条件

- 已有 GitHub 账户
- 已创建 `waywf.github.io` 仓库
- 本地已安装 Git 和 Node.js

## 部署步骤

### 1. 克隆或初始化本地仓库

如果您还没有本地仓库，请执行：

```bash
git clone https://github.com/waywf/waywf.github.io.git
cd waywf.github.io
```

### 2. 复制项目文件

将 `ancheng_blog` 项目中的所有文件复制到仓库根目录：

```bash
# 假设 ancheng_blog 在上级目录
cp -r ../ancheng_blog/* .
```

### 3. 安装依赖

```bash
pnpm install
```

### 4. 构建静态网站

```bash
pnpm build
```

这将生成 `dist` 目录，包含所有静态文件。

### 5. 配置 GitHub Actions（可选）

项目已包含 `.github/workflows/deploy.yml` 配置文件。这个工作流会在每次推送到 `main` 分支时自动构建和部署网站。

如果您想使用自动部署：

1. 确保 `.github/workflows/deploy.yml` 文件已存在
2. 推送代码到 GitHub
3. GitHub Actions 会自动运行构建和部署

### 6. 手动部署（如果不使用 GitHub Actions）

如果您想手动部署，可以使用 `gh-pages` 分支：

```bash
# 安装 gh-pages
npm install --save-dev gh-pages

# 添加部署脚本到 package.json
# "deploy": "pnpm build && gh-pages -d dist"

# 执行部署
pnpm deploy
```

### 7. 推送到 GitHub

```bash
git add .
git commit -m "Deploy blog to GitHub Pages"
git push origin main
```

## 验证部署

1. 访问 `https://waywf.github.io` 检查网站是否在线
2. 如果使用了 GitHub Actions，可以在仓库的 "Actions" 标签中查看部署状态

## 添加新文章

部署后，要添加新文章：

1. 在 `client/public/articles/` 目录下创建新的 Markdown 文件
2. 更新 `client/public/articles/manifest.json`
3. 运行 `pnpm build`
4. 提交并推送更改

```bash
git add .
git commit -m "Add new article: my-article"
git push origin main
```

## 常见问题

### Q: 网站没有显示？
A: 
1. 检查 GitHub Pages 是否已启用（Settings → Pages）
2. 确认 `dist` 目录中有 `index.html` 文件
3. 清除浏览器缓存并重新访问

### Q: 样式没有加载？
A: 
1. 检查 `vite.config.ts` 中的 `base` 配置
2. 确保所有资源路径都是相对路径或绝对路径

### Q: 文章没有显示？
A:
1. 检查 Markdown 文件是否在 `client/public/articles/` 目录中
2. 确认 `manifest.json` 中包含了文章文件名
3. 重新构建项目

## 自定义域名（可选）

如果您想使用自定义域名而不是 `waywf.github.io`：

1. 在 `client/public/` 目录下创建 `CNAME` 文件
2. 在文件中写入您的域名，例如 `ancheng.com`
3. 在您的域名提供商处配置 DNS 记录
4. 在 GitHub 仓库设置中配置自定义域名

## 更多信息

- [GitHub Pages 官方文档](https://docs.github.com/en/pages)
- [Vite 部署指南](https://vitejs.dev/guide/static-deploy.html)
- [Vue 3 官方文档](https://vuejs.org/)
