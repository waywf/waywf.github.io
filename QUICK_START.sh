#!/bin/bash

# 安澄的个人空间 - 快速部署脚本

echo "🚀 安澄的个人空间 - 快速部署"
echo "================================"

# 检查 Node.js 和 pnpm
if ! command -v node &> /dev/null; then
    echo "❌ 需要安装 Node.js"
    exit 1
fi

if ! command -v pnpm &> /dev/null; then
    echo "📦 安装 pnpm..."
    npm install -g pnpm
fi

echo "✅ 环境检查完成"

# 安装依赖
echo "📥 安装依赖..."
pnpm install

# 构建项目
echo "🔨 构建项目..."
pnpm build

echo ""
echo "✅ 构建完成！"
echo ""
echo "📁 静态文件位置: ./dist"
echo ""
echo "🌐 部署选项:"
echo "  1. GitHub Pages: 将 dist 目录推送到 GitHub"
echo "  2. 本地测试: pnpm preview"
echo "  3. 开发模式: pnpm dev"
echo ""
echo "📝 添加文章:"
echo "  1. 在 client/public/articles/ 创建 .md 文件"
echo "  2. 更新 client/public/articles/manifest.json"
echo "  3. 运行 pnpm build"
echo ""
