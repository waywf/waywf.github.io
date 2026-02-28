#!/bin/bash

# AI Blog 发布脚本
# 使用方法: ./publish.sh

set -e  # 遇到错误立即退出

echo "🚀 开始发布流程..."

# 1. 构建项目
echo "📦 正在构建项目..."
npm run build

# 2. 添加所有更改到git
echo "📋 添加更改到git..."
git add .

# 3. 提交更改
echo "💾 提交更改..."
git commit -m "init: 迁移文章"

# 4. 推送到远程仓库
echo "🚀 推送到远程仓库..."
git push

echo "✅ 发布完成！"
