<template>
  <div class="min-h-screen flex flex-col bg-[var(--background)]">
    <Navigation />

    <!-- Header Section -->
    <section class="py-12 md:py-16 border-b border-[var(--border)]">
      <div class="container">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">
          <span class="text-[#00FF41]">{'$ publish'}</span>
        </h1>
        <p class="text-[var(--muted-foreground)] text-lg">
          创建和发布新的 Markdown 文章。
        </p>
      </div>
    </section>

    <!-- Publish Form -->
    <section class="py-12 md:py-16 flex-1">
      <div class="container max-w-4xl">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <!-- Form Section -->
          <div class="space-y-6">
            <div>
              <label class="block text-[#00FF41] font-bold mb-2">文章标题</label>
              <input
                v-model="form.title"
                type="text"
                placeholder="输入文章标题..."
                class="w-full px-4 py-3 bg-[var(--card)] border-2 border-[var(--border)] rounded text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:border-[#00FF41] focus:outline-none transition-colors"
              />
            </div>

            <div>
              <label class="block text-[#00FF41] font-bold mb-2">文章摘要</label>
              <textarea
                v-model="form.excerpt"
                placeholder="输入文章摘要..."
                rows="3"
                class="w-full px-4 py-3 bg-[var(--card)] border-2 border-[var(--border)] rounded text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:border-[#00FF41] focus:outline-none transition-colors resize-none"
              ></textarea>
            </div>

            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-[#00FF41] font-bold mb-2">分类</label>
                <select
                  v-model="form.category"
                  class="w-full px-4 py-3 bg-[var(--card)] border-2 border-[var(--border)] rounded text-[var(--foreground)] focus:border-[#00FF41] focus:outline-none transition-colors"
                >
                  <option value="技术">技术</option>
                  <option value="生活">生活</option>
                  <option value="思考">思考</option>
                </select>
              </div>

              <div>
                <label class="block text-[#00FF41] font-bold mb-2">阅读时间 (分钟)</label>
                <input
                  v-model.number="form.readTime"
                  type="number"
                  min="1"
                  class="w-full px-4 py-3 bg-[var(--card)] border-2 border-[var(--border)] rounded text-[var(--foreground)] focus:border-[#00FF41] focus:outline-none transition-colors"
                />
              </div>
            </div>

            <div>
              <label class="block text-[#00FF41] font-bold mb-2">标签 (用逗号分隔)</label>
              <input
                v-model="form.tagsInput"
                type="text"
                placeholder="例如: Vue, TypeScript, 前端"
                class="w-full px-4 py-3 bg-[var(--card)] border-2 border-[var(--border)] rounded text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:border-[#00FF41] focus:outline-none transition-colors"
              />
            </div>

            <div class="flex gap-4">
              <button
                @click="publishArticle"
                class="flex-1 px-6 py-3 bg-[#00FF41] text-[var(--primary-foreground)] font-bold rounded hover:shadow-lg hover:shadow-[#00FF41]/50 transition-all"
              >
                发布文章
              </button>
              <button
                @click="resetForm"
                class="flex-1 px-6 py-3 border-2 border-[#FF006E] text-[#FF006E] font-bold rounded hover:bg-[#FF006E]/10 transition-all"
              >
                重置
              </button>
            </div>

            <div v-if="publishMessage" :class="['p-4 rounded text-center font-bold', publishMessage.includes('成功') ? 'bg-[#00FF41]/20 text-[#00FF41]' : 'bg-[#FF006E]/20 text-[#FF006E]']">
              {{ publishMessage }}
            </div>
          </div>

          <!-- Preview Section -->
          <div>
            <label class="block text-[#00FF41] font-bold mb-2">Markdown 内容</label>
            <textarea
              v-model="form.content"
              placeholder="输入 Markdown 内容..."
              rows="20"
              class="w-full px-4 py-3 bg-[var(--card)] border-2 border-[var(--border)] rounded text-[var(--foreground)] placeholder-[var(--muted-foreground)] focus:border-[#00FF41] focus:outline-none transition-colors resize-none font-mono text-sm"
            ></textarea>

            <div class="mt-4 p-4 border-2 border-[var(--border)] rounded bg-[var(--card)]/50 max-h-96 overflow-y-auto">
              <h3 class="text-[#00FF41] font-bold mb-3">预览</h3>
              <div class="prose prose-invert max-w-none text-sm">
                <Streamdown :content="form.content" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
import Streamdown from '../components/Streamdown.vue'
import { articles } from '../lib/articles'

const form = ref({
  title: '',
  excerpt: '',
  category: '技术',
  readTime: 5,
  tagsInput: '',
  content: '',
})

const publishMessage = ref('')

const publishArticle = () => {
  if (!form.value.title || !form.value.excerpt || !form.value.content) {
    publishMessage.value = '请填写所有必填字段'
    return
  }

  const tags = form.value.tagsInput
    .split(',')
    .map((tag) => tag.trim())
    .filter((tag) => tag)

  const newArticle = {
    id: String(articles.length + 1),
    title: form.value.title,
    excerpt: form.value.excerpt,
    content: form.value.content,
    category: form.value.category,
    date: new Date().toISOString().split('T')[0],
    readTime: form.value.readTime,
    tags,
  }

  articles.unshift(newArticle)
  publishMessage.value = '文章发布成功！'

  setTimeout(() => {
    resetForm()
  }, 2000)
}

const resetForm = () => {
  form.value = {
    title: '',
    excerpt: '',
    category: '技术',
    readTime: 5,
    tagsInput: '',
    content: '',
  }
  publishMessage.value = ''
}
</script>

<style scoped>
</style>
