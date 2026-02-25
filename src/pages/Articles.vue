<template>
  <div class="min-h-screen flex flex-col bg-[#0F1419]">
    <Navigation />

    <!-- Header Section -->
    <section class="py-12 md:py-16 border-b border-[#2D3447]">
      <div class="container">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">
          <span class="text-[#00FF41]">{'$ ls '}</span>
          <span>articles/</span>
        </h1>
        <p class="text-[#A0A0A0] text-lg">
          探索我的所有文章和想法。使用搜索和分类来找到你感兴趣的内容。
        </p>
      </div>
    </section>

    <!-- Search and Filter Section -->
    <section class="py-8 border-b border-[#2D3447]">
      <div class="container">
        <!-- Search Bar -->
        <div class="mb-8">
          <div class="relative">
            <svg class="absolute left-4 top-1/2 transform -translate-y-1/2 text-[#A0A0A0] w-5 h-5" fill="none"
              stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input v-model="searchQuery" type="text" placeholder="搜索文章..."
              class="w-full pl-12 pr-4 py-3 bg-[#1A1F2E] border-2 border-[#2D3447] rounded text-[#E0E0E0] placeholder-[#A0A0A0] focus:border-[#00FF41] focus:outline-none transition-colors" />
          </div>
        </div>

        <!-- Category Filter -->
        <div class="flex flex-wrap gap-3">
          <button v-for="category in ['全部', ...availableCategories]" :key="category"
            @click="selectedCategory = category" :class="[
              'px-4 py-2 rounded font-bold transition-all',
              selectedCategory === category
                ? 'bg-[#00FF41] text-[#0F1419] shadow-lg shadow-[#00FF41]/50'
                : 'border-2 border-[#2D3447] text-[#E0E0E0] hover:border-[#00FF41]'
            ]">
            {{ category }}
          </button>
        </div>
      </div>
    </section>

    <!-- Articles Grid -->
    <section class="py-12 md:py-16 flex-1">
      <div class="container">
        <div v-if="filteredArticles.length > 0" class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <RouterLink v-for="article in filteredArticles" :key="article.id" :to="`/article/${article.id}`"
            class="group p-6 border-2 border-[#2D3447] rounded bg-[#1A1F2E]/50 hover:border-[#00FF41] transition-all hover:shadow-lg hover:shadow-[#00FF41]/20">
            <!-- <div class="absolute inset-0 bg-[#1A1F2E]/70 rounded"></div> -->
            <div class="relative z-10">
              <div class="flex justify-between items-start mb-3">
                <span class="text-xs px-3 py-1 bg-[#00FF41]/20 text-[#00FF41] rounded font-bold">
                  {{ article.category }}
                </span>
                <span class="text-xs text-[#A0A0A0]">{{ article.readTime }} min read</span>
              </div>

              <h3 class="text-xl font-bold mb-3 text-[#E0E0E0] group-hover:text-[#00FF41] transition-colors">
                {{ article.title }}
              </h3>

              <p class="text-[#A0A0A0] text-sm mb-4 line-clamp-2">
                {{ article.excerpt }}
              </p>

              <div class="flex items-center justify-between">
                <div class="flex flex-wrap gap-2">
                  <span v-for="tag in article.tags" :key="tag" class="text-xs text-[#FF006E]">
                    #{{ tag }}
                  </span>
                </div>
                <span class="text-xs text-[#A0A0A0]">{{ formatDate(article.date) }}</span>
              </div>
            </div>
          </RouterLink>
        </div>

        <div v-else class="text-center py-12">
          <p class="text-[#A0A0A0] text-lg mb-4">
            <span class="text-[#FF006E]">{'>> '}</span>
            未找到匹配的文章
          </p>
          <button @click="resetFilters"
            class="px-4 py-2 border-2 border-[#00FF41] text-[#00FF41] rounded hover:bg-[#00FF41]/10 transition-all">
            重置筛选
          </button>
        </div>

        <!-- Results Count -->
        <div class="mt-8 text-center text-[#A0A0A0]">
          <p>
            <span class="text-[#00FF41]">{'{'}</span>
            <span> 找到 {{ filteredArticles.length }} 篇文章 </span>
            <span class="text-[#FF006E]">{'}'}</span>
          </p>
        </div>
      </div>
    </section>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
import { loadAllArticles, getCategories, filterArticlesByCategory, searchArticles, type Article } from '../lib/articles-loader'

const selectedCategory = ref('全部')
const searchQuery = ref('')
const articles = ref<Article[]>([])
const availableCategories = ref<string[]>([])

onMounted(async () => {
  const loadedArticles = await loadAllArticles()
  articles.value = loadedArticles
  availableCategories.value = getCategories(loadedArticles)
})

const filteredArticles = computed(() => {
  let result = articles.value

  // 按分类过滤
  if (selectedCategory.value !== '全部') {
    result = filterArticlesByCategory(result, selectedCategory.value)
  }

  // 按搜索词过滤
  if (searchQuery.value) {
    result = searchArticles(result, searchQuery.value)
  }

  return result
})

const resetFilters = () => {
  selectedCategory.value = '全部'
  searchQuery.value = ''
}

const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
}
</script>

<style scoped></style>
