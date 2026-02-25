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
    <section class="py-6 md:py-8 border-b border-[#2D3447]">
      <div class="container px-4 md:px-0">
        <!-- Search Bar -->
        <div class="mb-6 md:mb-8">
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
        <div class="mb-4 md:mb-6 flex flex-wrap gap-2 md:gap-3">
          <button v-for="category in ['全部', ...availableCategories]" :key="category"
            @click="selectedCategory = category" :class="[
              'px-3 py-2 md:px-4 md:py-2 rounded font-bold transition-all text-sm md:text-base',
              selectedCategory === category
                ? 'bg-[#00FF41] text-[#0F1419] shadow-lg shadow-[#00FF41]/50'
                : 'border-2 border-[#2D3447] text-[#E0E0E0] hover:border-[#00FF41]'
            ]">
            {{ category }}
          </button>
        </div>

        <!-- Tags Filter -->
        <div class="relative">
          <div
            :class="['flex flex-wrap gap-2 transition-all duration-300', showAllTags ? 'max-h-none' : 'max-h-[5rem] overflow-hidden']">
            <button v-for="tag in ['全部标签', ...filteredTags]" :key="tag" @click="selectedTag = tag" :class="[
              'px-3 py-1 rounded text-xs md:text-sm transition-all',
              selectedTag === tag
                ? 'bg-[#FF006E] text-[#FFFFFF] shadow-lg shadow-[#FF006E]/50'
                : 'border border-[#2D3447] text-[#A0A0A0] hover:border-[#FF006E]'
            ]">
              {{ tag }}
            </button>
          </div>
          <button v-if="filteredTags.length > 8" @click="showAllTags = !showAllTags"
            class="absolute bottom-0 right-0 p-2 bg-[#1A1F2E] rounded-full hover:bg-[#2D3447] transition-colors">
            <svg :class="['w-4 h-4 text-[#FF006E] transition-transform duration-300', showAllTags ? 'rotate-180' : '']"
              fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>
    </section>

    <!-- Articles Grid -->
    <section class="py-8 md:py-12 flex-1">
      <div class="container px-4 md:px-0">
        <div v-if="filteredArticles.length > 0" class="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          <RouterLink v-for="article in displayedArticles" :key="article.id" :to="`/article/${article.id}`"
            class="group p-6 border-2 border-[#2D3447] rounded bg-[#1A1F2E]/50 hover:border-[#00FF41] transition-all hover:shadow-lg hover:shadow-[#00FF41]/20">
            <!-- <div class="absolute inset-0 bg-[#1A1F2E]/70 rounded"></div> -->
            <div class="relative z-10">
              <div class="flex justify-between items-start mb-3">
                <span class="text-xs px-3 py-1 bg-[#00FF41]/20 text-[#00FF41] rounded font-bold">
                  {{ article.category }}
                </span>
                <span class="text-xs text-[#A0A0A0]">{{ article.readTime }} min read</span>
              </div>

              <h3 class="text-lg md:text-xl font-bold mb-3 text-[#E0E0E0] group-hover:text-[#00FF41] transition-colors">
                {{ article.title }}
              </h3>

              <p class="text-[#A0A0A0] text-sm mb-4 truncate min-h-[1.25rem]">
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

        <!-- Load More Button -->
        <div v-if="displayedArticles.length > 0 && displayedArticles.length < filteredArticles.length"
          class="mt-12 text-center">
          <button @click="loadMore" :disabled="isLoading"
            class="px-6 py-3 border-2 border-[#00FF41] text-[#00FF41] rounded font-bold hover:bg-[#00FF41]/10 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
            {{ isLoading ? '加载中...' : '加载更多文章' }}
          </button>
        </div>

        <div v-else class="text-center py-12">
          <p class="text-[#A0A0A0] text-lg mb-4">
            <span class="text-[#FF006E]">{'>> '}</span>
            未找到更多的文章
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
            <span> 总计找到 {{ filteredArticles.length }} 篇文章 </span>
            <span class="text-[#FF006E]">{'}'}</span>
          </p>
        </div>
      </div>
    </section>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
import { loadAllArticles, getCategories, getAllTags, filterArticlesByCategory, searchArticles, type Article } from '../lib/articles-loader'

// 使用sessionStorage保存筛选条件和分页信息
const getInitialValue = (key: string, defaultValue: string) => {
  if (typeof sessionStorage !== 'undefined') {
    const saved = sessionStorage.getItem(key)
    return saved ? saved : defaultValue
  }
  return defaultValue
}

const getInitialNumber = (key: string, defaultValue: number) => {
  if (typeof sessionStorage !== 'undefined') {
    const saved = sessionStorage.getItem(key)
    return saved ? parseInt(saved) : defaultValue
  }
  return defaultValue
}

const selectedCategory = ref(getInitialValue('selectedCategory', '全部'))
const selectedTag = ref(getInitialValue('selectedTag', '全部标签'))
const searchQuery = ref(getInitialValue('searchQuery', ''))
const currentPage = ref(getInitialNumber('currentPage', 1))
const showAllTags = ref(false)
const articles = ref<Article[]>([])
const displayedArticles = ref<Article[]>([])
const availableCategories = ref<string[]>([])
const availableTags = ref<string[]>([])
const filteredTags = computed(() => {
  if (selectedCategory.value === '全部') {
    return availableTags.value
  }
  const categoryArticles = articles.value.filter(article => article.category === selectedCategory.value)
  const tags = new Set<string>()
  categoryArticles.forEach(article => {
    article.tags.forEach(tag => tags.add(tag))
  })
  return Array.from(tags).sort()
})
const articlesPerPage = ref(10)

onMounted(async () => {
  const loadedArticles = await loadAllArticles()
  articles.value = loadedArticles
  availableCategories.value = getCategories(loadedArticles)
  availableTags.value = getAllTags(loadedArticles)
  updateDisplayedArticles()
})

const filteredArticles = computed(() => {
  let result = articles.value

  // 按分类过滤
  if (selectedCategory.value !== '全部') {
    result = filterArticlesByCategory(result, selectedCategory.value)
  }

  // 按标签过滤
  if (selectedTag.value !== '全部标签') {
    result = result.filter(article => article.tags.includes(selectedTag.value))
  }

  // 按搜索词过滤
  if (searchQuery.value) {
    result = searchArticles(result, searchQuery.value)
  }

  return result
})

// 监听过滤条件变化，重置分页并保存到sessionStorage
watch([selectedCategory, selectedTag, searchQuery], () => {
  if (typeof sessionStorage !== 'undefined') {
    sessionStorage.setItem('selectedCategory', selectedCategory.value)
    sessionStorage.setItem('selectedTag', selectedTag.value)
    sessionStorage.setItem('searchQuery', searchQuery.value)
    sessionStorage.setItem('currentPage', '1')
  }
  currentPage.value = 1
  updateDisplayedArticles()
})

// 监听分类变化，重置标签选择（使用同步刷新避免重复触发过滤监听）
watch(selectedCategory, () => {
  selectedTag.value = '全部标签'
}, { flush: 'sync' })

const resetFilters = () => {
  selectedCategory.value = '全部'
  selectedTag.value = '全部标签'
  searchQuery.value = ''
  currentPage.value = 1
  if (typeof sessionStorage !== 'undefined') {
    sessionStorage.removeItem('selectedCategory')
    sessionStorage.removeItem('selectedTag')
    sessionStorage.removeItem('searchQuery')
    sessionStorage.removeItem('currentPage')
  }
}

const updateDisplayedArticles = () => {
  // 直接使用filteredArticles.value，避免重复计算
  displayedArticles.value = filteredArticles.value.slice(0, currentPage.value * articlesPerPage.value)
}

const isLoading = ref(false)

const loadMore = async () => {
  isLoading.value = true
  // Simulate asynchronous loading delay to improve UX
  await new Promise(resolve => setTimeout(resolve, 600))
  currentPage.value++
  if (typeof sessionStorage !== 'undefined') {
    sessionStorage.setItem('currentPage', currentPage.value.toString())
  }
  updateDisplayedArticles()
  isLoading.value = false
}

const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
}
</script>

<style scoped></style>
