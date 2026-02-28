<template>
  <div class="min-h-screen flex flex-col bg-background">
    <Navigation />

    <!-- Header Section -->
    <section class="py-12 md:py-16 border-b border-border">
      <div class="container px-4">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">
          <span class="text-primary">{'$ls'}</span>
          <span>articles/</span>
        </h1>
        <p class="text-muted-foreground text-lg">
          探索我的所有文章和想法。使用搜索和分类来找到你感兴趣的内容。
        </p>
      </div>
    </section>

    <!-- Search and Filter Section -->
    <section class="py-6 md:py-8 border-b border-border">
      <div class="container px-4 md:px-0">
        <!-- Search Bar -->
        <div class="mb-6 md:mb-8">
          <div class="relative">
            <svg class="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" fill="none"
              stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input v-model="searchQuery" type="text" placeholder="搜索文章..."
              class="w-full pl-12 pr-4 py-3 bg-card border-2 border-border rounded text-foreground placeholder-muted-foreground focus:border-primary focus:outline-none transition-colors" />
          </div>
        </div>

        <!-- Category Filter -->
        <div class="mb-4 md:mb-6 flex flex-wrap gap-2 md:gap-3">
          <button v-for="category in ['全部', ...availableCategories]" :key="category"
            @click="selectedCategory = category" :class="[
              'px-3 py-2 md:px-4 md:py-2 rounded font-bold transition-all text-sm md:text-base',
              selectedCategory === category
                ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/50'
                : 'border-2 border-border text-foreground hover:border-primary'
            ]">
            {{ category }}
          </button>
        </div>

        <!-- Tags Filter -->
        <div class="flex gap-2">
          <div
            :class="['flex flex-wrap gap-2 transition-all duration-300 flex-1 relative', showAllTags ? 'max-h-none' : 'max-h-[5rem] overflow-hidden']">
            <button v-for="tag in ['全部标签', ...filteredTags]" :key="tag" @click="selectedTag = tag" :class="[
              'px-3 py-1 rounded text-xs md:text-sm transition-all',
              selectedTag === tag
                ? 'bg-accent text-accent-foreground shadow-lg shadow-accent/50'
                : 'border border-border text-muted-foreground hover:border-accent'
            ]">
              {{ tag }}
            </button>
            <!-- 省略号指示器 -->
            <div v-if="!showAllTags && filteredTags.length > 8"
              class="absolute bottom-0 right-0 left-0 h-8 bg-gradient-to-t from-background to-transparent pointer-events-none">
            </div>
          </div>
          <button v-if="filteredTags.length > 8" @click="showAllTags = !showAllTags"
            class="p-2 bg-card rounded-full hover:bg-muted transition-colors flex-shrink-0 self-start shadow-lg shadow-accent/30">
            <svg :class="['w-4 h-4 text-accent transition-transform duration-300', showAllTags ? 'rotate-180' : '']"
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
        <div v-if="displayedArticles.length > 0" class="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          <RouterLink v-for="article in displayedArticles" :key="article.id" :to="`/article/${article.id}`"
            class="group p-6 border-2 border-border rounded bg-card/50 hover:border-primary transition-all hover:shadow-lg hover:shadow-primary/20">
            <!-- <div class="absolute inset-0 bg-card/70 rounded"></div> -->
            <div class="relative z-10">
              <div class="flex justify-between items-start mb-3">
                <span class="text-xs px-3 py-1 bg-primary/20 text-primary rounded font-bold">
                  {{ article.category }}
                </span>
                <span class="text-xs text-muted-foreground">{{ article.readTime }} min read</span>
              </div>

              <h3 class="text-lg md:text-xl font-bold mb-3 text-foreground group-hover:text-primary transition-colors">
                {{ article.title }}
              </h3>

              <p class="text-muted-foreground text-sm mb-4 truncate min-h-[1.25rem]">
                {{ article.excerpt }}
              </p>

              <div class="flex items-center justify-between">
                <div class="flex flex-wrap gap-2">
                  <span v-for="tag in article.tags" :key="tag" class="text-xs text-accent">
                    #{{ tag }}
                  </span>
                </div>
                <span class="text-xs text-muted-foreground">{{ formatDate(article.date) }}</span>
              </div>
            </div>
          </RouterLink>
        </div>

        <!-- Load More Button -->
        <div v-if="displayedArticles.length > 0 && displayedArticles.length < totalFilteredCount"
          class="mt-12 text-center">
          <button @click="loadMore" :disabled="isLoading"
            class="px-6 py-3 border-2 border-primary text-primary rounded font-bold hover:bg-primary/10 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
            {{ isLoading ? '加载中...' : '加载更多文章' }}
          </button>
        </div>

        <div v-else-if="displayedArticles.length === 0" class="text-center py-12">
          <p class="text-muted-foreground text-lg mb-4">
            <span class="text-accent">{'>> '}</span>
            未找到更多的文章
          </p>
          <button @click="resetFilters"
            class="px-4 py-2 border-2 border-primary text-primary rounded hover:bg-primary/10 transition-all">
            重置筛选
          </button>
        </div>

        <!-- Results Count -->
        <div class="mt-8 text-center text-muted-foreground">
          <p>
            <span class="text-primary">{'{'}</span>
            <span> 总计找到 {{ totalFilteredCount }} 篇文章 </span>
            <span class="text-accent">{'}'}</span>
          </p>
        </div>
      </div>
    </section>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
import { loadArticlesByIds, loadAllArticlesMetadata, getArticleIdsByCategory, getArticleIdsByTag, loadManifest, type Article, type ManifestItem } from '../lib/articles-loader'

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
const displayedArticles = ref<Article[]>([])
const availableCategories = ref<string[]>([])
const availableTags = ref<string[]>([])
const manifest = ref<ManifestItem[]>([])
const articlesPerPage = ref(10)

// 根据manifest和筛选条件获取文章ID列表
const getFilteredArticleIds = (): string[] => {
  let result = manifest.value

  // 按分类过滤
  if (selectedCategory.value !== '全部') {
    result = result.filter(item => item.category === selectedCategory.value)
  }

  // 按标签过滤
  if (selectedTag.value !== '全部标签') {
    result = result.filter(item => item.tags && item.tags.includes(selectedTag.value))
  }

  // 按搜索词过滤（搜索标题）
  if (searchQuery.value) {
    const lowerQuery = searchQuery.value.toLowerCase()
    result = result.filter(item =>
      item.id.toLowerCase().includes(lowerQuery) ||
      item.filename.toLowerCase().includes(lowerQuery)
    )
  }

  return result.map(item => item.id)
}

// 计算筛选后的文章总数
const totalFilteredCount = computed(() => {
  return getFilteredArticleIds().length
})

const filteredTags = computed(() => {
  if (selectedCategory.value === '全部') {
    return availableTags.value
  }
  // 从manifest中筛选当前分类下的标签
  const categoryItems = manifest.value.filter(item => item.category === selectedCategory.value)
  const tags = new Set<string>()
  categoryItems.forEach(item => {
    if (item.tags) {
      item.tags.forEach(tag => tags.add(tag))
    }
  })
  return Array.from(tags).sort()
})

onMounted(async () => {
  // 加载manifest
  manifest.value = await loadManifest()

  // 加载元数据（分类和标签）
  const metadata = await loadAllArticlesMetadata()
  availableCategories.value = metadata.categories
  availableTags.value = metadata.tags

  // 加载第一页文章
  await loadArticles()
})

// 加载文章（根据当前筛选条件和页码）
const loadArticles = async () => {
  const ids = getFilteredArticleIds()
  const pageIds = ids.slice(0, currentPage.value * articlesPerPage.value)
  const articles = await loadArticlesByIds(pageIds)
  displayedArticles.value = articles
}

// 监听过滤条件变化，重置分页并重新加载
watch([selectedCategory, selectedTag, searchQuery], async () => {
  if (typeof sessionStorage !== 'undefined') {
    sessionStorage.setItem('selectedCategory', selectedCategory.value)
    sessionStorage.setItem('selectedTag', selectedTag.value)
    sessionStorage.setItem('searchQuery', searchQuery.value)
    sessionStorage.setItem('currentPage', '1')
  }
  currentPage.value = 1
  await loadArticles()
})

// 监听分类变化，重置标签选择（使用同步刷新避免重复触发过滤监听）
watch(selectedCategory, () => {
  selectedTag.value = '全部标签'
}, { flush: 'sync' })

const resetFilters = async () => {
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
  await loadArticles()
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
  // 加载更多文章
  const ids = getFilteredArticleIds()
  const pageIds = ids.slice((currentPage.value - 1) * articlesPerPage.value, currentPage.value * articlesPerPage.value)
  const newArticles = await loadArticlesByIds(pageIds)
  displayedArticles.value = [...displayedArticles.value, ...newArticles]
  isLoading.value = false
}

const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
}
</script>

<style scoped></style>
