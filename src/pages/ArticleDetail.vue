<template>
  <div class="min-h-screen flex flex-col bg-background">
    <Navigation />

    <div v-if="article" class="flex-1">
      <!-- Article Header -->
      <section class="py-12 md:py-16 border-b border-border">
        <div class="container max-w-4xl">
          <RouterLink to="/articles"
            class="inline-flex items-center gap-2 text-primary hover:text-accent transition-colors mb-8">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
            返回文章列表
          </RouterLink>

          <h1 class="text-4xl md:text-5xl font-bold mb-6 text-foreground">
            {{ article.title }}
          </h1>

          <div class="flex flex-wrap gap-6 text-muted-foreground">
            <div class="flex items-center gap-2">
              <svg class="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span>{{ formatDate(article.date) }}</span>
            </div>
            <div class="flex items-center gap-2">
              <svg class="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{{ article.readTime }} min read</span>
            </div>
            <div class="flex items-center gap-2">
              <svg class="w-5 h-5 text-chart-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              <span>{{ article.category }}</span>
            </div>
          </div>
        </div>
      </section>

      <!-- Article Content -->
      <section class="py-12 md:py-16">
        <div class="container max-w-4xl">
          <div class="prose prose-invert max-w-none">
            <Streamdown :content="article.content" />
          </div>

          <!-- Tags -->
          <div class="mt-12 pt-8 border-t border-border">
            <h3 class="text-primary font-bold mb-4">标签</h3>
            <div class="flex flex-wrap gap-3">
              <span v-for="tag in article.tags" :key="tag"
                class="px-4 py-2 bg-accent/20 text-accent rounded font-bold hover:bg-accent/30 transition-colors cursor-pointer">
                #{{ tag }}
              </span>
            </div>
          </div>
        </div>
      </section>

      <!-- Navigation -->
      <section class="py-12 md:py-16 border-t border-border">
        <div class="container max-w-4xl">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <a v-if="prevArticle" :href="`/article/${prevArticle.id}`"
              class="group p-6 border-2 border-border rounded bg-card/50 hover:border-primary transition-all">
              <div class="flex items-center gap-2 text-primary mb-2">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
                <span>上一篇</span>
              </div>
              <h3 class="text-foreground group-hover:text-primary transition-colors">
                {{ prevArticle.title }}
              </h3>
            </a>
            <div v-else></div>

            <a v-if="nextArticle" :href="`/article/${nextArticle.id}`"
              class="group p-6 border-2 border-border rounded bg-card/50 hover:border-primary transition-all text-right">
              <div class="flex items-center justify-end gap-2 text-primary mb-2">
                <span>下一篇</span>
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                </svg>
              </div>
              <h3 class="text-foreground group-hover:text-primary transition-colors">
                {{ nextArticle.title }}
              </h3>
            </a>
            <div v-else></div>
          </div>
        </div>
      </section>
    </div>

    <div v-else class="flex-1 flex items-center justify-center">
      <div class="text-center">
        <h1 class="text-4xl font-bold text-accent mb-4">404</h1>
        <p class="text-muted-foreground mb-8">文章未找到</p>
        <RouterLink to="/articles"
          class="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground font-bold rounded hover:shadow-lg hover:shadow-primary/50 transition-all">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
          返回文章列表
        </RouterLink>
      </div>
    </div>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { RouterLink } from 'vue-router'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
import Streamdown from '../components/Streamdown.vue'
import { loadArticle, loadAllArticles, type Article } from '../lib/articles-loader'

const route = useRoute()
const article = ref<Article | null>(null)
const allArticles = ref<Article[]>([])

onMounted(async () => {
  try {
    const id = route.params.id as string
    const loadedArticle = await loadArticle(`${id}.md`)
    if (loadedArticle) {
      article.value = loadedArticle
    }

    // 加载所有文章以便导航
    const articles = await loadAllArticles()
    allArticles.value = articles
    console.log('All articles:', articles.length)
    console.log('Current article:', article.value?.id)
    console.log('Current index:', currentIndex.value)
  } catch (error) {
    console.error('Failed to load article:', error)
  }
})

const currentIndex = computed(() => {
  if (!article.value || allArticles.value.length === 0) return -1
  return allArticles.value.findIndex(a => a.id === article.value.id)
})

const prevArticle = computed(() => {
  const index = currentIndex.value
  console.log('Prev article index:', index)
  if (index > 0) {
    return allArticles.value[index - 1]
  }
  return null
})

const nextArticle = computed(() => {
  const index = currentIndex.value
  console.log('Next article index:', index)
  if (index >= 0 && index < allArticles.value.length - 1) {
    return allArticles.value[index + 1]
  }
  return null
})

const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
}
</script>

<style scoped></style>
