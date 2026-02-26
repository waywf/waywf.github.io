<template>
  <div class="min-h-screen flex flex-col bg-background">
    <Navigation />

    <!-- Hero Section -->
    <section class="relative py-20 md:py-32 overflow-hidden" :style="{
      backgroundImage: 'url(https://files.manuscdn.com/user_upload_by_module/session_file/310519663323678943/tuoDQCiRsjwEvCRX.png)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
    }">
      <div class="absolute inset-0 bg-background/70"></div>
      <div class="container relative z-10">
        <div class="max-w-3xl animate-slide-in-left">
          <div class="mb-6 inline-block">
            <span class="text-primary text-sm font-bold tracking-widest">
              {'>> WELCOME TO MY SPACE <<'} </span>
          </div>

          <h1 class="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span class="text-primary">{'<'} </span>
                <span class="text-foreground">70KG</span>
                <span class="text-accent">{'>'}</span>
          </h1>

          <p class="text-xl text-muted-foreground mb-8 leading-relaxed">
            一个热爱编程、设计和生活的开发者。在这里分享技术见解、设计思考和生活感悟。
          </p>

          <div class="flex flex-col sm:flex-row gap-4">
            <RouterLink to="/articles"
              class="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground font-bold rounded hover:shadow-lg hover:shadow-primary/50 transition-all">
              开始阅读
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </RouterLink>
            <RouterLink to="/about"
              class="inline-flex items-center gap-2 px-6 py-3 border-2 border-accent text-accent font-bold rounded hover:bg-accent/10 transition-all">
              了解更多
            </RouterLink>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="py-16 md:py-24 border-b border-border">
      <div class="container">
        <h2 class="text-3xl md:text-4xl font-bold mb-12 text-center">
          <span class="text-primary">{'$ '}</span>
          <span>核心特性</span>
        </h2>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <!-- Feature 1 -->
          <div
            class="p-6 border-2 border-border rounded hover:border-primary transition-all hover:shadow-lg hover:shadow-primary/20">
            <svg class="w-12 h-12 text-primary mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M10 20l4-16m4 4l4 4m0 0l-4 4m4-4H3" />
            </svg>
            <h3 class="text-xl font-bold mb-3 text-foreground">技术分享</h3>
            <p class="text-muted-foreground">
              深入浅出地分享前端、后端和全栈开发的知识和经验。
            </p>
          </div>

          <!-- Feature 2 -->
          <div
            class="p-6 border-2 border-border rounded hover:border-accent transition-all hover:shadow-lg hover:shadow-accent/20">
            <svg class="w-12 h-12 text-accent mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <h3 class="text-xl font-bold mb-3 text-foreground">设计思考</h3>
            <p class="text-muted-foreground">
              探讨用户体验、产品设计和视觉美学的思考和实践。
            </p>
          </div>

          <!-- Feature 3 -->
          <div
            class="p-6 border-2 border-border rounded hover:border-chart-3 transition-all hover:shadow-lg hover:shadow-chart-3/20">
            <svg class="w-12 h-12 text-chart-3 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 6.253v13m0-13C6.5 6.253 2 10.998 2 17s4.5 10.747 10 10.747c5.5 0 10-4.998 10-10.747S17.5 6.253 12 6.253z" />
            </svg>
            <h3 class="text-xl font-bold mb-3 text-foreground">生活感悟</h3>
            <p class="text-muted-foreground">
              分享生活中的所思所想，探索生活的意义和价值。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- Latest Articles Section -->
    <section class="py-16 md:py-24">
      <div class="container">
        <div class="flex justify-between items-center mb-12">
          <h2 class="text-3xl md:text-4xl font-bold">
            <span class="text-primary">{'$ '}</span>
            <span>最新文章</span>
          </h2>
          <RouterLink to="/articles"
            class="text-primary hover:text-accent transition-colors flex items-center gap-2">
            查看全部
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </RouterLink>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <RouterLink v-for="(article, index) in latestArticles" :key="article.id" :to="`/article/${article.id}`"
            class="group p-6 border-2 border-border rounded bg-card/50 hover:border-primary transition-all hover:shadow-lg hover:shadow-primary/20 animate-slide-in-up"
            :style="{ animationDelay: `${0.3 + index * 0.1}s` }">
            <!-- <div class="absolute inset-0 bg-card/70 rounded"></div> -->
            <div class="relative z-10">
              <div class="flex justify-between items-start mb-3">
                <span class="text-xs px-3 py-1 bg-primary/20 text-primary rounded font-bold">
                  {{ article.category }}
                </span>
                <span class="text-xs text-muted-foreground">{{ article.readTime }} min</span>
              </div>

              <h3 class="text-xl font-bold mb-3 text-foreground group-hover:text-primary transition-colors">
                {{ article.title }}
              </h3>

              <p
                class="text-muted-foreground text-sm mb-4 line-clamp-2 group-hover:text-primary transition-colors min-h-[2.5rem]">
                {{ article.excerpt }}
              </p>

              <div class="flex justify-between items-center pt-4 border-t border-border">
                <span class="text-xs text-muted-foreground">{{ article.date }}</span>
                <svg class="w-4 h-4 text-primary group-hover:translate-x-1 transition-transform" fill="none"
                  stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </div>
            </div>
          </RouterLink>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="py-16 md:py-24 border-t border-border bg-gradient-to-b from-background to-card">
      <div class="container text-center">
        <h2 class="text-3xl md:text-4xl font-bold mb-6">
          <span class="text-accent">{'>> '}</span>
          <span>准备好开始了吗？</span>
        </h2>
        <p class="text-muted-foreground mb-8 max-w-2xl mx-auto">
          探索我的文章、了解我的想法，或者直接联系我。让我们一起创造更好的东西。
        </p>
        <RouterLink to="/articles"
          class="inline-flex items-center gap-2 px-8 py-3 bg-accent text-accent-foreground font-bold rounded hover:shadow-lg hover:shadow-accent/50 transition-all">
          开始探索
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </RouterLink>
      </div>
    </section>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
import { loadAllArticles, type Article } from '../lib/articles-loader'

const latestArticles = ref<Article[]>([])

onMounted(async () => {
  const articles = await loadAllArticles()
  latestArticles.value = articles.slice(0, 3)
})
</script>

<style scoped></style>
