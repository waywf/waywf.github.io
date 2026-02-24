<template>
  <div class="min-h-screen flex flex-col bg-[#0F1419]">
    <Navigation />

    <!-- Hero Section -->
    <section class="relative py-20 md:py-32 overflow-hidden" :style="{
      backgroundImage: 'url(https://files.manuscdn.com/user_upload_by_module/session_file/310519663323678943/tuoDQCiRsjwEvCRX.png)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
    }">
      <div class="absolute inset-0 bg-[#0F1419]/70"></div>
      <div class="container relative z-10">
        <div class="max-w-2xl animate-slide-in-left">
          <div class="mb-6 inline-block">
            <span class="text-[#00FF41] text-sm font-bold tracking-widest">
              {'>> WELCOME TO MY SPACE <<'} </span>
          </div>

          <h1 class="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span class="text-[#00FF41]">{'<'} </span>
                <span class="text-[#E0E0E0]">安澄</span>
                <span class="text-[#FF006E]">{'>'}</span>
          </h1>

          <p class="text-xl text-[#A0A0A0] mb-8 leading-relaxed">
            一个热爱编程、设计和生活的开发者。在这里分享技术见解、设计思考和生活感悟。
          </p>

          <div class="flex flex-col sm:flex-row gap-4">
            <RouterLink to="/articles"
              class="inline-flex items-center gap-2 px-6 py-3 bg-[#00FF41] text-[#0F1419] font-bold rounded hover:shadow-lg hover:shadow-[#00FF41]/50 transition-all">
              开始阅读
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </RouterLink>
            <RouterLink to="/about"
              class="inline-flex items-center gap-2 px-6 py-3 border-2 border-[#FF006E] text-[#FF006E] font-bold rounded hover:bg-[#FF006E]/10 transition-all">
              了解更多
            </RouterLink>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="py-16 md:py-24 border-b border-[#2D3447]">
      <div class="container">
        <h2 class="text-3xl md:text-4xl font-bold mb-12 text-center">
          <span class="text-[#00FF41]">{'$ '}</span>
          <span>核心特性</span>
        </h2>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <!-- Feature 1 -->
          <div
            class="p-6 border-2 border-[#2D3447] rounded hover:border-[#00FF41] transition-all hover:shadow-lg hover:shadow-[#00FF41]/20">
            <svg class="w-12 h-12 text-[#00FF41] mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M10 20l4-16m4 4l4 4m0 0l-4 4m4-4H3" />
            </svg>
            <h3 class="text-xl font-bold mb-3 text-[#E0E0E0]">技术分享</h3>
            <p class="text-[#A0A0A0]">
              深入浅出地分享前端、后端和全栈开发的知识和经验。
            </p>
          </div>

          <!-- Feature 2 -->
          <div
            class="p-6 border-2 border-[#2D3447] rounded hover:border-[#FF006E] transition-all hover:shadow-lg hover:shadow-[#FF006E]/20">
            <svg class="w-12 h-12 text-[#FF006E] mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <h3 class="text-xl font-bold mb-3 text-[#E0E0E0]">设计思考</h3>
            <p class="text-[#A0A0A0]">
              探讨用户体验、产品设计和视觉美学的思考和实践。
            </p>
          </div>

          <!-- Feature 3 -->
          <div
            class="p-6 border-2 border-[#2D3447] rounded hover:border-[#00D9FF] transition-all hover:shadow-lg hover:shadow-[#00D9FF]/20">
            <svg class="w-12 h-12 text-[#00D9FF] mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 6.253v13m0-13C6.5 6.253 2 10.998 2 17s4.5 10.747 10 10.747c5.5 0 10-4.998 10-10.747S17.5 6.253 12 6.253z" />
            </svg>
            <h3 class="text-xl font-bold mb-3 text-[#E0E0E0]">生活感悟</h3>
            <p class="text-[#A0A0A0]">
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
            <span class="text-[#00FF41]">{'$ '}</span>
            <span>最新文章</span>
          </h2>
          <RouterLink to="/articles"
            class="text-[#00FF41] hover:text-[#FF006E] transition-colors flex items-center gap-2">
            查看全部
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </RouterLink>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <RouterLink v-for="(article, index) in latestArticles" :key="article.id" :to="`/article/${article.id}`"
            class="group p-6 border-2 border-[#2D3447] rounded bg-[#1A1F2E]/50 hover:border-[#00FF41] transition-all hover:shadow-lg hover:shadow-[#00FF41]/20 animate-slide-in-up"
            :style="{ animationDelay: `${0.3 + index * 0.1}s` }">
            <!-- <div class="absolute inset-0 bg-[#1A1F2E]/70 rounded"></div> -->
            <div class="relative z-10">
              <div class="flex justify-between items-start mb-3">
                <span class="text-xs px-3 py-1 bg-[#00FF41]/20 text-[#00FF41] rounded font-bold">
                  {{ article.category }}
                </span>
                <span class="text-xs text-[#A0A0A0]">{{ article.readTime }} min</span>
              </div>

              <h3 class="text-xl font-bold mb-3 text-[#E0E0E0] group-hover:text-[#00FF41] transition-colors">
                {{ article.title }}
              </h3>

              <p class="text-[#A0A0A0] text-sm mb-4 line-clamp-2">
                {{ article.excerpt }}
              </p>

              <div class="flex justify-between items-center pt-4 border-t border-[#2D3447]">
                <span class="text-xs text-[#A0A0A0]">{{ article.date }}</span>
                <svg class="w-4 h-4 text-[#00FF41] group-hover:translate-x-1 transition-transform" fill="none"
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
    <section class="py-16 md:py-24 border-t border-[#2D3447] bg-gradient-to-b from-[#0F1419] to-[#1A1F2E]">
      <div class="container text-center">
        <h2 class="text-3xl md:text-4xl font-bold mb-6">
          <span class="text-[#FF006E]">{'>> '}</span>
          <span>准备好开始了吗？</span>
        </h2>
        <p class="text-[#A0A0A0] mb-8 max-w-2xl mx-auto">
          探索我的文章、了解我的想法，或者直接联系我。让我们一起创造更好的东西。
        </p>
        <RouterLink to="/articles"
          class="inline-flex items-center gap-2 px-8 py-3 bg-[#FF006E] text-[#E0E0E0] font-bold rounded hover:shadow-lg hover:shadow-[#FF006E]/50 transition-all">
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
