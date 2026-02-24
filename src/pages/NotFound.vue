<template>
  <div class="min-h-screen flex flex-col bg-[#0F1419]">
    <Navigation />

    <section class="flex-1 flex items-center justify-center py-20">
      <div class="container text-center">
        <div class="mb-8">
          <svg class="w-24 h-24 text-[#FF006E] mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4v.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>

        <h1 class="text-6xl md:text-8xl font-bold mb-4">
          <span class="text-[#FF006E]">404</span>
        </h1>

        <h2 class="text-3xl md:text-4xl font-bold mb-4 text-[#E0E0E0]">
          <span class="text-[#00FF41]">{'>> '}</span>
          <span>页面未找到</span>
        </h2>

        <p class="text-[#A0A0A0] text-lg mb-8 max-w-2xl mx-auto">
          <span class="text-[#FF006E]">{'{'}</span>
          <span> 抱歉，您访问的页面不存在。可能是链接已过期或输入错误。 </span>
          <span class="text-[#FF006E]">{'}'}</span>
        </p>

        <div class="flex flex-col sm:flex-row gap-4 justify-center">
          <RouterLink to="/" class="inline-flex items-center gap-2 px-6 py-3 bg-[#00FF41] text-[#0F1419] font-bold rounded hover:shadow-lg hover:shadow-[#00FF41]/50 transition-all">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
            返回首页
          </RouterLink>
          <RouterLink to="/articles" class="inline-flex items-center gap-2 px-6 py-3 border-2 border-[#FF006E] text-[#FF006E] font-bold rounded hover:bg-[#FF006E]/10 transition-all">
            查看文章
          </RouterLink>
        </div>

        <div class="mt-12 text-[#A0A0A0] text-sm">
          <p>
            <span class="text-[#00D9FF]">{'<'}</span>
            <span> Error Code: 404 Not Found </span>
            <span class="text-[#00D9FF]">{'>'}</span>
          </p>
        </div>
      </div>
    </section>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { RouterLink } from 'vue-router'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
</script>

<style scoped>
</style>
