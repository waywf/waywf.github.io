<template>
  <footer class="border-t border-[#2D3447] bg-[#0F1419] mt-16">
    <!-- Terminal-style top line -->
    <div class="h-1 bg-gradient-to-r from-[#00FF41] via-[#FF006E] to-[#00D9FF]"></div>

    <div class="container py-12">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
        <!-- About -->
        <div>
          <h3 class="text-[#00FF41] font-bold mb-4 text-lg">
            {'<'} 安澄 {'>'}
          </h3>
          <p class="text-[#A0A0A0] text-sm leading-relaxed">
            一个热爱编程和生活的开发者。在这里分享技术见解和生活思考。
          </p>
        </div>

        <!-- Quick Links -->
        <div>
          <h3 class="text-[#00FF41] font-bold mb-4">快速链接</h3>
          <ul class="space-y-2 text-sm">
            <li>
              <RouterLink to="/" class="text-[#A0A0A0] hover:text-[#00FF41] transition-colors">
                首页
              </RouterLink>
            </li>
            <li>
              <RouterLink to="/articles" class="text-[#A0A0A0] hover:text-[#00FF41] transition-colors">
                文章
              </RouterLink>
            </li>
            <li>
              <RouterLink to="/about" class="text-[#A0A0A0] hover:text-[#00FF41] transition-colors">
                关于
              </RouterLink>
            </li>
          </ul>
        </div>

        <!-- Social Links -->
        <div>
          <h3 class="text-[#00FF41] font-bold mb-4">联系方式</h3>
          <div class="flex gap-4">
            <a href="#" class="text-[#A0A0A0] hover:text-[#FF006E] transition-colors" aria-label="GitHub">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v 3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
            <a href="#" class="text-[#A0A0A0] hover:text-[#FF006E] transition-colors" aria-label="Twitter">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2s9 5 20 5a9.5 9.5 0 00-9-5.5c4.75 2.25 7-7 7-7s1.1 5-5 8.5"/>
              </svg>
            </a>
            <a href="mailto:ancheng@example.com" class="text-[#A0A0A0] hover:text-[#FF006E] transition-colors" aria-label="Email">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </a>
          </div>
        </div>
      </div>

      <!-- Bottom line -->
      <div class="border-t border-[#2D3447] pt-8">
        <div class="flex flex-col md:flex-row justify-between items-center text-sm text-[#A0A0A0]">
          <p>
            <span class="text-[#00FF41]">{'{'}</span>
            <span> © {{ currentYear }} 安澄的个人空间 </span>
            <span class="text-[#FF006E]">{'}'}</span>
          </p>
          <p class="mt-4 md:mt-0">
            <span class="text-[#00D9FF]">{'<'}</span>
            <span> Built with Vue 3 & Tailwind </span>
            <span class="text-[#00D9FF]">{'>'}</span>
          </p>
        </div>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink } from 'vue-router'

const currentYear = ref(new Date().getFullYear())
</script>

<style scoped>
</style>
