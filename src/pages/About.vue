<template>
  <div class="min-h-screen flex flex-col bg-[#0F1419]">
    <Navigation />

    <!-- Header Section -->
    <section class="py-12 md:py-16 border-b border-[#2D3447]">
      <div class="container">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">
          <span class="text-[#00FF41]">{'$ whoami'}</span>
        </h1>
        <p class="text-[#A0A0A0] text-lg">
          了解更多关于我和这个项目的信息。
        </p>
      </div>
    </section>

    <!-- About Content -->
    <section class="py-12 md:py-16 flex-1">
      <div class="container max-w-3xl">
        <!-- Introduction -->
        <div class="mb-12 p-8 border-2 border-[#2D3447] rounded bg-[#1A1F2E]/50">
          <h2 class="text-2xl font-bold mb-4 text-[#00FF41]">
            {'<'} 关于我 {'>'}
          </h2>
          <p class="text-[#A0A0A0] leading-relaxed mb-4">
            我是70KG，一个热爱编程和设计的全栈开发者。我对创建美观、高效和用户友好的网络应用充满热情。
          </p>
          <p class="text-[#A0A0A0] leading-relaxed mb-4">
            在过去的几年里，我在前端开发、UI/UX 设计和全栈开发方面积累了丰富的经验。我相信好的设计不仅仅是美观，更重要的是要解决实际问题。
          </p>
          <p class="text-[#A0A0A0] leading-relaxed">
            在这个博客中，我分享我的技术见解、设计思考和生活感悟。希望能够帮助和启发更多的开发者。
          </p>
        </div>

        <!-- Skills -->
        <div class="mb-12">
          <h2 class="text-2xl font-bold mb-6 text-[#FF006E]">
            {'$ skills'}
          </h2>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="p-6 border-2 border-[#2D3447] rounded hover:border-[#00FF41] transition-all">
              <h3 class="text-lg font-bold text-[#00FF41] mb-3">前端开发</h3>
              <ul class="space-y-2 text-[#A0A0A0]">
                <li>• React & Vue.js</li>
                <li>• TypeScript</li>
                <li>• Tailwind CSS</li>
                <li>• Next.js & Nuxt</li>
                <li>• Web Performance</li>
              </ul>
            </div>

            <div class="p-6 border-2 border-[#2D3447] rounded hover:border-[#FF006E] transition-all">
              <h3 class="text-lg font-bold text-[#FF006E] mb-3">后端开发</h3>
              <ul class="space-y-2 text-[#A0A0A0]">
                <li>• Node.js & Express</li>
                <li>• Python & Django</li>
                <li>• SQL & NoSQL</li>
                <li>• RESTful API</li>
                <li>• System Design</li>
              </ul>
            </div>

            <div class="p-6 border-2 border-[#2D3447] rounded hover:border-[#00D9FF] transition-all">
              <h3 class="text-lg font-bold text-[#00D9FF] mb-3">设计工具</h3>
              <ul class="space-y-2 text-[#A0A0A0]">
                <li>• Figma</li>
                <li>• Adobe XD</li>
                <li>• Sketch</li>
                <li>• Prototyping</li>
                <li>• User Research</li>
              </ul>
            </div>

            <div class="p-6 border-2 border-[#2D3447] rounded hover:border-[#00FF41] transition-all">
              <h3 class="text-lg font-bold text-[#00FF41] mb-3">其他技能</h3>
              <ul class="space-y-2 text-[#A0A0A0]">
                <li>• Git & GitHub</li>
                <li>• Docker & DevOps</li>
                <li>• CI/CD</li>
                <li>• Technical Writing</li>
                <li>• Team Leadership</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- Experience -->
        <div class="mb-12">
          <h2 class="text-2xl font-bold mb-6 text-[#00D9FF]">
            {'$ experience'}
          </h2>
          <div class="space-y-6">
            <div v-for="(experience, index) in experiences" :key="experience.id"
              class="p-6 border-l-4 bg-[#1A1F2E]/50 rounded" :style="{ borderLeftColor: experience.color }">
              <div class="flex justify-between items-start mb-2">
                <div>
                  <h3 class="text-lg font-bold text-[#E0E0E0] mb-2">{{ experience.title }}</h3>
                  <p class="text-[#A0A0A0]">{{ experience.company }} | {{ experience.period }}</p>
                </div>
                <button @click="toggleExpand(index)" :class="['hover:text-[#00FF41] transition-colors']"
                  :style="{ color: experience.color }">
                  <svg :class="['w-5 h-5 transition-transform duration-300', expanded[index] ? 'rotate-180' : '']"
                    fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>1
              <div
                :class="['overflow-hidden transition-all duration-300', expanded[index] ? 'max-h-[1500px] opacity-100' : 'max-h-[100px] opacity-70']">
                <p class="text-[#A0A0A0]">{{ experience.description }}</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Contact -->
        <div class="p-8 border-2 border-[#2D3447] rounded bg-[#1A1F2E]/50">
          <h2 class="text-2xl font-bold mb-6 text-[#FF006E]">
            {'$ contact'}
          </h2>
          <p class="text-[#A0A0A0] mb-6">
            欢迎通过以下方式与我联系。我很乐意讨论技术、设计或任何有趣的想法。
          </p>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <a href="mailto:ancheng@example.com"
              class="flex items-center gap-2 p-4 border-2 border-[#2D3447] rounded hover:border-[#00FF41] transition-all hover:bg-[#00FF41]/10">
              <svg class="w-5 h-5 text-[#00FF41]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <span class="text-sm">Email</span>
            </a>
            <a href="#"
              class="flex items-center gap-2 p-4 border-2 border-[#2D3447] rounded hover:border-[#FF006E] transition-all hover:bg-[#FF006E]/10">
              <svg class="w-5 h-5 text-[#FF006E]" fill="currentColor" viewBox="0 0 24 24">
                <path
                  d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v 3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              <span class="text-sm">GitHub</span>
            </a>
            <a href="#"
              class="flex items-center gap-2 p-4 border-2 border-[#2D3447] rounded hover:border-[#00D9FF] transition-all hover:bg-[#00D9FF]/10">
              <svg class="w-5 h-5 text-[#00D9FF]" fill="currentColor" viewBox="0 0 24 24">
                <path
                  d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2s9 5 20 5a9.5 9.5 0 00-9-5.5c4.75 2.25 7-7 7-7s1.1 5-5 8.5" />
              </svg>
              <span class="text-sm">Twitter</span>
            </a>
            <a href="#"
              class="flex items-center gap-2 p-4 border-2 border-[#2D3447] rounded hover:border-[#00FF41] transition-all hover:bg-[#00FF41]/10">
              <svg class="w-5 h-5 text-[#00FF41]" fill="currentColor" viewBox="0 0 24 24">
                <path d="M16 8a6 6 0 016 6v7h-4v-7a2 2 0 00-2-2 2 2 0 00-2 2v7h-4v-7a6 6 0 016-6zM2 9h4v12H2z" />
              </svg>
              <span class="text-sm">LinkedIn</span>
            </a>
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
import { experiences, type Experience } from '../data/experience'

const expanded = ref<boolean[]>([false, false, false])

const toggleExpand = (index: number) => {
  expanded.value[index] = !expanded.value[index]
}
</script>

<style scoped></style>
