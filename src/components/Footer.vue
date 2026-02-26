<template>
  <footer class="border-t border-border bg-background mt-16">
    <!-- Terminal-style top line -->
    <div class="h-1 bg-gradient-to-r from-primary via-accent to-chart-3"></div>

    <div class="container py-10">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <!-- About -->
        <div>
          <h3 class="text-primary font-bold mb-4 text-lg">
            {'<'} 70KG {'>'}
          </h3>
          <p class="text-muted-foreground text-sm leading-relaxed">
            一个热爱编程和生活的开发者。在这里分享技术见解和生活思考。
          </p>
        </div>

        <!-- Quick Links -->
        <div>
          <h3 class="text-primary font-bold mb-4">快速链接</h3>
          <ul class="space-y-2 text-sm flex gap-10">
            <li>
              <RouterLink to="/" class="text-muted-foreground hover:text-primary transition-colors">
                首页
              </RouterLink>
            </li>
            <li>
              <RouterLink to="/articles" class="text-muted-foreground hover:text-primary transition-colors">
                文章
              </RouterLink>
            </li>
            <li>
              <RouterLink to="/about" class="text-muted-foreground hover:text-primary transition-colors">
                关于
              </RouterLink>
            </li>
          </ul>
        </div>

        <!-- Social Links -->
        <div>
          <h3 class="text-primary font-bold mb-4">联系方式</h3>
          <div class="flex gap-10">
            <a href="mailto:381531043@qq.com" class="text-muted-foreground hover:text-accent transition-colors"
              aria-label="Email">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </a>
            <a href="https://github.com/waywf" class="text-muted-foreground hover:text-accent transition-colors"
              aria-label="GitHub">
              <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path
                  d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v 3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
            </a>
            <button class="text-muted-foreground hover:text-accent transition-colors" @click="showWechat = !showWechat"
              aria-label="WeChat">
              <svg t="1772097691955" class="w-5 h-5" viewBox="0 0 1024 1024" version="1.1"
                xmlns="http://www.w3.org/2000/svg" p-id="2527">
                <path
                  d="M683.058 364.695c11 0 22 1.016 32.943 1.976C686.564 230.064 538.896 128 370.681 128c-188.104 0.66-342.237 127.793-342.237 289.226 0 93.068 51.379 169.827 136.725 229.256L130.72 748.43l119.796-59.368c42.918 8.395 77.37 16.79 119.742 16.79 11 0 21.46-0.48 31.914-1.442a259.168 259.168 0 0 1-10.455-71.358c0.485-148.002 128.744-268.297 291.403-268.297l-0.06-0.06z m-184.113-91.992c25.99 0 42.913 16.79 42.913 42.575 0 25.188-16.923 42.579-42.913 42.579-25.45 0-51.38-16.85-51.38-42.58 0-25.784 25.93-42.574 51.38-42.574z m-239.544 85.154c-25.384 0-51.374-16.85-51.374-42.58 0-25.784 25.99-42.574 51.374-42.574 25.45 0 42.918 16.79 42.918 42.575 0 25.188-16.924 42.579-42.918 42.579z m736.155 271.655c0-135.647-136.725-246.527-290.983-246.527-162.655 0-290.918 110.88-290.918 246.527 0 136.128 128.263 246.587 290.918 246.587 33.972 0 68.423-8.395 102.818-16.85l93.809 50.973-25.93-84.677c68.907-51.93 120.286-119.815 120.286-196.033z m-385.275-42.58c-16.923 0-34.452-16.79-34.452-34.179 0-16.79 17.529-34.18 34.452-34.18 25.99 0 42.918 16.85 42.918 34.18 0 17.39-16.928 34.18-42.918 34.18z m188.165 0c-16.984 0-33.972-16.79-33.972-34.179 0-16.79 16.927-34.18 33.972-34.18 25.93 0 42.913 16.85 42.913 34.18 0 17.39-16.983 34.18-42.913 34.18z"
                  fill="currentColor" p-id="2528"></path>
              </svg>
            </button>
            <div v-if="showWechat" class="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
              @click="showWechat = false">
              <div class="p-4 bg-card rounded-lg shadow-2xl" @click.stop>
                <button @click="showWechat = false" class="absolute top-4 right-4 text-foreground hover:text-primary">
                  <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
                <img src="/images/wechat.jpg" alt="WeChat QR Code" class="w-64 h-64 object-contain" />
                <p class="text-center mt-4 text-foreground">扫描二维码添加微信</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom line -->
    <div class="container border-t border-border pt-8 pb-8">
      <div class="flex flex-col md:flex-row justify-between items-center text-sm text-muted-foreground">
        <p>
          <span class="text-primary">{'{'}</span>
          <span> © {{ currentYear }} 70KG的个人空间 </span>
          <span class="text-accent">{'}'}</span>
        </p>
        <p class="mt-4 md:mt-0">
          <span class="text-chart-3"> {'<'} </span>
              <span> Built with Vue 3 & Vite & Tailwind </span>
              <span class="text-chart-3">{'>'}</span>
        </p>
      </div>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink } from 'vue-router'

const currentYear = ref(new Date().getFullYear())
const showWechat = ref(false)
</script>

<style scoped></style>
