<template>
  <div class="min-h-screen flex flex-col bg-background">
    <Navigation />

    <!-- Header Section -->
    <section class="py-12 md:py-16 border-b border-border">
      <div class="container px-4 max-w-4xl">
        <h1 class="text-4xl md:text-5xl font-bold mb-4">
          <span class="text-primary">{'$whoami'}</span>
        </h1>
        <p class="text-muted-foreground text-lg">
          了解更多关于我和这个项目的信息。
        </p>
      </div>
    </section>

    <!-- About Content -->
    <section class="py-12 md:py-16 flex-1">
      <div class="container max-w-4xl px-4">
        <!-- Introduction -->
        <div class="mb-12 p-8 border-2 border-border rounded bg-card/50">
          <h2 class="text-2xl font-bold mb-4 text-primary">
            {'<'} 关于我 {'>'}
          </h2>
          <p class="text-muted-foreground leading-relaxed mb-4">
            我是70KG，一个热爱编程和设计的前端攻城狮。我对创建美观、高效和用户友好的网络应用充满热情。
          </p>
          <p class="text-muted-foreground leading-relaxed mb-4">
            在过去的几年里，我在前端开发、UI/UX 设计和后端开发方面积累了丰富的经验。我相信好的设计不仅仅是美观，更重要的是要解决实际问题。
          </p>
          <p class="text-muted-foreground leading-relaxed">
            在这个博客中，我分享我的技术见解、设计思考和生活感悟。
          </p>
        </div>

        <!-- Experience -->
        <div class="mb-12">
          <h2 class="text-2xl font-bold mb-6 text-chart-3">
            {'$ experience'}
          </h2>
          <div class="space-y-6">
            <div v-for="(experience, index) in experiences" :key="experience.id"
              class="p-6 border-l-4 bg-card/50 rounded" :style="{ borderLeftColor: experience.color }">
              <div class="flex justify-between items-start mb-2">
                <div>
                  <h3 class="text-lg font-bold text-foreground mb-2">{{ experience.title }}</h3>
                  <p class="text-muted-foreground">{{ experience.company }} | {{ experience.period }}</p>
                </div>
                <button @click="toggleExpand(index)" :class="['hover:text-primary transition-colors']"
                  :style="{ color: experience.color }">
                  <svg :class="['w-5 h-5 transition-transform duration-300', expanded[index] ? 'rotate-180' : '']"
                    fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>
              <div
                :class="['overflow-hidden transition-all duration-300', expanded[index] ? 'max-h-[1500px] opacity-100' : 'max-h-[100px] opacity-70']">
                <p class="text-muted-foreground">{{ experience.description }}</p>
              </div>
            </div>
          </div>
        </div>
        <!-- Project -->
        <div class="mb-12">
          <h2 class="text-2xl font-bold mb-6 text-accent">
            {'$ projects'}
          </h2>
          <div class="space-y-6">
            <div v-for="(project, index) in projects" :key="project.id"
              class="p-6 border-2 bg-card/50 rounded hover:shadow-lg transition-all hover:border-primary">
              <div class="flex justify-between items-start mb-3">
                <div>
                  <h3 class="text-lg font-bold text-foreground mb-1">{{ project.name }}</h3>
                  <p class="text-sm text-muted-foreground">{{ project.role }} | {{ project.period }}</p>
                </div>
                <div class="flex flex-wrap gap-2 justify-end">
                  <div v-for="tag in project.tags" :key="tag"
                    class="px-3 py-1 text-xs rounded-full bg-primary/10 text-primary whitespace-nowrap">
                    {{ tag }}
                  </div>
                </div>
              </div>
              <div
                :class="['overflow-hidden transition-all duration-300', projectExpanded[index] ? 'max-h-[1500px] opacity-100' : 'max-h-[100px] opacity-70']">
                <p class="text-muted-foreground leading-relaxed">{{ project.description }}</p>
              </div>
              <button @click="toggleProjectExpand(index)" class="mt-4 text-primary hover:text-accent transition-colors">
                <span v-if="!projectExpanded[index]">查看更多</span>
                <span v-else>收起</span>
              </button>
            </div>
          </div>
        </div>
        <!-- Skills -->
        <div class="mb-12">
          <h2 class="text-2xl font-bold mb-6 text-accent">
            {'$ skills'}
          </h2>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="p-6 border-2 border-border rounded hover:border-primary transition-all">
              <h3 class="text-lg font-bold text-accent mb-3">前端开发(精通)</h3>
              <ul class="space-y-2 text-muted-foreground">
                <li>• HTML & CSS & JavaScript</li>
                <li>• Vue2 & Vue3 & React</li>
                <li>• Taro & Uni-app & 微信小程序</li>
                <li>• Echart & AntV</li>
                <li>• ---</li>
              </ul>
            </div>

            <div class="p-6 border-2 border-border rounded hover:border-accent transition-all">
              <h3 class="text-lg font-bold text-primary mb-3">后端开发(见习)</h3>
              <ul class="space-y-2 text-muted-foreground">
                <li>• Node.js & Express & Python</li>
                <li>• MySQL & MongoDB</li>
                <li>• RESTful API</li>
                <li>• System Design</li>
                <li>• ---</li>
              </ul>
            </div>

            <div class="p-6 border-2 border-border rounded hover:border-chart-3 transition-all">
              <h3 class="text-lg font-bold text-primary mb-3">设计工具(见习)</h3>
              <ul class="space-y-2 text-muted-foreground">
                <li>• Figma</li>
                <li>• Axure</li>
                <li>• Sketch</li>
                <li>• Pencil</li>
                <li>• ---</li>
              </ul>
            </div>

            <div class="p-6 border-2 border-border rounded hover:border-primary transition-all">
              <h3 class="text-lg font-bold text-chart-3 mb-3">其他技能(兴趣)</h3>
              <ul class="space-y-2 text-muted-foreground">
                <li>• AI & Machine Learning</li>
                <li>• Docker & DevOps</li>
                <li>• 软件设计师</li>
                <li>• CI/CD</li>
                <li>• ---</li>
              </ul>
            </div>
          </div>
        </div>
        <!-- Contact -->
        <div class="p-8 border-2 border-border rounded bg-card/50">
          <h2 class="text-2xl font-bold mb-6 text-accent">
            {'$ contact'}
          </h2>
          <p class="text-muted-foreground mb-6">
            欢迎通过以下方式与我联系。我很乐意讨论技术、设计或任何有趣的想法。
          </p>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <!-- wechat -->
            <button @click="toggleWechat"
              class="flex items-center gap-2 p-4 border-2 border-border rounded hover:border-primary transition-all hover:bg-primary/10">
              <svg t="1772097691955" class="w-5 h-5 text-primary" viewBox="0 0 1024 1024" version="1.1"
                xmlns="http://www.w3.org/2000/svg" p-id="2527">
                <path
                  d="M683.058 364.695c11 0 22 1.016 32.943 1.976C686.564 230.064 538.896 128 370.681 128c-188.104 0.66-342.237 127.793-342.237 289.226 0 93.068 51.379 169.827 136.725 229.256L130.72 748.43l119.796-59.368c42.918 8.395 77.37 16.79 119.742 16.79 11 0 21.46-0.48 31.914-1.442a259.168 259.168 0 0 1-10.455-71.358c0.485-148.002 128.744-268.297 291.403-268.297l-0.06-0.06z m-184.113-91.992c25.99 0 42.913 16.79 42.913 42.575 0 25.188-16.923 42.579-42.913 42.579-25.45 0-51.38-16.85-51.38-42.58 0-25.784 25.93-42.574 51.38-42.574z m-239.544 85.154c-25.384 0-51.374-16.85-51.374-42.58 0-25.784 25.99-42.574 51.374-42.574 25.45 0 42.918 16.79 42.918 42.575 0 25.188-16.924 42.579-42.918 42.579z m736.155 271.655c0-135.647-136.725-246.527-290.983-246.527-162.655 0-290.918 110.88-290.918 246.527 0 136.128 128.263 246.587 290.918 246.587 33.972 0 68.423-8.395 102.818-16.85l93.809 50.973-25.93-84.677c68.907-51.93 120.286-119.815 120.286-196.033z m-385.275-42.58c-16.923 0-34.452-16.79-34.452-34.179 0-16.79 17.529-34.18 34.452-34.18 25.99 0 42.918 16.85 42.918 34.18 0 17.39-16.928 34.18-42.918 34.18z m188.165 0c-16.984 0-33.972-16.79-33.972-34.179 0-16.79 16.927-34.18 33.972-34.18 25.93 0 42.913 16.85 42.913 34.18 0 17.39-16.983 34.18-42.913 34.18z"
                  fill="#09BB07" p-id="2528"></path>
              </svg>
              <span class="text-sm text-foreground">WeChat</span>
            </button>
            <a href="mailto:381531043@qq.com"
              class="flex items-center gap-2 p-4 border-2 border-border rounded hover:border-primary transition-all hover:bg-primary/10">
              <svg class="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <span class="text-sm text-foreground">Email</span>
            </a>
            <a href="https://github.com/waywf"
              class="flex items-center gap-2 p-4 border-2 border-border rounded hover:border-accent transition-all hover:bg-accent/10">
              <svg class="w-5 h-5 text-accent" fill="currentColor" viewBox="0 0 24 24">
                <path
                  d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v 3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              <span class="text-sm text-foreground">GitHub</span>
            </a>

          </div>
          <!-- WeChat QR Code Modal -->
          <div v-if="showWechat" class="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
            @click="toggleWechat">
            <div class="p-4 bg-card rounded-lg shadow-2xl" @click.stop>
              <button @click="toggleWechat" class="absolute top-4 right-4 text-foreground hover:text-primary">
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
    </section>

    <Footer />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import Navigation from '../components/Navigation.vue'
import Footer from '../components/Footer.vue'
import { experiences, type Experience } from '../data/experience'
import { projects, type Project } from '../data/projects'

const expanded = ref<boolean[]>([false, false, false])
const projectExpanded = ref<boolean[]>([false, false, false, false])
const showWechat = ref<boolean>(false)

const toggleExpand = (index: number) => {
  expanded.value[index] = !expanded.value[index]
}

const toggleProjectExpand = (index: number) => {
  projectExpanded.value[index] = !projectExpanded.value[index]
}

const toggleWechat = () => {
  showWechat.value = !showWechat.value
}
</script>

<style scoped></style>
