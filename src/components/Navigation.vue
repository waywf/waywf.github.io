<template>
  <nav class="sticky top-0 z-50 border-b border-[#2D3447] bg-[#0F1419]/95 backdrop-blur-sm scan-lines">
    <div class="container flex items-center justify-between h-16">
      <!-- Logo -->
      <RouterLink to="/" class="flex items-center gap-2 text-xl font-bold neon-glow hover:text-[#FF006E] transition-colors">
        <span class="text-[#00FF41]">{'<'}</span>
        <span>安澄</span>
        <span class="text-[#FF006E]">{'>'}</span>
      </RouterLink>

      <!-- Desktop Navigation -->
      <div class="hidden md:flex items-center gap-8">
        <RouterLink to="/" class="text-[#E0E0E0] hover:text-[#00FF41] transition-colors duration-200">
          首页
        </RouterLink>
        <RouterLink to="/articles" class="text-[#E0E0E0] hover:text-[#00FF41] transition-colors duration-200">
          文章
        </RouterLink>
        <RouterLink to="/about" class="text-[#E0E0E0] hover:text-[#00FF41] transition-colors duration-200">
          关于
        </RouterLink>

      </div>

      <!-- Mobile Menu Button -->
      <button
        @click="isOpen = !isOpen"
        class="md:hidden p-2 hover:text-[#00FF41] transition-colors"
        aria-label="Toggle menu"
      >
        <svg v-if="!isOpen" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
        <svg v-else class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>

    <!-- Mobile Navigation -->
    <div v-if="isOpen" class="md:hidden border-t border-[#2D3447] bg-[#0F1419]">
      <div class="container py-4 flex flex-col gap-4">
        <RouterLink 
          to="/" 
          class="text-[#E0E0E0] hover:text-[#00FF41] transition-colors"
          @click="isOpen = false"
        >
          首页
        </RouterLink>
        <RouterLink 
          to="/articles" 
          class="text-[#E0E0E0] hover:text-[#00FF41] transition-colors"
          @click="isOpen = false"
        >
          文章
        </RouterLink>
        <RouterLink 
          to="/about" 
          class="text-[#E0E0E0] hover:text-[#00FF41] transition-colors"
          @click="isOpen = false"
        >
          关于
        </RouterLink>

      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink } from 'vue-router'

const isOpen = ref(false)
</script>

<style scoped>
</style>
