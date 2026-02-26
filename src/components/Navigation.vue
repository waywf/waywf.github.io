<template>
  <nav class="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur-sm dark:scan-lines">
    <div class="container flex items-center justify-between h-16">
      <!-- Logo -->
      <RouterLink to="/"
        class="flex items-center gap-2 text-xl font-bold dark:neon-glow hover:text-accent transition-colors">
        <span class="text-primary">{'<'} </span>
            <span>70KG</span>
            <span class="text-accent">{'>'}</span>
      </RouterLink>

      <!-- Desktop Navigation -->
      <div class="hidden md:flex items-center gap-8">
        <RouterLink to="/" class="text-foreground hover:text-primary transition-colors duration-200">
          首页
        </RouterLink>
        <RouterLink to="/articles" class="text-foreground hover:text-primary transition-colors duration-200">
          文章
        </RouterLink>
        <RouterLink to="/about" class="text-foreground hover:text-primary transition-colors duration-200">
          关于
        </RouterLink>

        <!-- Theme Toggle Button -->
        <button @click="toggleTheme" class="p-2 rounded-lg hover:bg-muted transition-colors" aria-label="Toggle theme">
          <!-- Sun icon for dark mode (click to switch to light) -->
          <svg v-if="currentTheme === 'dark'" class="w-5 h-5 text-primary" fill="none" stroke="currentColor"
            viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
          <!-- Moon icon for light mode (click to switch to dark) -->
          <svg v-else class="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>

      </div>

      <!-- Mobile Menu Button -->
      <div class="md:hidden flex items-center gap-2">
        <!-- Mobile Theme Toggle -->
        <button @click="toggleTheme" class="p-2 rounded-lg hover:bg-muted transition-colors" aria-label="Toggle theme">
          <svg v-if="currentTheme === 'dark'" class="w-5 h-5 text-primary" fill="none" stroke="currentColor"
            viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
          <svg v-else class="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>
        <button @click="isOpen = !isOpen" class="p-2 hover:text-primary transition-colors" aria-label="Toggle menu">
          <svg v-if="!isOpen" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
          <svg v-else class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>

    <!-- Mobile Navigation -->
    <div v-if="isOpen" class="md:hidden border-t border-border bg-background">
      <div class="container py-4 flex flex-col gap-4">
        <RouterLink to="/" class="text-foreground hover:text-primary transition-colors" @click="isOpen = false">
          首页
        </RouterLink>
        <RouterLink to="/articles" class="text-foreground hover:text-primary transition-colors" @click="isOpen = false">
          文章
        </RouterLink>
        <RouterLink to="/about" class="text-foreground hover:text-primary transition-colors" @click="isOpen = false">
          关于
        </RouterLink>

      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { RouterLink } from 'vue-router'
import { useTheme } from '../composables/useTheme'

const isOpen = ref(false)
const { currentTheme, toggleTheme } = useTheme()
</script>

<style scoped></style>
