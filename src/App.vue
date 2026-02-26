<template>
  <div class="min-h-screen flex flex-col bg-background">
    <RouterView />
  </div>
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'
import { useTheme } from './composables/useTheme'

// Initialize theme
useTheme()
</script>

<style scoped></style>
