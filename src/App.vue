<template>
  <div class="min-h-screen flex flex-col bg-background">
    <RouterView v-slot="{ Component }">
      <KeepAlive include="Articles">
        <component :is="Component" />
      </KeepAlive>
    </RouterView>
  </div>
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'
import { KeepAlive } from 'vue'
import { useTheme } from './composables/useTheme'

// Initialize theme
useTheme()
</script>

<style scoped></style>
