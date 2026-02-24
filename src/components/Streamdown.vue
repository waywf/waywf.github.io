<template>
  <div class="prose prose-invert max-w-none" v-html="renderedContent"></div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import markdownItAnchor from 'markdown-it-anchor'
import markdownItHighlightJs from 'markdown-it-highlightjs'

interface Props {
  content: string
}

const props = defineProps<Props>()

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  highlight: (str, lang) => {
    if (lang && lang.match(/^ts$|^js$|^vue$|^html$|^css$|^python$/)) {
      try {
        return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`
      } catch {
        return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`
      }
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`
  }
})
  .use(markdownItAnchor, {
    permalink: markdownItAnchor.permalink.headerLink()
  })
  .use(markdownItHighlightJs)

const renderedContent = computed(() => {
  try {
    return md.render(props.content)
  } catch (error) {
    console.error('Failed to render markdown:', error)
    return '<p>Failed to render content</p>'
  }
})
</script>

<style scoped>
</style>
