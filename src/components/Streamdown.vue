<template>
  <div class="prose max-w-none" v-html="renderedContent"></div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import markdownit from 'markdown-it'
// import markdownItAnchor from 'markdown-it-anchor'
// import markdownItHighlightJs from 'markdown-it-highlightjs'
import hljs from 'highlight.js'
import 'highlight.js/styles/atom-one-dark.css'
interface Props {
  content: string
}

const props = defineProps<Props>()

const md = markdownit({
  html: true,
  linkify: true,
  typographer: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return '<pre><code class="hljs">' +
          hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
          '</code></pre>';
      } catch (__) { }
    }
    return '<pre><code class="hljs">' + md.utils.escapeHtml(str) + '</code></pre>';
  }
})
// .use(markdownItAnchor, {
//   permalink: markdownItAnchor.permalink.headerLink()
// })
// .use(markdownItHighlightJs)

const renderedContent = computed(() => {
  try {
    return md.render(props.content)
  } catch (error) {
    console.error('Failed to render markdown:', error)
    return '<p>Failed to render content</p>'
  }
})
</script>

<style scoped></style>
