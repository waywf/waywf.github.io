import MarkdownIt from 'markdown-it'
import markdownItAnchor from 'markdown-it-anchor'
import markdownItHighlightJs from 'markdown-it-highlightjs'

// 创建 markdown-it 实例
const md: MarkdownIt = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  highlight: (str: string, lang: string): string => {
    if (lang) {
      try {
        return `<pre class="hljs language-${lang}"><code>${md.utils.escapeHtml(str)}</code></pre>`
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

/**
 * 渲染 Markdown 内容
 */
export function renderMarkdown(content: string): string {
  try {
    return md.render(content)
  } catch (error) {
    console.error('Failed to render markdown:', error)
    return '<p>Failed to render content</p>'
  }
}

/**
 * 解析 Markdown frontmatter
 */
export interface ArticleFrontmatter {
  title?: string
  date?: string
  category?: string
  tags?: string[]
  excerpt?: string
  author?: string
}

export function parseFrontmatter(content: string): {
  frontmatter: ArticleFrontmatter
  body: string
} {
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n/)
  
  if (!frontmatterMatch) {
    return {
      frontmatter: {},
      body: content
    }
  }

  const frontmatterText = frontmatterMatch[1]
  const body = content.slice(frontmatterMatch[0].length)
  const frontmatter: ArticleFrontmatter = {}

  const lines = frontmatterText.split('\n')
  for (const line of lines) {
    const [key, ...valueParts] = line.split(':')
    if (!key || !valueParts.length) continue

    const value = valueParts.join(':').trim()
    const cleanKey = key.trim()

    if (cleanKey === 'tags') {
      frontmatter.tags = value.split(',').map(t => t.trim()).filter(Boolean)
    } else if (cleanKey === 'title') {
      frontmatter.title = value
    } else if (cleanKey === 'date') {
      frontmatter.date = value
    } else if (cleanKey === 'category') {
      frontmatter.category = value
    } else if (cleanKey === 'excerpt') {
      frontmatter.excerpt = value
    } else if (cleanKey === 'author') {
      frontmatter.author = value
    }
  }

  return { frontmatter, body }
}

/**
 * 计算阅读时间（分钟）
 */
export function calculateReadTime(content: string): number {
  const words = content.split(/\s+/).length
  return Math.ceil(words / 200)
}
