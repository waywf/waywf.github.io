import type { Article } from './articles'

interface FrontMatter {
  title: string
  excerpt: string
  category: string
  date: string
  readTime: number
  tags: string[]
}

interface ArticleManifest {
  id: string
  filename: string
}

/**
 * 解析 Markdown 文件的 Front Matter
 * Front Matter 格式:
 * ---
 * title: 文章标题
 * excerpt: 文章摘要
 * category: 分类
 * date: 2026-02-23
 * readTime: 8
 * tags: [tag1, tag2]
 * ---
 */
function parseFrontMatter(content: string): { frontMatter: FrontMatter; body: string } {
  const frontMatterRegex = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/
  const match = content.match(frontMatterRegex)

  if (!match) {
    throw new Error('Invalid Markdown format: Missing front matter')
  }

  const frontMatterStr = match[1]
  const body = match[2]

  const frontMatter: FrontMatter = {
    title: '',
    excerpt: '',
    category: '技术',
    date: new Date().toISOString().split('T')[0],
    readTime: 5,
    tags: [],
  }

  // 解析 YAML 格式的 Front Matter
  const lines = frontMatterStr.split('\n')
  for (const line of lines) {
    if (!line.trim()) continue

    const [key, ...valueParts] = line.split(':')
    const value = valueParts.join(':').trim()

    if (key === 'title') {
      frontMatter.title = value
    } else if (key === 'excerpt') {
      frontMatter.excerpt = value
    } else if (key === 'category') {
      frontMatter.category = value
    } else if (key === 'date') {
      frontMatter.date = value
    } else if (key === 'readTime') {
      frontMatter.readTime = parseInt(value, 10)
    } else if (key === 'tags') {
      // 解析数组格式: [tag1, tag2, tag3]
      const tagsStr = value.replace(/[\[\]]/g, '')
      frontMatter.tags = tagsStr
        .split(',')
        .map((tag) => tag.trim())
        .filter((tag) => tag)
    }
  }

  return { frontMatter, body }
}

/**
 * 加载所有 Markdown 文章
 */
export async function loadArticles(): Promise<Article[]> {
  try {
    // 加载清单文件
    const manifestResponse = await fetch('/articles/manifest.json')
    if (!manifestResponse.ok) {
      console.warn('Failed to load manifest.json, using default articles')
      return []
    }

    const manifest: ArticleManifest[] = await manifestResponse.json()

    // 加载每个文件
    const articles: Article[] = []
    for (const item of manifest) {
      try {
        const article = await loadArticle(item.filename)
        articles.push(article)
      } catch (error) {
        console.error(`Failed to load article: ${item.filename}`, error)
      }
    }

    // 按日期排序（最新的在前）
    articles.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())

    return articles
  } catch (error) {
    console.error('Failed to load articles:', error)
    return []
  }
}

/**
 * 加载单个 Markdown 文章
 */
export async function loadArticle(filename: string): Promise<Article> {
  const response = await fetch(`/articles/${filename}`)
  if (!response.ok) {
    throw new Error(`Failed to load article: ${filename}`)
  }

  const content = await response.text()
  const { frontMatter, body } = parseFrontMatter(content)

  // 从文件名提取 ID
  const id = filename.replace(/\.md$/, '')

  return {
    id,
    title: frontMatter.title,
    excerpt: frontMatter.excerpt,
    content: body,
    category: frontMatter.category,
    date: frontMatter.date,
    readTime: frontMatter.readTime,
    tags: frontMatter.tags,
  }
}

/**
 * 通过 ID 加载文章
 */
export async function loadArticleById(id: string): Promise<Article | null> {
  try {
    // 先加载清单以找到正确的文件名
    const manifestResponse = await fetch('/articles/manifest.json')
    if (!manifestResponse.ok) {
      return null
    }

    const manifest: ArticleManifest[] = await manifestResponse.json()
    const item = manifest.find((m) => m.id === id)

    if (!item) {
      return null
    }

    return await loadArticle(item.filename)
  } catch (error) {
    console.error(`Failed to load article with id: ${id}`, error)
    return null
  }
}
