import { parseFrontmatter, calculateReadTime, type ArticleFrontmatter } from './markdown'

export interface Article {
  id: string
  filename: string
  title: string
  excerpt: string
  category: string
  date: string
  tags: string[]
  readTime: number
  content: string
  frontmatter: ArticleFrontmatter
}

/**
 * 从 manifest.json 加载文章列表
 */
export async function loadArticlesList(): Promise<string[]> {
  try {
    const response = await fetch('/articles/manifest.json')
    if (!response.ok) throw new Error('Failed to load manifest')
    const data = await response.json()
    // manifest 是一个数组，每个元素有 filename 属性
    if (Array.isArray(data)) {
      return data.map(item => item.filename)
    }
    return data.articles || []
  } catch (error) {
    console.error('Failed to load articles manifest:', error)
    return []
  }
}

/**
 * 加载单篇文章
 */
export async function loadArticle(filename: string): Promise<Article | null> {
  try {
    const response = await fetch(`/articles/${filename}`)
    if (!response.ok) throw new Error('Failed to load article')
    const content = await response.text()
    
    const { frontmatter, body } = parseFrontmatter(content)
    const readTime = calculateReadTime(body)
    
    return {
      id: filename.replace(/\.md$/, ''),
      filename,
      title: frontmatter.title || 'Untitled',
      excerpt: frontmatter.excerpt || '',
      category: frontmatter.category || 'General',
      date: frontmatter.date || new Date().toISOString().split('T')[0],
      tags: frontmatter.tags || [],
      readTime,
      content: body,
      frontmatter
    }
  } catch (error) {
    console.error(`Failed to load article ${filename}:`, error)
    return null
  }
}

/**
 * 加载所有文章
 */
export async function loadAllArticles(): Promise<Article[]> {
  const filenames = await loadArticlesList()
  const articles: Article[] = []
  
  for (const filename of filenames) {
    const article = await loadArticle(filename)
    if (article) {
      articles.push(article)
    }
  }
  
  // 按日期降序排序
  articles.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  
  return articles
}

/**
 * 加载最新文章（只加载指定数量的文章）
 */
export async function loadLatestArticles(count: number): Promise<Article[]> {
  const filenames = await loadArticlesList()
  const articles: Article[] = []
  
  // 只加载前 count 篇文章
  const limitedFilenames = filenames.slice(0, count)
  
  for (const filename of limitedFilenames) {
    const article = await loadArticle(filename)
    if (article) {
      articles.push(article)
    }
  }
  
  // 按日期降序排序
  articles.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  
  return articles
}

/**
 * 按分类过滤文章
 */
export function filterArticlesByCategory(articles: Article[], category: string): Article[] {
  if (category === 'all') return articles
  return articles.filter(article => article.category === category)
}

/**
 * 搜索文章
 */
export function searchArticles(articles: Article[], query: string): Article[] {
  const lowerQuery = query.toLowerCase()
  return articles.filter(article =>
    article.title.toLowerCase().includes(lowerQuery) ||
    article.excerpt.toLowerCase().includes(lowerQuery) ||
    article.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
  )
}

/**
 * 获取所有分类
 */
export function getCategories(articles: Article[]): string[] {
  const categories = new Set<string>()
  articles.forEach(article => {
    if (article.category) {
      categories.add(article.category)
    }
  })
  return Array.from(categories).sort()
}

/**
 * 获取所有标签
 */
export function getAllTags(articles: Article[]): string[] {
  const tags = new Set<string>()
  articles.forEach(article => {
    article.tags.forEach(tag => tags.add(tag))
  })
  return Array.from(tags).sort()
}
