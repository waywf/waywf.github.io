import {
  parseFrontmatter,
  calculateReadTime,
  type ArticleFrontmatter,
} from "./markdown";

export interface Article {
  id: string;
  filename: string;
  title: string;
  excerpt: string;
  category: string;
  date: string;
  tags: string[];
  readTime: number;
  content: string;
  frontmatter: ArticleFrontmatter;
}

export interface ManifestItem {
  id: string;
  filename: string;
  title: string;
  category: string;
  tags: string[];
  date: string;
}

/**
 * 从 manifest.json 加载文章列表
 */
export async function loadArticlesList(): Promise<string[]> {
  try {
    const response = await fetch("/articles/manifest.json");
    if (!response.ok) throw new Error("Failed to load manifest");
    const data = await response.json();
    // manifest 是一个数组，每个元素有 filename 属性
    if (Array.isArray(data)) {
      return data.map(item => item.filename);
    }
    return data.articles || [];
  } catch (error) {
    console.error("Failed to load articles manifest:", error);
    return [];
  }
}

/**
 * 从 manifest.json 加载完整的文章元数据
 */
export async function loadManifest(): Promise<ManifestItem[]> {
  try {
    const response = await fetch("/articles/manifest.json");
    if (!response.ok) throw new Error("Failed to load manifest");
    const data = await response.json();
    if (Array.isArray(data)) {
      return data;
    }
    return [];
  } catch (error) {
    console.error("Failed to load articles manifest:", error);
    return [];
  }
}

/**
 * 加载单篇文章
 */
export async function loadArticle(filename: string): Promise<Article | null> {
  try {
    const response = await fetch(`/articles/${filename}`);
    if (!response.ok) throw new Error("Failed to load article");
    const content = await response.text();

    const { frontmatter, body } = parseFrontmatter(content);
    const readTime = calculateReadTime(body);

    return {
      id: filename.replace(/\.md$/, ""),
      filename,
      title: frontmatter.title || "Untitled",
      excerpt: frontmatter.excerpt || "",
      category: frontmatter.category || "General",
      date: frontmatter.date || new Date().toISOString().split("T")[0],
      tags: frontmatter.tags || [],
      readTime,
      content: body,
      frontmatter,
    };
  } catch (error) {
    console.error(`Failed to load article ${filename}:`, error);
    return null;
  }
}

/**
 * 加载所有文章
 */
export async function loadAllArticles(): Promise<Article[]> {
  const filenames = await loadArticlesList();
  const articles: Article[] = [];

  for (const filename of filenames) {
    const article = await loadArticle(filename);
    if (article) {
      articles.push(article);
    }
  }

  // 按日期降序排序
  articles.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );

  return articles;
}

/**
 * 加载最新文章（只加载指定数量的文章）
 * 使用manifest中的date字段排序，再通过id加载文章
 */
export async function loadLatestArticles(count: number): Promise<Article[]> {
  const manifest = await loadManifest();

  // 按date降序排序
  const sortedManifest = [...manifest].sort((a, b) => {
    return new Date(b.date).getTime() - new Date(a.date).getTime();
  });

  // 取前count个
  const topItems = sortedManifest.slice(0, count);

  // 通过id加载文章
  const articles: Article[] = [];
  for (const item of topItems) {
    const article = await loadArticle(item.filename);
    if (article) {
      articles.push(article);
    }
  }

  return articles;
}

/**
 * 加载相邻文章（上一篇和下一篇）
 */
export async function loadNeighborArticles(
  currentId: string
): Promise<{ prev: Article | null; next: Article | null }> {
  const filenames = await loadArticlesList();
  const currentIndex = filenames.findIndex(
    filename => filename.replace(/\.md$/, "") === currentId
  );

  if (currentIndex === -1) {
    return { prev: null, next: null };
  }

  const prevArticle =
    currentIndex > 0 ? await loadArticle(filenames[currentIndex - 1]) : null;
  const nextArticle =
    currentIndex < filenames.length - 1
      ? await loadArticle(filenames[currentIndex + 1])
      : null;

  return { prev: prevArticle, next: nextArticle };
}

/**
 * 分页加载文章
 */
export async function loadArticlesByPage(
  page: number,
  pageSize: number
): Promise<Article[]> {
  const filenames = await loadArticlesList();
  const startIndex = (page - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  const pageFilenames = filenames.slice(startIndex, endIndex);

  const articles: Article[] = [];
  for (const filename of pageFilenames) {
    const article = await loadArticle(filename);
    if (article) {
      articles.push(article);
    }
  }

  // 按日期降序排序
  articles.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );

  return articles;
}

/**
 * 加载所有文章的元数据（用于分类和标签过滤）
 * 直接从manifest读取，无需加载完整文章内容
 */
export async function loadAllArticlesMetadata(): Promise<{
  categories: string[];
  tags: string[];
}> {
  const manifest = await loadManifest();
  const categories = new Set<string>();
  const tags = new Set<string>();

  manifest.forEach(item => {
    if (item.category) {
      categories.add(item.category);
    }
    if (item.tags && Array.isArray(item.tags)) {
      item.tags.forEach(tag => tags.add(tag));
    }
  });

  return {
    categories: Array.from(categories).sort(),
    tags: Array.from(tags).sort(),
  };
}

/**
 * 根据分类从manifest获取文章ID列表
 */
export async function getArticleIdsByCategory(
  category: string
): Promise<string[]> {
  const manifest = await loadManifest();
  return manifest
    .filter(item => item.category === category)
    .map(item => item.id);
}

/**
 * 根据标签从manifest获取文章ID列表
 */
export async function getArticleIdsByTag(tag: string): Promise<string[]> {
  const manifest = await loadManifest();
  return manifest
    .filter(item => item.tags && item.tags.includes(tag))
    .map(item => item.id);
}

/**
 * 根据ID列表加载文章
 */
export async function loadArticlesByIds(ids: string[]): Promise<Article[]> {
  const articles: Article[] = [];
  for (const id of ids) {
    const article = await loadArticle(`${id}.md`);
    if (article) {
      articles.push(article);
    }
  }
  // 按日期降序排序
  articles.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
  return articles;
}

/**
 * 按分类过滤文章
 */
export function filterArticlesByCategory(
  articles: Article[],
  category: string
): Article[] {
  if (category === "all") return articles;
  return articles.filter(article => article.category === category);
}

/**
 * 搜索文章
 */
export function searchArticles(articles: Article[], query: string): Article[] {
  const lowerQuery = query.toLowerCase();
  return articles.filter(
    article =>
      article.title.toLowerCase().includes(lowerQuery) ||
      article.excerpt.toLowerCase().includes(lowerQuery) ||
      article.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
  );
}

/**
 * 获取所有分类
 */
export function getCategories(articles: Article[]): string[] {
  const categories = new Set<string>();
  articles.forEach(article => {
    if (article.category) {
      categories.add(article.category);
    }
  });
  return Array.from(categories).sort();
}

/**
 * 获取所有标签
 */
export function getAllTags(articles: Article[]): string[] {
  const tags = new Set<string>();
  articles.forEach(article => {
    article.tags.forEach(tag => tags.add(tag));
  });
  return Array.from(tags).sort();
}
