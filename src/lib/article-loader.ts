import type { Article } from "./articles";

/**
 * 从 Markdown 文件加载文章
 * 文章格式：
 * ---
 * title: 文章标题
 * date: 2026-01-25
 * category: 前端开发
 * tags: tag1, tag2
 * ---
 * # 文章内容
 */
export async function loadArticleFromMarkdown(
  filePath: string
): Promise<Article | null> {
  try {
    const response = await fetch(filePath);
    if (!response.ok) return null;

    const content = await response.text();
    return parseMarkdownArticle(content, filePath);
  } catch (error) {
    console.error(`Failed to load article from ${filePath}:`, error);
    return null;
  }
}

/**
 * 解析 Markdown 文章
 */
function parseMarkdownArticle(
  content: string,
  filePath: string
): Article | null {
  // 提取 frontmatter
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!frontmatterMatch) return null;

  const frontmatter = parseFrontmatter(frontmatterMatch[1]);
  const body = content.slice(frontmatterMatch[0].length);

  // 计算阅读时间
  const readTime = Math.ceil(body.split(/\s+/).length / 200);

  // 提取文章 ID（从文件名）
  const id =
    filePath.split("/").pop()?.replace(".md", "") || Date.now().toString();

  return {
    id,
    title: frontmatter.title || "无标题",
    excerpt: frontmatter.excerpt || body.slice(0, 150).replace(/[#*_`]/g, ""),
    content: body,
    date: frontmatter.date || new Date().toISOString().split("T")[0],
    category: frontmatter.category || "其他",
    tags: frontmatter.tags || [],
    readTime,
  };
}

/**
 * 解析 YAML frontmatter
 */
function parseFrontmatter(yaml: string): Record<string, any> {
  const result: Record<string, any> = {};

  const lines = yaml.split("\n");
  for (const line of lines) {
    const [key, ...valueParts] = line.split(":");
    if (!key || !valueParts.length) continue;

    const value = valueParts.join(":").trim();
    const cleanKey = key.trim();

    if (cleanKey === "tags") {
      result[cleanKey] = value
        .split(",")
        .map(t => t.trim())
        .filter(Boolean);
    } else {
      result[cleanKey] = value;
    }
  }

  return result;
}

/**
 * 加载所有 Markdown 文章
 */
export async function loadAllArticles(): Promise<Article[]> {
  // 从 manifest.json 获取文章列表
  try {
    const response = await fetch("/articles/manifest.json");
    if (!response.ok) return [];

    const manifest = await response.json();
    const articles: Article[] = [];

    for (const filePath of manifest.articles || []) {
      const article = await loadArticleFromMarkdown(`/articles/${filePath}`);
      if (article) {
        articles.push(article);
      }
    }

    return articles.sort(
      (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
    );
  } catch (error) {
    console.error("Failed to load articles manifest:", error);
    return [];
  }
}
