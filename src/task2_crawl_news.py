"""Task 2: crawl public Shopee help-centre articles to JSON files."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"


def setup_directory() -> None:
    """Create the landing directory when it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Public Shopee Vietnam help-centre pages. Keep at least five URLs for the lab.
ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/79180-[Th%c3%a0nh-vi%c3%aan-m%e1%bb%9bi]-L%c3%a0m-sao-%c4%91%e1%bb%83-mua-h%c3%a0ng-%2F-%c4%91%e1%ba%b7t-h%c3%a0ng-tr%c3%aan-%e1%bb%a9ng-d%e1%bb%a5ng-Shopee%3F?previousPage=other+articles",
    "https://help.shopee.vn/portal/4/article/79543-[ShopeePay]-T%c3%b4i-c%c3%b3-%c4%91%c6%b0%e1%bb%a3c-ho%c3%a0n-ti%e1%bb%81n-khi-g%e1%ba%b7p-s%e1%bb%b1-c%e1%bb%91-giao-d%e1%bb%8bch-qua-V%c3%ad-ShopeePay-t%e1%ba%a1i-c%e1%bb%ada-h%c3%a0ng-kh%c3%b4ng%3F?previousPage=other+articles",
    "https://help.shopee.vn/portal/4/article/125827-%5BB%E1%BA%A3o-m%E1%BA%ADt-t%C3%A0i-kho%E1%BA%A3n%5D-T%C3%B4i-c%E1%BA%A7n-l%C3%A0m-g%C3%AC-n%E1%BA%BFu-c%C3%B3-giao-d%E1%BB%8Bch-l%E1%BA%A1-ph%C3%A1t-sinh-tr%C3%AAn-th%E1%BA%BB-t%C3%ADn-d%E1%BB%A5ng%2Ft%C3%A0i-kho%E1%BA%A3n-ng%C3%A2n-h%C3%A0ng-c%E1%BB%A7a-t%C3%B4i?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/79476-%5BMua-s%E1%BA%AFm-an-to%C3%A0n%5D-L%C3%A0m-sao-%C4%91%E1%BB%83-bi%E1%BA%BFt-s%E1%BA%A3n-ph%E1%BA%A9m-ch%C3%ADnh-h%C3%A3ng-hay-kh%C3%B4ng?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/79564-%5BC%E1%BA%A3nh-b%C3%A1o-l%E1%BB%ABa-%C4%91%E1%BA%A3o%5D-N%C3%AAn-v%C3%A0-kh%C3%B4ng-n%C3%AAn-l%C3%A0m-%C4%91%E1%BB%83-tr%C3%A1nh-nh%E1%BA%ADn-ph%E1%BA%A3i-%C4%91%C6%A1n-h%C3%A0ng-%E1%BA%A3o%2Fgi%E1%BA%A3-m%E1%BA%A1o?previousPage=secondary%20category",
]


def _markdown_text(markdown: Any) -> str:
    """Normalize markdown returned by different Crawl4AI releases."""
    if isinstance(markdown, str):
        return markdown.strip()
    if markdown is None:
        return ""

    # Newer Crawl4AI versions return a MarkdownGenerationResult object.
    for attribute in ("fit_markdown", "raw_markdown", "markdown_with_citations"):
        value = getattr(markdown, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(markdown).strip()


async def crawl_article(url: str) -> dict:
    """Crawl one article and return its source metadata and Markdown content."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid article URL: {url!r}")

    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError as exc:
        raise RuntimeError(
            "Crawl4AI is not installed. Run: pip install crawl4ai"
        ) from exc

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    if result is None:
        raise RuntimeError(f"Crawler returned no result for {url}")
    if getattr(result, "success", True) is False:
        detail = getattr(result, "error_message", "unknown crawl error")
        raise RuntimeError(f"Could not crawl {url}: {detail}")

    content = _markdown_text(getattr(result, "markdown", None))
    if not content:
        raise RuntimeError(
            f"No article content was rendered for {url}; choose another public article"
        )

    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") or metadata.get("og:title") or "Unknown"
    return {
        "url": url,
        "title": str(title).strip(),
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


async def crawl_all() -> None:
    """Crawl every configured article and save one UTF-8 JSON file per page."""
    if len(ARTICLE_URLS) < 5:
        raise ValueError("Task 2 requires at least five article URLs")

    setup_directory()
    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
