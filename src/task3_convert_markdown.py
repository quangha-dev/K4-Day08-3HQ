"""Task 3: convert landing-zone legal documents and articles to Markdown."""

import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}

# Stable metadata for source files that do not carry metadata internally.
# customer_role is intentionally explicit so retrieval can filter buyer/seller docs.
LEGAL_METADATA = {
    "payment-methods-shopee": {
        "title": "Phương thức thanh toán trên Shopee",
        "category": "payment",
        "customer_role": "buyer",
        "source_url": "https://help.shopee.vn/portal/4/article/79198",
        "tags": ["thanh toán", "ShopeePay", "ngân hàng"],
    },
    "privacy-policy-shopee": {
        "title": "Chính sách bảo mật Shopee",
        "category": "privacy",
        "customer_role": "both",
        "source_url": "https://help.shopee.vn/portal/4/article/77244",
        "tags": ["bảo mật", "quyền riêng tư", "dữ liệu cá nhân"],
    },
    "returns-refund-policy-shopee": {
        "title": "Chính sách trả hàng và hoàn tiền Shopee",
        "category": "returns_refunds",
        "customer_role": "both",
        "source_url": "https://help.shopee.vn/portal/4/article/77251",
        "tags": ["trả hàng", "hoàn tiền", "khiếu nại"],
    },
}

NEWS_METADATA = {
    "article_01": {"category": "ordering", "tags": ["đặt hàng", "người dùng mới"]},
    "article_02": {"category": "refunds", "tags": ["ShopeePay", "hoàn tiền"]},
    "article_03": {"category": "account_security", "tags": ["giao dịch lạ", "bảo mật"]},
    "article_04": {"category": "safe_shopping", "tags": ["chính hãng", "mua sắm an toàn"]},
    "article_05": {"category": "fraud_prevention", "tags": ["lừa đảo", "đơn hàng giả mạo"]},
}


def _yaml_front_matter(metadata: dict) -> str:
    """Serialize simple metadata as YAML-compatible front matter."""
    lines = ["---"]
    for key, value in metadata.items():
        # JSON scalars/lists are valid YAML and safely quote URLs and Vietnamese text.
        encoded = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {encoded}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


_NUMBERED_SECTION = re.compile(r"^(\d+(?:\.\d+)*\.?)[ \t]+(.+)$")


def _is_uppercase_heading(text: str) -> bool:
    """Return True for PDF section labels such as '3. ĐIỀU KIỆN ÁP DỤNG'."""
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and len(text) <= 140 and all(
        not char.isalpha() or char.isupper() for char in text
    )


def _structure_legal_markdown(content: str, title: str) -> str:
    """Add semantic headings to plain text extracted from policy PDFs."""
    output = [f"# {title}", ""]
    seen_numbers: dict[str, int] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = _NUMBERED_SECTION.match(line)
        if match:
            number, label = match.groups()
            key = number.rstrip(".")
            seen_numbers[key] = seen_numbers.get(key, 0) + 1
            depth = key.count(".") + 1

            # Top-level uppercase clauses are major sections. Short nested labels
            # are subsections. In payment documents, repeated numbered labels mark
            # the detailed section after an initial table-of-contents-like list.
            if depth == 1 and _is_uppercase_heading(label):
                line = f"## {number} {label}"
            elif depth > 1 and len(label) <= 90 and not label.endswith((".", ";", ":")):
                line = f"### {number} {label}"
            elif depth == 1 and seen_numbers[key] > 1 and len(label) <= 80:
                line = f"## {number} {label}"
        output.append(line)

    return "\n".join(output).strip()


def _structure_news_markdown(content: str, title: str) -> str:
    """Ensure a single H1 and nest headings supplied by Crawl4AI below it."""
    output = [f"# {title}", ""]
    for line in content.splitlines():
        match = re.match(r"^(#{1,5})\s+(.+)$", line)
        if match:
            hashes, label = match.groups()
            # The document title owns H1; crawled sections begin at H2.
            line = f"{'#' * (len(hashes) + 1)} {label}"
        output.append(line)
    return "\n".join(output).strip()


def _create_converter():
    """Create MarkItDown lazily and provide a useful installation error."""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            'MarkItDown is not installed. Run: pip install "markitdown[pdf]"'
        ) from exc
    return MarkItDown()


def convert_legal_docs() -> int:
    """Convert all supported legal files while preserving the legal subfolder."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"Skipping missing directory: {legal_dir}")
        return 0

    files = sorted(
        path for path in legal_dir.iterdir()
        if path.is_file() and path.suffix.lower() in LEGAL_EXTENSIONS
    )
    if not files:
        print("No PDF/DOCX/DOC legal documents found.")
        return 0

    converter = _create_converter()
    for filepath in files:
        print(f"Converting: {filepath.name}")
        result = converter.convert(str(filepath))
        content = (getattr(result, "text_content", None) or "").strip()
        if not content:
            raise RuntimeError(f"Conversion produced empty content: {filepath}")

        configured = LEGAL_METADATA.get(filepath.stem, {})
        metadata = {
            "schema_version": "1.0",
            "document_id": f"legal:{filepath.stem}",
            "title": configured.get("title", filepath.stem.replace("-", " ").title()),
            "document_type": "legal",
            "category": configured.get("category", "other"),
            "customer_role": configured.get("customer_role", "both"),
            "platform": "Shopee Vietnam",
            "language": "vi",
            "source_url": configured.get("source_url", ""),
            "source_file": filepath.name,
            "date_crawled": None,
            "tags": configured.get("tags", []),
        }
        output_path = output_dir / f"{filepath.stem}.md"
        structured_content = _structure_legal_markdown(
            content, str(metadata["title"])
        )
        output_path.write_text(
            _yaml_front_matter(metadata) + structured_content + "\n", encoding="utf-8"
        )
        print(f"  Saved: {output_path}")

    return len(files)


def convert_news_articles() -> int:
    """Convert crawled article JSON files and include their source metadata."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"Skipping missing directory: {news_dir}")
        return 0

    files = sorted(path for path in news_dir.glob("*.json") if path.is_file())
    for filepath in files:
        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {filepath}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {filepath}")

        title = str(data.get("title") or "Unknown")
        url = str(data.get("url") or "N/A")
        crawled = str(data.get("date_crawled") or "N/A")
        body = str(data.get("content_markdown") or data.get("content") or "").strip()
        if not body:
            raise ValueError(f"Article has no Markdown content: {filepath}")

        configured = NEWS_METADATA.get(filepath.stem, {})
        metadata = {
            "schema_version": "1.0",
            "document_id": f"news:{filepath.stem}",
            "title": title,
            "document_type": "news",
            "category": configured.get("category", "customer_support"),
            "customer_role": "buyer",
            "platform": "Shopee Vietnam",
            "language": "vi",
            "source_url": url,
            "source_file": filepath.name,
            "date_crawled": None if crawled == "N/A" else crawled,
            "tags": configured.get("tags", []),
        }
        structured_content = _structure_news_markdown(body, title)
        content = _yaml_front_matter(metadata) + structured_content + "\n"
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"  Saved: {output_path}")

    return len(files)


def convert_all() -> None:
    """Convert all supported landing files to data/standardized/."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    print(f"\nDone: {legal_count} legal documents, {news_count} news articles")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
