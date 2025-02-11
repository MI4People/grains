import os
import warnings
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

import openai
import tiktoken
from docling.document_converter import DocumentConverter

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL: str = "gpt-4"


def extract_all_to_markdown(input_dir: Path, md_dir: Path):
    """Convert all PDFs to Markdown files and return saved paths"""
    md_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in input_dir.glob("*.pdf"):
        try:
            _, md_path = extract_content(pdf_path, md_dir)
            print(f"Saved: {md_path}")
        except Exception as e:
            warnings.warn(f"Failed to process {pdf_path.name}: {str(e)}")


def merge_categories(all_categories: Iterable[str]) -> str:
    """Use LLM to create unified taxonomy."""
    prompt: str = f"""Create a comprehensive chapter structure that best organizes these hospitality categories:

    {"\n\n".join(all_categories)}

    Return final structure in markdown format with hierarchy. Follow these rules:
    1. Group similar concepts (e.g., merge "Hotel Operations" and "Resort Management")
    2. Maintain original technical terms
    3. Order logically from fundamentals to advanced topics
    4. Include clear hierarchy (##, ###, ####)"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content)


def extract_content(
    pdf_path: Path, md_dir: Path, overwrite: bool = False
) -> Tuple[str, Path]:
    """Extract content from a single PDF and save as Markdown"""
    markdown_content: str = ""
    md_path: Path = md_dir / f"{pdf_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if md_path.exists() and not overwrite:
        with md_path.open("r") as f:
            markdown_content = f.read()
    else:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown_content = result.document.export_to_markdown()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    return markdown_content, md_path


def analyze_document(content: str) -> str:
    """Use LLM to identify key categories/sections."""
    prompt: str = f"""Analyze this document and extract hierarchical categories/chapters.
    Return as markdown with maximum 3 levels (##, ###, ####).
    Include brief section summaries (1-2 sentences). Keep technical terminology specific to hospitality:

    {content[:12000]}"""

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content)


def analyze_documents(md_paths: Iterable[Path]) -> List[str]:
    """Process all Markdown files to extract categories"""
    categories = []
    for md_path in md_paths:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            analysis = analyze_document(content)
            categories.append(analysis)
        except Exception as e:
            warnings.warn(f"Failed to analyze {md_path.name}: {str(e)}")
    return categories


def count_okens_in_markdown(file_path: str, encoding_name: str = "cl100k_base") -> int:
    """Counts the tokens of a markdown document."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    encoder = tiktoken.get_encoding(encoding_name)
    tokens = encoder.encode(content)

    return len(tokens)


def categorize_content(content: str, taxonomy: str) -> str:
    """Map document content to unified taxonomy."""
    prompt: str = f"""Match this content to the taxonomy below. Return only category names and relevant excerpts:
    Content: {content[:12000]}
    Taxonomy:
    {taxonomy}"""

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return str(response.choices[0].message.content)


def categorize_and_merge_content(
    md_paths: Iterable[Path], merged_taxonomy: str
) -> Generator[str, None, None]:
    """Categorize content and generate merged sections"""
    yield merged_taxonomy  # Start with taxonomy

    for md_path in md_paths:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            categorized = categorize_content(content, merged_taxonomy)
            yield f"## Source: {md_path.name}\n{categorized}"
        except Exception as e:
            warnings.warn(f"Failed to categorize {md_path.name}: {str(e)}")


def save_final_document(
    output_file: Path, content_generator: Generator[str, None, None]
) -> None:
    """Save final merged document from generated content"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for section in content_generator:
            f.write(f"{section}\n\n")


def process_documents(input_dir: Path, output_file: Path, md_dir: Path) -> None:
    """Main processing pipeline with error resilience"""
    extract_all_to_markdown(input_dir, md_dir)
    md_paths = md_dir.glob("*.md")
    all_categories = analyze_documents(md_paths)
    merged_taxonomy = merge_categories(all_categories)
    content_generator = categorize_and_merge_content(md_paths, merged_taxonomy)
    save_final_document(output_file, content_generator)


if __name__ == "__main__":
    # Configuration with type-hinted Path objects
    input_dir: Path = Path("../../data/pdf")
    output_file: Path = Path("../../data/merged/hospitality_llm_merged.md")
    md_dir: Path = Path("../../data/md")

    # Run processing
    process_documents(input_dir, output_file, md_dir)
    print(f"\nProcessing complete!\nMarkdown files saved to: {md_dir}")
