"""
Taxonomy Generation Module for Hospitality Documents.

This module provides functionality to convert PDF documents into a unified
taxonomy structure using LLM analysis. It processes multiple hospitality
documents, extracts their content structure, and merges them into a
comprehensive categorized document.

Main workflow:
1. Convert PDFs to Markdown using Docling
2. Analyze each document for categories using LLM
3. Merge categories into unified taxonomy
4. Categorize content against taxonomy
5. Generate final merged document

Dependencies:
    - openai: For LLM analysis
    - docling: For PDF to Markdown conversion
    - tiktoken: For token counting
"""

import os
import warnings
from pathlib import Path
from typing import Generator, Iterable, List, Tuple, Optional

import openai
import tiktoken
from docling.document_converter import DocumentConverter

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL: str = "gpt-4"


def extract_all_to_markdown(input_dir: Path, md_dir: Path) -> None:
    """Convert all PDFs in input directory to Markdown files.
    
    Args:
        input_dir: Directory containing PDF files to convert
        md_dir: Directory where Markdown files will be saved
        
    Raises:
        Warning: If any PDF file fails to process
    """
    md_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in input_dir.glob("*.pdf"):
        try:
            _, md_path = extract_content(pdf_path, md_dir)
            print(f"Saved: {md_path}")
        except Exception as e:
            warnings.warn(f"Failed to process {pdf_path.name}: {str(e)}")


def merge_categories(all_categories: Iterable[str]) -> str:
    """Use LLM to create a unified taxonomy from multiple category lists.
    
    Args:
        all_categories: Collection of category strings from different documents
        
    Returns:
        Unified taxonomy structure in markdown format with hierarchy
        
    Raises:
        openai.OpenAIError: If LLM API call fails
    """
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
    """Extract content from a single PDF and save as Markdown.
    
    Args:
        pdf_path: Path to the PDF file to convert
        md_dir: Directory where Markdown file will be saved
        overwrite: Whether to overwrite existing Markdown file
        
    Returns:
        Tuple containing the markdown content and path to saved file
        
    Raises:
        IOError: If file operations fail
        DocumentConverter.ConversionError: If PDF conversion fails
    """
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
    """Use LLM to identify key categories and sections from document content.
    
    Args:
        content: Document content to analyze (truncated to 12000 chars)
        
    Returns:
        Hierarchical categories in markdown format with section summaries
        
    Raises:
        openai.OpenAIError: If LLM API call fails
    """
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
    """Process all Markdown files to extract categories and structure.
    
    Args:
        md_paths: Collection of paths to Markdown files
        
    Returns:
        List of category analyses, one per document
        
    Raises:
        Warning: If any document fails to analyze
    """
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


def count_tokens_in_markdown(file_path: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a markdown document.
    
    Args:
        file_path: Path to the markdown file
        encoding_name: Tiktoken encoding to use for counting
        
    Returns:
        Number of tokens in the document
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If file encoding is invalid
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    encoder = tiktoken.get_encoding(encoding_name)
    tokens = encoder.encode(content)

    return len(tokens)


def categorize_content(content: str, taxonomy: str) -> str:
    """Map document content to unified taxonomy categories.
    
    Args:
        content: Document content to categorize (truncated to 12000 chars)
        taxonomy: Unified taxonomy structure to map against
        
    Returns:
        Category mappings with relevant content excerpts
        
    Raises:
        openai.OpenAIError: If LLM API call fails
    """
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
    """Categorize content from multiple documents and generate merged sections.
    
    Args:
        md_paths: Collection of paths to Markdown files
        merged_taxonomy: Unified taxonomy structure for categorization
        
    Yields:
        Categorized content sections, starting with taxonomy structure
        
    Raises:
        Warning: If any document fails to categorize
    """
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
    """Save final merged document from generated content sections.
    
    Args:
        output_file: Path where the final document will be saved
        content_generator: Generator yielding content sections to write
        
    Raises:
        IOError: If file operations fail
        OSError: If directory creation fails
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for section in content_generator:
            f.write(f"{section}\n\n")


def process_documents(input_dir: Path, output_file: Path, md_dir: Path) -> None:
    """Main processing pipeline that converts PDFs to unified taxonomy document.
    
    This function orchestrates the entire workflow:
    1. Converts PDFs to Markdown
    2. Analyzes documents for categories
    3. Merges categories into unified taxonomy
    4. Categorizes and merges content
    5. Saves final document
    
    Args:
        input_dir: Directory containing PDF files to process
        output_file: Path for the final merged document
        md_dir: Directory for intermediate Markdown files
        
    Raises:
        Various exceptions from constituent functions
    """
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
