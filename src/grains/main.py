import json
import os
import warnings
from itertools import islice
from pathlib import Path
from pprint import pprint
from typing import Dict, Generator, Iterable, Iterator, List, NamedTuple, Tuple

import mistletoe
import openai
import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                TableFormerMode)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import PictureStackedBarChartData
from mistletoe.ast_renderer import AstRenderer
from pydantic import BaseModel, Field

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL: str = "gpt-4"


class AstData(NamedTuple):
    filename: str
    tokens: int
    data: Dict


class HeadingSummary(BaseModel):
    """Represents a heading with its text, level, content, and summary."""
    text: str = Field(..., description="The text of the heading")
    level: int = Field(..., description="The level of the heading (1-6)")
    content: str = Field("", description="The content text under this heading")
    summary: str = Field("", description="Generated summary of the heading content")
    children: List["HeadingSummary"] = Field(default_factory=list, description="Child headings")

HeadingSummary.model_rebuild()


def extract_all_to_markdown(
    input_files: Iterable[Path], md_dir: Path, overwrite: bool = False
) -> Iterator[Tuple[Path, str]]:
    """Convert all PDFs to Markdown files and yield saved paths and contents."""
    md_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in input_files:
        try:
            md_content, md_path = extract_content(pdf_path, md_dir, overwrite)
            yield md_path, md_content
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
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        # TODO: can also embed images as base64
        markdown_content = result.document.export_to_markdown()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            print(f"Saved: {md_path}")

    return markdown_content, md_path


def extract_headings(node: Dict, max_level=3, current_level=1):
    """
    Recursively builds a nested structure of headings as dictionaries up to the specified maximum level.

    Args:
        node (dict): The current node in the AST.
        max_level (int): The maximum heading level to include (inclusive).
        current_level (int): The current heading level being processed.

    Returns:
        list: A nested list representing the heading structure as dictionaries.
    """
    structure = []
    for child in node.get("children", []):
        if child["type"] == "Heading" and child["level"] <= max_level:
            heading_text = "".join(
                grandchild["content"]
                for grandchild in child["children"]
                if grandchild["type"] == "RawText"
            )
            sub_structure = extract_headings(child, max_level, child["level"])
            structure.append(
                {
                    "text": heading_text,
                    "children": sub_structure,
                    "level": current_level,
                }
            )
        else:
            sub_structure = extract_headings(child, max_level, current_level)
            if sub_structure:
                structure.extend(sub_structure)
    return structure


def extract_heading_content(node: Dict) -> str:
    """Extract text content from paragraphs under a heading."""
    text = ""
    for child in node.get("children", []):
        if child["type"] == "Paragraph":
            for paragraph_child in child.get("children", []):
                if paragraph_child["type"] == "RawText":
                    text += paragraph_child.get("content", "") + " "
    return text.strip()


def generate_heading_summary(content: str, heading_text: str) -> str:
    """Generate a concise summary (20-200 words) for heading content."""
    if not content.strip():
        return "No content available to summarize."
    
    client = openai.OpenAI()
    
    prompt = f"""Summarize the following content under the heading "{heading_text}".
    Use only factual information present in the text.
    Keep the summary between 20-200 words:
    
    {content[:4000]}"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )
        return str(response.choices[0].message.content).strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def analyze_document(content: str) -> str:
    """Use LLM to identify key categories/sections."""
    # prompt: str = f"""Analyze this document and extract hierarchical categories/chapters.
    # Return as markdown with maximum 3 levels (##, ###, ####).
    # Include brief section summaries (1-2 sentences). Keep technical terminology specific to hospitality:

    # {content[:12000]}"""

    # client = openai.OpenAI()
    # response = client.chat.completions.create(
    #     model=MODEL, messages=[{"role": "user", "content": prompt}]
    # )
    # return str(response.choices[0].message.content)
    ast = mistletoe.markdown(content, AstRenderer)
    return ast


def build_ast(md_tuples: Iterable[Tuple[Path, str]]) -> Iterable[AstData]:
    for md_path, md_content in md_tuples:
        ast_str = mistletoe.markdown(md_content, AstRenderer)
        ast_dict = json.loads(ast_str)
        tokens = count_tokens_in_markdown(md_content)
        yield AstData(filename=md_path.stem, data=ast_dict, tokens=tokens)


def analyze_documents(md_tuples: Iterable[Tuple[Path, str]]) -> List[str]:
    """Process all Markdown files to extract categories"""
    categories = []
    for md_path, md_content in islice(md_tuples, 2):
        try:
            analysis = analyze_document(md_content)
            categories.append(analysis)
        except Exception as e:
            warnings.warn(f"Failed to analyze {md_path.name}: {str(e)}")
    return categories


def count_tokens_in_markdown(
    md_content: str, encoding_name: str = "cl100k_base"
) -> int:
    """Counts the tokens of a markdown document."""
    encoder = tiktoken.get_encoding(encoding_name)
    tokens = encoder.encode(md_content)
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


def process_heading_with_summary(node: Dict, level: int = 0) -> HeadingSummary:
    """Process a heading node to create a HeadingSummary with content and summary."""
    heading_text = "".join(
        child["content"] for child in node.get("children", []) 
        if child["type"] == "RawText"
    )

    content = extract_heading_content(node)
    summary = generate_heading_summary(content, heading_text)

    return HeadingSummary(
        text=heading_text,
        level=level,
        content=content,
        summary=summary,
        children=[]
    )


def process_markdown(md_generator: Iterable[Tuple[Path, str]]) -> None:
    ast_generator = build_ast(md_generator)
    for ast in ast_generator:
        # pprint(ast.data["children"][:15])
        headings = extract_headings(ast.data)
        print(f"{ast.filename} with {ast.tokens} tokens.")
        
        summaries = []
        for heading_dict in headings:
            heading_node = next(
                (child for child in ast.data.get("children", []) 
                if child["type"] == "Heading" and 
                "".join(grandchild["content"] for grandchild in child["children"] 
                        if grandchild["type"] == "RawText") == heading_dict["text"]),
                None
            )
            
            if heading_node:
                summary = process_heading_with_summary(heading_node, heading_dict["level"])
                summaries.append(summary)
                print(f"Heading: {summary.text} (Level {summary.level})")
                print(f"Summary: {summary.summary}\n")
        
        output_file = Path(f"data/summaries/{ast.filename}_summaries.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([s.dict() for s in summaries], f, indent=2)
        print(f"Saved summaries to {output_file}")


def process_documents(input_dir: Path, output_file: Path, md_dir: Path) -> None:
    """Main processing pipeline with error resilience"""
    input_files = input_dir.glob("*.pdf")
    md_generator = extract_all_to_markdown(input_files, md_dir)
    process_markdown(md_generator)


if __name__ == "__main__":
    # Configuration with type-hinted Path objects
    input_dir: Path = Path("data/pdf")
    output_file: Path = Path("data/merged/hospitality_llm_merged.md")
    md_dir: Path = Path("data/md")
    # Run processing
    process_documents(input_dir, output_file, md_dir)