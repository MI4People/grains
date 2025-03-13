import json
import os
import warnings
from itertools import islice
from pathlib import Path
from pprint import pprint
from typing import (Any, Dict, Generator, Iterable, Iterator, List, NamedTuple,
                    Tuple)

import mistletoe
import openai
import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                TableFormerMode)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import PictureStackedBarChartData
from mistletoe.ast_renderer import AstRenderer

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL: str = "gpt-4"


class AstData(NamedTuple):
    filename: str
    tokens: int
    data: Dict[str, Any]


# =============== PDF to Markdown Extraction =============== #
def extract_content(pdf_path: Path, md_dir: Path, overwrite: bool = False) -> Tuple[str, Path]:
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
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(str(pdf_path))
        # TODO: can also embed images as base64
        markdown_content = result.document.export_to_markdown()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            print(f"Saved: {md_path}")

    return markdown_content, md_path


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


# =============== Markdown Analysis =============== #


def build_ast(md_tuples: Iterable[Tuple[Path, str]]) -> Iterable[AstData]:
    for md_path, md_content in md_tuples:
        ast_str = mistletoe.markdown(md_content, AstRenderer)
        ast_dict = json.loads(ast_str)
        tokens = count_tokens_in_markdown(md_content)
        yield AstData(filename=md_path.stem, data=ast_dict, tokens=tokens)


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
                grandchild["content"] for grandchild in child["children"] if grandchild["type"] == "RawText"
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


def count_tokens_in_markdown(md_content: str, encoding_name: str = "cl100k_base") -> int:
    """Counts the tokens of a markdown document."""
    encoder = tiktoken.get_encoding(encoding_name)
    tokens = encoder.encode(md_content)
    return len(tokens)


# =============== LLM Processing =============== #


def save_final_document(output_file: Path, content_generator: Generator[str, None, None]) -> None:
    """Save final merged document from generated content"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for section in content_generator:
            f.write(f"{section}\n\n")


def ast_to_document(ast_data: AstData) -> Document:
    """
    Converts an AST data structure (Mistletoe AST as a dictionary)
    to a Document object.

    Args:
        ast_data: An AstData NamedTuple containing the filename, token count,
                  and the Mistletoe AST represented as a Dict.

    Returns:
        A Document object representing the structured document.
    """

    data: Dict = ast_data.data
    filename: str = ast_data.filename
    tokens: int = ast_data.tokens
    sections: List[Section] = []
    current_section: Section | None = None

    def extract_text_content(node: Dict) -> str:
        """Extracts text content recursively from a node and its children."""
        text = ""
        if node.get("type") == "RawText":
            return node.get("content", "")

        for child in node.get("children", []):
            text += extract_text_content(child)
        return text

    for node in data.get("children", []):  # Top level 'children' key now present
        node_type = node.get("type")  # Use get because type may not always be present

        if node_type == "Heading":
            if current_section:
                sections.append(current_section)

            title = extract_text_content(node)  # Extract title more generically
            current_section = Section(
                title=title if title else "No Title",
                level=node.get("level", 1),
            )
        elif node_type == "Paragraph":
            paragraph_content = extract_text_content(node)

            if current_section:
                current_section.content += paragraph_content + "\n\n"  # append spacing
            else:
                sections.append(Section(title="Preamble", content=paragraph_content, level=0))
                current_section = None  # Important: reset var after preamble create
        # Add handling all other elements needed (quotes, list's etc.)
        else:
            pass

    if current_section:
        sections.append(current_section)

    return Document(filename=filename, tokens=tokens, sections=sections)
# =============== Processing Pipelines =============== #


def process_markdown(md_generator: Iterable[Tuple[Path, str]]) -> None:
    ast_generator = build_ast(md_generator)
    # TODO
    for ast in islice(ast_generator, 1):
        # for ast in ast_generator:
        pprint(ast.data["children"])
        headings = extract_headings(ast.data)
        print(f"{ast.filename} with {ast.tokens} tokens.")
        # TODO: need summary
        # for heading in headings:
        #     print((heading["level"] - 1) * "\t" + heading["text"])
    # all_categories = analyze_documents(md_generator)
    # merged_taxonomy = merge_categories(all_categories)
    # content_generator = categorize_and_merge_content(md_paths, merged_taxonomy)
    # save_final_document(output_file, content_generator)


def process_documents(input_dir: Path, md_dir: Path) -> None:
    """Main processing pipeline with error resilience"""
    input_files = input_dir.glob("*.pdf")
    md_generator = extract_all_to_markdown(input_files, md_dir)
    process_markdown(md_generator)


if __name__ == "__main__":
    # Configuration with type-hinted Path objects
    input_dir: Path = Path("data/j_in")
    md_dir: Path = Path("data/j_out")
    # Run processing
    process_documents(input_dir, md_dir)
