import json
import os
import uuid
import warnings
from itertools import islice
from pathlib import Path
from pprint import pprint
from typing import (Any, Dict, Generator, Iterable, Iterator, List, NamedTuple,
                    Tuple)
from uuid import UUID

import mistletoe
import openai
import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                TableFormerMode)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import PictureStackedBarChartData
from mistletoe.ast_renderer import AstRenderer

from grains.data_structures import AstData, Curriculum, Document, Section
from grains.llm_utils import add_summaries
from grains.mapping import load_or_create_mappings_for_docs
from grains.utils import (load_curriculum, save_pydantic_object,
                          try_loading_document_object)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "meta-llama/llama-3.3-70b-instruct"
SUMMARY_MODEL = "meta-llama/llama-3.3-70b-instruct"
MAPPINGS_MODEL = "meta-llama/llama-3.3-70b-instruct"

# =============== PDF to Markdown Extraction =============== #
def ocr(md_path: Path, md_dir: Path, overwrite: bool = False) -> str:
    """
    Extracts content from a single PDF and saves it as Markdown.

    Args:
        md_path (Path): Path to the Markdown file.
        md_dir (Path): Directory for the Markdown file.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.

    Returns:
        str: Markdown content of the PDF.
    """
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(str(md_path))
    # TODO: can also embed images as base64
    markdown_content = result.document.export_to_markdown()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        print(f"Saved: {md_path}")

    return markdown_content

def get_md_paths(input_files: Iterable[Path], md_dir: Path) -> Iterable[Path]:
    """
    Generates Markdown file paths for the given input files.

    Args:
        input_files (Iterable[Path]): List of input PDF files.
        md_dir (Path): Directory for the Markdown files.

    Returns:
        Iterable[Path]: List of Markdown file paths.
    """
    return [md_dir / f"{pdf_path.stem}.md" for pdf_path in input_files]

def extract_or_load_markdown(
        input_files: Iterable[Path], md_dir: Path, overwrite: bool = False, only_load: bool = True
) -> Iterator[Tuple[Path, str]]:
    """
    Converts PDFs to Markdown files and yields saved paths and contents.

    Args:
        input_files (Iterable[Path]): List of input PDF files.
        md_dir (Path): Directory for the Markdown files.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        only_load (bool): Whether to only load existing files. Defaults to True.

    Yields:
        Tuple[Path, str]: Markdown file path and content.
    """
    md_dir.mkdir(parents=True, exist_ok=True)
    md_paths = get_md_paths(input_files, md_dir)
    for md_path in md_paths:
        if md_path.exists() and not overwrite:
            with md_path.open("r") as f:
                md_content = f.read()
                yield md_path, md_content
        else:
            if not only_load:
                try:
                    md_content = ocr(md_path, md_dir, overwrite)
                    yield md_path, md_content
                except Exception as e:
                    warnings.warn(f"Failed to process {md_path.name}: {str(e)}")


# =============== Markdown Analysis =============== #


def build_ast(md_tuples: Iterable[Tuple[Path, str]]) -> Iterable[AstData]:
    """
    Builds an Abstract Syntax Tree (AST) for each Markdown file.

    Args:
        md_tuples (Iterable[Tuple[Path, str]]): List of Markdown file paths and contents.

    Yields:
        AstData: AST data for each Markdown file.
    """
    for md_path, md_content in md_tuples:
        ast_str = mistletoe.markdown(md_content, AstRenderer)
        ast_dict = json.loads(ast_str)
        tokens = count_tokens_in_markdown(md_content)
        yield AstData(filename=md_path.stem, data=ast_dict, tokens=tokens)


def extract_headings(node: Dict, max_level=3, current_level=1):
    """
    Recursively extracts headings from the AST up to the specified maximum level.

    Args:
        node (Dict): Current node in the AST.
        max_level (int): Maximum heading level to include. Defaults to 3.
        current_level (int): Current heading level. Defaults to 1.

    Returns:
        List[Dict]: Nested list of headings as dictionaries.
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
    """
    Counts the tokens in a Markdown document using the specified encoding.

    Args:
        md_content (str): Markdown content.
        encoding_name (str): Encoding name. Defaults to "cl100k_base".

    Returns:
        int: Number of tokens in the Markdown content.
    """
    encoder = tiktoken.get_encoding(encoding_name)
    tokens = encoder.encode(md_content)
    return len(tokens)


# =============== LLM Processing =============== #


def save_final_document(output_file: Path, content_generator: Generator[str, None, None]) -> None:
    """
    Saves the final merged document from the generated content.

    Args:
        output_file (Path): Path to the output file.
        content_generator (Generator[str, None, None]): Generator for the document content.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for section in content_generator:
            f.write(f"{section}\n\n")


def ast_to_document(ast_data: AstData) -> Document:
    """
    Converts AST data to a Document object.

    Args:
        ast_data (AstData): AST data containing filename, tokens, and AST structure.

    Returns:
        Document: Structured Document object.
    """
    data: Dict = ast_data.data
    filename: str = ast_data.filename
    tokens: int = ast_data.tokens
    sections: List[Section] = []
    current_section: Section | None = None

    def extract_text_content(node: Dict) -> str:
        """
        Recursively extracts text content from a node and its children.

        Args:
            node (Dict): Node in the AST.

        Returns:
            str: Extracted text content.
        """
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
        # Add handling for other elements as needed (e.g., quotes, lists)
        else:
            pass

    if current_section:
        sections.append(current_section)

    return Document(filename=filename, tokens=tokens, sections=sections)


# =============== Processing Pipelines =============== #


def asts_to_documents(ast_generator: Iterable[AstData]) -> Iterable[Document]:
    """
    Converts AST data to Document objects.

    Args:
        ast_generator (Iterable[AstData]): Generator for AST data.

    Returns:
        Iterable[Document]: List of Document objects.
    """
    docs = []
    for ast in ast_generator:
        doc = ast_to_document(ast)
        docs.append(doc)
    return docs


def get_object_paths(documents: Iterable[Document], base_dir: Path | str = "data/objects") -> Iterable[Path]:
    """
    Generates object file paths for the given documents.

    Args:
        documents (Iterable[Document]): List of Document objects.
        base_dir (Path | str): Base directory for object files. Defaults to "data/objects".

    Returns:
        Iterable[Path]: List of object file paths.
    """
    return [(Path(base_dir) / f"{doc.filename}.json", doc) for doc in documents]


def generate_and_store_summary(document: Document) -> Document:  # Pass base directory
    """
    Generates a summary for the Document object and stores it.

    Args:
        document (Document): Document object to summarize.

    Returns:
        Document: Document object with generated summary.
    """
    try:
        # Call OpenAI API
        document = add_summaries(document, SUMMARY_MODEL)
        save_pydantic_object(document, filepath)  # Save updated document
        return document
    except Exception as e:
        print(f"Error: Failed to generate summary: {e}")
        return document  # return original.


def generate_or_load_summary_documents(
    documents: Iterable[Document], overwrite: bool = False, base_dir: str = "data/objects"  # Pass base directory
) -> Iterable[Document]:
    """
    Generates or loads summary documents.

    Args:
        documents (Iterable[Document]): List of Document objects.
        overwrite (bool): Whether to overwrite existing summaries. Defaults to False.
        base_dir (str): Base directory for summaries. Defaults to "data/objects".

    Returns:
        Iterable[Document]: List of Document objects with summaries.
    """
    results = []
    os.makedirs(base_dir, exist_ok=True)
    obj_paths_docs = get_object_paths(documents, base_dir)
    for path, doc in obj_paths_docs:
        loaded_document = try_loading_document_object(path)
        if loaded_document:
            print(loaded_document.filename)
        if loaded_document and loaded_document.has_summaries and not overwrite:
            results.append(loaded_document)
        else:
            # TODO: if we do not want to wait for new documents outcomment the following line
            # continue
            result = generate_and_store_summary(doc, path)
            results.append(result)
        # yield generate_and_store_summary(d, base_dir)
    return results


def process_markdown(md_generator: Iterable[Tuple[Path, str]], overwrite: bool = False) -> Iterable[Document]:
    """
    Processes Markdown content to generate Document objects.

    Args:
        md_generator (Iterable[Tuple[Path, str]]): Generator for Markdown file paths and contents.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.

    Returns:
        Iterable[Document]: List of processed Document objects.
    """
    ast_generator = build_ast(md_generator)
    # TODO: remove =========================
    # ast_generator = islice(ast_generator, 1)
    # ======================================
    doc_generator = asts_to_documents(ast_generator)
    docs = generate_or_load_summary_documents(list(doc_generator))
    return docs


def load_or_create_aggregations(mappings_store, docs_with_summaries):
    """
    Loads or creates aggregations from the mappings store and summarized documents.

    Args:
        mappings_store: Store for relevance mappings.
        docs_with_summaries (Iterable[Document]): List of Document objects with summaries.
    """
    pass


def process_documents(input_dir: Path, md_dir: Path, curriculum) -> None:
    """
    Main processing pipeline for documents.

    Args:
        input_dir (Path): Directory containing input PDF files.
        md_dir (Path): Directory for Markdown files.
        curriculum (Curriculum): Curriculum object for mapping.
    """
    input_files = input_dir.glob("*.pdf")
    md_path_markdown_tuples = extract_or_load_markdown(input_files, md_dir)
    md_path_markdown_tuples = list(md_path_markdown_tuples)
    #print([it[0] for it in list(md_path_markdown_tuples)])
    print("Create Summaries")
    docs_with_summaries = process_markdown(md_path_markdown_tuples)
    # Create the mappings file
    print("Create Mappings")
    mappings_store = load_or_create_mappings_for_docs(docs_with_summaries, curriculum, MAPPINGS_MODEL)
    aggregations = load_or_create_aggregations(mappings_store, docs_with_summaries)


if __name__ == "__main__":
    # Configuration with type-hinted Path objects
    input_dir: Path = Path("data/pdf")
    md_dir: Path = Path("data/md")
    # Load desired curriculum
    curriculum = load_curriculum()
    # Run processing
    process_documents(input_dir, md_dir, curriculum)
