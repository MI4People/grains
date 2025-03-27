import json
import os
import warnings
from itertools import islice
from pathlib import Path
from pprint import pprint
from typing import (Any, Dict, Generator, Iterable, Iterator, List, NamedTuple,
                    Tuple)

import boto3
import mistletoe
import openai
import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                TableFormerMode)
from docling.document_converter import DocumentConverter, PdfFormatOption
from mistletoe.ast_renderer import AstRenderer

from grains.data_structures import AstData, Document, Section
from grains.llm_utils import add_summaries
from grains.utils import load_curriculum, try_loading_document_object

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL: str = "gpt-4"

# Initialize S3 client
S3_BUCKET_NAME: str = "grains-files"
S3_PREFIX: str = "house-keeping/"
# NOTE:
# Add a ~/.aws/credentials file
# with
# [grains]
# aws_access_key_id = "<id>"
# aws_secret_access_key = "<key>"
session = boto3.Session(profile_name="grains")
s3 = session.client("s3")


# =============== S3 Download =============== #


def check_s3_connection():
    """Check if the connection to S3 is successful."""
    try:
        s3.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=1)
        print("Successfully connected to S3.")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        print("AWS credentials are missing or incorrect.")
    except EndpointConnectionError:
        print("Unable to connect to S3 endpoint. Check your network or region settings.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False


def download_missing_files(input_dir: Path):
    """List S3 objects and download only those not present in the local directory."""
    os.makedirs(input_dir, exist_ok=True)
    
    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
    if "Contents" in response:
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            file_name = os.path.basename(s3_key)  # Get only the filename
            local_file_path = os.path.join(input_dir, file_name)

            if not os.path.exists(local_file_path):
                print(f"Downloading {s3_key} to {local_file_path}...")
                s3.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
            else:
                print(f"Skipping {file_name}, already exists.")
        print("Download complete.")


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


def asts_to_documents(ast_generator: Iterable[AstData]) -> Iterable[Document]:
    docs = []
    for ast in ast_generator:
        doc = ast_to_document(ast)
        docs.append(doc)
    return docs


def save_document(document: Document, filepath: Path) -> None:
    """Serializes and saves a Document to a JSON file."""
    with open(filepath, "w") as f:
        f.write(document.model_dump_json(indent=2))
    print(f"Saved document to {filepath}")


def generate_and_store_summary(
    document: Document, overwrite: bool = False, base_dir: str = "data/objects"
) -> Document:  # Pass base directory
    """
    Loads a Document object if it exists, otherwise generates summaries
    stores it, and returns the Document.

    Args:
        document:  The original document
        base_dir: Base directory to put files under

    Returns:
        The Document object (either loaded or newly summarized).
    """
    title = document.filename
    filepath = Path(base_dir) / f"{title}.json"
    os.makedirs(base_dir, exist_ok=True)
    loaded_document = try_loading_document_object(filepath)

    if loaded_document and not overwrite:
        return loaded_document
    else:
        print(f"Document not found at {filepath}. Generating summary...")
        try:
            # call openAI API KEY
            document = add_summaries(document)
            save_document(document, filepath)  # Save updated document
            return document
        except Exception as e:
            print(f"Error: Failed to generate summary: {e}")
            return document  # return original.


def generate_and_store_or_load_summary_documents(
    documents: Iterable[Document], overwrite: bool = False, base_dir: str = "data/objects"  # Pass base directory
) -> Iterable[Document]:
    results = []
    for d in documents:
        print(d.filename)
        result = generate_and_store_summary(d, overwrite, base_dir)
        # yield generate_and_store_summary(d, base_dir)
        results.append(result)
    return results


def process_markdown(md_generator: Iterable[Tuple[Path, str]]) -> None:
    ast_generator = build_ast(md_generator)
    # TODO: remove =========================
    # ast_generator = islice(ast_generator, 1)
    # ======================================
    doc_generator = asts_to_documents(ast_generator)
    doc_generator = generate_and_store_or_load_summary_documents(doc_generator)
    # merged_taxonomy = merge_categories(all_categories)
    # content_generator = categorize_and_merge_content(md_paths, merged_taxonomy)
    # save_final_document(output_file, content_generator)


def process_documents(input_dir: Path, md_dir: Path) -> None:
    """Main processing pipeline with error resilience"""
    if check_s3_connection():
        download_missing_files(input_dir)
    input_files = input_dir.glob("*.pdf")
    md_generator = extract_all_to_markdown(input_files, md_dir)
    process_markdown(md_generator)


if __name__ == "__main__":
    # Configuration with type-hinted Path objects
    input_dir: Path = Path("data/pdf")
    md_dir: Path = Path("data/md")
    # Load desired curriculum
    curriculum = load_curriculum()
    # Run processing
    process_documents(input_dir, md_dir)
