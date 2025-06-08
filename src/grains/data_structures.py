import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import (Any, Dict, Generator, Iterable, Iterator, List, NamedTuple,
                    Optional, Set, Tuple)
from uuid import UUID

import tiktoken
from openai import OpenAI
from pydantic import BaseModel, Field, root_validator

from grains.prompts import generate_aggregation_prompts


class AstData(NamedTuple):
    """
    Auxiliary data structure for storing AST-related information.

    Attributes:
        filename (str): Name of the file.
        tokens (int): Number of tokens in the file.
        data (Dict[str, Any]): Additional data related to the AST.
    """
    filename: str
    tokens: int
    data: Dict[str, Any]


class Topic(BaseModel):
    """
    Represents a topic within a curriculum module.

    Attributes:
        id (UUID): Unique identifier for the topic (auto-generated).
        name (str): Name of the topic.
        description (str): Brief description of the topic.
        content (list[str]): List of the original sections with relevance score >= threshold.
    """
    id: UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(..., description="Name of the topic")
    description: str = Field(..., description="Description of the topic")
    content: Optional[List[str]] = Field(default_factory=list, description="Content of the relevant sections")

    def __str__(self) -> str:
        content_str = (
            "\n    Content:\n" + "\n".join(f"      - {line}" for line in self.content)
            if self.content else "\n    Content: []"
        )
        return f"Topic: {self.name}\n    Description: {self.description}{content_str}"



class Module(BaseModel):
    """
    Represents a module within a curriculum.

    Attributes:
        id (UUID): Unique identifier for the module (auto-generated).
        name (str): Name of the module.
        topics (List[Topic]): List of topics covered in the module.
    """
    id: UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(..., description="Name of the module")
    topics: List[Topic] = Field(..., description="Topics covered in the module")

    def __str__(self) -> str:
        topics_str = "\n    ".join(str(topic) for topic in self.topics)
        return f"Module: {self.name}\n  Topics:\n    {topics_str}"

    def __getitem__(self, topic_name: str) -> Topic:
        """Allows dictionary-like access to topics by name, e.g., module["Topic Name"]"""
        for topic in self.topics:
            if topic.name == topic_name:
                return topic
        raise KeyError(f"Topic '{topic_name}' not found in module '{self.name}'.")


class Curriculum(BaseModel):
    """
    Represents a curriculum comprising multiple modules.

    Attributes:
        modules (List[Module]): List of modules in the curriculum.
    """
    modules: List[Module] = Field(..., description="List of modules in the curriculum")

    def __str__(self) -> str:
        modules_str = "\n\n".join(str(module) for module in self.modules)
        return f"Curriculum:\n{modules_str}"

    def __getitem__(self, module_name: str) -> Module:
        """Allows dictionary-like access to modules by name, e.g., curriculum["Module Name"]"""
        for module in self.modules:
            if module.name == module_name:
                return module
        raise KeyError(f"Module '{module_name}' not found in curriculum.")


class Section(BaseModel):
    """
    Represents a section within a document.

    Attributes:
        id (UUID): Unique identifier for the section (auto-generated).
        title (str): Title of the section.
        level (int): Heading level (1-6) of the section.
        content (Optional[str]): Content of the section.
        summary (Optional[str]): Summary of the section content.
        summary_title (Optional[str]): Section title based on the summary.
    """
    id: UUID = Field(default_factory=uuid.uuid4)
    title: str = Field(..., description="Title of the section")
    level: int = Field(..., description="Heading level (1-6)")
    content: Optional[str] = Field("", description="Content of the section")
    summary: Optional[str] = Field(None, description="Summary of the section content")
    summary_title: Optional[str] = Field(None, description="Section title based on summary")

    def __str__(self) -> str:
        """Returns a markdown representation of the Section object."""
        text = f"### {self.title} - {self.summary_title}\n"
        text += f"#### Summary:\n{self.summary}\n"
        text += f"#### Content:\n{self.content}"
        return text

    @property
    def tokens(self) -> int:
        """
        Calculates the number of tokens in the section's content.
        Uses tiktoken if available and an encoding is specified, otherwise falls back
        to a simple space-separated word count.
        """
        if not self.content:
            return 0

        encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(self.content)
        return len(token_ids)


class Document(BaseModel):
    """
    Represents a document with its metadata and content.

    Attributes:
        id (UUID): Unique identifier for the document (auto-generated).
        filename (Optional[str]): Name of the source file.
        tokens (Optional[int]): Number of tokens in the document.
        summary (Optional[str]): Summary of the entire document.
        summary_title (Optional[str]): Document title based on the summary.
        sections (List[Section]): List of sections in the document.
    """
    id: UUID = Field(default_factory=uuid.uuid4)
    filename: Optional[str] = Field(None, description="Name of the source file")
    #tokens: Optional[int] = Field(None, description="Number of tokens in the document")
    summary: Optional[str] = Field(None, description="Summary of the entire document, generated by an LLM")
    summary_title: Optional[str] = Field(None, description="Document title based on summary")
    sections: List[Section] = Field(
        ...,
        description="Sections of the document as an aggregation of the Sections summaries.",
    )

    def __str__(self) -> str:
        """Returns a markdown representation of the Document object."""
        tokens = sum([sec.tokens for sec in self.sections])
        markdown = f"# {self.filename} - {self.summary_title}\nTokens: {tokens}\n"
        markdown += f"## Summary\n{self.summary}\n"
        markdown += f"## Sections\n"
        if self.sections:
            markdown += "\n".join([str(section) for section in self.sections])
        return markdown

    @property
    def tokens(self) -> int:
        """
        Calculates the number of tokens in the section's content.
        Uses tiktoken if available and an encoding is specified, otherwise falls back
        to a simple space-separated word count.
        """
        if not self.sections:
            return 0
        return sum([sec.tokens for sec in self.sections])

    @property
    def has_summaries(self) -> bool:
        """Checks if the document has a summary and all sections have summaries."""
        if not self.summary:
            return False
        if any(section is None for section in self.sections):
            return False
        return True

    @property
    def section_titles(self) -> str:
        """Returns a concatenated string of section titles."""
        return "\n".join([s.title for s in self.sections])

    @property
    def section_ids(self) -> List[UUID]:
        """Returns a List of section ids for this document."""
        return {section.id: section for section in self.sections}
    
    def filter_by_section_ids(self, section_ids_to_keep: Iterable[UUID]) -> 'Document':
        """
        Creates a new Document containing only the sections whose IDs are in
        section_ids_to_keep and keeps the orginal order.

        Args:
            section_ids_to_keep: An iterable of UUIDs for the sections to retain,
                                 defining the order of sections in the new document.
        Returns:
            A new Document instance with the filtered and reordered sections.
        """
        filtered_sections = [
            section for section in self.sections if section.id in section_ids_to_keep
        ]
        return Document(
            # id is handled by default_factory in Pydantic model
            filename=self.filename,
            sections=filtered_sections,
            tokens=None,  # needs recalculation for the new subset
            summary=self.summary,
            summary_title=self.summary
        )

class RelevanceMapping(BaseModel):
    """
    Represents a relevance mapping between a document section and a curriculum topic.

    Attributes:
        module_id (UUID): Unique identifier for the document.
        document_id (UUID): Unique identifier for the module.
        section_id (UUID): Unique identifier for the section.
        topic_id (UUID): Unique identifier for the topic module.
        relevance_score (float): Relevance score between 0 and 1.
        reasoning (str): Explanation for the mapping.
    """
    module_id: UUID = Field(..., description="Unique identifier for the document")
    document_id: UUID = Field(..., description="Unique identifier for the module")
    section_id: UUID = Field(..., description="Unique identifier for the section")
    topic_id: UUID = Field(..., description="Unique identifier for the topic module")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")
    reasoning: str = Field(..., description="Explanation for the mapping")


class DocumentMappings(BaseModel):
    """
    Collection of relevance mappings for a document.

    Attributes:
        mappings (List[RelevanceMapping]): List of relevance mappings.
    """
    mappings: List[RelevanceMapping] = Field(..., description="List of relevance mappings")

    @classmethod
    def create_mappings_from_json(cls, file_path: str):
        """Deserializes a RelevanceStore from a JSON file using DocumentMappings."""
        if not Path(file_path).exists():
            return DocumentMappings(mappings=list())
        with open(file_path, "r") as f:
            data = json.load(f)
            mappings = DocumentMappings(**data)
        return mappings


class RelevanceStore:
    """
    Stores and manages relevance mappings between document sections and curriculum topics.
    """

    def __init__(self):
        # Key: (section_id, topic_id), Value: RelevanceMapping
        self.mappings: Dict[Tuple[UUID, UUID], RelevanceMapping] = {}
        self.topic_to_sections: Dict[UUID, Set[UUID]] = defaultdict(set)
        self.section_to_topics: Dict[UUID, Set[UUID]] = defaultdict(set)
        self.section_to_modules: Dict[UUID, Set[UUID]] = defaultdict(set)
        self.topic_to_documents: Dict[UUID, Set[UUID]] = defaultdict(set)

    def add_mappings(self, mappings: DocumentMappings, file_path: str = None):
        """
        Adds a list of relevance mappings to the store.

        Args:
            mappings (DocumentMappings): List of mappings to add.
            file_path (str, optional): Path to save the updated mappings. Defaults to None.
        """
        for mapping in mappings.mappings:
            self.add_mapping(mapping)
        if file_path:
            self.save_or_update(file_path)

    def add_mapping(self, mapping: RelevanceMapping):
        """
        Adds a single relevance mapping to the store.

        Args:
            mapping (RelevanceMapping): Mapping to add.
        """
        key = (mapping.section_id, mapping.topic_id)
        self.mappings[key] = mapping
        self.topic_to_sections[mapping.topic_id].add(mapping.section_id)
        self.section_to_topics[mapping.section_id].add(mapping.topic_id)
        self.section_to_modules[mapping.section_id].add(mapping.module_id)
        self.topic_to_documents[mapping.topic_id].add(mapping.document_id)

    def section_already_mapped_to_module(self, section_id: UUID, module_id: UUID):
        """
        Checks if a section is already mapped to a module.

        Args:
            section_id (UUID): Unique identifier for the section.
            module_id (UUID): Unique identifier for the module.

        Returns:
            bool: True if the section is already mapped to the module, False otherwise.
        """
        return module_id in self.section_to_modules[section_id]

    def get_relevant_mappings(self, min_score: float = 0.0, sort_by_relevance: bool = False) -> List[RelevanceMapping]:
        """
        Retrieves relevant mappings based on a minimum score.

        Args:
            min_score (float, optional): Minimum relevance score. Defaults to 0.0.
            sort_by_relevance (bool, optional): Sort results by relevance score. Defaults to False.

        Returns:
            List[RelevanceMapping]: List of relevant mappings.
        """
        mappings = [m for m in self.mappings.values() if m.relevance_score >= min_score]
        if sort_by_relevance:
            mappings = sorted(mappings, key=lambda obj: obj.relevance_score, reverse=True)
        return mappings

    def get_relevant_section_mappings(self, topic_id: UUID, min_score: float = 0.0) -> List[RelevanceMapping]:
        """
        Retrieves relevant mappings for a specific topic.

        Args:
            topic_id (UUID): Unique identifier for the topic.
            min_score (float, optional): Minimum relevance score. Defaults to 0.0.

        Returns:
            List[RelevanceMapping]: List of relevant mappings for the topic.
        """
        return [
            self.mappings[(section_id, topic_id)]
            for section_id in self.topic_to_sections.get(topic_id, set())
            if (section_id, topic_id) in self.mappings and self.mappings[(section_id, topic_id)].relevance_score >= min_score
        ]

    def get_relevant_topic_mappings(self, section_id: UUID, min_score: float = 0.0) -> List[RelevanceMapping]:
        """
        Retrieves relevant mappings for a specific section.

        Args:
            section_id (UUID): Unique identifier for the section.
            min_score (float, optional): Minimum relevance score. Defaults to 0.0.

        Returns:
            List[RelevanceMapping]: List of relevant mappings for the section.
        """
        return [
            self.mappings[(section_id, topic_id)]
            for topic_id in self.section_to_topics.get(section_id, set())
            if (section_id, topic_id) in self.mappings and self.mappings[(section_id, topic_id)].relevance_score >= min_score
        ]

    def get_relevant_section_ids(self, topic_id: UUID, min_score: float = 0.0) -> List[UUID]:
        """
        Retrieves section IDs relevant to a specific topic.

        Args:
            topic_id (UUID): Unique identifier for the topic.
            min_score (float, optional): Minimum relevance score. Defaults to 0.0.

        Returns:
            List[UUID]: List of section IDs relevant to the topic.
        """
        return [
            mapping.section_id
            for mapping in self.get_relevant_section_mappings(topic_id, min_score=min_score)
        ]

    def get_relevant_section_ids_per_doc(self, topic_id: UUID, min_score: float = 0.0, docs : List[Document] = None) -> List[Document]:
        """
        Retrieves sections relevant to a specific topic per document.

        Args:
            topic_id (UUID): Unique identifier for the topic.
            min_score (float, optional): Minimum relevance score. Defaults to 0.0.

        Returns:
            List[UUID]: List of section IDs relevant to the topic.
        """
        section_ids = self.get_relevant_section_ids(topic_id, min_score)
        return [doc.filter_by_section_ids(section_ids) for doc in docs]


    def get_relevant_topic_ids(self, section_id: UUID, min_score: float = 0.0) -> List[UUID]:
        """
        Retrieves topic IDs relevant to a specific section.

        Args:
            section_id (UUID): Unique identifier for the section.
            min_score (float, optional): Minimum relevance score. Defaults to 0.0.

        Returns:
            List[UUID]: List of topic IDs relevant to the section.
        """
        return [
            mapping.topic_id
            for mapping in self.get_relevant_topic_mappings(section_id, min_score=min_score)
        ]

    def remove_mapping(self, section_id: UUID, topic_id: UUID):
        """
        Removes a mapping between a section and a topic.

        Args:
            section_id (UUID): Unique identifier for the section.
            topic_id (UUID): Unique identifier for the topic.
        """
        key = (section_id, topic_id)
        if key in self.mappings:
            mapping = self.mappings[key]
            del self.mappings[key]
            self.topic_to_sections[topic_id].discard(section_id)
            self.section_to_topics[section_id].discard(topic_id)
            self.section_to_modules[section_id].discard(mapping.module_id)
            self.topic_to_documents[topic_id].discard(mapping.document_id)

    def save_or_update(self, file_path: str):
        """
        Saves or updates the RelevanceStore to a JSON file.

        Args:
            file_path (str): Path to the output JSON file.
        """
        unique_mappings = []
        document_mappings = DocumentMappings.create_mappings_from_json(file_path)
        all_mappings = document_mappings.mappings + list(self.mappings.values())
        seen = set()
        for mapping in all_mappings:
            key = (mapping.section_id, mapping.topic_id)
            if key not in seen:
                unique_mappings.append(mapping)
                seen.add(key)
        updated_document_mappings = DocumentMappings(mappings=unique_mappings)
        with open(file_path, "w") as f:
            f.write(updated_document_mappings.model_dump_json(indent=2))

    def __str__(self) -> str:
        return f"Relevance store contains {len(self.mappings)} mappings"

    def create_aggregated_document(self, module_name, topic_name, curriculum, docs, min_score,
                                   client,
                                   llm_model,
                                   prompt_generation_function : callable = generate_aggregation_prompts):
        topic = curriculum[module_name][topic_name]
        rel_sections_per_doc : List[Document] = self.get_relevant_section_ids_per_doc(topic.id, min_score=min_score, docs=docs)
        rel_docs = [doc for doc in rel_sections_per_doc if len(doc.sections) > 0]
        SYSTEM_PROMPT, USER_PROMPT = prompt_generation_function(module_name, topic_name, topic.description, rel_docs)
        try:
            response_section = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT}],
                # max_tokens=SUMMARIZE_SECTION_MAX_TOKENS,
            )
            text = response_section.choices[0].message.content.strip()
            return text
        except Exception as e:
            print(f"Error summarizing section: {e}")

        # print(f"## Module Name: {module_name}\n")
        # print(f"## Topic Name: {topic_name}\n")
        # print(f"## Topic Description:\n {topic.description}")
        # print("## Documents:\n")
        # print(str(rel_docs[0]))
        # text = self._aggregate_sections(rel_docs, module_name, topic_name, currriculum)


    # def _aggregate_document(self, rel_docs, module_name, topic_name, currriculum, TOKEN_LIMIT : int):
    #     total_tokens = sum([doc.tokens for doc in rel_docs])
    #     if total_tokens > TOKEN_LIMIT:
    #       raise ...
    #     [for sec in rel_docs]

    @classmethod
    def create_store_from_json(cls, file_path: str):
        """
        Creates a RelevanceStore from a JSON file.

        Args:
            file_path (str): Path to the input JSON file.

        Returns:
            RelevanceStore: Instantiated RelevanceStore.
        """
        store = cls()
        document_mappings = DocumentMappings.create_mappings_from_json(file_path)
        store.add_mappings(document_mappings)
        return store
