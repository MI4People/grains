import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import UUID

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from grains.data_structures import Curriculum, Document, DocumentMappings
from grains.data_structures import Module
from grains.data_structures import Module as CurriculumModule
from grains.data_structures import RelevanceMapping, RelevanceStore, Section
from grains.prompts import MAPPING_PROMPT, MAPPING_SYSTM_PROMPT
from grains.utils import load_curriculum, try_loading_document_object

#MODEL: str = "anthropic/claude-3.5-sonnet"
#MODEL = "google/gemini-2.0-flash-001"
# Works
#MODEL = "openai/gpt-4o-mini"
MODEL = "meta-llama/llama-3.3-70b-instruct"  # Use a single model


def create_mapping_prompt(
    document_id: UUID, section: Section, module: CurriculumModule, prompt: str = MAPPING_PROMPT
) -> str:
    """
    Generates a formatted prompt for mapping a document section to a curriculum module.

    Args:
        document_id (UUID): Unique identifier of the document.
        section (Section): Document section to be mapped.
        module (CurriculumModule): Curriculum module to map to.
        prompt (str, optional): Base prompt template. Defaults to MAPPING_PROMPT.

    Returns:
        str: Formatted prompt string.
    """
    section_info = f"\tSection id: {section.id}\n\tDescription: {section.title}\n\tSection Summary: {section.summary}"
    module_info = f"\tModule id: {module.id}\n\tDescription: {module.name}\n\tTopics:\n"
    for topic in module.topics:
        module_info += f"\t- {topic.name}\n\t  topic id: {topic.id}\n\t  {topic.description}\n"
    formatted_prompt = prompt.format(document_id=document_id, module_info=module_info, section_info=section_info)
    return formatted_prompt


async def _run_async_agent_with_semaphore(prompt: str, async_agent : Agent, store : RelevanceStore, sema : asyncio.Semaphore, lock : asyncio.Lock) -> str:
    """
    Wrapper function to run the async_agent while respecting the semaphore.
    """
    mapping = f"sec {prompt}"
    async with sema:
        try:
            print(f"Acquired semaphore ... {sema._value} available.")
            result = await async_agent.run(prompt)
            print(f"Released semaphore")
            if not result or not result.data:
                raise ValueError(f"Received empty model response {result}")
            async with lock:
                print("Updated data/mappings.json")
                store.add_mappings(result.data, "data/mappings.json")
        except asyncio.TimeoutError:
            print(f"Timeout: Section exceeded 60s")
            return
        except Exception as e:
            print(f"[ERROR] Agent failed for prompt {e}")
            return
        return result.data


async def create_mappings_store(
        document: Document, curriculum: Curriculum, agent: Agent, store: RelevanceStore, sema : asyncio.Semaphore, lock : asyncio.Lock
) -> RelevanceStore:
    """
    Gathers all concurrent tasks and then executes them concurrently 
    Args:
        document (Document): Document to process.
        curriculum (Curriculum): Curriculum to map to.
        agent (Agent): LLM agent for generating mappings.
        store (RelevanceStore): Existing RelevanceStore to update.

    Returns:
        RelevanceStore: Updated RelevanceStore with new mappings.
    """
    tasks = []
    for section in document.sections:
        for module in curriculum.modules:
            if not store.section_already_mapped_to_module(section.id, module.id):
                prompt = create_mapping_prompt(document.id, section, module)
                tasks.append(asyncio.create_task(_run_async_agent_with_semaphore(prompt, agent, store, sema, lock)))
            else:
                print(f"Section {section.id} already mapped to module {module.id}")

    print(f"Running {len(tasks)} task for document {document.filename}")
    await asyncio.gather(*tasks, return_exceptions=True)

    return store



async def load_or_create_mappings_for_docs(
        documents: Iterable[Document], curriculum: Curriculum, llm_model: str, mappings_store_file_path: str = "data/mappings.json",
        MAX_CONCURRENT_LLM_CALLS : int = 20, MAX_RETRIES : int =3) -> RelevanceStore:
    """
    Loads existing mappings or creates new ones for a set of documents against a curriculum.

    Args:
        documents (Iterable[Document]): Documents to process.
        curriculum (Curriculum): Curriculum to map to.
        model (str): LLM model to use for mapping generation.
        mappings_store_file_path (str, optional): Path to existing mappings file. Defaults to "data/mappings.json".

    Returns:
        RelevanceStore: RelevanceStore containing all mappings.
    """
    mappings_store = RelevanceStore.create_store_from_json(mappings_store_file_path)
    client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY")
    )
    async_llm_model = OpenAIModel(
        llm_model,
        openai_client=client
    )
    agent = Agent(
        model=async_llm_model,
        result_type=DocumentMappings,
        system_prompt=MAPPING_SYSTM_PROMPT,
        retries=MAX_RETRIES
    )
    sema = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    lock = asyncio.Lock()
    for doc in documents:
        print(f"Create mappings for {doc.filename}")
        store = await create_mappings_store(doc, curriculum, agent, mappings_store, sema, lock)
    return store

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process document mappings.")
    parser.add_argument(
        "-l",
        "--test_with_llm",
        action="store_true",
        help="Enable testing with LLM (requires OPENAI_API_KEY environment variable).",
    )
    args = parser.parse_args()
    document = try_loading_document_object("data/objects/w26579.json")
    curriculum = load_curriculum()

    if not document:
        raise RuntimeError("Failed to load document")

    mappings_store = RelevanceStore.create_store_from_json("data/mappings_test.json")
    if args.test_with_llm:
        async def main_async():
            agent = Agent(
                OpenAIModel(
                    MODEL,
                    provider=OpenAIProvider(
                        base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY")
                    ),
                ),
                result_type=DocumentMappings,
                system_prompt=MAPPING_SYSTM_PROMPT,
            )
            sema = asyncio.Semaphore(2)
            lock = asyncio.Lock()
            await create_mappings_store(document, curriculum, agent, mappings_store, sema, lock)
            print(mappings_store.get_relevant_topic_ids(UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), min_score=0.1))
            print(mappings_store.get_relevant_mappings(min_score = 0.0, sort_by_relevance=True))
        asyncio.run(main_async())
    else:
        mappings_store = RelevanceStore()
        dummy_llm_mappings_output=DocumentMappings(mappings=[RelevanceMapping(module_id=UUID('b06e46cc-13ee-431c-8295-a116edd60650'), document_id=UUID('7bf4e67f-c426-4813-a0f0-c63768e33fc3'), section_id=UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), topic_id=UUID('8c50023c-6b17-4e90-b4a7-923d566e15e7'), relevance_score=0.0, reasoning="The section only contains 'NBER WORKING PAPER SERIES' which has no direct relation to hospitality philosophy or psychology."), RelevanceMapping(module_id=UUID('b06e46cc-13ee-431c-8295-a116edd60650'), document_id=UUID('7bf4e67f-c426-4813-a0f0-c63768e33fc3'), section_id=UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), topic_id=UUID('61be82d6-2b25-4747-be13-e078530616f6'), relevance_score=0.0, reasoning='The section content does not address service or guest needs fulfillment in any way.'), RelevanceMapping(module_id=UUID('b06e46cc-13ee-431c-8295-a116edd60650'), document_id=UUID('7bf4e67f-c426-4813-a0f0-c63768e33fc3'), section_id=UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), topic_id=UUID('12432a09-d7b6-4bdf-82e2-b91c3d6968b1'), relevance_score=0.0, reasoning='The section has no content related to service, hospitality or housekeeping videos.'), RelevanceMapping(module_id=UUID('b06e46cc-13ee-431c-8295-a116edd60650'), document_id=UUID('7bf4e67f-c426-4813-a0f0-c63768e33fc3'), section_id=UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), topic_id=UUID('49246b6b-f6b8-41d6-8783-43dd399f31a9'), relevance_score=0.0, reasoning='The section contains no information about hotel classification, structure, or values.'), RelevanceMapping(module_id=UUID('b06e46cc-13ee-431c-8295-a116edd60650'), document_id=UUID('7bf4e67f-c426-4813-a0f0-c63768e33fc3'), section_id=UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), topic_id=UUID('db083dcc-bc8c-4fb0-9e6c-7908ebd0ce89'), relevance_score=0.0, reasoning='The section does not discuss any guest archetypes or examples.'), RelevanceMapping(module_id=UUID('b06e46cc-13ee-431c-8295-a116edd60650'), document_id=UUID('7bf4e67f-c426-4813-a0f0-c63768e33fc3'), section_id=UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), topic_id=UUID('fd9c49bd-f731-4886-aeed-ad787210ae73'), relevance_score=0.0, reasoning='The section does not address career prospects or personal development in the hospitality industry.'), RelevanceMapping(module_id=UUID('b06e46cc-13ee-431c-8295-a116edd60650'), document_id=UUID('7bf4e67f-c426-4813-a0f0-c63768e33fc3'), section_id=UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'), topic_id=UUID('93bd02c8-7480-4058-ba1a-a6ad4e7360fb'), relevance_score=0.0, reasoning='The section contains no discussion of personal goals or student objectives.')])
        mappings_store.add_mappings(dummy_llm_mappings_output, "data/mappings_test.json")
        print(mappings_store.get_relevant_topic_ids(UUID('a0ba7c9c-b40c-440e-adbc-323a89b63214'),min_score=0.1))
        print(mappings_store.get_relevant_mappings(min_score = 0.0, sort_by_relevance=True))
