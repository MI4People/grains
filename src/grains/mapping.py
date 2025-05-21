import argparse
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import UUID
import time 
import asyncio
import random 

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from grains.data_structures import Curriculum, Document, DocumentMappings, Section, Module
from grains.data_structures import Module as CurriculumModule
from grains.data_structures import RelevanceMapping, RelevanceStore, Section
from grains.prompts import MAPPING_PROMPT, MAPPING_SYSTM_PROMPT
from grains.utils import load_curriculum, try_loading_document_object

#MODEL: str = "anthropic/claude-3.5-sonnet"
#MODEL = "google/gemini-2.0-flash-001"
# Works
#MODEL = "openai/gpt-4o-mini"
MODEL = "meta-llama/llama-3.3-70b-instruct"  # Use a single model

# Async wrapper - restricts the concurrenct running actions
sem = asyncio.Semaphore(5)

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


async def map_section_to_module(
        document: Document, section: Section, module: Module, agent: Agent, store: RelevanceStore
        ):
    """
    Calculates the mappings between the given section and module

    Args and output 
    """
    # Start async with above defined setup restrictor
    async with sem:
        if not store.section_already_mapped_to_module(section.id,module.id):
            
            # Create prompt if not mapped 
            prompt = create_mapping_prompt(document.id, section, module)

            # Set retries to loop over for somehow failed attempts
            retries = 3
            for attempt in range(retries):

                try:
                    map_start = time.time()             # Tracking timer

                    # Quit after one minute
                    result = await asyncio.wait_for(
                        await asyncio.to_thread(agent.run_sync, prompt),
                        timeout=60)
                    
                    map_end = time.time()
                    print(f"[{section.id} <-> {module.id}] took {map_end - map_start:.2f}s")
                    
                    if not result or not result.data:
                        raise ValueError("Received empty model response")
                    
                    store.add_mappings(result.data, "data/mappings.json")
                    break  # success
                
                # Timeout error
                except asyncio.TimeoutError:
                    print(f"Timeout: Section {section.id} <-> Module {module.id} exceeded 60s")

                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt + random.random())
                    else:
                        print(f"Failed to map section {section.id} to module {module.id} after {retries} attempts due to: {e}")
        else:
            print("already mapped")


async def create_mappings_store(
    document: Document, curriculum: Curriculum, agent: Agent, store: RelevanceStore
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
            tasks.append(map_section_to_module(document,section,module,agent,store))
    
    # Trigger the async run
    await asyncio.gather(*tasks)

    return store 

async def load_or_create_mappings_for_docs(
    documents: Iterable[Document], curriculum: Curriculum, model: str, mappings_store_file_path: str = "data/mappings.json"
) -> RelevanceStore:
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
    agent = Agent(
        OpenAIModel(
            model,
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY")
            ),
        ),
        result_type=DocumentMappings,
        system_prompt=MAPPING_SYSTM_PROMPT,
    )
    for doc in documents:
        print(f"Create mappings for {doc.filename}")
        store = await create_mappings_store(doc, curriculum, agent, mappings_store)
    return store