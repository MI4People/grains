import asyncio
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import (OpenAIModel, OpenAIProvider, from,
                                       import, pydantic_ai.providers.openai)

from grains.data_structures import (Curriculum, Document, Module, Section,
                                    from, grains.utils, import,
                                    load_curriculum,
                                    try_loading_document_object)

# Initialize OpenAI client
open_router_key = os.getenv("OPENAI_API_KEY")
MODEL: str = "anthropic/claude-3.5-sonnet"


class TopicMapping(BaseModel):
     relevance_score: float = Field(..., description="Relevance score between 0 and 1")
     reasoning: str = Field(..., description="Explanation (Chain-of-Thought) for the mapping")
     timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the mapping")


class ModuleMapping(BaseModel):
     section_title: str = Field(..., description="Title of the document section")
     topics: Dict[str, TopicMapping] = Field(..., description="Mappings from topic names to TopicMapping objects")


class ModuleMappings(BaseModel):
     module_name: str = Field(..., description="Module Name")
     mappings: Dict[str, ModuleMapping] = Field(..., description="Dict from document tilte to ModuelMapping")


model = OpenAIModel(
     MODEL,
     provider=OpenAIProvider(
         base_url="https://openrouter.ai/api/v1",
         api_key=open_router_key,
     ),
)


def create_mapping_prompt(section: Section, module: Module) -> str:
     section_info = f"Section Title: {section.title}\nSection Summary: {section.summary}"
     module_info = f"Module: {module.name}\nTopics:\n" + "\n".join(
         f"- {topic.name}: {topic.summary}" for topic in module.topics
     )
     return f"""
         Module:
         {module_info}

         Section:
         {section_info}

         Provide the mapping for this section as a JSON object with the
                        following structure:
                                ```json
                                {{
                                    "mappings": [
                                        {{
                                            "section_title": "Section Title",
                                            "topics": {{
                                                "Topic Name 1": {{"relevance_score": 0.9, "reasoning": "Reasoning..."}},
                                                "Topic Name 2": {{"relevance_score": 0.7, "reasoning": "Reasoning..."}},
                                                    ...
                        }}
                 }}
             ]
         }}
     """


async def map_section_to_module_topics(section: Section, module: Module, agent: Agent) -> ModuleMappings:
     """
     TODO
     """
     prompt = create_mapping_prompt(section, module)
     result = agent.run_sync(prompt)  # Call the LLM and parse the response into Pydantic objects
     return result.data  # type: ignore


async def document_to_module_mapping(document: Document, curriculum:
Curriculum, agent: Agent) -> List[ModuleMapping]:
     """
     Map all sections of a document to all topics in a module, returning
    a list of ModuleMapping objects.
     """
     all_mappings = []
     for module in curriculum.modules:
         for section in document.sections:
             mappings = await map_section_to_module_topics(section, module, agent)
             all_mappings.extend(mappings.mappings)
     return all_mappings


def get_most_relevant_sections(
     mappings: List[ModuleMapping], module_name: str, topic_name: str, k: int = 3
) -> List[Tuple[str, TopicMapping]]:
     """
     Retrieve the top k most relevant sections for a specific topic in a module, based on relevance score.
     """
     relevant_mappings = [(mapping.section_title, mapping.topics[topic_name]) for mapping in mappings if topic_name in mapping.topics]
     sorted_mappings = sorted(relevant_mappings, key=lambda x: x[1].relevance_score, reverse=True)
     return sorted_mappings[:k]


if __name__ == "__main__":
     agent = Agent(
         model,
         result_type=ModuleMappings,
         system_prompt=(
             "You are an expert in mapping document sections to module topics for the hotel and service industry. "
             "Your task is to analyze the following section summary below and calculate its relevance for each topic of the following module. "
             "For each topic, provide:\n"
             "1. A relevance score [0-1], where 1 means highly relevant and 0 means no relevance.\n"
             "2. A concise and short reasoning (Chain-of-Thought) explaining why the section is relevant to the topic."
         ),
     )
     # Configuration with type-hinted Path objects
     document = try_loading_document_object("data/objects/w26579.json")
     curriculum = load_curriculum()
     if document:
         mapping = map_section_to_module_topics(document.sections[4], curriculum.modules[0], agent)
         print(mapping)
