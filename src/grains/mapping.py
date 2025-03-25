import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from grains.data_structures import Curriculum, Document, Module, Section
from grains.utils import load_curriculum, try_loading_document_object

MODEL: str = "anthropic/claude-3.5-sonnet"

class TopicMapping(BaseModel):
    """Represents the mapping of a document section to a specific topic."""
    topic_name: str = Field(..., description="Name of the topic")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")
    reasoning: str = Field(..., description="Explanation (Chain-of-Thought) for the mapping")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the mapping")
    
    # Custom JSON serialization for datetime
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SectionMapping(BaseModel):
    """Represents all topic mappings for a specific document section."""
    section_title: str = Field(..., description="Title of the document section")
    section_id: Optional[str] = Field(None, description="Unique identifier for the section")
    topics: List[TopicMapping] = Field(default_factory=list, description="List of topic mappings for this section")


class ModuleMapping(BaseModel):
    """Represents all section mappings for a specific module."""
    module_name: str = Field(..., description="Name of the module")
    module_id: Optional[str] = Field(None, description="Unique identifier for the module")
    sections: List[SectionMapping] = Field(default_factory=list, description="List of section mappings for this module")


class DocumentMappings(BaseModel):
    """Represents all module mappings for a document."""
    document_title: str = Field(..., description="Title of the document")
    document_id: Optional[str] = Field(None, description="Unique identifier for the document")
    modules: List[ModuleMapping] = Field(default_factory=list, description="List of module mappings for this document")


class AITopicMappings(BaseModel):
    """Model for the AI's response when mapping a section to topics."""
    mappings: List[dict] = Field(..., description="List of topic mappings with relevance scores and reasoning")


def create_mapping_prompt(section: Section, module: Module) -> str:
    """Creates a prompt for the AI to map a section to module topics."""
    section_info = f"Section Title: {section.title}\nSection Summary: {section.summary}"
    
    # Create a list of topics with their descriptions
    topics_list = ""
    for topic in module.topics:
        topics_list += f"- {topic.name}: {topic.description}\n"
    
    module_info = f"Module: {module.name}\nTopics:\n{topics_list}"
    
    return f"""
Module:
{module_info}

Section:
{section_info}

For the section above, evaluate its relevance to each topic in the module.
For EACH topic, provide:
1. The exact topic name as listed above
2. A relevance score between 0 and 1 (where 1 is highly relevant, 0 is not relevant)
3. A brief explanation of why this section is relevant or not to the topic

Return your response in the following JSON format:

```json
{{
    "mappings": [
        {{
            "topic_name": "EXACT TOPIC NAME 1",
            "relevance_score": 0.9,
            "reasoning": "Brief explanation of relevance..."
        }},
        {{
            "topic_name": "EXACT TOPIC NAME 2",
            "relevance_score": 0.7,
            "reasoning": "Brief explanation of relevance..."
        }},
        ...one entry for EACH topic in the module...
    ]
}}
```

IMPORTANT: Make sure to include ALL topics from the module in your response, even if they have low relevance scores.
"""


async def map_section_to_module_topics(section: Section, module: Module, agent: Agent) -> List[TopicMapping]:
    """Maps a section to all topics in a module."""
    prompt = create_mapping_prompt(section, module)
    result = await agent.run(prompt)
    
    topic_mappings = []
    
    mappings_data = None
    
    if hasattr(result, 'data') and hasattr(result.data, 'mappings'):
        mappings_data = result.data.mappings
    elif hasattr(result, 'mappings'):
        mappings_data = result.mappings
    elif hasattr(result, 'data') and isinstance(result.data, list):
        mappings_data = result.data
    elif isinstance(result, list):
        mappings_data = result
    
    if mappings_data:
        for mapping in mappings_data:
            if isinstance(mapping, dict):
                if 'topic_name' in mapping and 'relevance_score' in mapping and 'reasoning' in mapping:
                    topic_mappings.append(
                        TopicMapping(
                            topic_name=mapping['topic_name'],
                            relevance_score=mapping['relevance_score'],
                            reasoning=mapping['reasoning']
                        )
                    )

                elif 'topics' in mapping and isinstance(mapping['topics'], dict):
                    for topic_name, topic_data in mapping['topics'].items():
                        if isinstance(topic_data, dict) and 'relevance_score' in topic_data and 'reasoning' in topic_data:
                            topic_mappings.append(
                                TopicMapping(
                                    topic_name=topic_name,
                                    relevance_score=topic_data['relevance_score'],
                                    reasoning=topic_data['reasoning']
                                )
                            )
    
    if not topic_mappings:
        print("WARNING: Could not extract topic mappings from AI response. Creating default mappings.")
        for topic in module.topics:
            topic_mappings.append(
                TopicMapping(
                    topic_name=topic.name,
                    relevance_score=0.0,
                    reasoning="Failed to extract mapping from AI response."
                )
            )
    
    return topic_mappings


async def document_to_module_mapping(document: Document, curriculum: Curriculum, agent: Agent) -> DocumentMappings:
    """
    Maps all sections of a document to all topics in all modules.
    Returns a comprehensive DocumentMappings object.
    """

    doc_id = getattr(document, 'id', None)
    doc_title = None
    for attr in ['title', 'name', 'document_name', 'filename']:
        if hasattr(document, attr):
            doc_title = getattr(document, attr)
            break
    
    if not doc_title:
        doc_title = f"Document {doc_id}" if doc_id else "Unnamed Document"
    
    doc_mappings = DocumentMappings(
        document_title=doc_title,
        document_id=doc_id
    )
    
    for module in curriculum.modules:
        module_mapping = ModuleMapping(
            module_name=module.name,
            module_id=getattr(module, 'id', None)
        )
        
        for section in document.sections:
            topic_mappings = await map_section_to_module_topics(section, module, agent)
            
            section_mapping = SectionMapping(
                section_title=section.title,
                section_id=getattr(section, 'id', None),
                topics=topic_mappings
            )
            
            module_mapping.sections.append(section_mapping)
        
        doc_mappings.modules.append(module_mapping)
    
    return doc_mappings


def get_most_relevant_sections(
    doc_mappings: DocumentMappings, module_name: str, topic_name: str, k: int = 3
) -> List[Tuple[str, float, str]]:
    """
    Retrieves the top k most relevant sections for a specific topic in a module.
    Returns a list of tuples: (section_title, relevance_score, reasoning)
    """
    relevant_sections = []
    
    for module in doc_mappings.modules:
        if module.module_name == module_name:
            for section in module.sections:
                for topic in section.topics:
                    if topic.topic_name == topic_name:
                        relevant_sections.append(
                            (section.section_title, topic.relevance_score, topic.reasoning)
                        )

    sorted_sections = sorted(relevant_sections, key=lambda x: x[1], reverse=True)
    return sorted_sections[:k]


def save_mappings_to_file(doc_mappings: DocumentMappings, output_dir: str = "data/mappings") -> str:
    """
    Save the document mappings to a JSON file.
    
    Args:
        doc_mappings: The DocumentMappings object to save
        output_dir: Directory where the file should be saved
        
    Returns:
        The path to the saved file
    """

    os.makedirs(output_dir, exist_ok=True)
    
    safe_title = "".join(c if c.isalnum() else "_" for c in doc_mappings.document_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_mappings_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    mappings_dict = doc_mappings.dict()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(mappings_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Mappings saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    # Create the agent
    agent = Agent(
        OpenAIModel(
            MODEL,
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENAI_API_KEY")
            ),
        ),
        result_type=AITopicMappings,
        system_prompt=(
            "You are an expert in mapping document sections to module topics for the hotel and service industry. "
            "Your task is to analyze the section summary below and calculate its relevance for each topic of the module. "
            "For each topic, provide:\n"
            "1. A relevance score [0-1], where 1 means highly relevant and 0 means no relevance.\n"
            "2. A concise and short reasoning (Chain-of-Thought) explaining why the section is relevant to the topic."
        ),
    )
    
    document = try_loading_document_object("data/objects/w26579.json")
    curriculum = load_curriculum()
    
    if document:
        print(f"Processing document with {len(document.sections)} sections")
        print(f"Curriculum has {len(curriculum.modules)} modules")
        
        doc_mappings = asyncio.run(document_to_module_mapping(document, curriculum, agent))
        
        output_file = save_mappings_to_file(doc_mappings)
        
        print(f"\n\nMappings for document: {doc_mappings.document_title}")
        for module in doc_mappings.modules:
            print(f"\nModule: {module.module_name}")
            for section in module.sections:
                print(f"  Section: {section.section_title}")
                for topic in section.topics:
                    print(f"    Topic: {topic.topic_name}")
                    print(f"      Relevance: {topic.relevance_score:.2f}")
                    print(f"      Reasoning: {topic.reasoning[:100]}...")
        
        if curriculum.modules and curriculum.modules[0].topics:
            module_name = curriculum.modules[0].name
            topic_name = curriculum.modules[0].topics[0].name
            
            print(f"\nMost relevant sections for {topic_name} in {module_name}:")
            relevant_sections = get_most_relevant_sections(doc_mappings, module_name, topic_name)
            
            for section_title, score, reasoning in relevant_sections:
                print(f"  {section_title} (Score: {score:.2f})")
                print(f"  Reasoning: {reasoning[:100]}...")
                
        print(f"\nMappings saved to: {output_file}")
    else:
        print("Failed to load document.")