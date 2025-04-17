import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from grains.data_structures import Curriculum, Document, Module as CurriculumModule, Section

MODEL: str = "anthropic/claude-3.5-sonnet"

class TopicRelevance(BaseModel):
    topic_name: str = Field(..., description="Name of the topic")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")
    reasoning: str = Field(..., description="Explanation for the mapping")

class ModuleMapping(BaseModel):
    module_name: str = Field(..., description="Name of the module")
    topics: List[TopicRelevance] = Field(default_factory=list)

class SectionMapping(BaseModel):
    section_title: str = Field(..., description="Title of the document section")
    modules: List[ModuleMapping] = Field(default_factory=list)

class DocumentMappings(BaseModel):
    document_title: str = Field(..., description="Title of the document")
    sections: List[SectionMapping] = Field(default_factory=list)

class AITopicMappings(BaseModel):
    mappings: List[dict] = Field(..., description="List of topic mappings")

def create_mapping_prompt(section: Section, module: CurriculumModule) -> str:
    section_info = f"Section Title: {section.title}\nSection Summary: {section.summary}"
    
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
Consider BOTH the topic name AND its description when determining relevance.

For EACH topic, provide:
1. The EXACT topic name as listed above - DO NOT CHANGE OR MODIFY THE TOPIC NAMES IN ANY WAY
2. A relevance score between 0 and 1 (where 1 is highly relevant, 0 is not relevant)
3. A brief explanation of why this section is relevant or not to the topic, specifically referencing how the section content relates to the topic description

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
        }}
    ]
}}
```

CRITICAL: You must use the EXACT topic names from the list above without any modifications, additions or omissions.
Make sure to include ALL topics from the module in your response, even if they have low relevance scores.
When determining relevance, pay careful attention to both the topic name and its description.
"""

async def map_section_to_module_topics(section: Section, module: CurriculumModule, agent: Agent) -> ModuleMapping:
    prompt = create_mapping_prompt(section, module)
    try:
        result = await agent.run(prompt)
        
        mappings_data = None
        if hasattr(result, 'data') and hasattr(result.data, 'mappings'):
            mappings_data = result.data.mappings

        valid_topic_names = {topic.name for topic in module.topics}
        
        topic_relevances = []
        if mappings_data:
            for mapping in mappings_data:
                if 'topic_name' in mapping and 'relevance_score' in mapping and 'reasoning' in mapping:
                    topic_name = mapping['topic_name']
                    
                    if topic_name not in valid_topic_names:
                        closest_match = None
                        for valid_name in valid_topic_names:
                            if valid_name in topic_name or topic_name in valid_name:
                                closest_match = valid_name
                                break
                        
                        if closest_match:
                            topic_name = closest_match
                        else:
                            topic_name = next(iter(valid_topic_names))
                    
                    topic_relevances.append(
                        TopicRelevance(
                            topic_name=topic_name,
                            relevance_score=mapping['relevance_score'],
                            reasoning=mapping['reasoning']
                        )
                    )
        
        # Make sure all topics are included
        covered_topics = {topic.topic_name for topic in topic_relevances}
        for topic_name in valid_topic_names:
            if topic_name not in covered_topics:
                topic_relevances.append(
                    TopicRelevance(
                        topic_name=topic_name,
                        relevance_score=0.0,
                        reasoning="No mapping available"
                    )
                )
        
        return ModuleMapping(module_name=module.name, topics=topic_relevances)
    
    except Exception as e:
        print(f"Error: {e}")
        default_topics = [
            TopicRelevance(
                topic_name=topic.name,
                relevance_score=0.0,
                reasoning="Error in mapping process"
            ) for topic in module.topics
        ]
        return ModuleMapping(module_name=module.name, topics=default_topics)

async def document_to_mappings(document: Document, curriculum: Curriculum, agent: Agent) -> DocumentMappings:
    doc_title = getattr(document, 'title', None) or document.filename or "Unnamed Document"
    
    doc_mappings = DocumentMappings(document_title=doc_title)
    
    for section in document.sections:
        section_mapping = SectionMapping(section_title=section.title)
        
        for module in curriculum.modules:
            module_mapping = await map_section_to_module_topics(section, module, agent)
            section_mapping.modules.append(module_mapping)
        
        doc_mappings.sections.append(section_mapping)
    
    return doc_mappings

def get_most_relevant_sections_for_topic(
    doc_mappings: DocumentMappings, module_name: str, topic_name: str, k: int = 3
) -> List[Tuple[str, float, str]]:
    relevant_sections = []
    
    for section in doc_mappings.sections:
        for module in section.modules:
            if module.module_name == module_name:
                for topic in module.topics:
                    if topic.topic_name == topic_name:
                        relevant_sections.append(
                            (section.section_title, topic.relevance_score, topic.reasoning)
                        )
    
    sorted_sections = sorted(relevant_sections, key=lambda x: x[1], reverse=True)
    return sorted_sections[:k]

def get_modules_for_section(
    doc_mappings: DocumentMappings, section_title: str, min_score: float = 0.5
) -> List[Tuple[str, str, float]]:
    relevant_modules = []
    
    for section in doc_mappings.sections:
        if section.section_title == section_title:
            for module in section.modules:
                for topic in module.topics:
                    if topic.relevance_score >= min_score:
                        relevant_modules.append(
                            (module.module_name, topic.topic_name, topic.relevance_score)
                        )
    
    return sorted(relevant_modules, key=lambda x: x[2], reverse=True)

def save_mappings_to_file(doc_mappings: DocumentMappings, output_dir: str = "data/mappings") -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    safe_title = "".join(c if c.isalnum() else "_" for c in doc_mappings.document_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_mappings_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(doc_mappings.dict(), f, indent=2, ensure_ascii=False, default=str)
    
    return filepath

if __name__ == "__main__":
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
            "For each topic, ensure you consider BOTH the topic name AND its detailed description when evaluating relevance. "
            "The topic descriptions provide crucial context about what each topic covers and should heavily inform your relevance assessment.\n\n"
            "For each topic, provide:\n"
            "1. A relevance score [0-1], where 1 means highly relevant and 0 means no relevance.\n"
            "2. A concise and short reasoning (Chain-of-Thought) explaining why the section is relevant to the topic, "
            "referencing specific elements from the topic description where applicable.\n"
            "ALWAYS use the EXACT topic names without any modifications."
        ),
    )

    from grains.utils import load_curriculum, try_loading_document_object
    document = try_loading_document_object("data/objects/The future of hospitality jobs.json")
    curriculum = load_curriculum()
    
    if document:
        doc_mappings = asyncio.run(document_to_mappings(document, curriculum, agent))
        
        output_file = save_mappings_to_file(doc_mappings)
        print(f"Mappings saved to: {output_file}")
        
        print(f"\nProcessed {len(doc_mappings.sections)} sections across {len(curriculum.modules)} modules")
    else:
        print("Failed to load document")