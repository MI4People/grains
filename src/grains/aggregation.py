from typing import List, Tuple
from pydantic import BaseModel

from grains.data_structures import Curriculum, Path, Document, RelevanceStore
from uuid import UUID 
import json
import copy
import os 
import glob
import glob
from dotenv import load_dotenv

import asyncio 

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from grains.utils import load_curriculum

def load_documents_from_dir(json_dir: Path) -> list[Document]:
    """
    Loads all JSON files from a directory and parses them into Document objects.

    Args:
        json_dir (Path): Path to the directory containing JSON files.

    Returns:
        List[Document]: List of validated Document objects.
    """
    documents = []
    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc = Document.model_validate(data)
            documents.append(doc)
        except Exception as e:
            print(f"Failed to load {json_file.name}: {e}")
    return documents

def get_relevant_sections(map_store: RelevanceStore, threshold):
    '''
    Exracts all relevant mappings with relevance score higher than the threshold.
    
    Args: 
        map_store (RelevanceStore). pydantic Relevance store object containing all mappings

    Output: 
        relevance (List): An orderd list containing 
                - moduleId 
                - documentId
                - sectionId
                - topicId
            from the mappings with score >= threshold
    '''
    mapp = map_store.mappings
    relevant: List[Tuple[UUID,UUID,UUID,UUID]] = []
    for key in mapp:
        r_score = mapp[key].relevance_score
        if r_score >= threshold:
            relevant.append((mapp[key].module_id,
                mapp[key].document_id,
                mapp[key].section_id,
                mapp[key].topic_id))
    
    relevant_sorted = sorted(relevant, key=lambda x: (x[0], x[1])) 

    return relevant_sorted 

def insert_section_content(cur: Curriculum, relevant, docs):
    '''
    Insert the section content into the curriculum for all relevant sections

    Args:
        cur (Curriculum): Source curriculum where the content gets added
        relevant (List): list of relevant sections and it's ids
        docs (List): list of documents  
    '''
    last_module_id = last_topic_id = None
    current_module = current_topic = None

    for r in relevant:
        module_id, document_id, section_id, topic_id = r

        if module_id != last_module_id:
            current_module = next((m for m in cur.modules if m.id == module_id), None)
            last_module_id = module_id
            current_topic = None
            last_topic_id = None

        if current_module and topic_id != last_topic_id:
            current_topic = next((t for t in current_module.topics if t.id == topic_id), None)
            last_topic_id = topic_id

        if current_module and current_topic:
            for d in docs:
                if d.id == document_id:
                    for s in d.sections:
                        if s.id == section_id:
                            section_content = s.content

            if section_content:
                current_topic.content.append(section_content)
    
    return cur 

def generate_system_prompt(module_name, topic_name, topic_description, section_content):
    """
    Returns a system prompt to summarize section content into one coherent paragraph.
    """

    prompt = f"""
            You are an expert in creating learning content out of document excerpts from differnt documents.
            The excerpts are given in a list of strings, where each string represents one whole section from one document.

            You are currently working with the following context:
            - **Module**: {module_name}
            - **Topic**: {topic_name}
            - **Topic Description**: {topic_description}

            The list of strings representing sections is given by:

            {section_content}
            
            ---

            Your task is to:
            1. Go - in detail - through all sections and get to know the content.
            2. Write a single and readable **CHAPTER**. It should contain all all information of all the different sections aggregated.
            3. The chapter should be extremely detailed as well as of appropriate length, since it is used as learning content for refugees.
        
        The output should be an alone standing whole chapter of a learning material - especially in terms of length.
        Just output the content, never include suggestions, introductions etc.
    """.strip()

    return prompt

async def process_curriculum(source_cur: Curriculum, target_cur: Curriculum, agent) -> Curriculum:
    """
    Takes content from source curriculum, summarizes it with AI, and adds it to the target curriculum.
    """

    for s_module, t_module in zip(source_cur.modules, target_cur.modules):
        for s_topic, t_topic in zip(s_module.topics, t_module.topics):
            # Skip if no section content
            if not len(s_topic.content):
                print("No Content")
                continue
            print(f"Processing: {s_topic.name}")
            # Generate prompt
            prompt = generate_system_prompt(
                s_module.name,
                s_topic.name,
                s_topic.description,
                s_topic.content
            )

            try:
                result = await agent.run(prompt)
                t_topic.content = [result.data]  # assuming result.data is a string
                print(f"✓ Added content to: {s_topic.name}")
            except Exception as e:
                print(f"✗ Error: {str(e)}")

    return target_cur



async def main():

    """Main function to execute the entire process"""

    MODEL: str = "meta-llama/llama-3.3-70b-instruct"

    agent = Agent(
               OpenAIModel(
                   MODEL,
                   provider=OpenAIProvider(
                       base_url="https://openrouter.ai/api/v1",
                       api_key=os.getenv("OPENAI_API_KEY")
                   ),
               )
       )

    source_cur = load_curriculum()
    target_cur = load_curriculum()

    dir = Path("data/objects")
    docs = load_documents_from_dir(dir)

    mapping_store = RelevanceStore.create_store_from_json("data/mappings.json")
    relevant = get_relevant_sections(mapping_store, 0.8)
    source_cur = insert_section_content(cur=source_cur,relevant=relevant,docs=docs)

    target_cur = await process_curriculum(source_cur=source_cur,target_cur=target_cur,agent=agent)

    # Save the result
    with open("data/target/curicculum-house-keeping-with-content.json", 'w', encoding='utf-8') as f:
        json.dump(target_cur, f, indent=3, ensure_ascii=False)
    
if __name__ == "__main__":
    asyncio.run(main())
