import asyncio
import copy
import glob
import json
import os
from collections import defaultdict
from typing import List, Tuple
from uuid import UUID

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from grains.clients import client
from grains.data_structures import (Curriculum, Document, Module, Path,
                                    RelevanceStore, Topic)
from grains.utils import load_curriculum, load_documents_from_obj_dir


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

from typing import List


def make_aggregation_prompt(module_name: str, topic_name: str, topic_description: str, rel_docs: List[Document]) -> dict:
    # Build the SYSTEM prompt
    system_prompt = (
        "You are an expert curriculum content integrator. "
        "Your job is to synthesize and aggregate all relevant knowledge from the provided document sections "
        "to create a cohesive, comprehensive, and logically structured learning document for a specific topic. "
        "Filter information for relevance to the topic, eliminate redundancy, and organize content clearly with appropriate headings. "
        "Do NOT simply concatenate the source material. Use clear academic or professional language for hospitality education."
    )

    # Build the USER prompt (with embedded variable values)
    user_prompt = f"""\
Module Name: {module_name}
Topic Name: {topic_name}
Topic Description:
{topic_description}

Here are relevant documents, each with a summary and several sections. Each section includes a summary and the full content.
Aggregate and synthesize the topic-relevant points into ONE coherent learning document. Structure the document for clarity and flow. Avoid redundancy.

Relevant documents and sections:
"""

    for doc in rel_docs:
        user_prompt += f"\n---\nDocument: {doc.filename or doc.summary_title or 'Untitled'}\n"
        if doc.summary_title:
            user_prompt += f"Document Title: {doc.summary_title}\n"
        if doc.summary:
            user_prompt += f"Document Summary:\n{doc.summary}\n"
        user_prompt += f"Sections:\n"
        for sec in doc.sections:
            user_prompt += (
                f"\n### {sec.title or 'Untitled Section'}"
                f"\n#### Summary:\n{sec.summary or '(No summary provided)'}"
                f"\n#### Content:\n{sec.content or '(No content provided)'}\n"
            )
    user_prompt += (
        "\n---\n"
        f"Now, aggregate and synthesize the above information into a single, structured educational document relevant to the topic: \"{topic_name}\".\n"
        "If possible, use subheadings, paragraphs, and clear transitions."
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }



def main(mappings_store_file_path: str = "data/mappings.json", dir : Path = Path("data/objects")):
    """Main function to execute the entire process"""
    MODEL: str = "google/gemini-2.5-flash-preview-05-20"
    curriculum = load_curriculum()
    test_module = curriculum.modules[1]
    test_section = test_module.topics[3]
    docs = load_documents_from_obj_dir(dir)
    store = RelevanceStore.create_store_from_json(mappings_store_file_path)
    print(store)
    THRESHOLD = 0.8
    print(f"Threshold {THRESHOLD}\n")
    to_be_aggregated = defaultdict(dict)
    for module in curriculum.modules:
        print(f"Module {module.name}:")
        for topic in module.topics:
            releveant = store.get_relevant_section_ids(topic.id, min_score=THRESHOLD)
            padding_width = 4-len(str(len(releveant)))
            padding_spaces = " " * padding_width
            print(f"{len(releveant)}{padding_spaces} relevant sections for topic {topic.name}.")
            to_be_aggregated[module.name][topic.name] = store.get_relevant_section_ids_per_doc(topic.id, min_score=THRESHOLD, docs=docs)
        print()
    print(f"Module:\n'{test_module.name}':\nSection:\n'{test_section.name}'")
    docs_for_section = to_be_aggregated[test_module.name][test_section.name]
    non_relevant_documents = []
    for doc in docs_for_section:
        if(len(doc.sections)>0):
            padding_width = 10-len(f"{len(doc.sections)}/{doc.tokens}")
            padding_spaces = " " * padding_width
            print(f"{len(doc.sections)}/{doc.tokens}{padding_spaces}[rel.sections/tokens] for doc '{doc.filename}'")
            #print(f"{doc.summary}")
        else:
            non_relevant_documents.append(doc)
    print(f"{len(non_relevant_documents)}/{len(docs)} documents have no relevant sections.")
    test = [[sec.content for sec in doc.sections] for doc in docs_for_section if doc.filename in ["1771058994_0"]]
    #print(test)



    text = store.create_aggregated_document(module.name, topic.name, curriculum, docs, THRESHOLD, client, MODEL)
    print(text)

    # source_cur = insert_section_content(cur=source_cur,relevant=relevant,docs=docs)
    # target_cur = await process_curriculum(source_cur=source_cur,target_cur=target_cur,agent=agent)
    # # Save the result
    # with open("data/target/curicculum-house-keeping-with-content.json", 'w', encoding='utf-8') as f:
    #     json.dump(target_cur, f, indent=3, ensure_ascii=False)

if __name__ == "__main__":
    main()
