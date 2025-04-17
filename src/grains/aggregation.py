from typing import List, Tuple
from pydantic import BaseModel
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


def add_field_to_innermost(data, field_name="sections", field_value=None):
    """
    Adds a new custom field to each innermost dictionary.
    Creates a new independent copy of the field_value for each dictionary.
    
    Args:
        data: The data structure to modify
        field_name: Name of the field to add (default: "sections")
        field_value: Value to add for each field (default: empty list)
                    This value will be deep-copied for each dictionary.
    
    Returns:
        Modified data structure
    """
    # Handle None input
    if data is None:
        return None
    
    # Set default field_value if not provided
    if field_value is None:
        field_value = []
    
    # Convert Pydantic models to dicts
    if isinstance(data, BaseModel):
        data = data.model_dump()

    # Process dictionaries
    if isinstance(data, dict):
        # For topics, add the custom field
        if "topics" in data and isinstance(data["topics"], list):
            for topic in data["topics"]:
                if isinstance(topic, dict):
                    # Important: Create a deep copy of field_value for each topic
                    # This ensures each topic gets its own independent value
                    import copy
                    topic[field_name] = copy.deepcopy(field_value)
        
        # Process all other keys recursively
        for k, v in data.items():
            data[k] = add_field_to_innermost(v, field_name, field_value)
    
    # Process lists
    elif isinstance(data, list):
        data = [add_field_to_innermost(item, field_name, field_value) for item in data]

    return data

def extract_relevant_sections(data) -> List[Tuple[str, str, str]]:
    """
    Extract relevant sections from data where relevance score >= 0.75.
    
    Returns:
        List of tuples: (module_name, topic_name, section_title)
    """
    matches: List[Tuple[str, str, str, str]] = []

    for section in data.get("sections", []):
        section_title = section.get("section_title")
    
        for module in section.get("modules", []):
            module_name = module.get("module_name")
            
            for topic in module.get("topics", []):
                relevance = topic.get("relevance_score", 0)
                topic_name = topic.get("topic_name")

                if relevance >= 0.75:
                    matches.append((module_name, topic_name, section_title, data.get("document_title")))

    return matches

def insert_sections_into_dictionary(matches, target_dict):
    """
    Insert relevant section names from matches list into the target dictionary.
    
    Args:
        matches: List of tuples (module_name, topic_name, section_name, document_title)
        target_dict: The target dictionary structure
    
    Returns:
        Updated dictionary with a list of section names which are relavant for the topic
    """
    # Make a deep copy to ensure we don't modify the original
    result_dict = copy.deepcopy(target_dict)
    
    # Process each match tuple
    for module_name, topic_name, section_name, document_title in matches:
        # Find matching module in target dictionary
        for module in result_dict["modules"]:
            if module["name"] == module_name:
                # Find matching topic in module
                for topic in module["topics"]:
                    if topic["name"] == topic_name:
                        # Add section to the sections list
                        topic["sections"].append(section_name)
                        topic["document"].append(document_title)
                        break  # Found the matching topic, no need to continue inner loop
                
                break  # Found the matching module, no need to continue outer loop
    
    return result_dict

def find_section_in_document(section_title, document):
    """Find a section by title in the document"""
    for section in document.get("sections", []):
        if section.get("title") == section_title:
            return section
    return None

def add_section_content(curriculum, document):
    """Adding actual content of the sections which are assigned to a topic"""

    # For each module and topic in the curriculum
    for module in curriculum["modules"]:
        
        for topic in module.get("topics", []):
            
            # Skip topics with empty sections list
            if not topic.get("sections") or len(topic["sections"]) == 0:
                continue
            
            # Initialize section_content if it doesn't exist
            if "section_content" not in topic:
                topic["section_content"] = []

            # Process each section title in the topic
            for section_title in topic["sections"]:
                # Find the section in the document
                section = find_section_in_document(section_title, document)
                
                if section:
                # Adding section content to the list
                    section_content = section.get("content", "")
                    topic["section_content"].append(section_content)
    
    return curriculum

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

async def process_curriculum(source_cur, target_cur, agent):

    """
    Takes content from source curriculum, summarizes it with AI, and adds it to target curriculum
    """
    for s_module, t_module in zip(source_cur["modules"], target_cur["modules"]):
        for s_topic, t_topic in zip(s_module["topics"], t_module["topics"]):
            # Skip if no section content
            if not s_topic.get("section_content") or not s_topic["section_content"]:
                continue
            
            print(f"Processing: {s_topic['name']}")

            # Generate prompt
            prompt = generate_system_prompt(
                s_module["name"],
                s_topic["name"],
                s_topic["description"],
                s_topic["section_content"]
            )
        
            try:
                result = await agent.run(prompt)
                t_topic["content"] = result.data
                print(f"✓ Added content to: {s_topic['name']}")
            except Exception as e:
                print(f"✗ Error: {str(e)}")
    
    return target_cur

async def main():

    """Main function to execute the entire process"""

    # Load enviroinment and setup agent, which will be replaced by normal LLM call
    load_dotenv()
    MODEL: str = "meta-llama/llama-3.3-70b-instruct"
    #MODEL: str = "anthropic/claude-3.5-sonnet"
    #MODEL: str = "openai/gpt-4o-2024-11-20"

    agent = Agent(
               OpenAIModel(
                   MODEL,
                   provider=OpenAIProvider(
                       base_url="https://openrouter.ai/api/v1",
                       api_key=os.getenv("OPENAI_API_KEY")
                   ),
               )
       )

    with open("conf/curicculum-house-keeping.json","r") as f:
            target_cur = json.load(f)

    print("1. Loading ground structure to work with...")
    source_cur = load_curriculum()

    print("2. Adding empty section lists as well as empty document string to topics...")
    source_cur = add_field_to_innermost(source_cur)
    source_cur = add_field_to_innermost(source_cur,field_name="document")

    # Define ordered lists containing the paths to the documents and to the mappings
    object_files = sorted(glob.glob("data/objects/*.json"))
    mapping_files = sorted(glob.glob("data/mappings/*.json"))
    paired_files = list(zip(object_files, mapping_files))

    # Load the data containing section information
    for obj_file, map_file in paired_files:
        with open(obj_file,"rb") as of:
            obj = json.load(of)
        with open(map_file,"rb") as mf:
            map = json.load(mf)
    
        # Get all relevant sections out of the mapping files
        matches = extract_relevant_sections(map)
        
        # Add section names to the source curriculum
        source_cur = insert_sections_into_dictionary(matches, source_cur)
    
        # Save to file for debugging reasons
        with open("data/target/curriculum_with_sections_and_docname.json", "w") as f:
            json.dump(source_cur, f, indent=2)
        # Add section content to the source curriculum
        source_cur = add_section_content(curriculum=source_cur,document=obj)

    target_cur = await process_curriculum(source_cur=source_cur, target_cur=target_cur, agent=agent)

    # Save the result
    with open("data/target/curicculum-house-keeping-with-content.json", 'w', encoding='utf-8') as f:
        json.dump(target_cur, f, indent=3, ensure_ascii=False)
    
if __name__ == "__main__":
    asyncio.run(main())
