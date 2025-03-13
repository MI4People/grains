import os
import json

import openai
import numpy as np

from grains.data_structures import Document, Section
from grains.prompts import (MODEL_NAME, SUMMARIZE_DOCUMENT_MAX_TOKENS,
                            SUMMARIZE_DOCUMENT_PROMPT,
                            SUMMARIZE_SECTION_MAX_TOKENS,
                            SUMMARIZE_SECTION_PROMPT,
                            SUMMARIZE_TITLE_MAX_TOKENS, SUMMARIZE_TITLE_PROMPT,
                            SYSTEM_PROMPT)

# Ensure environment variable is set (replace with your actual key or alternative method)
# os.environ["OPENAI_API_KEY"] = ("sk-your-open-API-key")  # VERY SECURE ...


def add_summaries(document: Document) -> Document:
    """
    Generates summaries and titles for sections and the overall document using an LLM.

    Args:
        document: The Document object to summarize.

    Returns:
        The Document object with generated summaries and titles.
    """
    client = openai.OpenAI()
    for section in document.sections:
        # 1. Generate Section Summary
        prompt_section = SUMMARIZE_SECTION_PROMPT.format(section=section.content)
        try:
            response_section = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt_section}],
                max_tokens=SUMMARIZE_SECTION_MAX_TOKENS,
            )
            section.summary = response_section.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error summarizing section: {e}")
            section.summary = "Failed to summarize section."

        # 2. Generate Section Title (from the summary)
        prompt_title = SUMMARIZE_TITLE_PROMPT.format(summary=section.summary)
        try:
            response_title = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt_title}],
                max_tokens=SUMMARIZE_TITLE_MAX_TOKENS,
            )
            section.summary_title = response_title.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error summarizing section title: {e}")
            section.summary_title = "Failed to summarize section title."

    # 3. Generate Document Summary (from section summaries)
    section_summaries = "\n".join([s.summary for s in document.sections if s.summary is not None])
    prompt_doc = SUMMARIZE_DOCUMENT_PROMPT.format(section_summaries=section_summaries)
    try:
        response_doc = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt_doc}],
            max_tokens=SUMMARIZE_DOCUMENT_MAX_TOKENS,
        )
        document.summary = response_doc.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing document: {e}")
        document.summary = "Failed to summarize document."

    # 4. Generate Document Title (from the document summary)
    prompt_document_title = SUMMARIZE_TITLE_PROMPT.format(summary=document.summary)
    try:
        response_document_title = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_document_title}],
            max_tokens=SUMMARIZE_TITLE_MAX_TOKENS,
        )
        document.summary_title = response_document_title.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing document Title: {e}")
        document.summary_title = "Failed to summarize document Title."

    return document

def classify(document: Document, curriculum)-> Document:
    """
    Classifies sections of a document to categories using an LLM.

    Args:
        document: The Document object to classify.
        curriculum: The curriculum of interest.

    Returns:
        The Document object with mappings.
    """
    client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<>",
    )
    # Infer LLM classifier function
    def get_class(text, description, labels, label_descriptions):
        function = {
        "type": "function",
        "function":{
        "name": "Classify",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "prediction": {
                    "type": "string",
                    "description": label_descriptions,
                    "enum": labels
                },
            },
            "required": ["prediction"]
        },
        }}
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-preview-02-05:free", 
            messages=[{"role": "system", "content": "Only use the functions you have been provided with."},
                        {"role": "user", "content": text}],
            tools=[function],
            tool_choice= {
                "type": "function",
                "function": {
                    "name": "Classify"
                }
                },
            temperature=0
        )
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)["prediction"]
     
    # Classify each document section for each curriculum module 
    for i in range(len(curriculum.modules)):
        module_desc = [topic.description for topic in curriculum.modules[i].topics]+["Other"]
        module_label = [topic.name for topic in curriculum.modules[i].topics]+["Other"]
        for j in range(len(document.sections)):
            # Classify from description of topics and summary of section
            text_topic = get_class(text = document.sections[j].summary, description="topic of a section",
                                labels=module_desc,
                                label_descriptions="The section the text belongs to")
            # Lookup topic name
            label = module_label[np.where(np.array(module_desc) == np.array(text_topic))[0][0]]
            document.sections[j].mappings += [[curriculum.modules[i].name, label]]

    return document