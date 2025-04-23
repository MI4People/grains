import os

import openai

from grains.data_structures import Document, Section
from grains.prompts import (SUMMARIZE_DOCUMENT_MAX_TOKENS,
                            SUMMARIZE_DOCUMENT_PROMPT,
                            SUMMARIZE_SECTION_MAX_TOKENS,
                            SUMMARIZE_SECTION_PROMPT,
                            SUMMARIZE_TITLE_MAX_TOKENS, SUMMARIZE_TITLE_PROMPT,
                            SYSTEM_PROMPT)

# Ensure environment variable is set (replace with your actual key or alternative method)
# os.environ["OPENAI_API_KEY"] = ("sk-your-open-API-key")  # VERY SECURE ...


def add_summaries(document: Document, model: str) -> Document:
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
                model=model,
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
                model=model,
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
            model=model,
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
            model=model,
            messages=[{"role": "user", "content": prompt_document_title}],
            max_tokens=SUMMARIZE_TITLE_MAX_TOKENS,
        )
        document.summary_title = response_document_title.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing document Title: {e}")
        document.summary_title = "Failed to summarize document Title."

    return document
