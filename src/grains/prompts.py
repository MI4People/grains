SUMMARIZE_TITLE_PROMPT = (
    "Create a short title in less than 8 words that captures the content of the following summary: {summary}"
)
SUMMARIZE_DOCUMENT_PROMPT = "Based on the following section summaries create an overall summary of the document in less 170 words:\n{section_summaries}"
SUMMARIZE_SECTION_PROMPT = (
    "Create a short summary less than 70 words and shorter than the section itself:\n\n===Section===\n{section}"
)
SUMMARIZE_TITLE_MAX_TOKENS = 30
SUMMARIZE_DOCUMENT_MAX_TOKENS = 250  # Increased max_tokens to accommodate a 170-word summary.
SUMMARIZE_SECTION_MAX_TOKENS = 150  # Increased max_tokens for summaries less than 70 words.
SYSTEM_PROMPT = (
    "You are a document summarization expert. Your goal is to provide concise, accurate, and informative summaries."
)
MAPPING_SYSTM_PROMPT = (
            "You are an expert in mapping document sections to module topics for the hotel and service industry. "
            "Your task is to analyze the section summary below and calculate its relevance for each topic of the module. "
            "For each topic, ensure you consider BOTH the topic name AND its detailed description when evaluating relevance. "
            "The topic descriptions provide crucial context about what each topic covers and should heavily inform your relevance assessment.\n\n"
            "For each topic, provide:\n"
            "1. A relevance score [0-1], where 1 means highly relevant and 0 means no relevance.\n"
            "2. A concise and short reasoning (Chain-of-Thought) explaining why the section is relevant to the topic, "
            "referencing specific elements from the topic description where applicable.\n"
            "ALWAYS use the EXACT topic names without any modifications."
        )

MAPPING_PROMPT = """
For the section of the corresponding Document above, evaluate its relevance to each topic in the module.
Consider BOTH the topic name AND its description when determining relevance.

For EACH topic, provide:
1. The EXACT topic id as listed above - DO NOT CHANGE OR MODIFY THE TOPIC ids IN ANY WAY
2. A relevance score between 0 and 1 (where 1 is highly relevant, 0 is not relevant)
3. A brief explanation of why this section is relevant or not to the topic, specifically referencing how the section content relates to the topic description

CRITICAL: You must use the EXACT topic ids from the list above without any modifications, additions or omissions.
Make sure to include ALL topics from the module in your response, even if they have low relevance scores.
When determining relevance, pay careful attention to both the topic id and its description.

Document Id: {document_id}
{section_info}
Module:
{module_info}
"""


def generate_aggregation_prompts(module_name: str, topic_name: str, topic_description: str, rel_docs: list) -> tuple[str, str]:
    """
    Generate OpenAI-compatible system and user prompts for content aggregation.

    Args:
        module_name (str): Name of the curriculum module
        topic_name (str): Name of the specific topic
        topic_description (str): Description of the topic and learning objectives
        rel_docs (list): List of Document objects containing relevant sections

    Returns:
        tuple[str, str]: (system_prompt, user_prompt)
    """

    system_prompt = """You are an expert curriculum content aggregator for the hospitality industry.
    Your task is to synthesize information from multiple document sections into a comprehensive,
    well-structured learning material that directly addresses a specific curriculum topic.

    Your responsibilities:
    - Analyze the topic description to identify key learning objectives
    - Extract relevant information from provided sections that directly relates to the topic
    - Synthesize information into a coherent, well-structured document
    - Organize content logically, building from foundational concepts to complex applications
    - Ensure all key aspects mentioned in the topic description are adequately addressed
    - Maintain academic rigor while making content accessible for learning
    - Create content in university lecture document format suitable for generating questions and answers
    - Preserve all important details and information without compression or omission


    Quality Requirements:
    - Eliminate redundancy while preserving important details
    - Create smooth transitions between concepts from different sources
    - Highlight connections and relationships between ideas
    - Ensure content directly supports the topic's learning objectives
    - Maintain professional, educational tone appropriate for hospitality students
    - Include specific examples and practical applications where available"""

    # Build the documents section
    documents_content = ""
    for doc in rel_docs:
        documents_content += f"**Document**: {doc.filename}\n"
        documents_content += f"**Document Summary**: {doc.summary if doc.summary else 'No summary available'}\n\n"
        documents_content += f"**Relevant Sections**:\n"

        for section in doc.sections:
            documents_content += f"- **Section**: {section.title}\n"
            if section.content:
                documents_content += f"  **Content**: {section.content}\n"
            documents_content += "\n"
        documents_content += "---\n\n"

    user_prompt = f"""
    **Instructions for Your Output:**
    1.  **Analyze and Organize Content:** The sections within each document are ordered logically, but the documents themselves are provided in random order. Before synthesizing, consider the logical flow and relationships between concepts across all sections to determine the most coherent order for presenting the information in your output.
    2.  **Focus on the Topic:** Your primary goal is to create a text that directly and comprehensively addresses the **Topic Name** and answers the questions or covers the areas specified in the **Topic Description**.
    3.  **Synthesize Information:** Integrate information from the various relevant "Section Content" parts while preserving all important details. Paraphrase and rephrase in your own words while retaining the original meaning and ALL significant information. Do not compress or omit details.
    4.  **Use Only Provided Information:** Base your output *solely* on the information present in the "Section Content" of the provided document sections. Do not introduce external knowledge or make assumptions beyond what is given.
    5.  **Structure and Coherence:** Organize the aggregated information logically. Use clear language. You may use paragraphs, headings, subheadings, or bullet points if they improve clarity and readability for the topic. The final output should read like a well-structured explanation or part of a textbook.
    6.  **Comprehensive Coverage:** Attempt to cover all aspects mentioned in the **Topic Description**, using the provided sections. If information for a specific aspect is missing in the sections, you do not need to invent it or state that it's missing; simply focus on what is available.
    7.  **Preserve Information:** If multiple sections provide similar information, integrate them comprehensively rather than condensing. Ensure all important details from each source are retained in the final output.
    8.  **Neutral and Informative Tone:** Maintain an educational and objective tone suitable for curriculum material.
    9.  **No Meta-Commentary:** Do not refer to the process of aggregation (e.g., "I have combined information from...", "This section discusses..."). Just provide the aggregated text itself as if it were a standalone piece of content for the topic.

    **Begin your response directly with the aggregated text for the topic.**
    Here is the information:
    --------------------------------------------------------------------------------

    **MODULE**: {module_name}
    **TOPIC**: {topic_name}
    **TOPIC DESCRIPTION**: {topic_description}

    **CONTENT SOURCES**:

    {documents_content}

    **SYNTHESIS REQUIREMENTS**:
    - Focus specifically on how the provided content relates to and supports the topic description
    - Create educationally valuable content that addresses all aspects mentioned in the topic description
    - Properly synthesize rather than just concatenate information
    - Maintain relevance to the hospitality industry context
    - Produce well-organized, coherent learning materials
    - Ensure comprehensive coverage of topic description requirements
    - Include logical flow and clear organization
    - Integrate multiple source perspectives where applicable
    - Emphasize practical relevance to hospitality professionals
    - Use clear, engaging writing suitable for educational use
    - Format output as a comprehensive university lecture document with clear headings and subheadings
    - Preserve all important information without compression - comprehensiveness is more important than brevity
    - Ensure the document contains sufficient detail for generating meaningful questions and answers
    - Structure content to support educational assessment and discussion


    Please create a comprehensive learning document following these guidelines."""

    return system_prompt.strip(), user_prompt.strip()
