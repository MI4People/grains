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
