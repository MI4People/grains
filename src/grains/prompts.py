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
MODEL_NAME = "gpt-3.5-turbo"  # Or a more recent, suitable model.
