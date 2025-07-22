import json
import os

import streamlit as st
from grains.data_structures import Curriculum, Module, Topic
from grains.utils import load_curriculum
from openai import OpenAI


@st.cache_resource
def get_curriculum():
    return load_curriculum()

curriculum: Curriculum = get_curriculum()

module_names = [module.name for module in curriculum.modules]
selected_module_name = st.selectbox("Select Module", module_names   )
module: Module = curriculum[selected_module_name]

topic_names = [topic.name for topic in module.topics]
selected_topic_name = st.selectbox("Select Topic", topic_names)
topic: Topic = module[selected_topic_name]

# --- 3. Show topic description & content (joined lines) ---
st.markdown(f"**{topic.name}**")
st.write(f"*{topic.description}*")

if topic.content:
    content_str = topic.content if isinstance(topic.content, str) else "\n\n".join(topic.content)
    st.markdown("#### Topic Content")
    st.markdown(content_str)
else:
    st.info("No content available for this topic.")

OPENROUTER_URL = 'https://openrouter.ai/api/v1'
client = OpenAI(
  base_url=OPENROUTER_URL,
  api_key=os.getenv('OPENAI_API_KEY'),
)

def generate_mcqs(content, n=5):
    SYSTEM_PROMPT = f"""You are an expert question writer.
Create {n} multiple choice questions (4 options each, only one correct answer marked) based solely on the text.
Output JSON:
[
  {{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "option_text",
    "explanation": "..."
  }},
  ...
]
"""
    user_prompt = f"Text:\n{content}\n"
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1200
    )
    message = response.choices[0].message.content
    # Parse out first JSON block returned
    try:
        start = message.index("[")
        end = message.rindex("]") + 1
        mcq_json = json.loads(message[start:end])
        return mcq_json
    except Exception as e:
        st.error(f"MCQ parsing error: {e}")
        st.text_area("Raw LLM Output", message, height=200)
        return []

n_questions = st.slider("Number of questions", 1, 10, 5)
if st.button("Generate MCQs"):
    if not topic.content:
        st.warning("No topic content found!")
    else:
        content_str = topic.content if isinstance(topic.content, str) else "\n\n".join(topic.content)
        with st.spinner("Generating multiple choice questions..."):
            mcqs = generate_mcqs(content_str, n=n_questions)
        for idx, q in enumerate(mcqs, 1):
            st.markdown(f"**Q{idx}. {q['question']}**")
            st.radio(f"Select answer for Q{idx}", options=q['options'], key=f"q_{idx}")
            st.markdown(f"**Answer:** {q['answer']}")
            if "explanation" in q:
                st.expander("Explanation").markdown(q["explanation"])
            st.write("---")
