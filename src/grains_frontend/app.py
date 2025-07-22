"""
Streamlit Frontend for Grains Curriculum Explorer.

This module provides a web interface for exploring curriculum content
and generating multiple choice questions using LLM.
"""

import json
import os
from typing import List, Dict, Any, Optional

import streamlit as st
from grains.data_structures import Curriculum, Module, Topic
from grains.utils import load_curriculum
from openai import OpenAI


@st.cache_resource
def get_curriculum() -> Curriculum:
    """Load and cache the curriculum data.
    
    Returns:
        Curriculum object loaded from configuration
    """
    return load_curriculum()


@st.cache_resource
def get_openai_client() -> OpenAI:
    """Initialize and cache OpenAI client.
    
    Returns:
        Configured OpenAI client for API calls
        
    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key,
    )

def generate_mcqs(content: str, n: int = 5) -> List[Dict[str, Any]]:
    """Generate multiple choice questions from content using LLM.
    
    Args:
        content: Text content to generate questions from
        n: Number of questions to generate
        
    Returns:
        List of MCQ dictionaries with question, options, answer, explanation
        
    Raises:
        Exception: If LLM API call fails or JSON parsing fails
    """
    client = get_openai_client()
    
    system_prompt = f"""You are an expert question writer.
Create {n} multiple choice questions (4 options each, only one correct answer) based solely on the provided text.
Output valid JSON array format:
[
  {{
    "question": "Clear, specific question text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option A",
    "explanation": "Brief explanation of why this answer is correct"
  }}
]
Ensure questions test understanding, not just memorization."""

    user_prompt = f"Generate MCQs from this text:\n\n{content}"
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        message = response.choices[0].message.content
        
        # Extract JSON from response
        start = message.find("[")
        end = message.rfind("]") + 1
        
        if start == -1 or end == 0:
            raise ValueError("No valid JSON array found in response")
            
        mcq_json = json.loads(message[start:end])
        return mcq_json
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse MCQ JSON: {e}")
        st.text_area("Raw LLM Output", message, height=200)
        return []
    except Exception as e:
        st.error(f"Error generating MCQs: {e}")
        return []


def display_quiz(mcqs: List[Dict[str, Any]]) -> None:
    """Display interactive quiz with scoring.
    
    Args:
        mcqs: List of MCQ dictionaries to display
    """
    if not mcqs:
        st.warning("No questions available.")
        return
    
    st.markdown("### 📝 Quiz Time!")
    
    # Initialize session state for answers
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # Display questions
    for idx, q in enumerate(mcqs, 1):
        st.markdown(f"**Q{idx}. {q['question']}**")
        
        # Radio button for answer selection
        answer_key = f"q_{idx}"
        selected = st.radio(
            f"Select answer for Q{idx}:",
            options=q['options'],
            key=answer_key,
            index=None
        )
        
        # Store user answer
        if selected:
            st.session_state.user_answers[idx] = selected
        
        st.markdown("---")
    
    # Submit and Results buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Submit Quiz", type="primary"):
            st.session_state.show_results = True
    
    with col2:
        if st.button("Reset Quiz"):
            st.session_state.user_answers = {}
            st.session_state.show_results = False
            st.rerun()
    
    # Show results if submitted
    if st.session_state.show_results:
        display_results(mcqs)


def display_results(mcqs: List[Dict[str, Any]]) -> None:
    """Display quiz results with scoring and explanations.
    
    Args:
        mcqs: List of MCQ dictionaries with correct answers
    """
    st.markdown("### 📊 Quiz Results")
    
    correct_count = 0
    total_questions = len(mcqs)
    
    for idx, q in enumerate(mcqs, 1):
        user_answer = st.session_state.user_answers.get(idx, "No answer")
        correct_answer = q['answer']
        is_correct = user_answer == correct_answer
        
        if is_correct:
            correct_count += 1
            st.success(f"Q{idx}: ✅ Correct!")
        else:
            st.error(f"Q{idx}: ❌ Incorrect")
            st.write(f"Your answer: {user_answer}")
            st.write(f"Correct answer: {correct_answer}")
        
        # Show explanation
        if 'explanation' in q and q['explanation']:
            with st.expander(f"Explanation for Q{idx}"):
                st.write(q['explanation'])
    
    # Overall score
    score_percentage = (correct_count / total_questions) * 100
    st.markdown(f"### Final Score: {correct_count}/{total_questions} ({score_percentage:.1f}%)")
    
    if score_percentage >= 80:
        st.balloons()
        st.success("Excellent work! 🎉")
    elif score_percentage >= 60:
        st.info("Good job! Keep practicing! 👍")
    else:
        st.warning("Keep studying and try again! 📚")


def main() -> None:
    """Main Streamlit application function."""
    st.set_page_config(
        page_title="Grains Curriculum Explorer",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Grains Curriculum Explorer")
    st.markdown("Explore curriculum content and test your knowledge with AI-generated quizzes!")
    
    try:
        # Load curriculum
        curriculum: Curriculum = get_curriculum()
        
        # Module selection
        module_names = [module.name for module in curriculum.modules]
        selected_module_name = st.selectbox("Select Module", module_names)
        module: Module = curriculum[selected_module_name]
        
        # Topic selection
        topic_names = [topic.name for topic in module.topics]
        selected_topic_name = st.selectbox("Select Topic", topic_names)
        topic: Topic = module[selected_topic_name]
        
        # Display topic information
        st.markdown(f"## 📖 {topic.name}")
        st.markdown(f"*{topic.description}*")
        
        # Display content
        if topic.content:
            content_str = topic.content if isinstance(topic.content, str) else "\n\n".join(topic.content)
            
            with st.expander("📄 Topic Content", expanded=True):
                st.markdown(content_str)
            
            # MCQ Generation Section
            st.markdown("## 🎯 Generate Quiz")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                n_questions = st.slider("Number of questions", 1, 10, 5)
            
            with col2:
                generate_button = st.button("Generate MCQs", type="primary")
            
            if generate_button:
                if not content_str.strip():
                    st.warning("No content available to generate questions from!")
                else:
                    with st.spinner("🤖 Generating multiple choice questions..."):
                        mcqs = generate_mcqs(content_str, n=n_questions)
                    
                    if mcqs:
                        st.session_state.current_mcqs = mcqs
                        st.session_state.user_answers = {}
                        st.session_state.show_results = False
                        st.success(f"Generated {len(mcqs)} questions!")
            
            # Display quiz if available
            if hasattr(st.session_state, 'current_mcqs') and st.session_state.current_mcqs:
                display_quiz(st.session_state.current_mcqs)
                
        else:
            st.info("No content available for this topic.")
            
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
