import streamlit as st
from workflow import process_with_langflow

st.title("Grains")
st.write("Welcome to Grains AI app!")

user_input = st.text_input("Enter a prompt:", "")
if user_input:
    response = process_with_langflow(user_input)
    st.write(f"LangFlow Response: {response}")
