# This sets up a global client using a .env file
# If you're using a different file please enhance accordingly

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

base_url="https://openrouter.ai/api/v1"
api_key=os.getenv("OPENAI_API_KEY")

client = OpenAI(
  base_url=base_url,
  api_key=api_key,
)
