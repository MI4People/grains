# This sets up a global client using a .env file
# If you're using a different file please enhance accordingly

from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("url")
key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  base_url=url,
  api_key=key,
)
