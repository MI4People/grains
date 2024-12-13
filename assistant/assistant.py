import os
import requests

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = "https://api.openai.com/v1/assistants"


def load_instructions(name):
    filename = name + ".txt"
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return ""
    except IOError as e:
        print(f"Error reading the file '{filename}': {e}")
        return ""


def list_assistants():
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }

    assistants = []
    url = f"{BASE_URL}?order=desc&limit=20"
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            assistants.extend(data.get("data", []))
            url = data.get("next", None)
        else:
            print(f"Error listing assistants: {response.status_code}, {response.text}")
            break
    return assistants


def post_with_retries(url, headers, payload, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code < 500:
                return response
            print(f"Attempt {attempt + 1} failed with status {response.status_code}. Retrying...")
        except requests.RequestException as e:
            print(f"Network error on attempt {attempt + 1}: {e}. Retrying...")
        time.sleep(delay)
    print("All retry attempts failed.")
    return None


def get_or_create_assistant(name):
    try:
        existing_assistants = list_assistants()
        for assistant in existing_assistants:
            if assistant.get("name", None) == name:
                print(f"Assistant '{name}' already exists.")
                return assistant

        instructions = load_instructions(name)
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        payload = {
            "name": name,
            "instructions": instructions,
            "model": "gpt-4o",
            "tools": [],
            "temperature": 0.2
        }
        response = post_with_retries(BASE_URL, headers, payload)
        if response and response.status_code == 200:
            print(f"Assistant '{name}' created successfully.")
            return response.json()
        else:
            print(f"Error creating assistant '{name}': {response.status_code if response else 'No response'}, {response.text if response else 'No response text'}")
            return None
    except Exception as e:
        print(f"Unexpected error while creating assistant '{name}': {e}")


if __name__ == "__main__":
    assistant_names = ["writer", "critic", "student"]
    assistants = {}
    for name in assistant_names:
        assistants[name] = get_or_create_assistant(name)

    for name, assistant in assistants.items():
        if assistant:
            print(f"{name.capitalize()} assistant: {assistant}")
