from langflow.custom import Component
from langflow.helpers.data import data_to_text
from langflow.io import MessageTextInput, Output
from langflow.schema import Data, Message
import requests
import time

class OpenAIChatComponent(Component):
    display_name = "Chat with student"
    description = "A custom component to interact with the OpenAI Student Assistant."
    documentation: str = "http://docs.langflow.org/components/custom"
    icon = "chat"
    name = "OpenAIChatComponent"

    inputs = [
        MessageTextInput(
            name="task",
            display_name="Task",
            info="The task to send to the assistant.",
            value="Hello, Assistant!",
            tool_mode=True,
        ),
        MessageTextInput(
            name="writer_output",
            display_name="Writer Output",
            info="Writer output to send to student",
            value="",
            tool_mode=True,
        ),
        MessageTextInput(
            name="api_key",
            display_name="API Key",
            info="Your OpenAI API key.",
            value="",
            tool_mode=False,
        ),
    ]

    outputs = [
        Output(display_name="Student Response", name="student_response", method="build_output"),
    ]

    def get_assistant_id(self, api_key):
        BASE_URL = "https://api.openai.com/v1/assistants"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        assistants = []
        url = f"{BASE_URL}?order=desc&limit=20"
        while url:
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                assistants.extend(data.get("data", []))
                url = data.get("next", None)
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error fetching assistant ID: {str(e)}")
        for assistant in assistants:
            if assistant.get("name") == "student":
                return assistant.get("id")
        raise RuntimeError("Assistant named 'student' not found.")

    def create_thread(self, api_key):
        url = f"https://api.openai.com/v1/threads"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        payload = {}
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("id")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error creating thread: {str(e)}")

    def add_message(self, api_key, thread_id, user_message, writer_output):
        url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        combined_message = f"Task: {user_message}\nWriter Output: {writer_output}"
        payload = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": combined_message
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error adding message to thread: {str(e)}")

    def run_thread(self, api_key, thread_id, assistant_id):
        url = f"https://api.openai.com/v1/threads/{thread_id}/runs"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        payload = {
            "assistant_id": assistant_id,
            "response_format": {
                "type": "json_object"
            }
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("id")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error running thread: {str(e)}")

    def wait_for_completion(self, api_key, thread_id, run_id):
        url = f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        start_time = time.time()
        while True:
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                status = data.get("status")
                if status == "completed":
                    messages = self.list_messages(api_key, thread_id)
                    responses = []
                    for msg in messages:
                        if msg['role'] == 'assistant':
                            responses.append(msg['content'][0]['text']['value'])
                    return responses
                elif time.time() - start_time > 300:
                    return data
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error checking run status: {str(e)}")

    def list_messages(self, api_key, thread_id):
        url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2"
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error listing messages: {str(e)}")

    def build_output(self) -> Message:
        user_message = self.task.strip()
        writer_output = self.writer_output.strip()
        api_key = self.api_key.strip()

        if not user_message:
            raise ValueError("User input cannot be empty.")
        if not api_key:
            raise ValueError("API key is required.")

        assistant_id = self.get_assistant_id(api_key)
        thread_id = self.create_thread(api_key)
        self.add_message(api_key, thread_id, user_message, writer_output)
        run_id = self.run_thread(api_key, thread_id, assistant_id)
        responses = self.wait_for_completion(api_key, thread_id, run_id)

        result_string = data_to_text("{text}", responses, sep="\n")
        self.status = result_string
        return Message(text=result_string)
