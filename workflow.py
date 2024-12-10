from langflow import Workflow

def process_with_langflow(prompt: str) -> str:
    workflow = Workflow.load_from_file("workflow.json")
    return workflow.run(prompt)
