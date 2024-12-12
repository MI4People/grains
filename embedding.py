import boto3
import requests
import os
import json

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_PREFIX = os.environ["S3_PREFIX"]
VECTOR_STORE_NAME = os.environ["VECTOR_STORE_NAME"]
FILE_UPLOAD_ENDPOINT = "https://api.openai.com/v1/files"
VECTOR_STORE_ENDPOINT = "https://api.openai.com/v1/vector_stores"

s3 = boto3.client("s3")

processed_files_path = "processed_documents.json"
if os.path.exists(processed_files_path):
    with open(processed_files_path, "r") as file:
        processed_files = json.load(file)
else:
    processed_files = []

def upload_file_to_openai(file_path, purpose="assistants"):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    files = {
        "file": (file_path, open(file_path, "rb")),
        "purpose": (None, purpose)
    }
    response = requests.post(FILE_UPLOAD_ENDPOINT, headers=headers, files=files)
    response.raise_for_status()
    return response.json()["id"]

def create_vector_store(name, metadata=None):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"  # Added required header
    }
    data = {"name": name, "metadata": metadata or {}}
    response = requests.post(VECTOR_STORE_ENDPOINT, headers=headers, json=data)

    if response.status_code != 200:
        print(f"Error creating vector store: {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()

    return response.json()["id"]

def attach_file_to_vector_store(vector_store_id, file_id, chunking_strategy="auto"):
    endpoint = f"{VECTOR_STORE_ENDPOINT}/{vector_store_id}/files"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"  # Added required header
    }
    data = {"file_id": file_id, "chunking_strategy": chunking_strategy}
    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code != 200:
        print(f"Error attaching file to vector store: {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()

    return response.json()

def process_files(bucket_name, vector_store_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in response:
        return

    vector_store_id = create_vector_store(vector_store_name)

    for obj in response["Contents"]:
        key = obj["Key"]
        if key in processed_files:
            continue

        local_file_name = key.split("/")[-1]
        s3.download_file(bucket_name, key, local_file_name)

        file_id = upload_file_to_openai(local_file_name)
        attach_file_to_vector_store(vector_store_id, file_id)

        processed_files.append(key)
        with open(processed_files_path, "w") as file:
            json.dump(processed_files, file)

if __name__ == "__main__":
    process_files(S3_BUCKET_NAME, vector_store_name=VECTOR_STORE_NAME, prefix=S3_PREFIX)
