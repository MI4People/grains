import boto3
import requests
import os
import json
import time

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_PREFIX = os.environ["S3_PREFIX"]
PROCESSED_FILES_KEY = os.environ["PROCESSED_FILES_KEY"]
VECTOR_STORE_NAME = os.environ["VECTOR_STORE_NAME"]
FILES_ENDPOINT = "https://api.openai.com/v1/files"
VECTOR_STORE_ENDPOINT = "https://api.openai.com/v1/vector_stores"

s3 = boto3.client("s3")

try:
    response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=PROCESSED_FILES_KEY)
    processed_files = json.loads(response["Body"].read().decode("utf-8"))
except s3.exceptions.NoSuchKey:
    processed_files = {}

def upload_file_to_openai(file_path, purpose="assistants"):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "assistants=v2"
    }
    files = {"file": (file_path, open(file_path, "rb")), "purpose": (None, purpose)}
    response = requests.post(FILES_ENDPOINT, headers=headers, files=files)
    response.raise_for_status()
    file_id = response.json()["id"]
    print(f"File uploaded successfully: {file_id}")
    return file_id

def wait_for_file_upload(file_id):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    while True:
        response = requests.get(FILES_ENDPOINT, headers=headers)
        response.raise_for_status()
        files = response.json()["data"]
        if any(f["id"] == file_id for f in files):
            print(f"File {file_id} is now available.")
            return
        print(f"Waiting for file {file_id} to be available...")
        time.sleep(5)

def list_files_in_vector_store(vector_store_id):
    endpoint = f"{VECTOR_STORE_ENDPOINT}/{vector_store_id}/files"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()
    files = [file["id"] for file in response.json()["data"]]
    return files

def get_vector_store_by_name(name):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "assistants=v2"
    }
    response = requests.get(VECTOR_STORE_ENDPOINT, headers=headers)
    response.raise_for_status()
    vector_stores = response.json()["data"]
    for store in vector_stores:
        if store["name"] == name:
            print(f"Found existing vector store ID: {store['id']} for name: {name}")
            return store["id"]
    return None

def create_vector_store(name, metadata=None):
    existing_store_id = get_vector_store_by_name(name)
    if existing_store_id:
        return existing_store_id

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    data = {"name": name, "metadata": metadata or {}}
    response = requests.post(VECTOR_STORE_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    vector_store_id = response.json()["id"]
    print(f"Created vector store ID: {vector_store_id}")
    return vector_store_id

def attach_file_to_vector_store(vector_store_id, file_id):
    existing_files = list_files_in_vector_store(vector_store_id)
    if file_id in existing_files:
        print(f"File {file_id} is already attached to vector store {vector_store_id}. Skipping.")
        return {"status": "already_attached"}

    endpoint = f"{VECTOR_STORE_ENDPOINT}/{vector_store_id}/files"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }

    data = {"file_id": file_id}

    try:
        print(f"Attaching file_id: {file_id} to vector_store_id: {vector_store_id}")
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error attaching file to vector store. Response: {response.text}")
        raise

    return response.json()

def save_processed_files(bucket_name, key, processed_files):
    s3.put_object(Bucket=bucket_name, Key=key, Body=json.dumps(processed_files))
    print(f"Processed files saved to S3 at {key}")

def process_files(bucket_name, vector_store_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in response:
        return

    vector_store_id = create_vector_store(vector_store_name)

    for obj in response["Contents"]:
        key = obj["Key"]
        if key in processed_files:
            print(f"File {key} already marked as processed. Skipping.")
            continue

        local_file_name = key.split("/")[-1]
        s3.download_file(bucket_name, key, local_file_name)

        file_id = upload_file_to_openai(local_file_name)
        wait_for_file_upload(file_id)
        attach_file_to_vector_store(vector_store_id, file_id)

        processed_files[key] = file_id
        save_processed_files(bucket_name, PROCESSED_FILES_KEY, processed_files)

if __name__ == "__main__":
    process_files(S3_BUCKET_NAME, vector_store_name=VECTOR_STORE_NAME, prefix=S3_PREFIX)
