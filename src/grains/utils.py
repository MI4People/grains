import json
import os
from pathlib import Path

# TODO: eventually remove
from botocore.exceptions import (EndpointConnectionError, NoCredentialsError,
                                 PartialCredentialsError)
from pydantic import BaseModel

from grains.data_structures import Curriculum, Document


def load_curriculum(filepath: str = "conf/curriculum-house-keeping.json") -> Curriculum:
    """
    Loads a curriculum from a JSON file and returns a Curriculum instance.

    Args:
        filepath (str): Path to the JSON file containing curriculum data.
                         Defaults to "conf/curriculum-house-keeping.json".

    Returns:
        Curriculum: A Curriculum object with validated modules and topics.

    Raises:
        ValidationError: If the JSON data does not conform to the Curriculum model.
        FileNotFoundError: If the file at the specified filepath does not exist.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Curriculum file not found: {filepath}") from e
    # Instantiate the Curriculum using Pydantic's parsing capabilities.
    curriculum = Curriculum.model_validate(data)
    return curriculum


def try_loading_document_object(filepath: str | Path) -> Document | None:
    """
    Attempts to load a Document from a JSON file.

    Args:
        filepath (str | Path): Path to the JSON file containing document data.

    Returns:
        Document | None: A Document object if loading is successful, otherwise None.

    Note:
        If the file does not exist or an error occurs during loading, returns None.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return Document(**data)
    except Exception as e:
        print(f"Problem loading {filepath}: {e}")
        return None
    

# =============== S3 Download =============== #


def check_s3_connection(s3, S3_BUCKET_NAME: str, S3_PREFIX: str) -> bool:
    """Check if the connection to S3 is successful."""
    try:
        s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX, MaxKeys=1)
        print("Successfully connected to S3.")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        print("AWS credentials are missing or incorrect.")
    except EndpointConnectionError:
        print("Unable to connect to S3 endpoint. Check your network or region settings.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False


def download_missing_files(s3, input_dir: Path, S3_BUCKET_NAME: str, S3_PREFIX: str) -> None:
    """List S3 objects and download only those not present in the local directory."""
    os.makedirs(input_dir, exist_ok=True)
    
    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
    if "Contents" in response:
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            file_name = os.path.basename(s3_key)  # Get only the filename
            local_file_path = os.path.join(input_dir, file_name)

            if not os.path.exists(local_file_path):
                print(f"Downloading {s3_key} to {local_file_path}...")
                s3.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
        print("S3 file check complete.")

def save_pydantic_object(obj: BaseModel, filepath: Path) -> None:
    """
    Serializes a Pydantic object to JSON and saves it to a file.

    Args:
        obj (BaseModel): The Pydantic object to be saved.
        filepath (Path): Path to the output JSON file.

    Returns:
        None
    """
    with open(filepath, "w") as f:
        f.write(obj.model_dump_json(indent=2))
    print(f"Saved document to {filepath}")


# =============== S3 Download =============== #


def check_s3_connection(s3, S3_BUCKET_NAME: str, S3_PREFIX: str) -> bool:
    """Check if the connection to S3 is successful."""
    try:
        s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX, MaxKeys=1)
        print("Successfully connected to S3.")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        print("AWS credentials are missing or incorrect.")
    except EndpointConnectionError:
        print("Unable to connect to S3 endpoint. Check your network or region settings.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False


def download_missing_files(s3, input_dir: Path, S3_BUCKET_NAME: str, S3_PREFIX: str) -> None:
    """List S3 objects and download only those not present in the local directory."""
    os.makedirs(input_dir, exist_ok=True)
    
    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
    if "Contents" in response:
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            file_name = os.path.basename(s3_key)  # Get only the filename
            local_file_path = os.path.join(input_dir, file_name)

            if not os.path.exists(local_file_path):
                print(f"Downloading {s3_key} to {local_file_path}...")
                s3.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
        print("S3 file check complete.")

def load_documents_from_obj_dir(json_dir: Path) -> list[Document]:
    """
    Loads all JSON files from a directory and parses them into Document objects.

    Args:
        json_dir (Path): Path to the directory containing JSON files.

    Returns:
        List[Document]: List of validated Document objects.
    """
    documents = []
    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc = Document.model_validate(data)
            documents.append(doc)
        except Exception as e:
            print(f"Failed to load {json_file.name}: {e}")
    return documents

if __name__ == "__main__":
    curriculum = load_curriculum()
    print(curriculum)
