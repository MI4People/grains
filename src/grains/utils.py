import json
import os
from pathlib import Path

from botocore.exceptions import (EndpointConnectionError, NoCredentialsError,
                                 PartialCredentialsError)

from grains.data_structures import Curriculum, Document


def load_curriculum(filepath: str = "conf/curicculum-house-keeping.json") -> Curriculum:
    """
    Loads the curriculum JSON file, instantiates and returns a Curriculum object.

    code

    :param filepath: Path to the JSON file.
    :return: Curriculum instance containing modules and topics.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Instantiate the Curriculum using Pydantic's parsing capabilities.
    curriculum = Curriculum.model_validate(data)
    return curriculum


def try_loading_document_object(filepath: str | Path) -> Document | None:
    """Loads a Document from a JSON file.  Returns None if the file doesn't exist."""
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

if __name__ == "__main__":
    curriculum = load_curriculum()
    print(curriculum)
