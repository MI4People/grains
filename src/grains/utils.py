import json
from pathlib import Path

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


if __name__ == "__main__":
    curriculum = load_curriculum()
    print(curriculum)
