import json
from pathlib import Path

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


if __name__ == "__main__":
    curriculum = load_curriculum()
    print(curriculum)
