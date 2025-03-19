import json

from grains.data_structures import Curriculum


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


if __name__ == "__main__":
    curriculum = load_curriculum()
    print(curriculum)
