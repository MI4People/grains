'''
INPUT: 
    JSON of the form

        {
            doc title
            doc id
            modules: 
                [
                    mod name
                    mod id
                    sections:
                        [
                            sec name
                            sec id
                            topics:
                                [
                                    topic name
                                    relevance score
                                    reasoning
                                    timestamp
                                ]
                        ]
                ]
        }

    TODO:

    write a function

    def extract_relevant_sections(mapping, output format)
    which extracts all topic relevant sections with a threshold >= x
'''

from typing import Any, Dict, Union
from pydantic import BaseModel

from grains.utils import load_curriculum

# Define intermediate format. Takes given Curriculum and adds "Sections" field 
def generate_inter_curr(data: Union[BaseModel, Dict, list], mapped_secs: list[str]):
    if isinstance(data, BaseModel):
        data = data.model_dump()

    if isinstance(data, dict):
        # If none of the values are dicts or lists, it's a leaf
        if all(not isinstance(v, (dict, list)) for v in data.values()):
            data["sections"] = mapped_secs
        else:
            for k, v in data.items():
                data[k] = generate_inter_curr(v, mapped_secs)

    elif isinstance(data, list):
        data = [generate_inter_curr(item, mapped_secs) for item in data]

    return data