from typing import TypeVar, Literal, Union
from pydantic import BaseModel
from pathlib import Path
import json


def read_jsonl(fname: str) -> list[dict]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        fname: Path to the JSONL file

    Returns:
        A list of dictionaries, one for each line in the file
    """
    results = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                results.append(json.loads(line))

    return results


T = TypeVar("T", bound=BaseModel)


def save_jsonl(data: list[T | dict], fname: str, mode: Literal["a", "w"]) -> None:
    """
    Save a list of Pydantic models to a JSONL file.

    Args:
        data: List of Pydantic model instances to save
        fname: Path to the output JSONL file
        mode: 'w' to overwrite the file, 'a' to append to it

    Returns:
        None
    """
    with open(fname, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                # Use model_dump_json for proper JSON serialization
                json_str = item.model_dump_json()
                f.write(json_str + "\n")
            else:
                f.write(json.dumps(item) + "\n")


def save_json(data: Union[BaseModel, dict, list], fname: str) -> None:
    """
    Save a Pydantic model, dictionary, or list to a JSON file.

    Args:
        data: Pydantic model instance, dictionary, or list to save
        fname: Path to the output JSON file

    Returns:
        None
    """
    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        # Use model_dump_json for proper JSON serialization
        json_str = data.model_dump_json(indent=2)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
    elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
        # Handle list of BaseModel objects
        json_data = [item.model_dump() for item in data]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    else:
        # Handle regular dict or list
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
