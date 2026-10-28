import sys
import json
from typing import Any


def read_file(file_name: str) -> Any:
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            content = json.load(file)
    except FileNotFoundError:
        print(
            f"Error: file not found: {file_name}",
            file=sys.stderr
        )
        sys.exit(1)
    except PermissionError:
        print(
            f"Error: could not read {file_name}: permission denied",
            file=sys.stderr
        )
        sys.exit(1)
    except IsADirectoryError:
        print(
            f"Error: could not read {file_name}: is a directory, not a file",
            file=sys.stderr
        )
        sys.exit(1)
    except UnicodeDecodeError:
        print(
            f"Error: {file_name} is not valid UTF-8 text",
            file=sys.stderr
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"Error: invalid JSON in {file_name}: {e}",
            file=sys.stderr
        )
        sys.exit(1)
    return content


def parsing_prompts(file_name: str) -> list[str]:
    prompts = []
    prompt_list = read_file(file_name)
    for item in prompt_list:
        for key, value in item.items():
            if key == "prompt":
                prompts.append(value)
    return prompts


def parsing_functions(file_name: str) -> dict[str | dict]:
    functions = {}
    functions_list = read_file(file_name)
    for item in functions_list:
        name = item["name"]
        parameters = {}
        for key, value in item["parameters"].items():
            ptype = value["type"]
            parameters[key] = ptype
        functions[name] = parameters
    return functions
