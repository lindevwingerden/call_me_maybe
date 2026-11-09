import sys
import json
from typing import Any


class InvalidFileError(ValueError):
    pass


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


def validate_prompt_list(content: Any) -> None:
    if not isinstance(content, list):
        raise InvalidFileError("Prompt file does not contain list")
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            raise InvalidFileError(
                f"Item {index} in prompt list is not a dict"
            )
        if len(item) != 1:
            raise InvalidFileError(
                f"Item {index} of prompt list has {len(item)} keys, expected 1"
            )
        for key, value in item.items():
            if key != "prompt":
                raise InvalidFileError(
                    'Other key than "prompt" found'
                    f" in item {index} of prompt list"
                )
            if not isinstance(value, str):
                raise InvalidFileError(
                    f"Prompt in item {index} of prompt list is not a string"
                )
            if len(value) < 1:
                raise InvalidFileError(
                    f"Empty prompt in item {index} of prompt list"
                )


def parsing_prompts(file_name: str) -> list[str]:
    prompt_list = read_file(file_name)
    try:
        validate_prompt_list(prompt_list)
    except InvalidFileError as e:
        print(f"Error in prompt file: {e}", file=sys.stderr)
        sys.exit(1)
    prompts = [item["prompt"] for item in prompt_list]
    return prompts


def validate_function_list(content: Any) -> None:
    if not isinstance(content, list):
        raise InvalidFileError("Function definition file does not contain list")
    function_names: list[str] = []
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            raise InvalidFileError(
                f"Item {index} in function list is not a dict"
            )
        if len(item) != 4:
            raise InvalidFileError(
                f"Item {index} of function list has {len(item)} keys, expected 4"
            )
        valid_keys = ["name", "description", "parameters", "returns"]
        for key, value in item.items():
            if key not in valid_keys:
                raise InvalidFileError(
                    f"Item {index} of function list has double / invalid key {key}"
                )
            else:
                valid_keys.remove(key)
                if (key == "name" or key == "description"):
                    if not isinstance(value, str):
                        raise InvalidFileError(
                            f'Value of key "{key}" in item {index} of function list is not a string'
                        )
                    if len(value) < 1:
                        raise InvalidFileError(
                            f'Value of key "{key}" in item {index} of function list is empty'
                        )
                    if key == "name":
                        if value in function_names:
                            raise InvalidFileError(
                                f'Double function name "{value}" in item {index} of function list'
                            )
                        else:
                            function_names.append(value)
                elif key == "parameters":
                    if not isinstance(value, dict):
                        raise InvalidFileError(
                            f'Value of key "{key}" in item {index} of function list is not a string'
                        )
                    for param_key, param_value in value.items():
                        try:
                            check_type(param_value)
                        except InvalidFileError as e:
                            print(f'Error in function definition file: parameter {param_key} in item {index} of function list has invalid type: {e}', file=sys.stderr)
                            sys.exit(1)
                elif key == "returns":
                    try:
                        check_type(value)
                    except InvalidFileError as e:
                        print(f'Error in function definition file: returns of item {index} of function list have invalid type: {e}', file=sys.stderr)
                        sys.exit(1)


def check_type(type_dict: Any) -> None:
    valid_types = ["null", "number", "integer", "boolean", "string"]
    if not isinstance(type_dict, dict):
        raise InvalidFileError(
            "Type is not a dict"
        )
    if len(type_dict) != 1:
        raise InvalidFileError(
            f"Type dict has {len(type_dict)} keys, expected 1"
        )
    for key, value in type_dict.items():
        if key != "type":
            raise InvalidFileError(
                'Other key than "type" found in type dict'
            )
        if not isinstance(value, str):
            raise InvalidFileError(
                "Type is not a string"
            )
        if len(value) < 1:
            raise InvalidFileError(
                "Type is empty"
            )
        if value not in valid_types:
            raise InvalidFileError(
                f'Type "{value}" is invalid'
            )


def parsing_functions(file_name: str) -> dict[str, dict[str, str]]:
    functions = {}
    functions_list = read_file(file_name)
    try:
        validate_function_list(functions_list)
    except InvalidFileError as e:
        print(f"Error in function definition file: {e}", file=sys.stderr)
        sys.exit(1)
    for item in functions_list:
        name = item["name"]
        parameters = {}
        for key, value in item["parameters"].items():
            ptype = value["type"]
            parameters[key] = ptype
        functions[name] = parameters
    return functions
