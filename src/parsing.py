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
    except OSError as e:
        print(
            f"Error: could not read {file_name}: {e}",
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
                f"Item {index} is not a dict"
            )
        if len(item) != 1:
            raise InvalidFileError(
                f"Item {index} has {len(item)} keys, expected 1"
            )
        for key, value in item.items():
            if key != "prompt":
                raise InvalidFileError(
                    f'Other key than "prompt" found in item {index}'
                )
            if not isinstance(value, str):
                raise InvalidFileError(
                    f"Prompt in item {index} is not a string"
                )
            if len(value) < 1:
                raise InvalidFileError(
                    f"Empty prompt in item {index}"
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
        raise InvalidFileError(
            "Function definition file does not contain list"
        )
    if len(content) < 1:
        raise InvalidFileError(
            "Function list is empty"
        )
    function_names: list[str] = []
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            raise InvalidFileError(
                f"Item {index} is not a dict"
            )
        valid_keys = {"name", "description", "parameters", "returns"}
        if item.keys() != valid_keys:
            raise InvalidFileError(
                f'Item {index} has invalid keys;'
                ' expected "name", "description", "parameters", "returns"'
            )
        try:
            check_name(item["name"], function_names)
            function_names.append(item["name"])
        except InvalidFileError as e:
            raise InvalidFileError(
                f'Item {index} has invalid name: {e}'
            ) from e
        try:
            check_description(item["description"])
        except InvalidFileError as e:
            raise InvalidFileError(
                f'Item {index} has invalid description: {e}'
            ) from e
        try:
            check_parameters(item["parameters"])
        except InvalidFileError as e:
            raise InvalidFileError(
                f'Item {index} has an invalid parameter: {e}'
            ) from e
        try:
            check_type(item["returns"])
        except InvalidFileError as e:
            raise InvalidFileError(
                f"Returns of item {index} have invalid type: {e}"
            ) from e


def check_name(function_name: Any, function_names_list: list[str]) -> None:
    if not isinstance(function_name, str):
        raise InvalidFileError(
            'Function name is not a string'
        )
    if len(function_name) < 1:
        raise InvalidFileError(
            'Function name is empty'
        )
    if function_name in function_names_list:
        raise InvalidFileError(
            'Function name already exists'
        )


def check_description(description: Any) -> None:
    if not isinstance(description, str):
        raise InvalidFileError(
            'Description is not a string'
        )
    if len(description) < 1:
        raise InvalidFileError(
            'Description is empty'
        )


def check_parameters(parameters: Any) -> None:
    if not isinstance(parameters, dict):
        raise InvalidFileError(
            'Parameters are not a dict'
        )
    for param_key, param_value in parameters.items():
        if len(param_key) < 1:
            raise InvalidFileError(
                'Parameter name is empty'
            )
        try:
            check_type(param_value)
        except InvalidFileError as e:
            raise InvalidFileError(
                f"Parameter {param_key} has invalid type: {e}"
            ) from e


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
        if value not in valid_types:
            raise InvalidFileError(
                f'Type "{value}" is invalid'
            )


def parsing_functions(file_name: str) -> dict[str, dict[str, str]]:
    functions: dict[str, dict[str, str]] = {}
    functions_list = read_file(file_name)
    try:
        validate_function_list(functions_list)
    except InvalidFileError as e:
        print(f"Error in function definition file: {e}", file=sys.stderr)
        sys.exit(1)
    for item in functions_list:
        name = item["name"]
        parameters: dict[str, str] = {}
        for key, value in item["parameters"].items():
            ptype = value["type"]
            parameters[key] = ptype
        functions[name] = parameters
    return functions
