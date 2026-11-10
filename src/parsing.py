import sys
import json
from typing import Any


class InvalidFileError(ValueError):
    """Raised when the contents of an input file have an invalid format."""
    pass


def read_file(file_name: str) -> Any:
    """Read a JSON file and return its parsed contents.

    Exits the program with an error message when the file cannot be read
    or does not contain valid JSON.

    Args:
        file_name: Path to the JSON file.

    Returns:
        The parsed JSON content (for example a list or a dict).
    """
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
    """Check that the content of a prompt file has a valid format.

    A valid prompt file contains a list of dicts, where every dict has
    exactly one key, "prompt", with a non-empty string as its value.

    Args:
        content: The parsed JSON content of the prompt file.

    Raises:
        InvalidFileError: If the content does not have the expected format.
    """
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
    """Read and validate a prompt file and return its prompts.

    Exits the program with an error message when the file is invalid.

    Args:
        file_name: Path to the prompt file.

    Returns:
        A list with the text of every prompt, in file order.
    """
    prompt_list = read_file(file_name)
    try:
        validate_prompt_list(prompt_list)
    except InvalidFileError as e:
        print(f"Error in prompt file: {e}", file=sys.stderr)
        sys.exit(1)
    prompts = [item["prompt"] for item in prompt_list]
    return prompts


def validate_function_list(content: Any) -> None:
    """Check that the content of a function definition file is valid.

    A valid file contains a non-empty list of dicts. Every dict has
    exactly the keys "name", "description", "parameters" and "returns",
    and every function name is unique.

    Args:
        content: The parsed JSON content of the function definition file.

    Raises:
        InvalidFileError: If the content does not have the expected format.
    """
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
    """Check that a function name is a unique, non-empty string.

    Args:
        function_name: The value of the "name" key of a function.
        function_names_list: The names of the functions checked so far.

    Raises:
        InvalidFileError: If the name is not a string, is empty or is
            already used by another function.
    """
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
    """Check that a function description is a non-empty string.

    Args:
        description: The value of the "description" key of a function.

    Raises:
        InvalidFileError: If the description is not a string or is empty.
    """
    if not isinstance(description, str):
        raise InvalidFileError(
            'Description is not a string'
        )
    if len(description) < 1:
        raise InvalidFileError(
            'Description is empty'
        )


def check_parameters(parameters: Any) -> None:
    """Check that the parameters of a function have a valid format.

    The parameters must be a dict that maps non-empty parameter names to
    a valid type dict (see check_type).

    Args:
        parameters: The value of the "parameters" key of a function.

    Raises:
        InvalidFileError: If the parameters are not a dict, a parameter
            name is empty or a parameter has an invalid type.
    """
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
    """Check that a type dict has a valid format.

    A valid type dict has exactly one key, "type", with one of the
    supported types as its value: "null", "number", "integer",
    "boolean" or "string".

    Args:
        type_dict: The type dict of a parameter or of a return value.

    Raises:
        InvalidFileError: If the type dict has an invalid format or
            contains an unsupported type.
    """
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
    """Read and validate a function definition file.

    Exits the program with an error message when the file is invalid.

    Args:
        file_name: Path to the function definition file.

    Returns:
        A dict that maps every function name to a dict of its parameters,
        which in turn maps every parameter name to its type, for example
        {"fn_add_numbers": {"a": "number", "b": "number"}}.
    """
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
