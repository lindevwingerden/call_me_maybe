import sys
from typing import NoReturn

try:
    from pydantic import ValidationError
    from src.function_caller import FunctionCaller
except ModuleNotFoundError as e:
    print(
        f'Error: missing module "{e.name}"\n'
        'Install the dependencies with "make install" or "uv sync"'
        ' and run the program with "make", "make all", "make run"'
        ' or "uv run python -m src"', file=sys.stderr
    )
    sys.exit(1)


def argument_error(message: str) -> NoReturn:
    print(
        f"Argument error: {message}\n"
        "Usage: uv run python -m src [--functions_definition"
        " <function_definition_file>] [--input <input_file>]"
        " [--output <output_file>]", file=sys.stderr
    )
    sys.exit(1)


def main() -> None:
    path_functions = "data/input/functions_definition.json"
    path_input = "data/input/function_calling_tests.json"
    path_output = "data/output/function_calling_results.json"
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--functions_definition":
            if (
                i < len(sys.argv) - 1
                and not sys.argv[i + 1].startswith("--")
            ):
                path_functions = sys.argv[i + 1]
                i = i + 2
            else:
                argument_error("Missing value for --functions_definition")
        elif sys.argv[i] == "--input":
            if (
                i < len(sys.argv) - 1
                and not sys.argv[i + 1].startswith("--")
            ):
                path_input = sys.argv[i + 1]
                i = i + 2
            else:
                argument_error("Missing value for --input")
        elif sys.argv[i] == "--output":
            if (
                i < len(sys.argv) - 1
                and not sys.argv[i + 1].startswith("--")
            ):
                path_output = sys.argv[i + 1]
                i = i + 2
            else:
                argument_error("Missing value for --output")
        else:
            argument_error(f"Unknown argument: {sys.argv[i]}")
    try:
        fc = FunctionCaller(
            path_functions=path_functions,
            path_input=path_input,
            path_output=path_output
        )
    except ValidationError as e:
        print(f"Validation error for FunctionCaller: {e}", file=sys.stderr)
        sys.exit(1)
    fc.call_functions_from_prompts()

    # output: list[dict[str, str | dict[str, ParameterType]]] = []
    # for prompt in fc.prompts:
    #     instructions = fc.create_instructions_for_prompt(prompt)
    #     function_name, instructions = fc.get_function_name(instructions)
    #     parameters = fc.get_parameters(instructions, function_name)
    #     fc.append_output(output, prompt, function_name, parameters)
    #     print(f"Current output: {output}\n")
    # print(output)

    # prompt = fc.prompts[2]
    # instructions = fc.create_instructions_for_prompt(prompt)
    # function_name, instructions = fc.get_function_name(instructions)
    # parameters = fc.get_parameters(instructions, function_name)
    # fc.append_output(output, prompt, function_name, parameters)
    # print(f"Output: {output}\n")


if __name__ == "__main__":
    main()
