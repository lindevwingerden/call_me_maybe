import sys
from src.function_caller import FunctionCaller


def main() -> None:
    path_functions = "data/input/functions_definition.json"
    path_input = "data/input/function_calling_tests.json"
    path_output = "data/output/function_calling_results.json"
    for i in range(1, len(sys.argv) - 1):
        if sys.argv[i] == "--functions_definition":
            path_functions = sys.argv[i + 1]
        if sys.argv[i] == "--input":
            path_input = sys.argv[i + 1]
        if sys.argv[i] == "--output":
            path_output = sys.argv[i + 1]
    fc = FunctionCaller(path_functions, path_input, path_output)
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
