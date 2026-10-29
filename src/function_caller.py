from llm_sdk import Small_LLM_Model
import numpy as np
from parsing import read_file, parsing_prompts, parsing_functions


class FunctionCaller:
    def __init__(
            self,
            path_prompts: str,
            path_functions: str
    ):
        self._llm = Small_LLM_Model()
        self.prompts = parsing_prompts(path_prompts)
        self._functions_total = read_file(path_functions)
        self._functions_crop = parsing_functions(path_functions)
        self._ids_valid_functions = []
        self._ids_valid_numbers = []
        self._ids_valid_bools = []
        self._instructions = (
            "For the following prompt, choose the correct function name "
            + "from the functions list, select the parameters that belong to "
            + "that function and fill in the correct value for each parameter.\n\n"
            + "Example:\n"
            + '{ "prompt": "What is the sum of 2 and 3?", "name": '
            + '"fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0} }\n\n'
            + "Functions list:\n"
            + str(self._functions_total) + "\n\n"
            + "Prompt:\n"
        )

    def get_ids_valid_numbers(self) -> list[int]:
        vocab_file = read_file(self._llm.get_path_to_vocab_file())
        chars_valid_numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "}"]
        ids_valid_numbers = {}
        for key, value in vocab_file.items():
            valid = True
            for char in key:
                if char not in chars_valid_numbers:
                    valid = False
                    break
            if valid:
                ids_valid_numbers[key] = value
        return ids_valid_numbers.values()
    
    def create_instructions_for_prompt(self, prompt: str) -> str:
        instructions = self._instructions + '{"prompt": "' + prompt
        return instructions

    def get_function_name(self, instructions: str) -> tuple[str, str]:
        llm = self._llm
        if len(self._ids_valid_functions) < 1:
            self._ids_valid_functions = [llm.encode(name + '"')[0].tolist() for name in self._functions_crop]
        instructions = instructions  + '", "name": "'
        id_nb = 0
        ids_function_name = []
        ids_total = llm.encode(instructions)[0].tolist()
        ids_functions = self._ids_valid_functions.copy()
        while '"' not in llm.decode(ids_function_name):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            for function in ids_functions:
                token_id = function[id_nb]
                masked[token_id] = logits[token_id]
            next_id = int(np.argmax(masked))
            ids_function_name.append(next_id)
            ids_total.append(next_id)
            for function in ids_functions:
                if function[id_nb] != next_id:
                    ids_functions.remove(function)
            id_nb = id_nb + 1
        function_name = llm.decode(ids_function_name)
        function_name = function_name.split('"')[0]
        instructions = instructions + function_name
        return function_name, instructions
    
    def get_parameter_number(self, instructions: str) -> tuple[float, str]:
        llm = self._llm
        if len(self._ids_valid_numbers) < 1:
            self._ids_valid_numbers = self.get_ids_valid_numbers()
        ids_parameter = []
        ids_total = llm.encode(instructions)[0].tolist()
        while (
            "," not in llm.decode(ids_parameter)
            and "}" not in llm.decode(ids_parameter)
            and len(ids_parameter) < 12
        ):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            for valid_id in self._ids_valid_numbers:
                masked[valid_id] = logits[valid_id]
            next_id = int(np.argmax(masked))
            ids_parameter.append(next_id)
            ids_total.append(next_id)
        parameter = llm.decode(ids_parameter)
        if "," in parameter:
            parameter = parameter.split(",")[0]
        if "}" in parameter:
            parameter = parameter.split("}")[0]
        instructions = instructions + parameter
        parameter_float = float(parameter)
        return parameter_float, instructions

    def get_parameter_bool(self, instructions: str) -> tuple[bool, str]:
        llm = self._llm
        if len(self._ids_valid_bools) < 1:
            self._ids_valid_bools = [llm.encode(boolean)[0].tolist() for boolean in ["True,", "False,"]]
        id_nb = 0
        ids_parameter = []
        ids_total = llm.encode(instructions)[0].tolist()
        ids_bools = self._ids_valid_bools.copy()
        while "," not in llm.decode(ids_parameter):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            for item in ids_bools:
                token_id = item[id_nb]
                masked[token_id] = logits[token_id]
            next_id = int(np.argmax(masked))
            ids_parameter.append(next_id)
            ids_total.append(next_id)
            for item in ids_bools:
                if item[id_nb] != next_id:
                    ids_bools.remove(item)
            id_nb = id_nb + 1
        parameter = llm.decode(ids_parameter)
        parameter = parameter.split(",")[0]
        instructions = instructions + parameter
        parameter_bool = bool(parameter)
        return parameter_bool, instructions

    def get_parameter_string(self, instructions: str) -> tuple[str, str]:
        llm = self._llm
        instructions = instructions + '"'
        ids_parameter = []
        ids_total = llm.encode(instructions)[0].tolist()
        while (
            '"' not in llm.decode(ids_parameter)
            and len(ids_parameter) < 50 # is deze extra nog nodig of niet?
        ):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            next_id = int(np.argmax(logits))
            ids_parameter.append(next_id)
            ids_total.append(next_id)
        parameter = llm.decode(ids_parameter)
        parameter = parameter.split('"')[0]
        instructions = instructions + parameter
        return parameter, instructions

    def get_parameters(self, instructions: str, function_name: str) -> dict:
        instructions = instructions + '", "parameters": {'
        parameters = {}
        valid_parameters = self._functions_crop[function_name]
        for key, value in valid_parameters.items():
            instructions = instructions + f'"{key}": '
            if value == "number":
                parameter, instructions = self.get_parameter_number(instructions)
            elif value == "boolean":
                parameter, instructions = self.get_parameter_bool(instructions)
            else:
                parameter, instructions = self.get_parameter_string(instructions)
            parameters[key] = parameter
            instructions = instructions + ", "
        return parameters

    def append_output(self, output: list, prompt: str, function_name: str, parameters: dict) -> None:
        current = {}
        current["prompt"] = prompt
        current["name"] = function_name
        current["parameters"] = parameters
        output.append(current)

    def output_to_file(self) -> None:
        pass # nog maken met json dump


def main() -> None:
    fc = FunctionCaller("data/input/function_calling_tests.json", "data/input/functions_definition.json")
    output = []
    for prompt in fc.prompts:
        instructions = fc.create_instructions_for_prompt(prompt)
        function_name, instructions = fc.get_function_name(instructions)
        parameters = fc.get_parameters(instructions, function_name)
        fc.append_output(output, prompt, function_name, parameters)
        print(f"Current output: {output}\n")
    print(output)

    # prompt = fc.prompts[2]
    # instructions = fc.create_instructions_for_prompt(prompt)
    # function_name, instructions = fc.get_function_name(instructions)
    # parameters = fc.get_parameters(instructions, function_name)
    # fc.append_output(output, prompt, function_name, parameters)
    # print(f"Output: {output}\n")


if __name__ == "__main__":
    main()
