import numpy as np
import json
import sys
from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.parsing import read_file, parsing_prompts, parsing_functions


ParameterType = float | bool | str


class FunctionCaller:
    def __init__(
            self,
            path_functions: str,
            path_input: str,
            path_output: str
    ):
        self._llm = Small_LLM_Model()
        self._functions_crop = parsing_functions(path_functions)
        self._functions_total = read_file(path_functions)
        self._prompts = parsing_prompts(path_input)
        self._path_output = path_output
        self._ids_valid_functions: list[list[int]] = []
        self._ids_valid_numbers: list[int] = []
        self._ids_valid_stops: list[int] = []
        self._ids_valid_ints: list[int] = []
        self._ids_valid_bools: list[list[int]] = []
        self._instructions = (
            "For the following prompt, choose the correct function name "
            + "from the functions list, select the parameters that belong to "
            + "that function and fill in the correct value for each parameter."
            + "\n\nExample:\n"
            + '{ "prompt": "What is the sum of 2 and 3?", "name": '
            + '"fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0} }\n\n'
            + "Functions list:\n"
            + str(self._functions_total) + "\n\n"
            + "Prompt:\n"
        )

    def call_functions_from_prompts(self) -> None:
        output: list[dict[str, str | dict[str, ParameterType]]] = []
        for prompt in self._prompts:
            instructions = self.create_instructions_for_prompt(prompt)
            function_name, instructions = self.get_function_name(instructions)
            parameters = self.get_parameters(instructions, function_name)
            self.append_output(output, prompt, function_name, parameters)
        self.output_to_file(output)

    def get_ids_valid_ints(self) -> tuple[list[int], list[int]]:
        vocab_file = read_file(self._llm.get_path_to_vocab_file())
        chars_valid_numbers = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "-", ",", "}"
        ]
        ids_valid_numbers = {}
        for key, value in vocab_file.items():
            valid = True
            for char in key:
                if char not in chars_valid_numbers:
                    valid = False
                    break
            if valid:
                ids_valid_numbers[key] = value
        return [int(value) for value in ids_valid_numbers.values()]

    def get_ids_valid_numbers(self) -> tuple[list[int], list[int]]:
        vocab_file = read_file(self._llm.get_path_to_vocab_file())
        chars_valid_numbers = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "-", "."
        ]
        chars_valid_stops = [",", "}"]
        ids_valid_numbers = {}
        ids_valid_stops = {}
        for key, value in vocab_file.items():
            if key in chars_valid_stops:
                ids_valid_stops[key] = value
                continue
            valid = True
            for char in key:
                if char not in chars_valid_numbers:
                    valid = False
                    break
            if valid:
                ids_valid_numbers[key] = value
        return (
            [int(value) for value in ids_valid_numbers.values()],
            [int(value) for value in ids_valid_stops.values()]
        )

    def create_instructions_for_prompt(self, prompt: str) -> str:
        instructions = self._instructions + '{"prompt": "' + prompt
        return instructions

    def get_function_name(self, instructions: str) -> tuple[str, str]:
        llm = self._llm
        if len(self._ids_valid_functions) < 1:
            self._ids_valid_functions = [
                llm.encode(name + '"')[0].tolist()
                for name in self._functions_crop
            ]
        instructions = instructions + '", "name": "'
        id_nb = 0
        ids_function_name: list[int] = []
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
            ids_functions = [
                function for function in ids_functions
                if function[id_nb] == next_id
            ]
            id_nb = id_nb + 1
        function_name = llm.decode(ids_function_name)
        function_name = function_name.split('"')[0]
        instructions = instructions + function_name
        return function_name, instructions

    def is_complete_number(self, text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

    def get_parameter_number(self, instructions: str) -> tuple[float, str]:
        llm = self._llm
        if len(self._ids_valid_numbers) < 1:
            numbers, stops = self.get_ids_valid_numbers()
            self._ids_valid_numbers = numbers
            self._ids_valid_stops = stops
        ids_parameter: list[int] = []
        ids_total = llm.encode(instructions)[0].tolist()
        valid_ids = self._ids_valid_numbers
        while (
            "," not in llm.decode(ids_parameter)
            and "}" not in llm.decode(ids_parameter)
            and len(ids_parameter) < 14
        ):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            if self.is_complete_number(llm.decode(ids_parameter)):
                valid_ids = valid_ids + self._ids_valid_stops
            for valid_id in valid_ids:
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

    def get_parameter_int(self, instructions: str) -> tuple[float, str]:
        llm = self._llm
        if len(self._ids_valid_ints) < 1:
            self._ids_valid_ints = self.get_ids_valid_ints()
        ids_parameter: list[int] = []
        ids_total = llm.encode(instructions)[0].tolist()
        valid_ids = self._ids_valid_ints
        while (
            "," not in llm.decode(ids_parameter)
            and "}" not in llm.decode(ids_parameter)
            and len(ids_parameter) < 12
        ):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            for valid_id in valid_ids:
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
        parameter_int = int(parameter)
        return parameter_int, instructions

    def get_parameter_bool(self, instructions: str) -> tuple[bool, str]:
        llm = self._llm
        if len(self._ids_valid_bools) < 1:
            self._ids_valid_bools = [
                llm.encode(boolean)[0].tolist()
                for boolean in ["True,", "False,"]
            ]
        id_nb = 0
        ids_parameter: list[int] = []
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
            ids_bools = [item for item in ids_bools if item[id_nb] == next_id]
            id_nb = id_nb + 1
        parameter = llm.decode(ids_parameter)
        parameter = parameter.split(",")[0]
        instructions = instructions + parameter
        if parameter == "True":
            parameter_bool = True
        else:
            parameter_bool = False
        return parameter_bool, instructions

    def get_parameter_string(self, instructions: str) -> tuple[str, str]:
        llm = self._llm
        instructions = instructions + '"'
        ids_parameter: list[int] = []
        ids_total = llm.encode(instructions)[0].tolist()
        while (
            '"' not in llm.decode(ids_parameter)
            and len(ids_parameter) < 50
        ):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            next_id = int(np.argmax(logits))
            ids_parameter.append(next_id)
            ids_total.append(next_id)
        parameter = llm.decode(ids_parameter)
        parameter = parameter.split('"')[0]
        instructions = instructions + parameter
        return parameter, instructions

    def get_parameters(
            self, instructions: str,
            function_name: str
    ) -> dict[str, ParameterType]:
        instructions = instructions + '", "parameters": {'
        parameters: dict[str, ParameterType] = {}
        valid_parameters = self._functions_crop[function_name]
        for key, value in valid_parameters.items():
            instructions = instructions + f'"{key}": '
            param: ParameterType
            if value == "null":
                param = None
                instructions = instructions + "null"
            elif value == "number":
                param, instructions = self.get_parameter_number(instructions)
            elif value == "integer":
                param, instructions = self.get_parameter_int(instructions)
            elif value == "boolean":
                param, instructions = self.get_parameter_bool(instructions)
            elif value == "string":
                param, instructions = self.get_parameter_string(instructions)
            else:
                pass # nog een goede foutmelding schrijven, of foutmelding in parsing?
                # en misschien nog iets voor als je een value error krijgt in een van deze ifs?
            parameters[key] = param
            instructions = instructions + ", "
        return parameters

    def append_output(
            self,
            output: list[dict[str, str | dict[str, ParameterType]]],
            prompt: str,
            function_name: str,
            parameters: dict[str, ParameterType]
    ) -> None:
        current: dict[str, str | dict[str, ParameterType]] = {}
        current["prompt"] = prompt
        current["name"] = function_name
        current["parameters"] = parameters
        output.append(current)

    def output_to_file(
            self,
            output: list[dict[str, str | dict[str, ParameterType]]]
    ) -> None:
        file_name = Path(self._path_output)
        try:
            file_name.parent.mkdir(parents=True, exist_ok=True)
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
        except PermissionError:
            print(
                f"Error: could not write to {file_name}: permission denied",
                file=sys.stderr
            )
            sys.exit(1)
        except IsADirectoryError:
            print(
                f"Error: could not write to {file_name}:",
                "is a directory, not a file",
                file=sys.stderr
            )
            sys.exit(1)
        except OSError as e:
            print(
                f"Error: could not write to {file_name}: {e}",
                file=sys.stderr
            )
            sys.exit(1)
