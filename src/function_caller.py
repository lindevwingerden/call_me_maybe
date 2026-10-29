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
        self._prompts = parsing_prompts(path_prompts)
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
    
    def create_instructions_for_prompt(self, prompt_nb: int) -> str:
        instructions = self._instructions + '{"prompt": "' + self._prompts[prompt_nb]
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
        while llm.encode('"') not in ids_function_name:
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            for function in ids_functions:
                token_id = function[id_nb]
                masked[token_id] = logits[token_id]
            next_id = np.argmax(masked)
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
    
    def get_parameter_number(self, instructions: str) -> tuple[str, str]:
        llm = self._llm
        print("in get_parameter_number: getting valid numbers....")
        if len(self._ids_valid_numbers) < 1:
            self._ids_valid_numbers = self.get_ids_valid_numbers()
        ids_parameter = []
        ids_total = llm.encode(instructions)[0].tolist()
        print("in get_parameter_number: going into while loop....")
        while (
            llm.encode(",") not in ids_parameter
            and llm.encode("}") not in ids_parameter
            and len(ids_parameter) < 12
        ):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            for valid_id in self._ids_valid_numbers:
                masked[valid_id] = logits[valid_id]
            next_id = np.argmax(masked)
            ids_parameter.append(next_id)
            ids_total.append(next_id)
            print(llm.decode(ids_parameter))
        parameter = llm.decode(ids_parameter)
        if "," in parameter:
            parameter = parameter.split(",")[0]
        if "}" in parameter:
            parameter = parameter.split("}")[0]
        instructions = instructions + parameter
        print("in get_parameter_number: end of function")
        return parameter, instructions

    def get_parameter_bool(self, instructions: str) -> tuple[str, str]:
        llm = self._llm
        if len(self._ids_valid_bools) < 1:
            self._ids_valid_bools = [llm.encode(boolean)[0].tolist() for boolean in ["True,", "False,"]]
        id_nb = 0
        ids_parameter = []
        ids_total = llm.encode(instructions)[0].tolist()
        ids_bools = self._ids_valid_bools.copy()
        while llm.encode(",") not in ids_parameter:
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            masked = np.full_like(logits, -np.inf)
            for item in ids_bools:
                token_id = item[id_nb]
                masked[token_id] = logits[token_id]
            next_id = np.argmax(masked)
            ids_parameter.append(next_id)
            ids_total.append(next_id)
            for item in ids_bools:
                if item[id_nb] != next_id:
                    ids_bools.remove(item)
            id_nb = id_nb + 1
        parameter = llm.decode(ids_parameter)
        parameter = parameter.split(",")[0]
        instructions = instructions + parameter
        return parameter, instructions

    def get_parameter_string(self, instructions: str) -> tuple[str, str]:
        llm = self._llm
        instructions = instructions + '"'
        ids_parameter = []
        ids_total = llm.encode(instructions)[0].tolist()
        while (
            llm.encode('"') not in ids_parameter
            and len(ids_parameter) < 50
        ):
            logits = np.array(llm.get_logits_from_input_ids(ids_total))
            next_id = np.argmax(logits)
            ids_parameter.append(next_id)
            ids_total.append(next_id)
        parameter = llm.decode(ids_parameter)
        instructions = instructions + parameter
        parameter = '"' + parameter
        return parameter, instructions

    def get_parameters(self, instructions: str, function_name: str) -> str:
        instructions = instructions + '", "parameters": {'
        parameters = ""
        parameters_dict = self._functions_crop[function_name]
        print(f"in get_parameters: parameters_dict: {parameters_dict}")
        for key, value in parameters_dict.items():
            print(f"in get_parameters: key, value: {key}, {value}")
            parameters = parameters + f'"{key}": '
            instructions = instructions + f'"{key}": ' # aan al deze dubbele dingen moet ik nog wat doen
            if value == "number":
                parameter, instructions = self.get_parameter_number(instructions)
                parameters = parameters + parameter
                print(f"in get_parameters: parameter: {parameter}")
            elif value == "boolean":
                parameter, instructions = self.get_parameter_bool(instructions)
                parameters = parameters + parameter
            else:
                parameter, instructions = self.get_parameter_string(instructions)
                parameters = parameters + parameter
            parameters = parameters + ", "
            instructions = instructions + ", "
            print(parameter)
        parameters = parameters[:-2] + "}"
        return parameters

    def get_output(self, function_name: str, parameters: str) -> str:
        # Misschien later nog maken dat ie m direct naar de output file stuurt, en dan -> None
        # Nee daar maak ik een losse functie voor als de hele output compleet is


def main() -> None:
    fc = FunctionCaller("data/input/function_calling_tests.json", "data/input/functions_definition.json")

    instructions = fc.create_instructions_for_prompt(0)
    function_name, instructions = fc.get_function_name(instructions)
    print(function_name)
    parameters = fc.get_parameters(instructions, function_name)
    print(instructions, parameters)



    # next_letters = ""
    # parameters = ""
    # while '}' not in next_letters:
    #     ids = llm.encode(total)[0].tolist()
    #     logits = llm.get_logits_from_input_ids(ids)
    #     logits = np.array(logits)
    #     next_id = np.argmax(logits)
    #     next_letters = llm.decode(next_id)
    #     parameters = parameters + next_letters
    #     total = total + next_letters
    # parameters = parameters.split('}')[0]
    # output = output + function_name + "} },"
    # total = output
    # print(total)


    # for prompt in prompts:
        # output = '{"prompt": "' + prompt + '","name": "'
        # total = instructions + output
        # ids = llm.encode(total)[0].tolist()
        # logits = llm.get_logits_from_input_ids(ids)
        # logits = np.array(logits)
        # next_id = np.argmax(logits)
        # next_word = llm.decode(next_id)
        # total = total + next_word
        # print(total)
        # print("\n\n")
        # output = output + next_word
        # print(output)
        # print("\n\n")


if __name__ == "__main__":
    main()
