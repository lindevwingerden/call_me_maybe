from llm_sdk import Small_LLM_Model
import numpy as np
from parsing import read_file, parsing_prompts, parsing_functions


def create_instructions(prompts: str, prompt_nb: int, functions_list: str) -> str:
    instructions = (
        "For the following prompt, choose the correct function name "
        + "from the functions list, select the parameters that belong to "
        + "that function and fill in the correct value for each parameter.\n\n"
        + "Example:\n"
        + '{ "prompt": "What is the sum of 2 and 3?", "name": '
        + '"fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0} }\n\n'
        + "Functions list:\n"
        + str(functions_list) + "\n\n"
        + "Prompt:\n"
        + '{"prompt": "' + prompts[prompt_nb] + '", "name": "'
    )
    return instructions


def main() -> None:
    llm = Small_LLM_Model()
    prompts = parsing_prompts("data/input/function_calling_tests.json")
    functions_list = read_file("data/input/functions_definition.json")
    functions = parsing_functions("data/input/functions_definition.json")

    instructions = create_instructions(prompts, 4, functions_list)
    total = instructions
    id_nb = 0
    function_name_ids = []
    ids_total = llm.encode(total)[0].tolist()
    ids_functions = [llm.encode(name + '"')[0].tolist() for name in functions]
    while llm.encode('"') not in function_name_ids:
        logits = np.array(llm.get_logits_from_input_ids(ids_total))
        masked = np.full_like(logits, -np.inf)
        for function in ids_functions:
            token_id = function[id_nb]
            masked[token_id] = logits[token_id]
        next_id = np.argmax(masked)
        function_name_ids.append(next_id)
        ids_total.append(next_id)
        for function in ids_functions:
            if function[id_nb] != next_id:
                ids_functions.remove(function)
        id_nb = id_nb + 1
    function_name = llm.decode(function_name_ids)
    function_name = function_name.split('"')[0]
    instructions = instructions + function_name + '", "parameters": {'
    total = instructions
    print(total)
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
