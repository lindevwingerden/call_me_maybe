from llm_sdk import Small_LLM_Model
import numpy as np
from parsing import read_file, parsing_prompts


def llm_test(llm, prompt) -> None:
    ids = llm.encode(prompt)[0].tolist()
    print(ids)

    logits = llm.get_logits_from_input_ids(ids)
    logits = np.array(logits)
    print(logits.shape)

    next_id = np.argmax(logits)
    next_word = llm.decode(next_id)
    print(next_id)
    print(next_word)
    prompt = prompt + " " + next_word
    print(f"{prompt}\n")


def main() -> None:
    llm = Small_LLM_Model()
    prompts = parsing_prompts("data/input/function_calling_tests.json")
    functions = read_file("data/input/functions_definition.json")
    instructions = (
        "For the following prompt, choose the correct function name "
        + "from the functions list, select the parameters that belong to "
        + "that function and fill in the correct value for each parameter.\n\n"
        + "Example:\n"
        + '{ "prompt": "What is the sum of 2 and 3?", "name": '
        + '"fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0} }\n\n'
        + "Functions list:\n"
        + str(functions) + "\n\n"
        + '{"prompt": "' + prompts[2] + '", "name": "'
    )  # put instructions in a text file maybe? and read it

    output = instructions
    total = output
    next_letters = ""
    function_name = ""
    while '"' not in function_name:
        ids = llm.encode(total)[0].tolist()
        logits = llm.get_logits_from_input_ids(ids)
        logits = np.array(logits)
        next_id = np.argmax(logits)
        next_letters = llm.decode(next_id)
        function_name = function_name + next_letters
        total = total + next_letters
    # nog iets met if > 1 in next_letters dat je dan iets anders moet doen om te splitten
    function_name = function_name.split('"')[0]
    output = output + function_name + '", "parameters": {'
    total = output
    next_letters = ""
    parameters = ""
    while '}' not in next_letters:
        ids = llm.encode(total)[0].tolist()
        logits = llm.get_logits_from_input_ids(ids)
        logits = np.array(logits)
        next_id = np.argmax(logits)
        next_letters = llm.decode(next_id)
        parameters = parameters + next_letters
        total = total + next_letters
    parameters = parameters.split('}')[0]
    output = output + function_name + "} },"
    total = output
    print(total)
    # hierna de output er nog uithalen door te splitten op iets
    # misschien nog iets toevoegen voor de output zodat dat makkelijker is

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
