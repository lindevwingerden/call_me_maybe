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
    # functions = read_file("data/input/functions_definition.json")

    for prompt in prompts:
        output = '\t{\n\t\t"prompt": "' + prompt + '",\n\t\t"name": "'
        ids = llm.encode(output)[0].tolist()
        logits = llm.get_logits_from_input_ids(ids)
        logits = np.array(logits)
        next_id = np.argmax(logits)
        next_word = llm.decode(next_id)
        output = output + next_word
        print(output)


if __name__ == "__main__":
    main()