from llm_sdk import Small_LLM_Model
import numpy as np


def main() -> None:
    llm = Small_LLM_Model()
    prompt = "Happy birthday to"

    for _ in range(5):
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


if __name__ == "__main__":
    main()