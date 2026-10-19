from llm_sdk.llm_sdk import Small_LLM_Model
import data


def main() -> None:
    llm = Small_LLM_Model()
    prompt = "Hello"
    ids = llm.encode(prompt)[0]
    logits = llm.get_logits_from_input_ids(ids)
    print(logits)


if __name__ == "__main__":
    main()