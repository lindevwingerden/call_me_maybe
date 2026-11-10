*This project has been created as part of the 42 curriculum by lvan-win.*

# Call Me Maybe

## Description

A function calling tool that translates natural language prompts into structured function calls. For a prompt like *"What is the sum of 2 and 3?"* it does not answer `5`, but selects the right function and its arguments:
 
```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```
 
It uses the small **Qwen/Qwen3-0.6B** model together with **constrained decoding**, which guarantees valid JSON that matches the function definitions.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).
 
```bash
make install    # uv sync: creates the virtual environment and installs dependencies
make run        # uv run python -m src
make debug      # run with pdb
make lint       # flake8 + mypy
make clean      # remove caches
```

or, if you want to run the program directly:

```bash
uv sync
uv run python -m src
```

### Example usage

```bash
# With default paths:
# --functions_definition data/input/functions_definition.json
# --input data/input/function_calling_tests.json
# --output data/output/function_calling_results.json
uv run python -m src
 
# Custom paths:
uv run python -m src
    --functions_definition data/input/functions_definition.json
    --input data/input/function_calling_tests.json
    --output data/output/function_calls.json
```

You can look at the files in data/input/ for examples of valid formatted input files.

## Algorithm explanation
 
The output JSON is built step by step. Fixed parts (such as `{"prompt": ..., "name": "`) are added by the program; the model only generates the parts where a choice must be made. At each generation step, the logits of all tokens that would lead to invalid output are set to `-inf`, and the token with the highest remaining logit is chosen (greedy decoding).
 
- **Function name:** every function name (plus a closing `"`) is tokenized in advance. At each step only the next token of the names that still match the generated tokens is allowed, so the model can only choose an existing name.
- **Numbers / integers:** number tokens (only digits, `-` and `.`) and stop tokens (`,`, `}`) are collected once from the vocabulary file. A number token is only allowed if the text stays a valid start of a number (one `-` at the start, at most one `.`, no `.` for integers). Stop tokens are only allowed once the text is a complete number.
- **Booleans:** only tokens that continue to `true,` or `false,` are allowed.
- **Strings:** the model generates freely until it produces a closing quote (with a maximum length).
- **Null:** no generation needed; `null` is filled in directly.

## Design decisions
 
- Input files are validated by hand in `parsing.py` (structure, keys, types, duplicate names), with clear error messages instead of crashes.
- `FunctionCaller` is a pydantic model that validates its paths; the model and token lists are private attributes.
- Valid token sets are computed once and cached, to keep each generation step fast.
- Both the `parsing.py` file and the `FunctionCaller` class have extensive error handling. If the input files contain an empty string as a prompt or a function name for example, my program will give an error. I rather point the user towards how to work with the program than to let the program work with nonsensical data.


## Performance analysis
 
- Accuracy: 11 / 11 prompts correct (function + arguments).
- JSON validity: 100% by construction (constrained decoding).
- Speed: 4 minutes 22 for 11 prompts: 23,8 seconds per prompt.
- Reliability: sometimes a bit too literal. For example, for the prompt "Replace all vowels in 'Programming is fun' with asterisks" it will replace all vowels with the word 'asterisk', which is technically not wrong, but it is possible that something different is meant here.

## Challenges faced

In the beginning, it was a bit hard to understand what I actually had to do in this project. For example, at first I thought the LLM actually needed to call functions, instead of outputting a JSON file.

Also, the error handling was more work than I expected, because there are so many things that could go wrong in the input files.

## Testing strategy

At first, I tested with the provided examples. Later, I also tested for invalid and missing input files, for invalid JSON in the input files, for vague prompts and functions and negative and really large numbers as parameters.

## Resources

* [Aidan Cooper: A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
* [JSON Data Types](https://www.geeksforgeeks.org/javascript/json-data-types/)

## AI Usage

AI (Claude) was used as a tutor. If I had questions, sometimes I asked my peers, and sometimes I asked Claude to explain concepts (LLMs, logits, constrained decoding, uv and virtual environments). I also asked Claude to help debug environment and mypy/flake8 errors, to review error handling and code, and to write the first drafts of docstrings and this README.
