import json

def read_jsonl_to_array(file_path):
    """
    Reads a .jsonl file and converts it into a list of dictionaries.

    Args:
    - file_path (str): The path to the .jsonl file.

    Returns:
    - List[dict]: A list of dictionaries, each representing a JSON object from the .jsonl file.
    """
    data_array = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data_array.append(json.loads(line))
    return data_array

def entry_to_prompt(entry):
    """
    Converts an entry into a prompt.

    Args:
    - entry (dict): A dictionary representing a JSON data object.

    Returns:
    - str: A prompt.
    """
    prompt = f"Context: \"{entry['context']}\"\n"
    for i, choice in enumerate(entry['choices']):
        prompt += f"{i}: \"{choice}\"\n"
    prompt += f"Best Continuation: {entry['answer']}\n"
    return prompt

# Read in data
fewshot_examples = read_jsonl_to_array('./data/fewshot_examples.jsonl')
data_train = read_jsonl_to_array('./data/data_train.jsonl')
data_val = read_jsonl_to_array('./data/data_val.jsonl')

# Build up examples string
example_str = ""
for example in fewshot_examples:
    example_str += entry_to_prompt(example) + "\n"

def dataset_to_prompts(data_array):
    """
    Converts a dataset into a list of prompts.

    Args:
    - data_array (List[dict]): A list of dictionaries, each representing a JSON data object.

    Returns:
    - List[Tuple[str, str]]: A list of prompts and their corresponding answers.
    """
    prompts = []
    for entry in data_array:
        masked_entry = {
            'context': entry['context'],
            'choices': entry['choices'],
            'answer': ""
        }
        prompt = example_str + entry_to_prompt(masked_entry)
        prompts.append((prompt[:-1], entry['answer']))
    return prompts

prompts_train = dataset_to_prompts(data_train)
prompts_val = dataset_to_prompts(data_val)
