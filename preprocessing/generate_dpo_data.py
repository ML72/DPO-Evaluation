import json
import random

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

def write_array_to_json(data_array, file_path):
    """
    Writes a list of dictionaries to a .json file.

    Args:
    - data_array (List[dict]): A list of dictionaries, each representing a JSON object.
    - file_path (str): The path to the .json file to write to.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_array, file, ensure_ascii=False, indent=4)

def random_other(choice):
    """
    Takes in an answer option, and then returns one of the other options randomly.
    """
    possible = [0, 1, 2, 3]
    possible.remove(choice)
    return random.choice(possible)

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

def example_str(fewshot_examples, num=0):
    """
    Converts a random sample from a list of examples into a string.

    Args:
    - fewshot_examples (List[dict]): A list of dictionaries, each representing a JSON data object.

    Returns:
    - str: A string representing the examples.
    """
    random_examples = random.sample(fewshot_examples, num)
    random.shuffle(random_examples)
    example_str = ""
    for example in random_examples:
        example_str += entry_to_prompt(example) + "\n"
    return example_str

def dataset_to_prompts(fewshot_examples, data_array, noise=0.0):
    """
    Converts a dataset into a list of prompts.

    Args:
    - data_array (List[dict]): A list of dictionaries, each representing a JSON data object.
    - noise (float): The probability of flipping a label.

    Returns:
    - List[Tuple[str, str]]: A list of prompts and their corresponding answers.
    """
    if not (0 <= noise <= 1):
        raise ValueError("noise must be between 0 and 1")
    
    # Generate prompts in format (prompt, chosen, rejected)
    prompts = []
    for entry in data_array:
        masked_entry = {
            'context': entry['context'],
            'choices': entry['choices'],
            'answer': ""
        }
        prompt = example_str(fewshot_examples) + entry_to_prompt(masked_entry)
        chosen = f" {entry['answer']}"
        rejected = f" {random_other(entry['answer'])}"
        prompts.append((prompt[:-2], chosen, rejected))
    
    # Flip random labels according to noise; this flips chosen and rejected
    num_flip = int(len(prompts) * noise)
    idx_flip = random.sample(range(len(prompts)), num_flip)
    for idx in idx_flip:
        prompts[idx] = (prompts[idx][0], prompts[idx][2], prompts[idx][1])
    
    return prompts

# Read in data
fewshot_examples = read_jsonl_to_array('./data/fewshot_examples.jsonl')
data_val = read_jsonl_to_array('./data/data_val.jsonl')
data_train = read_jsonl_to_array('./data/data_train.jsonl')
data_train = data_train[:1000]  # Limit to a small subset

# Build up noisy prompts and write train results
noises = [("000", 0.0), ("025", 0.25), ("050", 0.50), ("075", 0.75), ("100", 1.00)]
for noise in noises:
    for trial in range(5):
        res = dataset_to_prompts(fewshot_examples, data_train, noise=noise[1])
        res = [{
            "prompt": subarray[0],
            "chosen": subarray[1],
            "rejected": subarray[2]
        } for subarray in res]
        filename = f'./data/dpo/dpo_{noise[0]}-{trial+1}.json'
        write_array_to_json(res, filename)
        print(f"Generated trial {trial+1} of {filename} with noise {noise[1]}")

# Build up validation results
for i in range(5):
    res = dataset_to_prompts(fewshot_examples, data_val, noise=0.0)
    res = [{
        "prompt": subarray[0],
        "chosen": subarray[1],
        "rejected": subarray[2]
    } for subarray in res]
    filename = f'./data/dpo/val-{i+1}.json'
    write_array_to_json(res, filename)
    print(f"Generated trial {i+1} of {filename} for validation")
