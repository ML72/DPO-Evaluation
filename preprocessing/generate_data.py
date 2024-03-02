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

def write_array_to_jsonl(data_array, file_path):
    """
    Writes a list of dictionaries to a .jsonl file.

    Args:
    - data_array (List[dict]): A list of dictionaries, each representing a JSON object.
    - file_path (str): The path to the .jsonl file to write to.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data_array:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

def clean_entries(data_array):
    """
    Cleans the entries in the data_array.

    Args:
    - data_array (List[dict]): A list of dictionaries, each representing a JSON data object.

    Returns:
    - List[dict]: A list of dictionaries, each representing a JSON data object cleaned.
    """
    for i in range(len(data_array)):
        new_entry = {
            'context': data_array[i]['ctx'],
            'choices': data_array[i]['endings'],
            'answer': data_array[i]['label']
        }
        data_array[i] = new_entry

# Read in raw HellaSwag data
hellaswag_train = read_jsonl_to_array('./data/raw/hellaswag_train.jsonl')
hellaswag_val = read_jsonl_to_array('./data/raw/hellaswag_val.jsonl')

# Split into few-shot examples, train dataset, and val dataset
# 8 few-shot examples are taken from hellaswag_train
# 10,000 train entries are taken from hellaswag_train
# 1,000 val entries are taken from hellaswag_val
fewshot_idx = [0, 4, 36, 13, 21, 41, 29, 26]
data_fewshot = [hellaswag_train[i] for i in fewshot_idx]
data_train = hellaswag_train[50:10050]
data_val = hellaswag_val[0:1000]

clean_entries(data_fewshot)
clean_entries(data_train)
clean_entries(data_val)

write_array_to_jsonl(data_fewshot, './data/fewshot_examples.jsonl')
write_array_to_jsonl(data_train, './data/data_train.jsonl')
write_array_to_jsonl(data_val, './data/data_val.jsonl')

print("Data cleaned and split into few-shot examples, train dataset, and val dataset.")
