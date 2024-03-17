# DPO-Evaluation

This repository contains code used for my group's final project for the University of Washington's winter 2024 offering of CSE493G1.

## Abstract

As Large Language Models (LLMs) are becoming increasingly popular and widespread, aligning LLMs with human intent has become an ever more important area of research. To do so, there are methods such as Reinforcement Learning with Human Feedback (RLHF) or Direct Preference Optimization (DPO). Because fine-tuning using RLHF or DPO requires high-quality data, the amount of data available for fine-tuning is often insufficient. This paper investigates what happens when fine-tuning data is not high-quality, and how much it is worth sacrificing data quality to source greater quantities of data for fine-tuning.

## Files

- `data/`: Data-related files
    - `dpo/`: Files used for DPO training
    - `raw/`: Raw data files borrowed from HellaSwag
- `preprocessing/`: Utilities used to for processing and training
    - `generate_data.py`: Used to convert the raw HellaSwag data into formatted train, validation, and fewshot example files
    - `generate_dpo_data.py`: Used to sample training and validation datasets with varying amounts of noise from the formatted train, validation, and fewshot example files
- `scripts/`: Scripts used for training
    - `prompts.py`: Demo code demonstrating how to read in the entire train and validation files, used for baseline
    - `training.py`: Fine-tunes an LLM using DPO on the generated datasets

## Usage

To generate new datasets, run:

```
$ python preprocessing/generate_dpo_data.py
```

To train a new model, run:

```
$ python scripts/training.py --output_dir <output-dir> --train_data_path <train-data-path> --val_data_path <val-data-path> --run_name <run-name>
```

Here is an example usage:

```
$ python training.py --output_dir ./results/dpo_000-1 --train_data_path ../data/dpo/dpo_000-1.json --val_data_path ../data/dpo/val-1.json --run_name gpt2large-000-1
```

## Disclaimer

This experiment was carried out for a school course's final project, and thus was constrained by several compute and time restrictions.