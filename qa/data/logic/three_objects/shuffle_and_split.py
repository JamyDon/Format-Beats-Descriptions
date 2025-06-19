import json
import random
from typing import List, Dict, Tuple, Union


def statistics(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
        examples: List[Dict] = data["examples"]
    return len(examples)

def shuffle_and_split(file_name, split_ratio: Union[int, float]=1012, seed: int=42) -> Tuple[int, int]:
    with open(file_name, "r") as f:
        data = json.load(f)
        examples: List[Dict] = data["examples"]
        
        if split_ratio < 0:
            raise ValueError("split_ratio must be positive")
        elif split_ratio < 1 and split_ratio > 0:
            test_number = int(len(examples) * split_ratio)
        elif split_ratio > 1 and isinstance(split_ratio, int) and split_ratio < len(examples):
            test_number = split_ratio
        else:
            raise ValueError("split_ratio must be a positive float or integer less than the number of examples")

        random.seed(seed)
        random.shuffle(examples)

        test_examples = examples[:test_number]
        train_examples = examples[test_number:]       

        with open("train.json", "w") as f:
            json.dump({"examples": train_examples}, f, indent=4)

        with open("test.json", "w") as f:
            json.dump({"examples": test_examples}, f, indent=4)

        return len(train_examples), len(test_examples)


if __name__ == "__main__":
    
    file_path = "task.json"
    split_ratio = statistics(file_path) - 4

    train_num, test_num = shuffle_and_split(file_path, split_ratio=split_ratio)
    print(f"Train examples: {train_num}, Test examples: {test_num}")
