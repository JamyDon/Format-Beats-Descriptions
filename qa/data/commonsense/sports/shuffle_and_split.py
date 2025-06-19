import json
import random
from typing import List, Dict, Tuple, Union


def statistics(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
        examples: List[Dict] = data["examples"]
        positive = [example for example in examples if example["target_scores"]["plausible"] == 1]
        negative = [example for example in examples if example["target_scores"]["plausible"] == 0]
    print(f"Total examples: {len(examples)}")
    print(f"Positive examples: {len(positive)}")
    print(f"Negative examples: {len(negative)}")

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

        positive = [example for example in examples if example["target_scores"]["plausible"] == 1]
        negative = [example for example in examples if example["target_scores"]["plausible"] == 0]

        random.seed(seed)
        random.shuffle(positive)
        random.shuffle(negative)

        test_data = positive[:test_number//2] + negative[:test_number//2]
        train_data = positive[test_number//2:] + negative[test_number//2:]

        random.shuffle(train_data)

        with open("train.json", "w") as f:
            json.dump({"examples": train_data}, f, indent=4)

        with open("test.json", "w") as f:
            json.dump({"examples": test_data}, f, indent=4)

        return len(train_data), len(test_data)


if __name__ == "__main__":
    
    file_path = "sports.json"
    total = 1000
    split_ratio = total - 4

    shuffle_and_split(file_path, split_ratio=split_ratio)
