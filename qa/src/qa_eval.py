import os
from collections import defaultdict
from pprint import pprint
import pandas as pd

from data_loader import get_dataset
from plot import plot_cross_models, plot_cross_datasets


dataset_map = {
    "logical_fallacy_detection": "logicalfallacy",
    "three_objects": "threeobjects",
    "known_unknowns": "knownunknowns",
}

def evaluate_cross_models(dataset_name, output_dir, models=["alpaca-7b", "llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]):
    if isinstance(models, str):
        models = [models]

    if dataset_name in dataset_map:
        dataset_name = dataset_map[dataset_name]

    dataset = get_dataset(dataset_name)
    output_dir = dataset.data_dir.replace("data", "output") if output_dir is None else output_dir

    results = defaultdict(dict)

    for model_name in models:
        model_output_dir = f"{output_dir}/{model_name}"
        output_files = os.listdir(model_output_dir)
        output_files = [f"{model_output_dir}/{file}" for file in output_files if file.startswith("4") and "ensemble_random" in file or "vanilla" in file]

        for file in output_files:
            # print(f'evaluating {file}')
            if "cot" in file:
                template = file.split(".")[-3] + " (w/ CoT)"
                template = template.replace("ensemble_random", "ERR")
                results[model_name][template] = dataset.evaluate_from_file(file)
            else:
                template = file.split(".")[-2] + " (w/o CoT)"
                template = template.replace("ensemble_random", "ERR")
                results[model_name][template] = dataset.evaluate_from_file(file)

    return results, output_dir.replace("output", "images")

def evaluate_cross_datasets(model, category):
    category_dir = f"../output/{category}"
    datasets = os.listdir(category_dir)
    output_dirs = [f"{category_dir}/{dataset}/" for dataset in datasets]

    results = defaultdict(dict)
    for dataset, output_dir in zip(datasets, output_dirs):
        tmp, _ = evaluate_cross_models(dataset, output_dir, model)
        results[dataset] = tmp[model]

    return results, category_dir.replace("output", "images")

def to_excel(results, save_path, sort_order):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        for category, result in results.items():
            df = pd.DataFrame(result).T
            df = df.reindex(sort_order, axis=1)
            df.to_excel(writer, sheet_name=category, float_format="%.16f")
            

if __name__ == "__main__":
    datasets = ["date", "knownunknowns", "logicalfallacy", "threeobjects", "csqa", "strategyqa", "sports", "aqua", "gsm8k"]
    models = ["alpaca-7b", "llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]
    sort_order = ["vanilla (w/o CoT)", "vanilla (w/ CoT)", "ERR (w/o CoT)", "ERR (w/ CoT)"]

    task2datasets = {
        "commonsense": ["date", "sports", "csqa", "strategyqa"],
        "logic": ["logicalfallacy", "threeobjects"],
        "math": ["aqua", "gsm8k"],
        "hallucination": ["knownunknowns"]
    }


    # Small models
    small_model_results = {}
    for dataset in datasets:
        results, save_dir = evaluate_cross_models(dataset, None, models=models)
        small_model_results[dataset] = results
        # plot_cross_models(results, dataset, save_path=f"{save_dir}/{dataset}.png", sort_order=sort_order)
    to_excel(small_model_results, "small_models.xlsx", sort_order)

    # GPT-3.5
    model = "gpt-3.5-turbo-0125"

    gpt_results = {}
    for category, datasets in task2datasets.items():
        results, save_dir = evaluate_cross_datasets(model, category)
        gpt_results[category] = results
        plot_cross_datasets(results, model, category, save_path=f"{save_dir}/{model}.png", sort_order=sort_order)
    to_excel(gpt_results, "gpt.xlsx", sort_order)
