# Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions

[![arXiv](https://img.shields.io/badge/EMNLP_2025_Finding-3-red.svg)](https://aclanthology.org/2025.findings-emnlp.3/)
[![arXiv](https://img.shields.io/badge/arXiv-2408.08780-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2408.08780)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?logo=apache)](LICENSE)

Source code of our EMNLP 2025 Findings paper **Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions**.

## Prerequisites
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download ru_core_news_sm
```

## Machine Translation (MT)

```bash
cd mt
```

### Preparation
Fetch and extract the training data. Note that test data have been provided for convenience.

```bash
cd data
sh prepare.sh
```

Prepare for in-context examples. Note that `retriv` might be incompatible with `torch` of some versions in some cases. We recommend to run `bm25.py` in another environment if this occurs.

```bash
cd ../src
sh prepare.sh
```

### Run Experiments and Evaluation
```bash
sh run.sh
```

## QA Tasks

### Data

The datasets corresponding to each task appearing in our paper have been placed in [data](). The file names of the training set (example database) and test set must contain *train* and *test* or *dev* respectively, and both end with *json* or *jsonl*.

### Example of Runing Experiments

```bash
cd qa/src

python qa_inference.py \
    --device 0 \
    --dataset csqa strategyqa date sports logicalfallacy threeobjects knownunknowns gsm8k aqua \
    --shot 4 \
    --batch_size 32 \
    --templates vanilla ensemble_random \
    --cot_mode 1 \
    --max_new_tokens 256 \
    --models alpaca llama2 mistral
```

For OpenAI's API model, you can refer to [qa_gpt_inference.py]().

### Evaluation

In order to evaluate our proposed prompt template in different dimensions, we set up two forms of evaluation, one is cross-model and the other is cross-dataset. Please refer to [qa_eval.py]() for details.

### How to Add New Datasets

1. Refer to the architecture (optional) and format of the existing datasets and put your new dataset into [data]().
2. Inherit `ReasoningData` defined in [data_loader.py]() and overwrite corresponding methods according to your new dataset.
3. Add `@dateset_register` for your child class.

## Citation
If you find our work helpful, feel free to cite us:
```
@misc{tang2025largelanguagemodelscare,
      title={Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions}, 
      author={Chenming Tang and Zhixiang Wang and Hao Sun and Yunfang Wu},
      year={2025},
      eprint={2408.08780},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.08780}, 
}
```
