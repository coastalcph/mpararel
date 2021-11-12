# mLAMA: multilingual LAnguage Model Analysis

This repository contains code for the paper "Factual Consistency of Multilingual Pretrained Language Models".
It extends the original [ParaRel 🤘](https://github.com/yanaiela/pararel) dataset to a multilingual setting.

The repository was forked from https://github.com/norakassner/mlama from where we used the translations scripts. 

## mParaRel

TODO: how to use the data

## Reproduce the results

### Create an environment and install the requirements

```bash
python3 -m venv mpararel-venv
source mpararel-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

### To reproduce the experiments

TODO: :)
```bash
```

## To recreate the dataset

To recreate the generation of the dataset follow the steps in `dataset/mpararel.sh`

## Reference:

```bibtex

```

## Acknowledgements

* [https://github.com/yanaiela/pararel](https://github.com/yanaiela/pararel)
* [https://github.com/norakassner/mlama](https://github.com/norakassner/mlama)
* [https://huggingface.co/bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
* [https://huggingface.co/xlm-roberta-large](https://huggingface.co/xlm-roberta-large)
