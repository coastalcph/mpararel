# mParaRel

This repository contains the code for the paper [`"Factual Consistency of Multilingual Pretrained Language Models"`](https://aclanthology.org/2022.findings-acl.240/). It extends the original [ParaRel 🤘](https://github.com/yanaiela/pararel) dataset to a multilingual setting, by using mutliple machine translation systems to translate the templates to 45 languages.

The repository was forked from https://github.com/norakassner/mlama from where we used the translations scripts. 

## Dataset

**Update:** You can now find the dataset also in HugginFace Datasets 🤗:
- [coastalcph/mpararel](https://huggingface.co/datasets/coastalcph/mpararel)
- [coastalcph/mpararel_autorr](https://huggingface.co/datasets/coastalcph/mpararel_autorr)

You can find the reviewed templates and the subject-object tuples in the folder [`data/mpararel_reviewed`](https://github.com/coastalcph/mpararel/tree/master/data).

Note that we did not report any numbers in Hindi (even though the data is available) since during the human review it was pointed out that the data looked really noisy.

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

1. Get the models predictions
```bash
python evaluate_consistency/get_model_predictions.py \
    --mpararel_folder=$WORKDIR/data/mpararel \
    --model_name="bert-base-multilingual-cased" --batch_size=32 \
    --output_folder=$WORKDIR/data/predictions_mpararel/mbert_cased \
    --cpus 10
```
You can also add the flags `--only_languages` to get the predictions only for a couple of languages and not all the ones in the mpararel folder, and you can add `--add_end_of_sentence_punctuation '.'` if you want to experiment adding a sentence-final punctuation.

2. Evaluate consistency
```bash
python evaluate_consistency/run_evaluation.py \
    --predictions_folder=$WORKDIR/data/predictions_mpararel/mbert_cased \
    --mpararel_folder=$WORKDIR/data/mpararel_reviewed_with_tag \
    --mlama_folder=$WORKDIR/data/mlama1.1 \
    --remove_repeated_subjects
```
You can also add the flags `--only_languages zh-hans` if you want don't want to get the numbers of all the languages in the mpararel_folder.

## Recreate the dataset

To recreate the generation of the dataset follow the steps in [`dataset/mpararel.sh`](https://github.com/coastalcph/mpararel/blob/master/dataset/mpararel.sh)

## Reference
```
@inproceedings{fierro-sogaard-2022-factual,
    title = "Factual Consistency of Multilingual Pretrained Language Models",
    author = "Fierro, Constanza  and
      S{\o}gaard, Anders",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.240",
    pages = "3046--3052",
}
```

## Acknowledgements

* [https://github.com/yanaiela/pararel](https://github.com/yanaiela/pararel)
* [https://github.com/norakassner/mlama](https://github.com/norakassner/mlama)
* [https://huggingface.co/bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
* [https://huggingface.co/xlm-roberta-large](https://huggingface.co/xlm-roberta-large)
