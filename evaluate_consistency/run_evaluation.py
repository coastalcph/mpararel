"""Evaluate consistency and accuracy of the predictions in the mpararel dataset.

python evaluate_consistency/run_evaluation.py \
    --predictions_folder=$WORKDIR/data/mpararel_predictions/mbert_cased_wrong
"""
import argparse
import collections
import json
import os

import numpy as np
import tqdm
import wandb
from logger_utils import get_logger

LOG = get_logger(__name__)


def compute_relation_metrics(tuple_to_prediction):
    """Computes the metrics for one relation predictions."""
    metrics = collections.defaultdict(float)
    # Iterate over the tuples.
    for _, predictions in tuple_to_prediction.items():
        consistency_count = 0.0
        accuracy_count = 0.0
        accuracy_consistency_count = 0.0
        # Iterate over the templates.
        for i, (_, prediction_i, rank_of_correct_i) in enumerate(predictions):
            if rank_of_correct_i == "0":
                accuracy_count += 1
            for _, prediction_j, _ in predictions[i + 1:]:
                if prediction_i == prediction_j:
                    consistency_count += 1
                    if rank_of_correct_i == "0":
                        accuracy_consistency_count += 1
        metrics["accuracy"] += accuracy_count / len(predictions)
        total_consistency_pairs = len(predictions) * (len(predictions) - 1) / 2
        metrics["consistency"] += consistency_count / total_consistency_pairs
        metrics["accuracy-consistency"] += (accuracy_consistency_count /
                                            total_consistency_pairs)
    for metric_name in metrics.keys():
        metrics[metric_name] /= len(tuple_to_prediction)
    return metrics


def main(args):
    wandb.init(project="mpararel-evaluate",
               name=os.path.basename(args.predictions_folder))
    wandb.config.update(args)
    language_to_metrics = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for language in tqdm.tqdm(os.listdir(args.predictions_folder)):
        language_dir = os.path.join(args.predictions_folder, language)
        relations = os.listdir(language_dir)
        for relation_file in relations:
            with open(os.path.join(language_dir, relation_file)) as f:
                tuple_to_prediction = json.load(f)
                for metric_name, value in compute_relation_metrics(
                        tuple_to_prediction).items():
                    language_to_metrics[language][metric_name] += value
        # We take the macro average across the relations.
        for metric_name in language_to_metrics[language].keys():
            language_to_metrics[language][metric_name] /= len(relations)
    english_metrics = list(language_to_metrics["en"].items())
    for metric, en_value in english_metrics:
        wandb.run.summary[f"en - {metric}"] = en_value
        r_metric = []
        for lang in language_to_metrics.keys():
            language_to_metrics[lang]["r-" + metric] = (
                language_to_metrics[lang][metric] / (en_value + 1e-13))
            r_metric.append(language_to_metrics[lang]["r-" + metric])
        wandb.run.summary["min r-" + metric] = min(r_metric)
        wandb.run.summary["max r-" + metric] = max(r_metric)
        wandb.run.summary["avg r-" + metric] = np.average(np.array(r_metric))
    for metric in language_to_metrics[language].keys():
        data = [(l, language_to_metrics[l][metric])
                for l in language_to_metrics.keys()]
        columns = ["language", "value"]
        wandb.log({
            metric:
            wandb.plot.bar(wandb.Table(data=data, columns=columns),
                           columns[0],
                           columns[1],
                           title=metric)
        })


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())
