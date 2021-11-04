"""Evaluate consistency and accuracy of the predictions in the mpararel dataset.

python evaluate_consistency/run_evaluation.py \
    --predictions_folder=$WORKDIR/data/predictions_mpararel/reviewed_mbert_base_cased_2 \
    --mpararel_folder=$WORKDIR/data/mpararel_reviewed_with_tag \
    --mlama_folder=$WORKDIR/data/mlama1.1
"""
import argparse
import collections
import json
import os

import numpy as np
import tqdm
import wandb
from logger_utils import get_logger
from dataset.create_mpararel import read_mlama
from mpararel_utils import read_mpararel_templates

LOG = get_logger(__name__)


def compute_relation_metrics(tuple_to_prediction, mlama_template=None):
    """Computes the metrics for one relation predictions."""
    metrics = collections.defaultdict(float)
    # Iterate over the tuples.
    for _, predictions in tuple_to_prediction:
        consistency_count = 0.0
        accuracy_count = 0.0
        accuracy_consistency_count = 0.0
        mlama_accuracy_count = 0.0
        # Iterate over the templates.
        for i, (template, prediction_i,
                rank_of_correct_i) in enumerate(predictions):
            if rank_of_correct_i == "0":
                accuracy_count += 1
                if mlama_template is not None and template == mlama_template:
                    mlama_accuracy_count += 1
            for _, prediction_j, _ in predictions[i + 1:]:
                if prediction_i == prediction_j:
                    consistency_count += 1
                    if rank_of_correct_i == "0":
                        accuracy_consistency_count += 1
        metrics["accuracy"] += accuracy_count / len(predictions)
        metrics["mlama-accuracy"] += mlama_accuracy_count
        total_consistency_pairs = len(predictions) * (len(predictions) - 1) / 2
        metrics["consistency"] += consistency_count / total_consistency_pairs
        metrics["accuracy-consistency"] += (accuracy_consistency_count /
                                            total_consistency_pairs)
    for metric_name in metrics.keys():
        metrics[metric_name] /= len(tuple_to_prediction)
    return metrics


def filter_predictions(mpararel_templates,
                       tuple_to_prediction,
                       remove_repeated_subjects=False):
    filtered_tuple_to_prediction = []
    subject_count = collections.Counter(
        [data.split('-')[0] for data, _ in tuple_to_prediction])
    stats = {"removed_repeated_subjects": 0, "total_phrases": 0}
    for data, predictions in tuple_to_prediction:
        if subject_count[data.split('-')[0]] > 1 and remove_repeated_subjects:
            stats["removed_repeated_subjects"] += 1
            continue
        templates = set([p[0] for p in predictions])
        if mpararel_templates.difference(templates):
            LOG.warning(
                "mpararel templates not found in the predictions: {}".format(
                    mpararel_templates.difference(templates)))
        stats["total_phrases"] += len(mpararel_templates)
        filtered_tuple_to_prediction.append(
            (data, [p for p in predictions if p[0] in mpararel_templates]))
    return filtered_tuple_to_prediction, stats


def compute_metrics_by_language(mpararel,
                                predictions_folder,
                                mlama,
                                remove_repeated_subjects=False):
    """Computes the metrics based on the templates in mpararel."""
    language_to_metrics = collections.defaultdict(
        lambda: collections.defaultdict(list))
    language_to_avg_metrics = collections.defaultdict(
        lambda: collections.defaultdict(float))
    language_to_std_metrics = collections.defaultdict(
        lambda: collections.defaultdict(float))
    language_to_stats = collections.defaultdict(
        lambda: collections.defaultdict(int))
    for language in tqdm.tqdm(mpararel.keys()):
        for relation in mpararel[language].keys():
            # When looking at only the reviewed templates this could happen.
            if len(mpararel[language][relation]) < 2:
                LOG.info(
                    "Skipping relation '{}' in language '{}' because there's {}"
                    " templates".format(relation, language,
                                        len(mpararel[language][relation])))
                continue
            # The corrected chinese codes are not in mLAMA.
            if (language in mlama
                    and relation[:-len(".jsonl")] in mlama[language]):
                mlama_template = mlama[language][relation[:-len(".jsonl")]]
            else:
                LOG.info(
                    "language or relation not in mLAMA (language={}, relation={})"
                    .format(language, relation))
                mlama_template = None
            with open(os.path.join(predictions_folder, language, relation),
                      'r') as f:
                tuple_to_prediction = json.load(f)
                tuple_to_prediction, stats = filter_predictions(
                    mpararel[language][relation], tuple_to_prediction.items(),
                    remove_repeated_subjects)
                for stat_name, value in stats.items():
                    language_to_stats[language][stat_name] += value
                metrics = compute_relation_metrics(tuple_to_prediction,
                                                   mlama_template)
            for metric_name, value in metrics.items():
                language_to_metrics[language][metric_name].append(value)
        # We take the macro average across the relations.
        for metric_name in language_to_metrics[language].keys():
            language_to_avg_metrics[language][metric_name] = np.average(
                language_to_metrics[language][metric_name])
            language_to_std_metrics[language][metric_name] = np.std(
                language_to_metrics[language][metric_name])
    return language_to_avg_metrics, language_to_std_metrics, language_to_stats


def main(args):
    wandb.init(project="mpararel-evaluate",
               name=os.path.basename(args.predictions_folder))
    wandb.config.update(args)
    mlama = read_mlama(args.mlama_folder)
    mpararel = read_mpararel_templates(args.mpararel_folder,
                                       args.only_human_reviewed)
    if args.only_languages:
        remove_languages = set(mpararel.keys()).difference(
            set(args.only_languages))
        for to_remove in remove_languages:
            mpararel.pop(to_remove)
    (language_to_avg_metrics, language_to_std_metrics,
     language_to_stats) = compute_metrics_by_language(
         mpararel, args.predictions_folder, mlama,
         args.remove_repeated_subjects, args.micro_average)
    metrics = list(language_to_avg_metrics.items())[0][1].keys()
    for metric in metrics:
        data = [(l, language_to_avg_metrics[l][metric],
                 language_to_std_metrics[l][metric])
                for l in language_to_avg_metrics.keys()]
        columns = ["language", "value", "std"]
        wandb.log({
            metric:
            wandb.plot.bar(wandb.Table(data=data, columns=columns),
                           columns[0],
                           columns[1],
                           title=metric)
        })
    stats = list(language_to_stats.items())[0][1].keys()
    for stat in stats:
        data = [(l, language_to_stats[l][stat])
                for l in language_to_stats.keys()]
        columns = ["language", "value"]
        wandb.log({
            stat:
            wandb.plot.bar(wandb.Table(data=data, columns=columns),
                           columns[0],
                           columns[1],
                           title=stat)
        })


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--mlama_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--mpararel_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--only_human_reviewed",
                        default=False,
                        action="store_true")
    parser.add_argument("--remove_repeated_subjects",
                        default=False,
                        action="store_true")
    parser.add_argument("--only_languages", nargs='*', help="")
    parser.add_argument(
        "--micro_average",
        default=False,
        action="store_true",
        help=
        "To compute the micro average between the relations, note that we'd still be computing the macro averages"
    )
    return parser


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())
