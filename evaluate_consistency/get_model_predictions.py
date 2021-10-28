"""Queries the model and saves the predictions.

python evaluate_consistency/get_model_predictions.py \
    --mpararel_folder=$WORKDIR/data/mpararel_with_mlama \
    --model_name="bert-base-multilingual-cased" --batch_size=32 \
    --output_folder=$WORKDIR/data/mpararel_predictions/mbert_cased_with_mlama_2 \
    --path_to_existing_predictions=$WORKDIR/data/mpararel_predictions/mbert_cased \
    --cpus 10
"""
import argparse
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from dataset.constants import OBJECT_KEY, SUBJECT_KEY
from logger_utils import get_logger
from mpararel_utils import VALID_RELATIONS
from tqdm import tqdm

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
from transformers import AutoModelForMaskedLM, AutoTokenizer

LOG = get_logger(
    __name__,
    '/home/wsr217/mpararel/evaluate_consistency/debug_logging_get_model_predictions.log',
    level=logging.INFO,
    only_file_level=logging.DEBUG)


@dataclass
class TemplateTuple():
    language: str
    relation: str
    template: str
    subject: str
    object: str


class GenerateTemplateTupleExamples():
    def __init__(self,
                 tokenizer,
                 languages,
                 relations,
                 get_candidates,
                 get_templates,
                 get_tuples,
                 add_point=False) -> None:
        self.tokenizer = tokenizer
        self.languages = languages
        self.relations = relations
        self.get_candidates = get_candidates
        self.get_templates = get_templates
        self.get_tuples = get_tuples
        self.add_point = add_point

    def load_existing_predictions(self, path):
        self.existing_predictions = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))
        for lang in os.listdir(path):
            for relation in os.listdir(os.path.join(path, lang)):
                with open(os.path.join(path, lang, relation)) as f:
                    tuple_to_predictions = json.load(f)
                    for tuple_, predictions in tuple_to_predictions.items():
                        for template, _, _ in predictions:
                            self.existing_predictions[lang][relation][
                                tuple_].add(template)

    def __iter__(self):
        """Yields the encoded batch, the indices of the masks, and the target.

        Each batch will contain examples of only one template and tuple, but
        they may have different number of masks.

        Returns:
            encoded_input: The tokenized batch.
            candidates_to_ids: List[Dict], list contains a dict for each
                example of the batch mapping from the possible objects strings
                to its tokens ids.
            template_tuple: TemplateTuple
        """
        inputs, candidates_to_ids = [], []
        for language in tqdm(self.languages):
            for relation in self.relations:
                tokens_count_to_obj_to_ids = defaultdict(
                    lambda: defaultdict(list))
                for candidate_obj in self.get_candidates(language, relation):
                    tokens = self.tokenizer.tokenize(candidate_obj)
                    ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    tokens_count_to_obj_to_ids[len(
                        tokens)][candidate_obj] = ids
                for template in self.get_templates(language, relation):
                    for i, tuple in enumerate(
                            self.get_tuples(language, relation), 1):
                        tuple_key = f"{tuple[SUBJECT_KEY]}-{tuple[OBJECT_KEY]}"
                        if (hasattr(self, 'existing_predictions') and template
                                in self.existing_predictions[language]
                            [relation][tuple_key]):
                            continue
                        for masks_count in tokens_count_to_obj_to_ids.keys():
                            inputs.append(
                                get_populated_template(
                                    template, tuple, self.tokenizer.mask_token,
                                    masks_count, self.add_point))
                            candidates_to_ids.append(
                                tokens_count_to_obj_to_ids[masks_count])
                        this_template_tuple = TemplateTuple(
                            language, relation, template, tuple[SUBJECT_KEY],
                            tuple[OBJECT_KEY])
                        yield (inputs, candidates_to_ids, this_template_tuple)
                        inputs, candidates_to_ids = [], []


def get_items(path_to_file, key_item=None):
    items = []
    if not os.path.exists(path_to_file):
        return items
    with open(path_to_file) as file:
        for line in file:
            data = json.loads(line)
            if data:
                if key_item:
                    items.append(data[key_item])
                else:
                    items.append(data)
    return items


def build_model_by_name(model_name, device):
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_populated_template(template,
                           tuple,
                           mask_token,
                           mask_token_count,
                           add_point=False):
    template = template.replace("[X]", tuple[SUBJECT_KEY])
    template = template.replace("[Y]",
                                ' '.join([mask_token] * mask_token_count))
    if add_point:
        template = template + ' .'
    return template


def get_candidates_probabilities(logits_i, mask_indexes_i,
                                 candidates_to_ids_i):
    """Returns the probabilities of each candidate."""
    candidates_to_prob = {}
    # This has the shape: [masks_count, vocab_size].
    masks_probabilities = logits_i[mask_indexes_i.nonzero(as_tuple=True)]
    # Each example has multiple possible candidates so we check the
    # probability of each.
    for candidate, token_ids in candidates_to_ids_i.items():
        candidates_to_prob[candidate] = np.mean([
            masks_probabilities[i][token_id].item()
            for i, token_id in enumerate(token_ids)
        ])
    return candidates_to_prob


def write_predictions(results, output_folder, path_to_existing_predictions):
    for language, relation_to_predictions in results.items():
        os.makedirs(os.path.join(output_folder, language), exist_ok=True)
        relations = set(relation_to_predictions.keys())
        if path_to_existing_predictions:
            existing_relations = os.listdir(
                os.path.join(path_to_existing_predictions, language))
            relations.update(set(existing_relations))
        for relation in relations:
            # relation_to_predictions is a defaultdict so it'd be an empty list
            # if there are no predictions for that relation.
            tuple_to_predictions = relation_to_predictions[relation]
            if path_to_existing_predictions and os.path.isfile(
                    os.path.join(path_to_existing_predictions, language,
                                 relation)):
                with open(
                        os.path.join(path_to_existing_predictions, language,
                                     relation), 'r') as existing_f:
                    existing_tuple_to_predictions = json.load(existing_f)
                    for (tuple,
                         predictions) in existing_tuple_to_predictions.items():
                        if tuple in tuple_to_predictions:
                            # We assumed only new templates are in
                            # tuple_to_predictions, so we can safely merge the
                            # predictions knowing there will be no duplicates.
                            tuple_to_predictions[tuple] += predictions
                        else:
                            tuple_to_predictions[tuple] = predictions
            filename = os.path.join(output_folder, language, relation)
            with open(filename, 'w') as f:
                json.dump(tuple_to_predictions, f)


def init_wandb(args):
    wandb.init(project="mpararel-get-predictions",
               name=os.path.basename(args.output_folder))
    wandb.config.update(args)


def get_data(args):
    languages = os.listdir(os.path.join(args.mpararel_folder, "patterns"))
    if args.only_languages:
        LOG.info("Going to iterate only over the languages: {}".format(
            args.only_languages))
        languages = args.only_languages
    relations = [relation + '.jsonl' for relation in VALID_RELATIONS]
    get_templates = lambda lang, relation: get_items(
        os.path.join(args.mpararel_folder, "patterns", lang, relation),
        "pattern")
    get_tuples = lambda lang, relation: get_items(
        os.path.join(args.mpararel_folder, "tuples", lang, relation))
    get_candidates = lambda lang, relation: get_items(
        os.path.join(args.mpararel_folder, "tuples", lang, relation),
        OBJECT_KEY)
    if args.different_tuples_folder:
        LOG.info(
            "Using tuples from '{}' instead of those from mpararel.".format(
                args.different_tuples_folder))
        get_tuples = lambda lang, relation: get_items(
            os.path.join(args.different_tuples_folder, lang, relation))
        get_candidates = lambda lang, relation: get_items(
            os.path.join(args.different_tuples_folder, lang, relation),
            OBJECT_KEY)
    return languages, relations, get_candidates, get_templates, get_tuples


def batchify(inputs, candidates_to_ids, batch_size):
    if len(inputs) != len(candidates_to_ids):
        raise Exception("Can't batchify lists of different sizes.")
    batches = []
    for batch_i in range(0, len(inputs), batch_size):
        input_batch = inputs[batch_i:min(len(inputs), batch_i + batch_size)]
        candidates_to_ids_batch = candidates_to_ids[
            batch_i:min(len(inputs), batch_i + batch_size)]
        batches.append((input_batch, candidates_to_ids_batch))
    return batches


def get_predicted_and_rank_of_correct(candidates_to_prob, correct):
    # Sort in descending probability.
    candidates_and_prob = sorted(candidates_to_prob.items(),
                                 key=lambda x: x[1],
                                 reverse=True)
    correct_rank = np.argwhere(
        np.array(candidates_and_prob)[:, 0] == correct)[0][0]
    return candidates_and_prob[0][0], correct_rank


def get_masks_indices(encoded_input, mask_token_id):
    return torch.where(encoded_input.input_ids == mask_token_id, 1, 0)


def get_k(list_, k):
    return list_[:min(len(list_), k)]


def main(args):
    """Queries the model and saves the predictions.

    Saves in the output folder a folder for each language, with a file for each
    relation. Each json file contains a dictionary mapping from 'subject-object'
    pair to (template, prediction, rank correct).
    """
    init_wandb(args)
    if torch.cuda.is_available():
        LOG.info("Using GPU")
        args.device = "cuda:" + str(torch.cuda.current_device())
    # Pararelism variables needed.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    processes_pool = mp.Pool(processes=args.cpus)

    model, tokenizer = build_model_by_name(args.model_name, args.device)
    tuples_predictions = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    (languages, relations, get_candidates, get_templates,
     get_tuples) = get_data(args)
    template_tuple_examples = GenerateTemplateTupleExamples(
        tokenizer, languages, relations, get_candidates, get_templates,
        get_tuples, args.add_period_to_sentences)
    if args.path_to_existing_predictions:
        template_tuple_examples.load_existing_predictions(
            args.path_to_existing_predictions)
    model_queries_count = 0
    for template_tuple_i, (inputs, candidates_to_ids,
                           this_template_tuple) in enumerate(
                               template_tuple_examples, 1):
        candidates_to_prob = {}
        for input_batch, candidates_to_ids_batch in batchify(
                inputs, candidates_to_ids, args.batch_size):
            encoded_input = tokenizer(input_batch,
                                      padding=True,
                                      return_tensors='pt')
            encoded_input = encoded_input.to(args.device)
            mask_indexes = get_masks_indices(encoded_input,
                                             tokenizer.mask_token_id)
            # Query the model.
            init_time_model_query = time.time()
            with torch.no_grad():
                output = model(**encoded_input)
            total_time_model_query = time.time() - init_time_model_query
            model_queries_count += 1
            init_time_example_iter = time.time()
            # Each arg corresponds to one example in the batch.
            threads_args = [(output.logits[i], m, c) for i, (
                m, c) in enumerate(zip(mask_indexes, candidates_to_ids_batch))]
            # We check the predictions of each example concurrently.
            processes_results = processes_pool.starmap(
                get_candidates_probabilities, threads_args)
            for process_result in processes_results:
                for candidate, probability in process_result.items():
                    candidates_to_prob[candidate] = probability
            total_time_iter = time.time() - init_time_example_iter
            wandb.log({
                "Model inference time":
                total_time_model_query,
                "Total time iterating over batch examples":
                total_time_iter,
                "Average time checking the candidates of each example":
                total_time_iter / len(input_batch)
            })
        predicted, correct_rank = get_predicted_and_rank_of_correct(
            candidates_to_prob, this_template_tuple.object)
        LOG.debug(
            "[{}/{}/{}-{}]\nQueried the model with: '{}'\nConsidered the "
            "candidates: '{} ...'\nGot predicted='{}' and rank of correct is "
            "'{}'".format(this_template_tuple.language,
                          this_template_tuple.relation,
                          this_template_tuple.subject,
                          this_template_tuple.object, inputs,
                          [c.keys() for c in get_k(candidates_to_ids, 5)],
                          predicted, correct_rank))
        tuples_predictions[this_template_tuple.language][
            this_template_tuple.relation][
                f"{this_template_tuple.subject}-{this_template_tuple.object}"].append(
                    (this_template_tuple.template, predicted,
                     str(correct_rank)))
    processes_pool.close()
    processes_pool.join()
    wandb.run.summary["#model_queries"] = model_queries_count
    wandb.run.summary["#template_tuple_examples"] = template_tuple_i
    write_predictions(tuples_predictions, args.output_folder,
                      args.path_to_existing_predictions)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpararel_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to the folder with the mpararel data.")
    parser.add_argument(
        "--different_tuples_folder",
        default=None,
        type=str,
        help="Select this folder to get the tuples from there instead of "
        "mpararel_folder/tuples.")
    parser.add_argument(
        "--only_languages",
        nargs='*',
        help="If you don't want to iterate over all languages.")
    parser.add_argument(
        "--add_period_to_sentences",
        action='store_true',
        help="If true ' .' is added at the end of the phrases. This shouldn't "
        "be used for all the languages as not all the languages use a period as"
        " end puntuaction.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="")
    parser.add_argument("--cpus",
                        default=None,
                        type=int,
                        required=True,
                        help="")
    parser.add_argument(
        "--path_to_existing_predictions",
        default=None,
        type=str,
        help="If this is provided then the sentences for which there's already"
        " a prediction are skipped.")
    parser.add_argument("--output_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="Where to write the predictions.")
    parser.add_argument("--device", default="cpu", type=str, help="")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())
