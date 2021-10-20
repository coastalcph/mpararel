"""Queries the model and saves the predictions.

python evaluate_consistency/get_model_predictions.py \
    --mpararel_folder=$WORKDIR/data/mpararel_with_mlama \
    --model_name="bert-base-multilingual-cased" --batch_size=32 \
    --output_folder=$WORKDIR/data/mpararel_predictions/mbert_cased_with_mlama \
    --path_to_existing_predictions=$WORKDIR/data/mpararel_predictions/mbert_cased \
    --cpus 10


TODO:
    Add tests for the ranking, the writing of the file, and the dataloader.

"""
import argparse
import json
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
from tqdm import tqdm

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
from transformers import AutoModelForMaskedLM, AutoTokenizer

LOG = get_logger(__name__)


@dataclass
class TemplateTuple():
    language: str
    relation: str
    template: str
    subject: str
    object: str


class GenerateTemplateTupleExamples():
    def __init__(self, tokenizer, languages, relations, get_candidates,
                 get_templates, get_tuples) -> None:
        self.tokenizer = tokenizer
        self.languages = languages
        self.relations = relations
        self.get_candidates = get_candidates
        self.get_templates = get_templates
        self.get_tuples = get_tuples

    def load_existing_predictions(self, path):
        self.existing_predictions = defaultdict(set)
        for lang in os.listdir(path):
            for relation in os.listdir(os.path.join(path, lang)):
                with open(os.path.join(path, lang, relation)) as f:
                    tuple_to_predictions = json.load(f)
                    for tuple, predictions in tuple_to_predictions.items():
                        for template, _, _ in predictions:
                            self.existing_predictions[tuple].add(template)

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
        for i, language in tqdm(enumerate(self.languages, 1)):
            for relation in self.relations:
                tokens_count_to_obj_to_ids = defaultdict(
                    lambda: defaultdict(list))
                max_tokens_count = 0
                for candidate_obj in self.get_candidates(language, relation):
                    tokens = self.tokenizer.tokenize(candidate_obj)
                    ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    tokens_count_to_obj_to_ids[len(
                        tokens)][candidate_obj] = ids
                    max_tokens_count = max(max_tokens_count, len(tokens))
                for template in self.get_templates(language, relation):
                    for tuple in self.get_tuples(language, relation):
                        tuple_key = f"{tuple[SUBJECT_KEY]}-{tuple[OBJECT_KEY]}"
                        if (hasattr(self, 'existing_predictions')
                                and tuple_key in self.existing_predictions
                                and template
                                in self.existing_predictions[tuple_key]):
                            continue
                        for masks_count in tokens_count_to_obj_to_ids.keys():
                            inputs.append(
                                get_populated_template(
                                    template, tuple, self.tokenizer.mask_token,
                                    masks_count))
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


def get_populated_template(template, tuple, mask_token, mask_token_count):
    template = template.replace("[X]", tuple[OBJECT_KEY])
    template = template.replace("[Y]",
                                ' '.join([mask_token] * mask_token_count))
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
                                     relation)) as existing_f:
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
    languages = os.listdir(os.path.join(args.mpararel_folder, "patterns"))
    relations = os.listdir(os.path.join(args.mpararel_folder, "tuples", "en"))
    get_candidates = lambda lang, relation: get_items(
        os.path.join(args.mpararel_folder, "tuples", lang, relation),
        OBJECT_KEY)
    get_templates = lambda lang, relation: get_items(
        os.path.join(args.mpararel_folder, "patterns", lang, relation),
        "pattern")
    get_tuples = lambda lang, relation: get_items(
        os.path.join(args.mpararel_folder, "tuples", lang, relation))
    template_tuple_examples = GenerateTemplateTupleExamples(
        tokenizer, languages, relations, get_candidates, get_templates,
        get_tuples)
    if args.path_to_existing_predictions:
        template_tuple_examples.load_existing_predictions(
            args.path_to_existing_predictions)
    model_queries_count = 0
    for template_tuple_i, (inputs, candidates_to_ids,
                           this_template_tuple) in enumerate(
                               template_tuple_examples, 1):
        candidates_to_prob = {}
        for batch_i in range(0, len(inputs), args.batch_size):
            # Prepare the batch.
            input_batch = inputs[batch_i:min(len(inputs), batch_i +
                                             args.batch_size)]
            candidates_to_ids_batch = candidates_to_ids[
                batch_i:min(len(inputs), batch_i + args.batch_size)]
            encoded_input = tokenizer(input_batch,
                                      padding=True,
                                      return_tensors='pt')
            encoded_input = encoded_input.to(args.device)
            mask_indexes = torch.where(
                encoded_input.input_ids == tokenizer.mask_token_id, 1, 0)
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
        # Sort in descending probability.
        candidates_and_prob = sorted(candidates_to_prob.items(),
                                     key=lambda x: x[1],
                                     reverse=True)
        correct_rank = np.argwhere(
            np.array(candidates_and_prob)[:, 0] ==
            this_template_tuple.object)[0][0]
        tuples_predictions[this_template_tuple.language][
            this_template_tuple.relation][
                f"{this_template_tuple.subject}-{this_template_tuple.object}"].append(
                    (this_template_tuple.template, candidates_and_prob[0][0],
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
