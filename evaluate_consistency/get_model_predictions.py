"""Queries the model and saves the predictions.

python evaluate_consistency/get_model_predictions.py \
    --mpararel_folder=$WORKDIR/data/mpararel_00_00_06_02_logging \
    --model_name="bert-base-multilingual-cased" --batch_size=32 \
    --output_folder=$WORKDIR/data/mpararel_predictions/mbert_cased

"""
import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from dataset.constants import OBJECT_KEY, SUBJECT_KEY
from logger_utils import get_logger
from tqdm import tqdm
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

    def __iter__(self):
        """Yields the encoded batch, the indices of the masks, and the targets.

        Returns:
            encoded_input: The tokenized batch
            mask_indexes: Tensor of same shape than encoded_input.input_ids with
                1's if the token is a mask and 0 if not.
            batch_candidates_to_ids: List[Dict], the list contains a dict for
                each example of the batch mapping from the possible objects
                strings to its tokens ids.
            batch_data: List[TemplateTuple] containing the data that defines
                each example in the batch.
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


def write_predictions(results, output_folder):
    for language, relation_to_predictions in results.items():
        os.makedirs(os.path.join(output_folder, language), exist_ok=True)
        for relation, tuple_to_predictions in relation_to_predictions.items():
            filename = os.path.join(output_folder, language, relation)
            with open(filename, 'w') as f:
                json.dump(tuple_to_predictions, f)


def init_wandb(args):
    wandb.init(project="mpararel-get-predictions", name=args.model_name)
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
    model, tokenizer = build_model_by_name(args.model_name, args.device)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
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
    model_queries_count = 0
    for template_tuple_i, (inputs, candidates_to_ids,
                           this_template_tuple) in enumerate(
                               template_tuple_examples, 1):
        candidates_to_prob = {}
        for batch_i in range(0, len(inputs), args.batch_size):
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
            init_time_model_query = time.time()
            output = model(**encoded_input)
            total_time_model_query = time.time() - init_time_model_query
            model_queries_count += 1
            # Iterate over each example in the batch to check the predictions.
            init_time_example_iter = time.time()
            for i, (mask_indexes_i, candidates_to_ids_i) in enumerate(
                    zip(mask_indexes, candidates_to_ids_batch)):
                # This has the shape: [masks_count, vocab_size].
                mask_predictions = output.logits[i][mask_indexes_i.nonzero(
                    as_tuple=True)]
                # Each example has multiple possible candidates so we check the
                # probability of each.
                for candidate, token_ids in candidates_to_ids_i.items():
                    candidates_to_prob[candidate] = np.mean([
                        mask_predictions[i][token_id].item()
                        for i, token_id in enumerate(token_ids)
                    ])
            total_time_iter = time.time() - init_time_example_iter
            wandb.log({
                "Model inference time":
                total_time_model_query,
                "Total time iterating over batch examples":
                total_time_iter,
                "Average time checking the candidates of each example":
                total_time_iter / len(input_batch)
            })
        candidates_and_prob = sorted(candidates_to_prob.items(),
                                     key=lambda x: x[1])
        correct_rank = np.argwhere(
            np.array(candidates_and_prob)[:, 0] ==
            this_template_tuple.object)[0][0]
        results[this_template_tuple.language][this_template_tuple.relation][
            f"{this_template_tuple.subject}-{this_template_tuple.object}"].append(
                (this_template_tuple.template, candidates_and_prob[0][0],
                 str(correct_rank)))
    wandb.run.summary["#model_queries"] = model_queries_count
    wandb.run.summary["#template_tuple_examples"] = template_tuple_i
    write_predictions(results, args.output_folder)


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
    parser.add_argument("--output_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--device", default="cpu", type=str, help="")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())
