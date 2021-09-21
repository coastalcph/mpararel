"""Store in a file the LM model representations for the masked token (the object
of the templates).
"""
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pararel.consistency import utils


def log_wandb(args):
    pattern = args.patterns_file.split('/')[-1].split('.')[0]
    lm = args.lm

    config = dict(pattern=pattern, lm=lm)

    wandb.init(
        entity='consistency',
        name=f'{pattern}_encode_{lm}',
        project="consistency",
        tags=["encode", pattern],
        config=config,
    )


def parse_prompt(prompt: str, subject_label: str, mask: str) -> str:
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)\
                   .replace(OBJ_SYMBOL, mask)
    return prompt


# get mlm model to predict masked token.
def build_model_by_name(lm: str):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """

    model = AutoModel.from_pretrained(lm)
    tokenizer = AutoTokenizer.from_pretrained(lm)
    return model, tokenizer


def run_query(model,
              tokenizer,
              vals_dic: List[Dict],
              prompt: str,
              bs: int = 20) -> Tuple(List[Dict], List[Dict]):
    """Obtains the model's representation of the object in the prompt.
    Args:
        model: the model that we're going to query.
        tokenizer: the tokenizer to prepare the input to the model.
        vals_dic: List containing the tuples subject-object.
        prompt: the template that's going to be populated with the data in
            vals_dic.
    Returns:
        A list with the output representation of the mask token in each of the
        phrases (populated prompt).
    """
    data = []

    mask_token = tokenizer.mask_token

    # create the text prompt
    for sample in vals_dic:
        data.append({
            'prompt':
            parse_prompt(prompt, sample["sub_label"], mask_token),
            'sub_label':
            sample["sub_label"],
            'obj_label':
            sample["obj_label"]
        })

    predictions = []
    for batch in tqdm(data):
        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            tokenized_text = tokenizer.tokenize(batch['prompt'])
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])

            outputs = model(tokens_tensor)
            encoded_layers = outputs[0]

        mask_ind = [
            ind for ind, x in enumerate(tokenized_text) if x == mask_token
        ]
        assert len(mask_ind) == 1
        mask_ind = mask_ind[0]

        mask_vec = encoded_layers[0, mask_ind]

        predictions.append(mask_vec.detach().cpu().numpy())

    return np.array(predictions)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--lm",
                       type=str,
                       help="name of the used masked language model",
                       default="bert-base-uncased")
    parse.add_argument("--patterns_file",
                       type=str,
                       help="Path to templates for each prompt",
                       default="data/pattern_data/graphs_json/P106.jsonl")
    parse.add_argument("--data_file",
                       type=str,
                       help="",
                       default="data/trex_lms_vocab/P106.jsonl")
    parse.add_argument("--pred_file",
                       type=str,
                       help="Path to store LM predictions for each prompt")
    parse.add_argument("--wandb", action='store_true')

    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    # Load data
    data = utils.read_jsonl_file(args.data_file)

    # Load prompts
    prompts = utils.load_prompts(args.patterns_file)

    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    results_dict = {}
    model, tokenizer = build_model_by_name(model_name)

    results_dict[model_name] = {}

    representations = []
    for prompt_id, prompt in enumerate(tqdm(prompts)):
        predictions = run_query(model, tokenizer, data, prompt)
        representations.append(predictions)

    representations = np.array(representations)
    np.save(args.pred_file, representations)


if __name__ == '__main__':
    main()
