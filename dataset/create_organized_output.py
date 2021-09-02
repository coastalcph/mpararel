import argparse
import os
import json
from tqdm import tqdm
from logger_utils import get_logger

LOG = get_logger(__name__)
TRIPLES_FOLDER_NAME = "triples"
PATTERNS_FOLDER_NAME = "patterns"


def clean_triple(line):
    """Keeps only relevant keys from the entities."""
    data = json.loads(line)
    relevant_keys = {"obj_label", "sub_label", "obj_uri", "sub_uri"}
    result = {k: v for k, v in data.items(
    ) if k in relevant_keys and data["from_english"] is False}
    return result


def clean_relation(line):
    """Keeps only relevant keys from the template."""
    data = json.loads(line)
    relevant_keys = {"relation", "pattern"}
    result = {k: v for k, v in data.items() if k in relevant_keys}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infolder", default=None,
                        type=str, required=True, help="")
    parser.add_argument(
        "--templates_folder_name", required=True, type=str,
        help="The folder name that contains the templates inside infolder.")
    parser.add_argument("--outfolder", default=None,
                        type=str, required=True, help="")
    args = parser.parse_args()

    langs = os.listdir(os.path.join(
        args.infolder, args.templates_folder_name))
    relations = [x.replace(".jsonl", "")
                 for x in os.listdir(os.path.join(args.infolder, args.templates_folder_name, "en"))]

    for lang in langs:
        os.makedirs(os.path.join(args.outfolder, lang, TRIPLES_FOLDER_NAME))
        os.makedirs(os.path.join(args.outfolder, lang, PATTERNS_FOLDER_NAME))

    LOG.info("langs: {}".format(langs))
    LOG.info("relations: {}".format(relations))
    for lang in tqdm(langs):
        # Copy subject-object file to the output folder.
        for relation in relations:
            entities_filename = os.path.join(
                args.infolder, lang, relation + ".jsonl")
            output_entities_filename = os.path.join(
                args.outfolder, lang, TRIPLES_FOLDER_NAME, relation + ".jsonl")
            if os.path.exists(entities_filename):
                with open(entities_filename) as fin, \
                        open(output_entities_filename, "w") as fout:
                    for i, line in enumerate(fin):
                        triple = clean_triple(line)
                        if triple:
                            triple["lineid"] = i
                            fout.write("{}\n".format(json.dumps(triple)))
            else:
                LOG.debug("The file doesn't exists: {}".format(
                    entities_filename))
            # Copy patterns to the output dir.
            templates_filename = os.path.join(
                args.infolder, args.templates_folder_name, lang,
                relation + ".jsonl")
            output_templates_filename = os.path.join(
                args.outfolder, lang, PATTERNS_FOLDER_NAME, relation + ".jsonl")
            if os.path.exists(templates_filename):
                with open(templates_filename) as fin, \
                        open(output_templates_filename, "a") as fout:
                    for line in fin:
                        template = clean_relation(line)
                        fout.write("{}\n".format(json.dumps(template)))
            else:
                LOG.debug("The file doesn't exists: {}".format(
                    templates_filename))


if __name__ == '__main__':
    main()
