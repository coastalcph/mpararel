import argparse
import os
import json
from utils import get_logger

LOG = get_logger(__name__)


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
    relevant_keys = {"relation", "template"}
    result = {k: v for k, v in data.items() if k in relevant_keys}
    return result


def get_languages(templates_folder):
    langs = []
    for filename in os.listdir(templates_folder):
        if "relations_" in filename:
            langs.append(filename[len("relations_"):-len(".jsonl")])
    return langs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infolder", default=None,
                        type=str, required=True, help="")
    parser.add_argument("--templates_folder_name", default="templates",
                        type=str, help="")
    parser.add_argument("--outfolder", default=None,
                        type=str, required=True, help="")
    args = parser.parse_args()

    langs = get_languages(os.path.join(
        args.infolder, args.templates_folder_name))
    relations = [x.replace(".jsonl", "")
                 for x in os.listdir(os.path.join(args.infolder, "en"))]

    for lang in langs:
        os.makedirs(os.path.join(args.outfolder, lang))

    LOG.info("langs: {}".format(langs))
    LOG.info("relations: {}".format(relations))
    for lang in langs:
        LOG.info("Copying data for language: {}".format(lang))
        # Copy subject-object file to the output folder.
        for relation in relations:
            entities_filename = os.path.join(
                args.infolder, lang, relation + ".jsonl")
            output_entities_filename = os.path.join(
                args.outfolder, lang, relation + ".jsonl")
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
        # Copy templates files to the output dir.
        templates_filename = os.path.join(
            args.infolder, args.templates_folder_name,
            "relations_{}.jsonl".format(lang))
        output_templates_filename = os.path.join(
            args.outfolder, lang, "templates.jsonl")
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
