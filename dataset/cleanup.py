import argparse
import os
import json
from utils import get_logger

LOG = get_logger(__name__)


def clean_triple(line):
    data = json.loads(line)
    relevant_keys = {"obj_label", "sub_label", "obj_uri", "sub_uri"}
    result = {k: v for k, v in data.items(
    ) if k in relevant_keys and data["from_english"] is False}
    return result


def clean_relation(line):
    data = json.loads(line)
    relevant_keys = {"relation", "template"}
    result = {k: v for k, v in data.items() if k in relevant_keys}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infolder", default=None,
                        type=str, required=True, help="")
    parser.add_argument("--outfolder", default=None,
                        type=str, required=True, help="")
    args = parser.parse_args()

    langs = [x[len("relations_"):-len(".jsonl")] for x in os.listdir(
        os.path.join(args.infolder, "templates")) if "relations_" in x]
    relations = [x.replace(".jsonl", "")
                 for x in os.listdir(os.path.join(args.infolder, "en"))]

    for lang in langs:
        os.makedirs(os.path.join(args.outfolder, lang))

    LOG.info("langs: {}".format(langs))
    LOG.info("relations: {}".format(relations))
    for lang in langs:
        LOG.info(lang)
        # transfer triples
        for relation in relations:
            current_path = os.path.join(
                args.infolder, lang, relation + ".jsonl")
            if os.path.exists(current_path):
                with open(current_path) as fin:
                    with open(os.path.join(args.outfolder, lang, relation + ".jsonl"), "w") as fout:
                        for i, line in enumerate(fin):
                            triple = clean_triple(line)
                            if triple:
                                triple["lineid"] = i
                                fout.write("{}\n".format(json.dumps(triple)))
            else:
                LOG.debug("The file doesn't exists: {}".format(current_path))
        # transfer templates
        with open(os.path.join(args.outfolder, lang, "templates.jsonl"), "a") as fout:
            f = os.path.join(args.infolder, "templates",
                             "relations_{}.jsonl".format(lang))
            if os.path.exists(f):
                with open(os.path.join(args.infolder, "templates", "relations_{}.jsonl".format(lang))) as fin:
                    for line in fin:
                        template = clean_relation(line)
                        fout.write("{}\n".format(json.dumps(template)))
            else:
                LOG.debug("The file doesn't exists: {}".format(f))


if __name__ == '__main__':
    main()
