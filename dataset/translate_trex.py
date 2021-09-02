import argparse
from logging import warning
from relations import Relations
from typing import Text
import tqdm
import os
import json
import collections
from logger_utils import get_logger
from translate_utils import get_wiki_languages

LOG = get_logger(__name__)


def get_entity_surface(basepath: Text, uri: Text, language: Text) -> Text:
    filename = os.path.join(basepath, uri + ".json")
    if not os.path.isfile(filename):
        LOG.info("File {} not found, skipping entity.")
        return ""
    try:
        with open(filename) as fp:
            data = json.load(fp)
        if uri not in data['entities']:
            LOG.info("File '{}' doesn't contain uri '{}', going to use the uri "
                     "found instead '{}'".format(filename, uri,
                                                 data['entities'].keys()))
            if len(data['entities'].keys()) > 1:
                LOG.warning("Defaulting to first uri from the list available.")
            uri = list(data['entities'].keys())[0]
        surfaces = data['entities'][uri]['labels']
        if language in surfaces:
            if surfaces[language]["language"] != language:
                raise Warning("Language mismatch in data: {}".format(surfaces))
            return surfaces[language]["value"]
        else:
            return ""
    except Exception as e:
        LOG.info("Catched exception: {} {}".format(type(e), e))
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        type=str, required=True, help="")
    parser.add_argument("--entities", default=None,
                        type=str, required=True, help="")
    parser.add_argument("--outpath", default=None,
                        type=str, required=True, help="")
    parser.add_argument("--languagemapping", default=None,
                        type=str, required=True, help="")
    args = parser.parse_args()
    wiki_languages = get_wiki_languages(args.languagemapping)

    for lang in wiki_languages:
        t = Relations(args.data)
        filenames = t.get_available_filenames()
        t.load_data(filenames)
        count = collections.Counter()
        logfile = open(os.path.join(args.outpath, lang + ".log"), "w")
        for filename, relations in t.data.items():
            LOG.info("Processing relation: {}".format(filename))
            outdirectory = os.path.join(args.outpath, lang)
            os.makedirs(outdirectory, exist_ok=True)
            with open(os.path.join(outdirectory, filename + ".jsonl"), "w") as fout:
                for relation in relations:
                    count["in_file"] += 1
                    if ("sub_uri" in relation and "obj_uri" in relation and "sub_label" in relation and "obj_label" in relation):
                        count["available"] += 1
                        obj_uri = relation["obj_uri"]
                        sub_uri = relation["sub_uri"]
                        # load entitiy information
                        obj_surface = get_entity_surface(
                            args.entities, obj_uri, lang)
                        sub_surface = get_entity_surface(
                            args.entities, sub_uri, lang)
                        # write out
                        if obj_surface and sub_surface:
                            count["converted"] += 1
                            to_write = {"sub_uri": sub_uri, "obj_uri": obj_uri,
                                        "obj_label": obj_surface, "sub_label": sub_surface, "from_english": False}
                        else:
                            # use english surface forms
                            to_write = {"sub_uri": sub_uri, "obj_uri": obj_uri,
                                        "obj_label": relation["obj_label"], "sub_label": relation["sub_label"], "from_english": True}
                        fout.write(json.dumps(to_write) + "\n")
            summary = "{}|{}|{}|(converted/available/in_file)".format(
                count["converted"], count["available"], count["in_file"])
            LOG.info(summary)
            logfile.write("{}|{}\n".format(filename, summary))
        logfile.close()


if __name__ == '__main__':
    main()
