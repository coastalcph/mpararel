import argparse
import json
import os
import re
import traceback
from collections import defaultdict
from glob import glob
from typing import List, Tuple

import wandb
from logger_utils import get_logger
from tqdm import tqdm

from dataset.translate_populated_templates import translate_populated_template
from dataset.translate_utils import (TRANSLATOR_TO_OBJECT, Translator,
                                     get_wiki_language_mapping)

LOG = get_logger(__name__)

TRANSLATIONS_COUNT_LOG = "translations_count"


def fix_template(template, lang):
    # General rules.
    # Remove extra spaces and extra brackets, and capitalize.
    template = re.sub('\[+ ?[xX] ?\]+', '[X]', template)
    template = re.sub('\[+ ?[yY] ?\]+', '[Y]', template)
    if "[X]" not in template:
        template = template.replace("X", "[X]", 1)
    if "[Y]" not in template:
        template = template.replace("Y", "[Y]", 1)

    if lang == "tl":
        template = template.replace("Naglalaro ang [X] sa posisyon.",
                                    "Naglalaro si [X] sa posisyon na [Y]", 1)
        template = template.replace("Sumali sa [X] ang [X].",
                                    "Sumali ang [X] sa [Y].", 1)
        template = template.replace("Naglalaro ang [X] ng musika.",
                                    "Naglalaro si [X] ng [Y] musika.", 1)
        template = template.replace("Naglalaro ang [X].",
                                    "Ginawa ni [X] ang [Y].", 1)
    if lang == "el":
        template = template.replace("[Χ]", "[X]", 1)
        template = template.replace("[Υ]", "[Y]", 1)
        if "[Y]" in template and "[X]" not in template:
            template = template.replace("[Ο]", "[X]", 1)
        if "[X]" in template and "[Y]" not in template:
            template = template.replace("[Ο]", "[Y]", 1)
    if lang == "ceb":
        # to be checked
        template = template.replace("Natawo sa [Y].", "Natawo ang [X] sa [Y].",
                                    1)
        template = template.replace("Nag-apil sa [X] ang [X].",
                                    "Ang [X] miapil sa [Y].", 1)

    if lang == "pa":
        template = template.replace("[ਐਕਸ]", "[X]", 1)
        template = template.replace("[ਵਾਈ]", "[Y]", 1)
    if lang == "ta":
        template = template.replace("[எக்ஸ்]", "[X]", 1)
        template = template.replace("[ஒய்]", "[Y]", 1)
    if lang == "mg":
        template = template.replace(
            "Tamin'ny voalohany, nalefan'i [Y] tany am-boalohany.",
            "Tamin'ny voalohany, ny X [X] dia nalefa tamin'ny [Y].", 1)
    if lang == "gu":
        template = template.replace("[એક્સ]", "[X]", 1)
        template = template.replace("[વાય]", "[Y]", 1)
    if lang == "mr":
        template = template.replace("[एक्स]", "[X]", 1)
        template = template.replace("[वाई]", "[Y]", 1)
        template = template.replace("[वाय]", "[Y]", 1)
    if lang == "sr":
        template = template.replace("[Кс]", "[X]", 1)
        template = template.replace("[И]", "[Y]", 1)
        template = template.replace("[X] је рођен у И.", "[X] је рођен у [Y].",
                                    1)
    if lang == "kk":
        template = template.replace("[Х] университетте білім алған.",
                                    "[X] [Y] университетінде білім алған.", 1)
        template = template.replace("Ана тілі [Х] болып табылады.",
                                    "[Х] -дің ана тілі - [Y].", 1)
        template = template.replace("[Х]", "[X]", 1)
        template = template.replace("[Y]", "[Y]", 1)
    if lang == "kn":
        template = template.replace("[ಎಕ್ಸ್]", "[X]", 1)
        template = template.replace("[ವೈ]", "[Y]", 1)
    if lang == "ne":
        template = template.replace("[एक्स]", "[X]", 1)
        template = template.replace("[Y]", "[Y]", 1)
    if lang == "hy":
        template = template.replace("[X]", "[X]", 1)
        template = template.replace("[Յ]", "[Y]", 1)
    if lang == "uz":
        template = template.replace("[X] universitetida tahsil olgan.",
                                    "[X] [Y] universitetida tahsil olgan.", 1)
        template = template.replace("[X] din bilan bog'liq.",
                                    "[X] [Y] diniga mansub.", 1)
    if lang == "tg":
        template = template.replace("[X] аз рӯи касб аст.",
                                    "[X] аз рӯи касб [Y] аст.", 1)
        template = template.replace("[Ю]", "[Y]", 1)
        template = template.replace("[Х]", "[X]", 1)
        template = template.replace("[Y]", "[Y]", 1)
    if lang == "lt":
        template = template.replace(
            "Buvo įgijęs išsilavinimą [Y] universitete.",
            "[X] įgijo išsilavinimą [Y] universitete.", 1)
    if lang == "bn":
        template = template.replace("[এক্স]", "[X]", 1)
        template = template.replace("[ওয়াই]", "[Y]", 1)
    if lang == "la":
        template = template.replace("[K]", "[Y]", 1)
        template = template.replace("[A]", "[Y]", 1)
        template = template.replace("[N]", "[Y]", 1)
        template = template.replace("[V]", "[Y]", 1)
        template = template.replace("[ego]", "[Y]", 1)
        template = template.replace("[Ego]", "[Y]", 1)
    if lang == "hi":
        if "[X]" not in template:
            template = template.replace("[एक्स]", "[X]", 1)
        if "[Y]" not in template:
            template = template.replace("[वाई]", "[Y]", 1)
    if lang == "ko":
        template = template.replace("【", "[")
        template = template.replace("】", "]")
        # Remove extra spaces and extra brackets, and capitalize.
        template = re.sub('\[+ ?[xX] ?\]+', '[X]', template)
        template = re.sub('\[+ ?[yY] ?\]+', '[Y]', template)
    # Remove extra brackets that could have been added by the [X] and [Y]
    # replacements above.
    template = re.sub('\[+[X]\]+', '[X]', template)
    template = re.sub('\[+[Y]\]+', '[Y]', template)
    return template


def get_language_from_filename(folder, filename):
    if filename.startswith("relations_"):
        return filename[len("relations_"):-len(".jsonl")]
    elif filename.startswith("P"):
        # The direct folder of the file contains the language code.
        return os.path.basename(os.path.normpath(folder))


def fix_translated_dirs(args):
    initial_broken_count, final_broken_count = 0, 0
    for dir_to_fix in tqdm(glob(args.templates_folder_glob)):
        LOG.info("Cleaning folder {}".format(dir_to_fix))
        for sub_folder in tqdm(os.listdir(dir_to_fix)):
            initial_broken, final_broken = _fix_translated_files(
                os.path.join(dir_to_fix, sub_folder), args.template_json_key)
            initial_broken_count += initial_broken
            final_broken_count += final_broken
    LOG.info("Initial broken templates {} remain broken {}".format(
        initial_broken_count, final_broken_count))


def _fix_translated_files(templates_folder, template_json_key):
    initial_broken = 0
    final_broken = 0
    inital_incorrect_files = set()
    for file in os.listdir(templates_folder):
        new_templates = []
        with open(os.path.join(templates_folder, file), "r") as fp:
            for i, line in enumerate(fp):
                if not line:
                    continue
                template = json.loads(line)
                try:
                    if (template[template_json_key].count("[X]") == 1
                            and template[template_json_key].count("[Y]") == 1):
                        new_templates.append(template)
                        continue
                    initial_broken += 1
                    inital_incorrect_files.add(file)
                    template[template_json_key] = fix_template(
                        template[template_json_key],
                        get_language_from_filename(templates_folder, file))
                    if (template[template_json_key].count("[X]") != 1
                            or template[template_json_key].count("[Y]") != 1):
                        final_broken += 1
                    new_templates.append(template)
                except Exception as e:
                    LOG.info(
                        "Exception {} while trying to fix template '{}' in file"
                        " '{}'".format(e, template,
                                       os.path.join(templates_folder, file)))
                    raise Exception(e)
        if file not in inital_incorrect_files:
            continue
        # Overwrite the fixed template.
        with open(os.path.join(templates_folder, file), "w") as fp:
            for line in new_templates:
                fp.write(json.dumps(line) + "\n")
    return initial_broken, final_broken


def get_templates(templates_filename):
    templates = []
    with open(templates_filename) as fp:
        for line in fp:
            if line:
                templates.append(json.loads(line))
    return templates


def translate_template(template: str, template_key: str, translator,
                       translate_to_id) -> Tuple[dict, dict]:
    """Translates each template to all the languages in the dict values."""
    translated_text = translator.translate(template[template_key],
                                           from_lang="en",
                                           to_lang=translate_to_id)
    translated_template = template.copy()
    translated_template[template_key] = translated_text
    return translated_template


def init_wandb(args):
    translator = str(args.translator.name.lower())
    name = translator
    tags = [translator]
    if args.translate_populated_templates:
        name += "_populated"
        tags.append("populated")
    wandb.init(project="pararel-translations", name=name, tags=tags)
    wandb.config.update(args)


def wandb_log_table(value_name, values):
    data = [[i + 1, val] for i, val in enumerate(values)]
    table = wandb.Table(data=data, columns=["index", "value_name"])
    wandb.log({value_name: table})


def write_translated_templates(wikiid_to_filename_to_templates, output_folder):
    for wikiid, filename_to_templates in wikiid_to_filename_to_templates.items(
    ):
        os.makedirs(os.path.join(output_folder, wikiid), exist_ok=True)
        for filename, templates in filename_to_templates.items():
            output_filename = os.path.join(output_folder, wikiid, filename)
            with open(output_filename, "w") as fout:
                for template in templates:
                    fout.write("{}\n".format(json.dumps(template)))


def export_counts_to_wandb(relation_to_lang_to_counts):
    table_data = []
    for relation, lang_to_counts in relation_to_lang_to_counts.items():
        for lang, counts in lang_to_counts.items():
            for key, count in counts.items():
                table_data.append([relation, lang, key, count])
    wandb.log({
        "Templates translations summary":
        wandb.Table(
            data=table_data,
            columns=["Relation", "Language", "Result", "Templates count"])
    })

def translate_folder(args):
    wiki_lang_to_translator_lang = get_wiki_language_mapping(
        args.language_mapping_file, args.translator)
    if args.only_wiki_codes:
        wiki_lang_to_translator_lang = {
            k: v
            for k, v in wiki_lang_to_translator_lang.items()
            if k in args.only_wiki_codes
        }
    translator = TRANSLATOR_TO_OBJECT[args.translator]
    init_wandb(args)
    wandb_log_table("relations", os.listdir(args.templates_folder))
    wikiid_to_filename_to_templates = defaultdict(lambda: defaultdict(list))
    wandb_stats = defaultdict(int)
    relation_to_lang_to_counts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    relations = os.listdir(args.templates_folder)[args.skip_first_k_relations:]
    for i_filename, relation_filename in tqdm(
            enumerate(relations, args.skip_first_k_relations + 1)):
        templates = get_templates(
            os.path.join(args.templates_folder, relation_filename))
        for wikiid, translate_to_id in wiki_lang_to_translator_lang.items():
            for template in templates:
                print(relation_to_lang_to_counts)
                try:
                    if args.translate_populated_templates:
                        translated_templates = translate_populated_template(
                            template, "pattern", args.tuples_folder,
                            relation_filename, translator, translate_to_id)
                    else:
                        translated_template = translate_template(
                            template, "pattern", translator, translate_to_id)
                        translated_templates = [translated_template]
                except Exception as e:
                    LOG.info(
                        "Exception: '{}' when translating relation '{}' (template '{}') to "
                        "language='{}'".format(e, relation_filename, template,
                                               wikiid))
                    wandb_stats[str(e)] += 1
                    relation_to_lang_to_counts[
                        relation_filename[:-len(".jsonl")]][wikiid][str(
                            e)] += 1
                    print(traceback.format_exc())
                    continue
                relation_to_lang_to_counts[relation_filename[:-len(".jsonl")]][
                    wikiid]["successful"] += 1
                wikiid_to_filename_to_templates[wikiid][
                    relation_filename].extend(translated_templates)
                wandb_stats[TRANSLATIONS_COUNT_LOG] += len(
                    translated_templates)
        wandb_stats["relation_translated"] = i_filename
        wandb.log({k: v for k, v in wandb_stats.items() if isinstance(v, int)})
    LOG.info("Writing translated templates...")
    write_translated_templates(wikiid_to_filename_to_templates,
                               args.output_folder)
    LOG.info("Logging translations counts to wandb...")
    export_counts_to_wandb(relation_to_lang_to_counts)


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_translate_folder = subparsers.add_parser('translate_folder')
    parser_translate_folder.set_defaults(func=translate_folder)
    parser_translate_folder.add_argument("--templates_folder",
                                         default=None,
                                         type=str,
                                         required=True,
                                         help="")
    parser_translate_folder.add_argument("--language_mapping_file",
                                         default=None,
                                         type=str,
                                         required=True,
                                         help="")
    parser_translate_folder.add_argument("--translator",
                                         type=Translator,
                                         default=Translator.GOOGLE,
                                         choices=list(Translator))
    parser_translate_folder.add_argument(
        "--translate_populated_templates",
        action='store_true',
        default=False,
        help="Translate the templates with [X] and [Y] filled with subjects and"
        " objects.")
    parser_translate_folder.add_argument(
        "--tuples_folder",
        default=None,
        type=str,
        help="The folder where to find the subjects and objects per language "
        "per relation.")
    parser_translate_folder.add_argument("--output_folder",
                                         default=None,
                                         type=str,
                                         required=True,
                                         help="")
    parser_translate_folder.add_argument("--skip_first_k_relations",
                                         default=0,
                                         type=int,
                                         required=True,
                                         help="")
    parser_translate_folder.add_argument("--only_wiki_codes",
                                         nargs="*",
                                         default=[],
                                         help="")

    parser_clean_dir = subparsers.add_parser('fix_translated_dirs')
    parser_clean_dir.set_defaults(func=fix_translated_dirs)
    parser_clean_dir.add_argument(
        "--templates_folder_glob",
        default=None,
        type=str,
        required=True,
        help=
        "The glob to the directories with the translations. Each directory has"
        " a folder for each language containing a json files with the patterns"
        " translations.")
    parser_clean_dir.add_argument(
        "--template_json_key",
        default="pattern",
        type=str,
        help="The key that contains the template or pattern in each line of the"
        " templates files.")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)
