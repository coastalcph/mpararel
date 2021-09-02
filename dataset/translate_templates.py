import collections
from googletrans import Translator
import json
import os
import argparse
from utils import get_logger
from tqdm import tqdm
import re

LOG = get_logger(__name__)


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
        template = template.replace(
            "Naglalaro ang [X] sa posisyon.", "Naglalaro si [X] sa posisyon na [Y]", 1)
        template = template.replace(
            "Sumali sa [X] ang [X].", "Sumali ang [X] sa [Y].", 1)
        template = template.replace(
            "Naglalaro ang [X] ng musika.", "Naglalaro si [X] ng [Y] musika.", 1)
        template = template.replace(
            "Naglalaro ang [X].", "Ginawa ni [X] ang [Y].", 1)
    if lang == "el":
        template = template.replace("[Χ]", "[X]", 1)
        template = template.replace("[Υ]", "[Y]", 1)
        if "[Y]" in template and "[X]" not in template:
            template = template.replace("[Ο]", "[X]", 1)
        if "[X]" in template and "[Y]" not in template:
            template = template.replace("[Ο]", "[Y]", 1)
    if lang == "ceb":
        # to be checked
        template = template.replace(
            "Natawo sa [Y].", "Natawo ang [X] sa [Y].", 1)
        template = template.replace(
            "Nag-apil sa [X] ang [X].", "Ang [X] miapil sa [Y].", 1)

    if lang == "pa":
        template = template.replace("[ਐਕਸ]", "[X]", 1)
        template = template.replace("[ਵਾਈ]", "[Y]", 1)
    if lang == "ta":
        template = template.replace("[எக்ஸ்]", "[X]", 1)
        template = template.replace("[ஒய்]", "[Y]", 1)
    if lang == "mg":
        template = template.replace(
            "Tamin'ny voalohany, nalefan'i [Y] tany am-boalohany.", "Tamin'ny voalohany, ny X [X] dia nalefa tamin'ny [Y].", 1)
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
        template = template.replace(
            "[X] је рођен у И.", "[X] је рођен у [Y].", 1)
    if lang == "kk":
        template = template.replace(
            "[Х] университетте білім алған.", "[X] [Y] университетінде білім алған.", 1)
        template = template.replace(
            "Ана тілі [Х] болып табылады.", "[Х] -дің ана тілі - [Y].", 1)
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
        template = template.replace(
            "[X] universitetida tahsil olgan.", "[X] [Y] universitetida tahsil olgan.", 1)
        template = template.replace(
            "[X] din bilan bog'liq.", "[X] [Y] diniga mansub.", 1)
    if lang == "tg":
        template = template.replace(
            "[X] аз рӯи касб аст.", "[X] аз рӯи касб [Y] аст.", 1)
        template = template.replace("[Ю]", "[Y]", 1)
        template = template.replace("[Х]", "[X]", 1)
        template = template.replace("[Y]", "[Y]", 1)
    if lang == "lt":
        template = template.replace(
            "Buvo įgijęs išsilavinimą [Y] universitete.", "[X] įgijo išsilavinimą [Y] universitete.", 1)
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


def clean_dir(args):
    for directory in os.listdir(args.templates_folder):
        LOG.info("Cleaning directory {}".format(directory))
        clean_folder(os.path.join(args.templates_folder, directory),
                     args.template_json_key)


def clean(args):
    clean_folder(args.templates_folder, args.template_json_key)


def clean_folder(templates_folder, template_json_key):
    initial_broken = 0
    final_broken = 0
    inital_incorrect_files = set()
    final_incorrect_files = set()
    for file in os.listdir(templates_folder):
        new_templates = []
        with open(os.path.join(templates_folder, file), "r") as fp:
            for i, line in enumerate(fp):
                if not line:
                    continue
                template = json.loads(line)
                if (template[template_json_key].count("[X]") == 1 and
                        template[template_json_key].count("[Y]") == 1):
                    new_templates.append(template)
                    continue
                initial_broken += 1
                inital_incorrect_files.add(file)
                template[template_json_key] = fix_template(
                    template[template_json_key],
                    get_language_from_filename(templates_folder, file))
                if (template[template_json_key].count("[X]") != 1 or
                        template[template_json_key].count("[Y]") != 1):
                    LOG.info(
                        "Wasn't able to fix [{} (line = {})]: {}".format(
                            file, i, template))
                    final_broken += 1
                    final_incorrect_files.add(file)
                new_templates.append(template)
        if file not in inital_incorrect_files:
            continue
        # Overwrite the fixed template.
        with open(os.path.join(templates_folder, file), "w") as fp:
            for line in new_templates:
                fp.write(json.dumps(line) + "\n")
    LOG.info("Initial broken templates {} (in {} files)".format(
        initial_broken, len(inital_incorrect_files)))
    LOG.info("Remain broken templates {} (in {} files)".format(
        final_broken, len(final_incorrect_files)))


def get_language_mapping(language_mapping_filename):
    lang2translateid = {}
    with open(language_mapping_filename) as fp:
        next(fp)
        for line in fp:
            if line:
                wikiid, _, _, googleid = line.split("\t")
                if not googleid:
                    # try the other id and see what comes out of goole translate
                    googleid = wikiid
                lang2translateid[wikiid.strip()] = googleid.strip()
    return lang2translateid


def get_templates(templates_filename):
    templates = []
    with open(templates_filename) as fp:
        for line in fp:
            if line:
                templates.append(json.loads(line))
    return templates


def translate_templates(templates, lang2translateid, template_key):
    """Translates each template to all the languages in lang2translateid."""
    translated_templates = {}
    for wikiid, googleid in lang2translateid.items():
        LOG.info("Translating {}".format(wikiid))
        translated_templates[wikiid] = []
        for template in templates:
            try:
                translator = Translator()
                result = translator.translate(
                    template[template_key], src="en", dest=googleid)
                translated_template = template.copy()
                translated_template[template_key] = result.text
            except Exception as e:
                LOG.info("Exception: {}".format(e))
                break
            translated_templates[wikiid].append(translated_template)
        if len(translated_templates[wikiid]) != len(templates):
            LOG.warning("Skipping language, not all translations succesful!")
            continue
    return translated_templates


def translate_folder(args):
    lang2translateid = get_language_mapping(args.languagemapping)
    wikiid_to_filename_to_templates = collections.defaultdict(dict)
    translations_count = 0
    for filename in tqdm(os.listdir(args.templates_folder)):
        templates = get_templates(os.path.join(
            args.templates_folder, filename))
        translations_count += len(templates)*len(lang2translateid)
        LOG.info("Translating file: {}, (will reach {} translations)".format(
            filename, translations_count))
        for wikiid, this_file_templates in translate_templates(
                templates, lang2translateid, "pattern").items():
            wikiid_to_filename_to_templates[wikiid][filename] = this_file_templates
    for wikiid, filename_to_templates in wikiid_to_filename_to_templates.items():
        os.makedirs(os.path.join(args.out_folder, wikiid))
        for filename, templates in filename_to_templates.items():
            output_filename = os.path.join(args.out_folder, wikiid, filename)
            with open(output_filename, "w") as fout:
                for template in templates:
                    fout.write("{}\n".format(json.dumps(template)))


def translate(args):
    lang2translateid = get_language_mapping(args.languagemapping)
    templates = get_templates(args.templates)
    wikiid_to_translated = translate_templates(
        templates, lang2translateid, "template")
    for wikiid, translated in wikiid_to_translated.items():
        with open(os.path.join(args.outfile, "relations_{}.jsonl".format(wikiid)), "w") as fout:
            for template in translated:
                fout.write("{}\n".format(json.dumps(template)))


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_translate = subparsers.add_parser('translate')
    parser_translate.set_defaults(func=translate)
    parser_translate.add_argument(
        "--templates", default=None, type=str, required=True, help="")
    parser_translate.add_argument(
        "--languagemapping", default=None, type=str, required=True, help="")
    parser_translate.add_argument(
        "--outfile", default=None, type=str, required=True, help="")

    parser_translate_folder = subparsers.add_parser('translate_folder')
    parser_translate_folder.set_defaults(func=translate_folder)
    parser_translate_folder.add_argument(
        "--templates_folder", default=None, type=str, required=True, help="")
    parser_translate_folder.add_argument(
        "--languagemapping", default=None, type=str, required=True, help="")
    parser_translate_folder.add_argument(
        "--out_folder", default=None, type=str, required=True, help="")

    parser_clean = subparsers.add_parser('clean')
    parser_clean.set_defaults(func=clean)
    parser_clean.add_argument(
        "--templates_folder", default=None, type=str, required=True, help="")
    parser_clean.add_argument(
        "--template_json_key", default="template", type=str,
        help="The key that contains the template or pattern in each line of the"
             " templates files.")

    parser_clean_dir = subparsers.add_parser('clean_dir')
    parser_clean_dir.set_defaults(func=clean_dir)
    parser_clean_dir.add_argument(
        "--templates_folder", default=None, type=str, required=True, help="")
    parser_clean_dir.add_argument(
        "--template_json_key", default="pattern", type=str,
        help="The key that contains the template or pattern in each line of the"
             " templates files.")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)
