import argparse
import json
import os
import re
from glob import glob

from logger_utils import get_logger
from tqdm import tqdm

LOG = get_logger(__name__)
TUPLES_FOLDER_NAME = "tuples"
PATTERNS_FOLDER_NAME = "patterns"
VALID_RELATIONS = set([
    'P937', 'P1412', 'P127', 'P103', 'P276', 'P159', 'P140', 'P136', 'P495',
    'P17', 'P361', 'P36', 'P740', 'P264', 'P407', 'P138', 'P30', 'P131',
    'P176', 'P449', 'P279', 'P19', 'P101', 'P364', 'P106', 'P1376', 'P178',
    'P37', 'P413', 'P27', 'P20', 'P190', 'P1303', 'P39', 'P108', 'P463',
    'P530', 'P47'
])


def is_template_valid(template):
    """Checks that the template has one [X], one [Y], and extra text."""
    return (template.count("[X]") == 1 and template.count("[Y]") == 1
            and not re.match(r'^[\b\[X\]\b\b\[Y\]\b., ]+$', template))


def clean_template(data):
    """Cleans the pattern and keeps only relevant keys."""
    template = data["pattern"].lower()
    # Remove extra spaces and extra brackets from the subject/object.
    template = re.sub('\[+ ?[X] ?\]+', '[X]', template)
    template = re.sub('\[+ ?[Y] ?\]+', '[Y]', template)
    # Remove final puntuaction
    template = re.sub(r'[.:。]', '', template)
    # Remove extra spaces
    template = re.sub(r' +', ' ', template)
    template = re.sub(r' $', '', template)
    data["pattern"] = template
    return data


def get_cleaned_valid_templates(templates_lines):
    """Returns the valid non repeated templates."""
    clean_templates_lines = []
    patterns = set()
    for line in templates_lines:
        if (is_template_valid(line["pattern"])
                and clean_template(line)["pattern"] not in patterns):
            clean_templates_lines.append(clean_template(line))
            patterns.add(line["pattern"])
    return clean_templates_lines


def clean_triple(line):
    """Keeps only relevant keys from the entities."""
    data = json.loads(line)
    relevant_keys = {"obj_label", "sub_label", "obj_uri", "sub_uri"}
    result = {
        k: v
        for k, v in data.items()
        if k in relevant_keys and data["from_english"] is False
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuples_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to folder where to find the translated "
                        "tuples, it should contain a folder for each language "
                        "code, inside these there is a json file for each "
                        "relation with the translated tuples.")
    parser.add_argument(
        "--templates_folders_glob",
        default="templates",
        type=str,
        help="The glob that contains all the folders with the "
        "translations of the templates. Each translation folder"
        " contains a folder by language, where there is a json "
        "file for each relation with the translated templates.")
    parser.add_argument("--out_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="The folder where to write the clean tuples and "
                        "the valid templates.")
    args = parser.parse_args()

    # Copy templates files to the output dir.
    LOG.info("Cleaning templates from the folders:\n{}".format('\n'.join(
        glob(args.templates_folders_glob))))
    for translation_folder in tqdm(glob(args.templates_folders_glob)):
        translator_name = os.path.basename(translation_folder)
        for language_dirname in os.listdir(translation_folder):
            language_folder_path = os.path.join(translation_folder,
                                                language_dirname)
            for relation_filename in os.listdir(language_folder_path):
                if relation_filename[:-len(".jsonl")] not in VALID_RELATIONS:
                    LOG.info(
                        f"Ignoring relation '{relation_filename}' as it's "
                        "not in VALID_RELATIONS.")
                    continue
                templates_filename = os.path.join(language_folder_path,
                                                  relation_filename)
                output_templates_filename = os.path.join(
                    args.out_folder, PATTERNS_FOLDER_NAME, translator_name,
                    language_dirname, relation_filename)
                os.makedirs(output_templates_filename[:-len(
                    os.path.basename(output_templates_filename))],
                            exist_ok=True)
                with open(templates_filename) as templates_file, \
                        open(output_templates_filename, "w") as fout:
                    cleaned_templates = get_cleaned_valid_templates(
                        [json.loads(line) for line in templates_file])
                    for line in cleaned_templates:
                        fout.write("{}\n".format(json.dumps(line)))
    LOG.info("Templates have been written to the output folder.")
    # Copy subject-object file to the output folder.
    LOG.info("Writing cleaned subject-object tuples to the output folder.")
    for language_dirname in tqdm(os.listdir(args.tuples_folder)):
        if os.path.isfile(language_dirname):
            continue
        # If a folder with the translated tuples already exists then we don't
        # write it again.
        if os.path.isdir(
                os.path.join(args.out_folder, TUPLES_FOLDER_NAME,
                             language_dirname)):
            continue
        language_folder_path = os.path.join(args.tuples_folder,
                                            language_dirname)
        if not os.path.isdir(language_folder_path):
            continue
        os.makedirs(
            os.path.join(args.out_folder, TUPLES_FOLDER_NAME,
                         language_dirname))
        for relation_filename in os.listdir(language_folder_path):
            tuples_filename = os.path.join(language_folder_path,
                                           relation_filename)
            output_tuples_filename = os.path.join(args.out_folder,
                                                  TUPLES_FOLDER_NAME,
                                                  language_dirname,
                                                  relation_filename)
            with open(tuples_filename) as tuples_file, \
                    open(output_tuples_filename, "w") as fout:
                for i, line in enumerate(tuples_file):
                    triple = clean_triple(line)
                    if triple:
                        triple["lineid"] = i
                        fout.write("{}\n".format(json.dumps(triple)))


if __name__ == '__main__':
    main()
