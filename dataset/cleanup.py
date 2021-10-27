import argparse
import json
import os
from glob import glob

from logger_utils import get_logger
from tqdm import tqdm
from mpararel_utils import is_template_valid, clean_template, VALID_RELATIONS

LOG = get_logger(__name__)
TUPLES_FOLDER_NAME = "tuples"
PATTERNS_FOLDER_NAME = "patterns"
CHINESE_CORRECTION = {'zh': 'zh-hans', 'zh-classical': 'zh-hant'}


def clean_template_line(data):
    """Cleans the pattern and keeps only relevant keys."""
    data["pattern"] = clean_template(data["pattern"])
    return data


def get_cleaned_valid_templates(templates_lines):
    """Returns the valid non repeated templates."""
    clean_templates_lines = []
    patterns = set()
    for line in templates_lines:
        if (is_template_valid(line["pattern"])
                and clean_template_line(line)["pattern"] not in patterns):
            clean_templates_lines.append(clean_template_line(line))
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
    parser.add_argument("--correct_chinese_wiki_code",
                        default=False,
                        action='store_true')
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
                out_language_dirname = language_dirname
                if language_dirname in CHINESE_CORRECTION and args.correct_chinese_wiki_code:
                    out_language_dirname = CHINESE_CORRECTION[language_dirname]
                output_templates_filename = os.path.join(
                    args.out_folder, PATTERNS_FOLDER_NAME, translator_name,
                    out_language_dirname, relation_filename)
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
