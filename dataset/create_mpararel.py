"""Writes mpararel dataset files based on the agreements between multiple
translations.
"""
import argparse
import collections
import json
import os
import re
from glob import glob

import pandas as pd
from logger_utils import get_logger
from tqdm import tqdm

LOG = get_logger(__name__)


# TODO: we should move this cleaning to the cleanup.py
def clean_template(template):
    template = re.sub(r'[.:]', '', template)
    template = re.sub(r' +', ' ', template)
    template = re.sub(r' $', '', template)
    return template


def get_agreed_translations_and_stats(translations_folders):
    relations = [
        x.replace(".jsonl", "")
        for x in os.listdir(os.path.join(translations_folders[0], "en"))
    ]
    lang_relation_counts = []
    agreed_translations = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for relation in tqdm(relations):
        lang_to_translations_to_votes = collections.defaultdict(
            lambda: collections.defaultdict(int))
        lang_to_translators_count = collections.defaultdict(int)
        all_languages = set()
        for translation_folder in translations_folders:
            for language_dirname in os.listdir(translation_folder):
                all_languages.add(language_dirname)
                patterns_file = os.path.join(translation_folder,
                                             language_dirname,
                                             relation + '.jsonl')
                if not os.path.exists(patterns_file):
                    continue
                lang_to_translators_count[language_dirname] += 1
                with open(patterns_file) as patterns:
                    for line in patterns:
                        data = json.loads(line)
                        # Since the patterns are unique for each trasnlation we
                        # only count one vote per translator.
                        lang_to_translations_to_votes[language_dirname][
                            clean_template(data["pattern"])] += 1
        for language in all_languages:
            # LOG.info("language '{}'".format(language))
            translations_to_votes = lang_to_translations_to_votes[language]
            agreed_templates_count = 0
            not_agreed_templates_count = 0
            for template_translation, votes in translations_to_votes.items():
                # LOG.info("({}) {}".format(votes, template_translation))
                if votes > 1:
                    agreed_templates_count += 1
                    agreed_translations[language][relation].append(
                        template_translation)
                else:
                    not_agreed_templates_count += 1
            lang_relation_counts.append(
                (language, relation, agreed_templates_count,
                 not_agreed_templates_count,
                 lang_to_translators_count[language]))
    return (agreed_translations,
            pd.DataFrame(lang_relation_counts,
                         columns=[
                             'language', 'relation', 'agreed_templates_count',
                             'not_agreed_templates_count', 'translators_count'
                         ]))


def write_mpararel(agreed_translations, out_folder):
    for language, relation_to_templates in agreed_translations.items():
        for relation, templates in relation_to_templates.items():
            this_folder = os.path.join(out_folder, language)
            os.makedirs(this_folder, exist_ok=True)
            with open(os.path.join(this_folder, relation + ".jsonl"),
                      'w') as fout:
                for template in templates:
                    fout.write("{}\n".format(json.dumps({"pattern":
                                                         template})))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--translations_folders_glob",
        default=None,
        type=str,
        required=True,
        help="The glob to find all the folders that contain translations. Each"
        " translation folder is expected to have folders named by language"
        " code, where each inside has a file for each relation templates.")
    parser.add_argument(
        "--min_templates_per_relation",
        type=float,
        default=0.2,
        help="The minimum number of templates per relation that a language has "
        "to have to be included. The number is the fraction (0-1) of "
        "templates compared to the english total available for that "
        "relation.")
    parser.add_argument(
        "--min_relations_count",
        type=float,
        default=1.0,
        help="The minimum number of valid relations that a language has to have"
        " to be included. A relation is consider valid if it has the "
        "minimum number of templates. The number is the fraction (0-1) of "
        "relations compared to the english total available.")
    parser.add_argument("--out_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    args = parser.parse_args()

    LOG.info("Getting agreed translations from folders:\n{}".format('\n'.join(
        glob(args.translations_folders_glob))))
    agreed_translations, df = get_agreed_translations_and_stats(
        glob(args.translations_folders_glob))
    for i in range(1, len(glob(args.translations_folders_glob)) + 1):
        LOG.info("Relations translated with {0} translators: {1:.2%}".format(
            i,
            len(df[df['translators_count'] == i]) / len(df)))

    # Add fraction of templates compared to the number of templates in english.
    df = df.sort_values(by=['language', 'relation'])
    en_total = df[df['language'] == 'en']['agreed_templates_count'].values
    df['agreed_templates_rate'] = -1
    for language in df['language'].unique():
        df.loc[df['language'] == language, 'agreed_templates_rate'] = (
            df[df['language'] == language]['agreed_templates_count'] /
            en_total)
    # Filter relations that don't have enough templates.
    valid_df = df[
        df['agreed_templates_rate'] >= args.min_templates_per_relation]
    LOG.info(
        "From a total of {} relations across all languages, {} relations have "
        "at least {} of the total patterns for the same relation in "
        "english.".format(len(df), len(valid_df),
                          args.min_templates_per_relation))
    # Filter languages that don't have enough relations.
    language_relations_count = []
    for language in df['language'].unique():
        relations_count = len(valid_df[valid_df['language'] == language])
        language_relations_count.append((language, relations_count))
    valid_df = pd.DataFrame(language_relations_count,
                            columns=['language', 'relations_count'])
    en_relations_count = valid_df[valid_df['language'] ==
                                  'en'].relations_count.values[0]
    valid_languages = valid_df[
        valid_df['relations_count'] >= en_relations_count *
        args.min_relations_count].language.values
    LOG.info(
        "{} languages have at least {}*{} relations with the minimum number of"
        " patterns.".format(len(valid_languages), en_relations_count,
                            args.min_relations_count))
    LOG.info("The valid languages are: {}".format(valid_languages))
    # Remove the non valid languages from the translations.
    not_valid_languages = set(df.language.unique()).difference(
        set(valid_languages))
    for language in not_valid_languages:
        if language in agreed_translations:
            agreed_translations.pop(language)
    LOG.info("Writing valid languages data to the output folder.")
    # Write mpararel.
    write_mpararel(agreed_translations, args.out_folder)


if __name__ == '__main__':
    main()
