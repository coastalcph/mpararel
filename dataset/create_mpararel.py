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
    language_and_relation_counts = []
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
            language_and_relation_counts.append(
                (language, relation, agreed_templates_count,
                 not_agreed_templates_count,
                 lang_to_translators_count[language]))
    return (agreed_translations,
            pd.DataFrame(language_and_relation_counts,
                         columns=[
                             'language', 'relation', 'agreed_templates_count',
                             'not_agreed_templates_count', 'translators_count'
                         ]))


def get_languages_to_tuples_count(tuples_folder):
    """Returns a list with the language, relation, and the number of subject-object pairs."""
    lang_to_relation_to_tuples_count = []
    LOG.info(
        "Counting subject object pairs in each relation of each language.")
    for language_folder in tqdm(os.listdir(tuples_folder)):
        for relation_file in os.listdir(
                os.path.join(tuples_folder, language_folder)):
            templates_filename = os.path.join(tuples_folder, language_folder,
                                              relation_file)
            with open(templates_filename) as templates_file:
                lang_to_relation_to_tuples_count.append(
                    (language_folder, relation_file[:-len(".jsonl")],
                     len([json.loads(line) for line in templates_file])))
    return lang_to_relation_to_tuples_count


def add_tuples_counts(df, tuples_folder):
    lang_relation_tuples_count = pd.DataFrame(
        get_languages_to_tuples_count(tuples_folder),
        columns=["language", "relation", "tuples_count"])
    return pd.merge(df,
                    lang_relation_tuples_count,
                    on=["language", "relation"],
                    how="left")


def add_ratio_column(df, count_column, base_lang="en"):
    """Adds a column with the rate: count_column/count_column[base_lang]."""
    new_column_name = f'{count_column[:-len("count")]}rate'
    en_total = df[df['language'] == base_lang][count_column].values
    df[new_column_name] = -1
    for language in df['language'].unique():
        df.loc[df['language'] == language, new_column_name] = (
            df[df['language'] == language][count_column] / en_total)


def get_language_and_relations_count(valid_df):
    language_relations_count = []
    for language in valid_df['language'].unique():
        relations_count = len(valid_df[valid_df['language'] == language])
        language_relations_count.append((language, relations_count))
    return language_relations_count


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
        "--tuples_folder",
        default=None,
        type=str,
        required=True,
        help="The folder with the subject and object pairs. It contains folders"
        " named by language code, where each inside has a file for each "
        "relation tuples.")
    parser.add_argument(
        "--min_templates_per_relation",
        type=float,
        default=0.2,
        help="The minimum number of templates that a relation (in each "
        "language) needs to have to be considered valid. The number is the "
        "fraction (0-1) of templates compared to the english total available "
        "for that relation.")
    parser.add_argument(
        "--min_phrases_per_relation",
        type=float,
        default=0.2,
        help="The minimum number of phrases (templates populated with the "
        "available subject and object pairs) that a relation (in each language)"
        " needs to have to be considered valid. The number is the fraction "
        "(0-1) of templates compared to the english total available for that "
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

    df = add_tuples_counts(df, args.tuples_folder)
    df["phrases_count"] = df["agreed_templates_count"] * df["tuples_count"]

    # Add ratio compared to the count in english.
    df = df.sort_values(by=['language', 'relation'])
    add_ratio_column(df, 'agreed_templates_count')
    add_ratio_column(df, 'phrases_count')

    # Filter relations that don't have enough templates.
    enough_templates_df = df[
        df['agreed_templates_rate'] >= args.min_templates_per_relation]
    LOG.info(
        "From a total of {} relations across all languages, {} relations have "
        "at least {} of the total patterns for the same relation in "
        "english.".format(len(df), len(enough_templates_df),
                          args.min_templates_per_relation))
    # Filter relations that don't have enough phrases.
    valid_df = enough_templates_df[
        df['phrases_rate'] >= args.min_phrases_per_relation]
    LOG.info(
        "From a total of {} relations across all languages, {} relations have "
        "at least {} of the total phrases for the same relation in "
        "english.".format(len(enough_templates_df), len(valid_df),
                          args.min_phrases_per_relation))
    # Filter languages that don't have enough valid relations.
    valid_df = pd.DataFrame(get_language_and_relations_count(valid_df),
                            columns=['language', 'relations_count'])
    en_relations_count = valid_df[valid_df['language'] ==
                                  'en'].relations_count.values[0]
    min_relations_count = en_relations_count * args.min_relations_count
    valid_languages = valid_df[
        valid_df['relations_count'] >= min_relations_count].language.values
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
