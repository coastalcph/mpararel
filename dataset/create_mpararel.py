"""Writes mpararel dataset files based on the agreements between multiple
translations.
"""
import argparse
import json
import os
from collections import defaultdict
from glob import glob
import wandb

import pandas as pd
from logger_utils import get_logger
from tqdm import tqdm

LOG = get_logger(__name__)


def filter_repeated_across_languages(agreed_translations,
                                     language_and_relation_counts):
    for i in range(len(language_and_relation_counts)):
        (language, relation, agreed_templates_count,
         translators_count) = language_and_relation_counts[i]
        remove_templates = []
        for template in agreed_translations[language][relation]:
            for other_relation, other_templates in agreed_translations[
                    language].items():
                if other_relation == relation:
                    continue
                if template in other_templates:
                    remove_templates.append(template)
        agreed_templates_count -= len(remove_templates)
        for template in remove_templates:
            agreed_translations[language][relation].remove(template)
        language_and_relation_counts[i] = (language, relation,
                                           agreed_templates_count,
                                           translators_count)
    return agreed_translations, language_and_relation_counts


def get_agreed_translations_and_stats(translations_folders):
    relations = [
        x.replace(".jsonl", "")
        for x in os.listdir(os.path.join(translations_folders[0], "en"))
    ]
    language_and_relation_counts = []
    agreed_translations = defaultdict(lambda: defaultdict(set))
    for relation in tqdm(relations):
        lang_to_translations_to_votes = defaultdict(lambda: defaultdict(int))
        lang_to_translators_count = defaultdict(int)
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
                        vote = 1
                        if ("bing" in translation_folder
                                and not "populated" in translation_folder):
                            vote = 2
                        lang_to_translations_to_votes[language_dirname][
                            data["pattern"]] += vote
        for language in all_languages:
            translations_to_votes = lang_to_translations_to_votes[language]
            agreed_templates_count = 0
            for template_translation, votes in translations_to_votes.items():
                if votes > 1:
                    agreed_templates_count += 1
                    agreed_translations[language][relation].add(
                        template_translation)
            language_and_relation_counts.append(
                (language, relation, agreed_templates_count,
                 lang_to_translators_count[language]))
    agreed_translations, counts = filter_repeated_across_languages(
        agreed_translations, language_and_relation_counts)
    return (agreed_translations,
            pd.DataFrame(counts,
                         columns=[
                             'language', 'relation', 'agreed_templates_count',
                             'translators_count'
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


def get_language_and_phrases_count(valid_df):
    language_relations_count = []
    for language in valid_df['language'].unique():
        phrases_count = sum(
            valid_df[valid_df['language'] == language].phrases_count)
        language_relations_count.append((language, phrases_count))
    return language_relations_count


def get_valid_langs(language_and_counts, min_en_fraction,
                    min_description_text):
    """Returns the languages that have the minimum fraction for the given counts compared to english."""
    valid_df = pd.DataFrame(language_and_counts,
                            columns=['language', 'this_count'])
    en_relations_count = valid_df[valid_df['language'] ==
                                  'en'].this_count.values[0]
    min_count = en_relations_count * min_en_fraction
    valid_languages = set(
        valid_df[valid_df['this_count'] >= min_count].language.values)
    LOG.info("{} languages have >= {}*{} number of {}".format(
        len(valid_languages), en_relations_count, min_en_fraction,
        min_description_text))
    return valid_languages


def write_mpararel(df_valid, agreed_translations, out_folder):
    """Writes to the output folder the translations in the df_valid."""
    for language in df_valid.language.unique():
        for relation in df_valid[df_valid.language ==
                                 language].relation.unique():
            templates = agreed_translations[language][relation]
            this_folder = os.path.join(out_folder, language)
            os.makedirs(this_folder, exist_ok=True)
            with open(os.path.join(this_folder, relation + ".jsonl"),
                      'w') as fout:
                for template in templates:
                    fout.write("{}\n".format(json.dumps({"pattern":
                                                         template})))


def log_statistics(df_valid, agreed_translations):
    df_data = []
    for language in df_valid.language.unique():
        for relation in df_valid[df_valid.language ==
                                 language].relation.unique():
            templates = agreed_translations[language][relation]
            #TODO: add syntactic and lexical variation.
            df_data.append((language, relation, len(templates)))
    df = pd.DataFrame(df_data, columns=["lang", "relation", "count_patterns"])
    wandb.run.summary["min #patterns in a relation"] = df[
        "count_patterns"].min()
    wandb.run.summary["max #patterns in a relation"] = df[
        "count_patterns"].max()
    wandb.run.summary["avg #patterns in a relation"] = df[
        "count_patterns"].mean()
    data = []
    columns = [
        "language", "#relations", "min #patterns", "max #patterns",
        "avg #patterns", "total patterns"
    ]
    for lang in df.lang.unique():
        this_df = df[df["lang"] == lang]
        relations = this_df.relation.unique()
        data.append(
            (lang, len(relations), this_df["count_patterns"].min(),
             this_df["count_patterns"].max(), this_df["count_patterns"].mean(),
             this_df["count_patterns"].sum()))
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"Patterns statistics per language": table})
    for column in columns[1:]:
        wandb.log(
            {column: wandb.plot.bar(table, "language", column, title=column)})


def log_translators_count_per_template(total_translators, df):
    translators_count = []
    for i in range(1, total_translators + 1):
        translators_count.append(
            [i, len(df[df['translators_count'] == i]) / len(df)])
    label = "#Translators used to translate template"
    value = "Templates count"
    table = wandb.Table(data=translators_count, columns=[label, value])
    wandb.log({
        "translators_count":
        wandb.plot.bar(
            table,
            label,
            value,
            title="Number of translators used to translate each template.")
    })


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
        default=0.0,
        help="The minimum number of templates that a relation (in each "
        "language) needs to have to be considered valid. The number is the "
        "fraction (0-1) of templates compared to the english total available "
        "for that relation.")
    parser.add_argument(
        "--min_phrases_per_relation",
        type=float,
        default=0.0,
        help="The minimum number of phrases (templates populated with the "
        "available subject and object pairs) that a relation (in each language)"
        " needs to have to be considered valid. The number is the fraction "
        "(0-1) of templates compared to the english total available for that "
        "relation.")
    parser.add_argument(
        "--min_relations_count",
        type=float,
        default=0.6,
        help="The minimum number of valid relations that a language has to have"
        " to be included. A relation is consider valid if it has the "
        "minimum number of templates. The number is the fraction (0-1) of "
        "relations compared to the english total available.")
    parser.add_argument(
        "--min_total_phrases",
        type=float,
        default=0.2,
        help="The minimum total number of phrases. The number is the fraction "
        "(0-1) of relations compared to the english total available.")
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--out_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    args = parser.parse_args()
    if not args.wandb_run_name:
        args.wandb_run_name = "mpararel_{}_{}_{}_{}".format(
            args.min_templates_per_relation, args.min_phrases_per_relation,
            args.min_relations_count, args.min_total_phrases)
    wandb.init(project="mpararel-creation", name=args.wandb_run_name)
    wandb.config.update(args)

    LOG.info("Getting agreed translations from folders:\n{}".format('\n'.join(
        glob(args.translations_folders_glob))))
    agreed_translations, df = get_agreed_translations_and_stats(
        glob(args.translations_folders_glob))

    log_translators_count_per_template(
        len(glob(args.translations_folders_glob)), df)

    df = add_tuples_counts(df, args.tuples_folder)
    df["phrases_count"] = df["agreed_templates_count"] * df["tuples_count"]

    # Add ratio compared to the count in english.
    df = df.sort_values(by=['language', 'relation'])
    add_ratio_column(df, 'agreed_templates_count')
    add_ratio_column(df, 'phrases_count')

    # Filter relations that don't have enough templates.
    enough_templates_df = df[df['agreed_templates_count'] > 1]
    enough_templates_df = enough_templates_df[
        enough_templates_df['agreed_templates_rate'] >=
        args.min_templates_per_relation]
    LOG.info(
        "From a total of {} relations across all languages, {} relations have "
        "more than 1 template and at least {} of the total patterns for the "
        "same relation in english.".format(len(df), len(enough_templates_df),
                                           args.min_templates_per_relation))
    # Filter relations that don't have enough phrases.
    valid_df = enough_templates_df[enough_templates_df['tuples_count'] > 0]
    valid_df = valid_df[
        valid_df['phrases_rate'] >= args.min_phrases_per_relation]
    LOG.info(
        "From a total of {} relations across all languages, {} relations have "
        "at least {} of the total phrases for the same relation in "
        "english.".format(len(enough_templates_df), len(valid_df),
                          args.min_phrases_per_relation))

    # Filter languages that don't have enough valid relations.
    enough_relations_langs = get_valid_langs(
        get_language_and_relations_count(valid_df), args.min_relations_count,
        "relations")
    # Filter languages that don't have enough total phrases.
    enough_phrases_langs = get_valid_langs(
        get_language_and_phrases_count(valid_df), args.min_total_phrases,
        "total phrases")
    valid_languages = enough_relations_langs.intersection(enough_phrases_langs)
    valid_df = valid_df[valid_df["language"].isin(valid_languages)]
    summary = {
        "#templates translated across all languages":
        len(df),
        "(1) #templates with minimum #templates per relation":
        len(enough_templates_df),
        "(2) #templates with minimum #phrases per relation and (1)":
        len(valid_df),
        "(3) #languages with minimum #relations and (2)":
        len(enough_relations_langs),
        "(4) #languages with minimum total #phrases and (2)":
        len(enough_phrases_langs),
        "valid #languages [(3) and (4)]":
        len(valid_languages)
    }
    for k, v in summary.items():
        wandb.run.summary[k] = v
    LOG.info("The valid languages are: {}".format(valid_languages))
    LOG.info("Writing valid languages data to the output folder.")
    # Write mpararel.
    write_mpararel(valid_df[['language', 'relation']], agreed_translations,
                   args.out_folder)
    log_statistics(valid_df[['language', 'relation']], agreed_translations)


if __name__ == '__main__':
    main()
