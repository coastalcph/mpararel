"""Process the human reviews of the paraphrases and creates the reviewed version
of the dataset.

python dataset/crowdsourcing/create_reviewed_mpararel.py \
    --language_mapping_file=$WORKDIR/dataset/languages_mapping.txt \
    --mpararel_folder=$WORKDIR/data/mpararel \
    --reviews_filepath=$WORKDIR/data/reviews_2.pickle \
    --use_reviews_in_file \
    --output_folder=$WORKDIR/data/mpararel_reviewed/patterns
"""

import argparse
import collections
import itertools
import json
import os
import pickle
import time
import traceback

import gspread
import matplotlib.pyplot as plt
import mpararel_utils
import nltk
import numpy as np
import seaborn as sns
import tqdm
import wandb
from dataset.crowdsourcing.generate_sheet import (CREDENTIALS_PATH,
                                                  REVIEWERS_SHEET, SCOPES)
from dataset.constants import HUMAN_CHECKED, PATTERN
from dataset.translate_utils import get_wiki_to_names
from logger_utils import get_logger
from oauth2client.service_account import ServiceAccountCredentials

LOG = get_logger(__name__)

REMOVE_LANGUAGES = ['hi']


def get_review(worksheet):
    # Remove all the headers and the empty column
    rows_cells = np.array(worksheet.get_all_values())[4:, 1:5]
    extra_templates_divider = -1
    for i, row_cells in enumerate(rows_cells):
        if "If you want to add other paraphrases not encountered above" in row_cells[
                0]:
            extra_templates_divider = i
            break
    return (rows_cells[:extra_templates_divider - 2, :],
            [t for t, _, _, _ in rows_cells[extra_templates_divider + 1:, :]])


def fetch_reviews_from_sheets():
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        CREDENTIALS_PATH, SCOPES)
    client = gspread.authorize(creds)
    reviewers_sheet = client.open_by_key(REVIEWERS_SHEET)
    created_sheets = reviewers_sheet.worksheet("Reviewers spreadsheets")
    created_sheets = np.array(created_sheets.get_all_values())[1:, 0:6]
    finished_sheets = set([
        (lang_code, reviewer_name, link)
        for lang_code, reviewer_name, _, link, _, finished in created_sheets
        if finished == 'TRUE'
    ])
    LOG.info("There are {} finished_sheets ({})".format(
        len(finished_sheets), [n for _, n, _ in finished_sheets]))
    reviews = collections.defaultdict(lambda: collections.defaultdict(dict))
    count_worksheets_read = 0
    init_time = time.time()
    for lang_code, reviewer_name, link in tqdm.tqdm(finished_sheets):
        reviewed_sheet = client.open_by_url(link)
        count_worksheets_read += 2
        for worksheet in reviewed_sheet.worksheets():
            if not worksheet.title.startswith('P'):
                continue
            relation = worksheet.title
            templates_answers, extra_templates = get_review(worksheet)
            reviews[reviewer_name][lang_code][relation] = (templates_answers,
                                                           extra_templates)
            count_worksheets_read += 1
            if count_worksheets_read > 60:
                LOG.info(
                    "Seconds passed {}, going to pause...".format(time.time() -
                                                                  init_time))
                time.sleep(60)
                count_worksheets_read = 0
                init_time = time.time()
    return reviews


def get_reviewed_version(existing_templates, templates_answers,
                         extra_templates, stats):
    existing_templates = existing_templates.copy()
    templates_reviewed_by_human = set()
    stats["Existing templates"] += len(existing_templates)
    stats["Reviewed templates"] += len(templates_answers)
    stats["Extra templates in sheet"] += len(extra_templates)
    # Remove incorrect ones
    for template, _, is_incorrect, correction in templates_answers:
        if template not in existing_templates:
            LOG.warning("The template '{}' was not found "
                        "in the pararel folder provided".format(template))
            stats["Template not found in mpararel"] += 1
            if not correction and is_incorrect == 'FALSE':
                stats["Template not found in mpararel was added"] += 1
                extra_templates.append(template)
        if correction:
            extra_templates.append(correction)
            # When there's a correction we assume the template is not completely
            # fine and so we remove it.
            if template in existing_templates:
                existing_templates.remove(template)
                stats["Removed as it has correction"] += 1
        elif is_incorrect == 'TRUE' and template in existing_templates:
            existing_templates.remove(template)
            stats["Removed wrong template"] += 1
        else:
            templates_reviewed_by_human.add(template)
    # Add corrections.
    for extra_template in extra_templates:
        cleaned_template = mpararel_utils.clean_template(extra_template)
        if not mpararel_utils.is_template_valid(cleaned_template):
            raise Exception(
                "The correction provided '{}' is not a valid template.".format(
                    cleaned_template))
        if cleaned_template not in existing_templates:
            stats["Corrections and extra templates"] += 1
        existing_templates.add(cleaned_template)
        templates_reviewed_by_human.add(cleaned_template)
    return existing_templates, templates_reviewed_by_human, stats


def get_all_reviews(args):
    if args.use_reviews_in_file and os.path.isfile(args.reviews_filepath):
        with open(args.reviews_filepath, 'rb') as f:
            return pickle.load(f)
    else:
        while True:
            try:
                reviews = fetch_reviews_from_sheets()
                reviews_to_file = {}
                for reviewer_name, lang_to_relations in reviews.items():
                    reviews_to_file[reviewer_name] = dict(lang_to_relations)
                with open(args.reviews_filepath, 'wb') as f:
                    pickle.dump(reviews_to_file, f)
                return reviews
            except Exception as e:
                LOG.info(
                    "Getting all the reviews failed with error '{}', waiting 1 "
                    "minute and re trying.".format(e))
                print(traceback.format_exc())
                time.sleep(60)


def get_reviewed_mpararel(mpararel, reviews):
    """Returns mpararel reviewed for the present languages in the reviews."""
    new_mpararel_def = collections.defaultdict(
        lambda: collections.defaultdict(set))
    templates_checked_by_human = collections.defaultdict(set)
    stats_by_language = {}
    for reviewer_name, language_to_relations in reviews.items():
        for language, relation_to_review in language_to_relations.items():
            stats = collections.defaultdict(int)
            for relation, (templates_answers,
                           extra_templates) in relation_to_review.items():
                try:
                    existing_templates = mpararel[language][relation +
                                                            '.jsonl']
                    reviewed_templates, subset_reviewed, stats = get_reviewed_version(
                        existing_templates, templates_answers, extra_templates,
                        stats)
                    templates_checked_by_human[
                        language] = templates_checked_by_human[language].union(
                            subset_reviewed)
                except Exception as e:
                    LOG.error(
                        "Got error '{}' when processing the worksheet '{}' "
                        "review from '{}'.".format(e, relation, reviewer_name))
                    print(traceback.format_exc())
                    raise Exception()
                # We consider a template as correct as long as one reviewer set
                # it as correct.
                new_mpararel_def[language][relation + '.jsonl'] = (
                    new_mpararel_def[language][relation + '.jsonl'].union(
                        reviewed_templates))
            stats_by_language["{} {}".format(language, reviewer_name)] = stats
    # We remove the defaultdicts to be able to count the number of non empty
    # relations.
    new_mpararel = {}
    for language in new_mpararel_def.keys():
        new_mpararel[language] = dict(new_mpararel_def[language])
    return new_mpararel, templates_checked_by_human, stats_by_language


def plot_barh_by_languages(column, old_data, new_data, title):
    data = [(c, o, n) for c, o, n in zip(column, old_data, new_data)]
    wandb.log({
        title + "_table":
        wandb.Table(data=data, columns=["Language", "old", "new"])
    })
    colors = sns.color_palette('colorblind', 10)
    plt.style.use('seaborn-colorblind')
    plt.figure(figsize=(4, 8), dpi=150)
    plt.barh(column, new_data, label="new", alpha=0.7, color=colors[0])
    plt.barh(column, old_data, label="old", alpha=0.7, color=colors[-2])
    plt.title(title)
    plt.tight_layout()
    plt.legend(loc='lower left')
    plt.show()
    wandb.log({title: wandb.Image(plt)})


def log_string_distances(mpararel,
                         new_mpararel,
                         wiki_code_to_name,
                         languages=None):
    if languages is None:
        languages = list(mpararel.keys())

    def get_edit_distances(templates_list):
        edit_distances = []
        for templates in templates_list:
            for template_i, template_j in itertools.combinations(templates,
                                                                 r=2):
                edit_distances.append(
                    nltk.edit_distance(template_i, template_j))
        return edit_distances

    old_str_distance = []
    new_str_distance = []
    for language in languages:
        old_str_distance.append(
            np.average(get_edit_distances(mpararel[language].values())))
        new_str_distance.append(
            np.average(get_edit_distances(new_mpararel[language].values())))
    wandb.run.summary["avg. string distance"] = np.average(new_str_distance)
    plot_barh_by_languages([wiki_code_to_name[code] for code in languages],
                           old_str_distance, new_str_distance,
                           "Average string distance")


def log_overall_patterns_stats(mpararel, new_mpararel):
    # Overall number of patterns stats.
    old_count_patterns = []
    new_count_patterns = []
    total_relations = []
    total_patterns = []
    for language, relation_to_templates in mpararel.items():
        old_count_patterns += [
            len(templates) for templates in relation_to_templates.values()
        ]
        new_count_patterns += [
            len(templates) for templates in new_mpararel[language].values()
        ]
        total_relations.append(len(new_mpararel[language]))
        total_patterns.append(
            sum([
                len(templates)
                for templates in new_mpararel[language].values()
            ]))
    columns = [
        "old min #patterns in a relation", "new min #patterns in a relation",
        "old avg #patterns in a relation", "new avg #patterns in a relation",
        "old max #patterns in a relation", "new max #patterns in a relation",
        "avg #relations per language", "avg total #patterns per language"
    ]
    data = [
        min(old_count_patterns),
        min(new_count_patterns),
        np.average(old_count_patterns),
        np.average(new_count_patterns),
        max(old_count_patterns),
        max(new_count_patterns),
        np.average(total_relations),
        np.average(total_patterns)
    ]
    for value, column in zip(data, columns):
        wandb.run.summary[column] = value


def log_plots_by_language(mpararel,
                          new_mpararel,
                          wiki_code_to_name,
                          languages=None):
    if languages is None:
        languages = list(mpararel.keys())
    languages_names = [wiki_code_to_name[lang] for lang in languages]
    plot_barh_by_languages(languages_names,
                           [len(mpararel[lang]) for lang in languages],
                           [len(new_mpararel[lang])
                            for lang in languages], "Number of relations")

    plot_barh_by_languages(languages_names, [
        min([len(patterns) for patterns in mpararel[lang].values()])
        for lang in languages
    ], [
        min([len(patterns) for patterns in new_mpararel[lang].values()])
        for lang in languages
    ], "Minimum number of patterns in a relation")
    plot_barh_by_languages(languages_names, [
        max([len(patterns) for patterns in mpararel[lang].values()])
        for lang in languages
    ], [
        max([len(patterns) for patterns in new_mpararel[lang].values()])
        for lang in languages
    ], "Maximum number of patterns in a relation")
    plot_barh_by_languages(languages_names, [
        np.average([len(patterns) for patterns in mpararel[lang].values()])
        for lang in languages
    ], [
        np.average([len(patterns) for patterns in new_mpararel[lang].values()])
        for lang in languages
    ], "Average number of patterns in a relation")
    plot_barh_by_languages(languages_names, [
        sum([len(patterns) for patterns in mpararel[lang].values()])
        for lang in languages
    ], [
        sum([len(patterns) for patterns in new_mpararel[lang].values()])
        for lang in languages
    ], "Total number of patterns")


def log_reviews_stats(stats_by_language):
    columns = set()
    for lang in stats_by_language.keys():
        columns.update(stats_by_language[lang].keys())
    columns = ["review"] + list(columns)
    data = []
    for language in stats_by_language.keys():
        this_data = [language]
        for column in columns[1:]:
            # stats_by_language[lang][column] is a defaultdict.
            this_data.append(stats_by_language[language][column])
        data.append(this_data)
    wandb.log({"Reviews stats": wandb.Table(data=data, columns=columns)})


def write_mpararel(output_folder, new_mpararel, templates_checked_by_human):
    if os.path.exists(output_folder):
        raise Exception("The output folder already exists.")
    for language in new_mpararel.keys():
        for relation in new_mpararel[language].keys():
            os.makedirs(os.path.join(output_folder, language), exist_ok=True)
            with open(os.path.join(output_folder, language, relation),
                      'w') as f:
                for template in new_mpararel[language][relation]:
                    f.write("{}\n".format(
                        json.dumps({
                            PATTERN:
                            template,
                            HUMAN_CHECKED:
                            template in templates_checked_by_human[language]
                        })))


def remove_relations_with_not_enough_templates(new_mpararel):
    for language in list(new_mpararel.keys()):
        for relation in list(new_mpararel[language].keys()):
            if len(new_mpararel[language][relation]) < 2:
                LOG.info(
                    "Relation '{}' in language '{}' now has '{}' templates so "
                    "we're removing it.".format(
                        relation, language,
                        len(new_mpararel[language][relation])))
                new_mpararel[language].pop(relation)


def main(args):
    wandb.init(project="create-reviewed-mpararel",
               name=os.path.basename(args.mpararel_folder))
    wandb.config.update(args)
    LOG.info("Getting the reviews")
    reviews = get_all_reviews(args)
    mpararel = mpararel_utils.read_mpararel_templates(args.mpararel_folder)
    LOG.info("Constructing reviewed mpararel")
    (new_mpararel, templates_checked_by_human,
     stats_by_language) = get_reviewed_mpararel(mpararel, reviews)
    remove_relations_with_not_enough_templates(new_mpararel)
    log_reviews_stats(stats_by_language)
    reviewed_languages = list(new_mpararel.keys())
    for lang_not_reviewed in set(mpararel.keys()).difference(
            set(new_mpararel.keys())):
        new_mpararel[lang_not_reviewed] = mpararel[lang_not_reviewed]
    for language in REMOVE_LANGUAGES:
        new_mpararel.pop(language)
        mpararel.pop(language)
    LOG.info("Logging stats to wandb")
    log_overall_patterns_stats(mpararel, new_mpararel)
    wiki_code_to_name = get_wiki_to_names(args.language_mapping_file)
    log_string_distances(mpararel, new_mpararel, wiki_code_to_name)
    log_plots_by_language(mpararel, new_mpararel, wiki_code_to_name,
                          reviewed_languages)
    LOG.info("Writing reviewed mpararel")
    write_mpararel(args.output_folder, new_mpararel,
                   templates_checked_by_human)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpararel_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to the folder with the mpararel data.")
    parser.add_argument("--reviews_filepath", required=True, type=str, help="")
    parser.add_argument("--use_reviews_in_file", action="store_true")
    parser.add_argument("--language_mapping_file",
                        required=True,
                        type=str,
                        help="")
    parser.add_argument("--output_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())
