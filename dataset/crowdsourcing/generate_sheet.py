"""Script to create a spreedsheet for review.

python dataset/crowdsourcing/generate_sheet.py \
    --mpararel_folder=$WORKDIR/data/mpararel_00_00_06_02_logging \
    --pararel_patterns_folder=$WORKDIR/data/pararel/pattern_data/graphs_json \
    --reviewer_name="constanza" --reviewer_mail="my_mail@gmail.com" \
    --language_code="es"
"""

from __future__ import print_function

import argparse
import json
import os
import time
from collections import defaultdict
from math import ceil

import gspread
import numpy as np
from logger_utils import get_logger
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

LOG = get_logger(__name__)

SCOPES = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
]
# Obtained from the Google Cloud App from the Drive API when creating service
# credentials.
CREDENTIALS_PATH = '/home/wsr217/mpararel/dataset/crowdsourcing/credentials.json'
REVIEWERS_SHEET = "1LT6qo2BwdDfRr1Eusq7avLysPg-yDmfIxM53bxACoY0"
TEMPLATE_SHEET = "1JYKEM6t12VzlYO0pM543Rv1wozpEvlMkEzbYVY65yeU"
PERCENTAGE_TO_REVIEW = 0.5
SECONDS_PER_TEMPLATE = 9
CONTEXT_WORKSHEET = "Context"
RELATION_TEMPLATE_WORKSHEET = "Relation template"
PERSONAL_MAIL = "c.fierro@di.ku.dk"

# Template coordinates
REVIEWER_NAME_CONTEXT_COLUMN = 8
TOTAL_TIME_CONTEXT_COLUMN = 20


def create_copy_spreadsheet(client, sheet_to_copy, new_sheet_name,
                            editors_mails):
    client.copy(sheet_to_copy.id, title=new_sheet_name, copy_permissions=True)
    new_sheet = client.open(new_sheet_name)
    for mail in editors_mails:
        new_sheet.share(mail, perm_type='user', role='writer', notify=False)
    return new_sheet


def edit_context_description(sheet, reviewer_name, templates_count):
    context = sheet.worksheet(CONTEXT_WORKSHEET)
    example = context.cell(REVIEWER_NAME_CONTEXT_COLUMN, 2).value
    example = example.replace("<reviewer_name>", reviewer_name)
    context.update_cell(REVIEWER_NAME_CONTEXT_COLUMN, 2, example)
    time_expectation = context.cell(TOTAL_TIME_CONTEXT_COLUMN, 2).value
    time_expectation = time_expectation.replace(
        "<minutes>", str(ceil(templates_count * SECONDS_PER_TEMPLATE / 60)))
    context.update_cell(TOTAL_TIME_CONTEXT_COLUMN, 2, time_expectation)


def edit_relation_description(worksheet, relation_lemmas, tuples_examples):
    relation = worksheet.cell(1, 2).value
    relation = relation.replace("<relation>", ', '.join(relation_lemmas))
    worksheet.update_cell(1, 2, relation)
    examples = worksheet.cell(2, 2).value
    for i, (sub, obj) in enumerate(tuples_examples, 1):
        examples = examples.replace(f"<example_{i}>", f"{sub}, {obj}")
    worksheet.update_cell(2, 2, examples)


def filter_longer_templates(relation_to_templates, percentage):
    """Keeps the percentage portion of the templates sorted by length."""
    total_templates = 0
    for relation in relation_to_templates.keys():
        templates_to_keep = max(
            ceil(len(relation_to_templates[relation]) * percentage), 2)
        templates_sorted = sorted(relation_to_templates[relation],
                                  key=lambda t: len(t))
        relation_to_templates[relation] = templates_sorted[:templates_to_keep]
        total_templates += templates_to_keep
    return total_templates


def main(args):
    LOG.info("Reading patterns, tuples, and relations descriptions...")
    relation_to_templates = defaultdict(list)
    relations_folder = os.path.join(args.mpararel_folder, "patterns",
                                    args.language_code)
    for relation_file in os.listdir(relations_folder):
        relation = relation_file[:-len(".jsonl")]
        with open(os.path.join(relations_folder, relation_file)) as templates:
            for line in templates:
                template = json.loads(line)
                relation_to_templates[relation].append(template["pattern"])
    total_templates = filter_longer_templates(relation_to_templates,
                                              PERCENTAGE_TO_REVIEW)
    relation_to_tuples = defaultdict(list)
    tuples_folder = os.path.join(args.mpararel_folder, "tuples",
                                 args.language_code)
    for relation_file in os.listdir(tuples_folder):
        relation = relation_file[:-len(".jsonl")]
        with open(os.path.join(tuples_folder, relation_file)) as tuples:
            for line in tuples:
                tuples_dict = json.loads(line)
                if np.random.rand() > 0.7:
                    relation_to_tuples[relation].append(
                        (tuples_dict["sub_label"], tuples_dict["obj_label"]))
                if len(relation_to_tuples[relation]) == 3:
                    break
    relation_to_lemma = defaultdict(set)
    for relation_file in os.listdir(args.pararel_patterns_folder):
        relation = relation_file[:-len(".jsonl")]
        with open(os.path.join(args.pararel_patterns_folder,
                               relation_file)) as pararel_patterns:
            for line in pararel_patterns:
                pattern_dict = json.loads(line)
                relation_to_lemma[relation].add(pattern_dict["extended_lemma"])

    # Connect to drive and create the spreadsheet.
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        CREDENTIALS_PATH, SCOPES)
    client = gspread.authorize(creds)
    template_sheet = client.open_by_key(TEMPLATE_SHEET)
    sheet = create_copy_spreadsheet(
        client, template_sheet, f"{args.reviewer_name} - {args.language_code}",
        [args.reviewer_mail, PERSONAL_MAIL])
    LOG.info(f"Sheet created {sheet.url}")
    edit_context_description(sheet, args.reviewer_name, total_templates)

    # Creation each relation worksheet and write the templates.
    LOG.info("Creating worksheets for each relation...")
    for relation in tqdm(relation_to_templates.keys()):
        worksheet_template = sheet.worksheet(RELATION_TEMPLATE_WORKSHEET)
        worksheet = worksheet_template.duplicate()
        worksheet.update_title(relation)
        edit_relation_description(worksheet, relation_to_lemma[relation],
                                  relation_to_tuples[relation])
        total_templates = len(relation_to_templates[relation])
        worksheet.update(f"B5:B{5+total_templates-2}",
                         [[row]
                          for row in relation_to_templates[relation][:-1]])
        worksheet.update_cell(81, 2, relation_to_templates[relation][-1])
        # Remove extra templates rows.
        worksheet.delete_rows(5 + len(relation_to_templates[relation]) - 1, 80)
        # There's a maximum number of requests per minute in gspread.
        time.sleep(5)
    sheet.del_worksheet(sheet.worksheet(RELATION_TEMPLATE_WORKSHEET))
    worksheets = sheet.worksheets()
    worksheets.reverse()
    sheet.reorder_worksheets(worksheets)

    # Add sheet url to
    reviewers_worksheet = client.open_by_key(REVIEWERS_SHEET).get_worksheet(1)
    next_empty_row = len(reviewers_worksheet.col_values(1)) + 1
    reviewers_worksheet.update(f"A{next_empty_row}:C{next_empty_row}", [[
        args.language_code, args.reviewer_name, args.reviewer_mail, sheet.url
    ]])


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpararel_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to the folder with the mpararel data.")
    parser.add_argument(
        "--pararel_patterns_folder",
        default=None,
        type=str,
        required=True,
        help="The path to the folder with the pararel json patterns.")
    parser.add_argument("--reviewer_name",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--reviewer_mail",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--language_code",
                        default=None,
                        type=str,
                        required=True,
                        help="The language code as it appears in the folders.")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())