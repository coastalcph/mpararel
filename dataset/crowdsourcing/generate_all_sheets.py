"""Creates the spreadsheets for review using the google form answers.

It only creates the spreadsheet that have not been created before. To check this
it uses the worksheet where the links of the generated spreadsheets are added.

python dataset/crowdsourcing/generate_all_sheets.py \
    --mpararel_folder=$WORKDIR/data/mpararel_00_00_06_02_logging \
    --pararel_patterns_folder=$WORKDIR/data/pararel/pattern_data/graphs_json \
    --language_mapping_file=$WORKDIR/dataset/languages_mapping.txt
"""
import argparse
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dataset.crowdsourcing.generate_sheet import SCOPES, REVIEWERS_SHEET, CREDENTIALS_PATH
from dataset.crowdsourcing.generate_sheet import main as generate_sheet
import pandas as pd
from logger_utils import get_logger

LOG = get_logger(__name__)


def main(args):
    language_mapping = pd.read_csv(args.language_mapping_file, sep='\t')
    lang_name_to_code = {
        name: code
        for code, name in language_mapping[["wiki", "name"]].values
    }
    # Connect to drive and read the form answers.
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        CREDENTIALS_PATH, SCOPES)
    client = gspread.authorize(creds)
    reviewers_sheet = client.open_by_key(REVIEWERS_SHEET)
    form_answers = reviewers_sheet.worksheet("Form Responses 1")
    form_answers = form_answers.get_all_values()[1:]

    # Read already created spreadsheets.
    created_sheets = reviewers_sheet.worksheet("Reviewers spreadsheets")
    created_sheets = created_sheets.get_all_values()[1:]
    created_sheets = set([
        lang_code + name + mail for lang_code, name, mail, _ in created_sheets
    ])

    # Create spreadsheets.
    for _, name, mail, languages in form_answers:
        for lang_name in languages.split(','):
            key = name + mail + lang_name_to_code[lang_name]
            if key in created_sheets or lang_name_to_code[lang_name] == "en":
                continue
            args.reviewer_name = name
            args.reviewer_mail = mail
            args.language_code = lang_name_to_code[lang_name]
            LOG.info("Generating spreadsheet for: {} {} {}".format(
                args.reviewer_name, args.reviewer_mail, args.language_code))
            try:
                generate_sheet(args)
            except Exception as e:
                LOG.error("Exception '{}' while generating spreadsheet for "
                          "language: '{}'".format(e, args.language_code))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpararel_folder",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to the folder with the mpararel data.")
    parser.add_argument("--language_mapping_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument(
        "--pararel_patterns_folder",
        default=None,
        type=str,
        required=True,
        help="The path to the folder with the pararel json patterns.")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    main(parser.parse_args())