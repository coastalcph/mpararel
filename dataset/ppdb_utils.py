import collections
from logger_utils import get_logger
import os
import re

LOG = get_logger(__name__)

DIR = "/home/wsr217/ppdb/"
COMMON_FILENAME = "ppdb-1.0-s-phrasal"

LANG_TO_PPDB_FILE = {
    "ar",  # Arabic
    "bg",  # Bulgarian
    "cs",  # Czech
    "de",  # German
    "el",  # Modern Greek
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "hu",  # Hungarian
    "it",  # Italian
    "lv",  # Latvian
    "lt",  # Lithuanian
    "nl",  # Dutch
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "ru",  # Russian
    "sk",  # Slovak
    "sl",  # Slovene
    "es",  # Spanish
}


def get_middle_text(template):
    matches = re.match(r'^\[[XY]\] (.*) \[[XY]\]$', template)
    if not matches:
        return None
    return matches.group(1)


def get_ppdb_paraphrases(lang):
    if lang not in LANG_TO_PPDB_FILE:
        return {}
    paraphrases = collections.defaultdict(set)
    file = os.path.join(DIR, f"{lang}-{COMMON_FILENAME}")
    if lang == "en":
        file = os.path.join(DIR, "en-ppdb-2.0-s-phrasal")
    with open(file) as f_ppdb:
        for line in f_ppdb:
            info = line.split(' ||| ')
            paraphrases[info[1]].add(info[2])
            paraphrases[info[2]].add(info[1])
    LOG.info("PPDB paraphrases read for '{}': {}".format(
        lang, len(paraphrases)))
    return paraphrases