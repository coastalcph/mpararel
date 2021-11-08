import os
import collections
import json
import re
from dataset.constants import PATTERN, HUMAN_CHECKED

VALID_RELATIONS = set([
    'P937', 'P1412', 'P127', 'P103', 'P276', 'P159', 'P140', 'P136', 'P495',
    'P17', 'P361', 'P36', 'P740', 'P264', 'P407', 'P138', 'P30', 'P131',
    'P176', 'P449', 'P279', 'P19', 'P101', 'P364', 'P106', 'P1376', 'P178',
    'P37', 'P413', 'P27', 'P20', 'P190', 'P1303', 'P39', 'P108', 'P463',
    'P530', 'P47'
])


# TODO: test
def read_mpararel_templates(mpararel_folder, only_human_reviewed=False):
    mpararel_folder = os.path.join(mpararel_folder, "patterns")
    patterns = collections.defaultdict(lambda: collections.defaultdict(set))
    for language in os.listdir(mpararel_folder):
        language_dir = os.path.join(mpararel_folder, language)
        for relation_file in os.listdir(language_dir):
            with open(os.path.join(language_dir, relation_file)) as f:
                for line in f:
                    if line:
                        line_data = json.loads(line)
                        if not only_human_reviewed or line_data[HUMAN_CHECKED]:
                            patterns[language][relation_file].add(
                                line_data[PATTERN])
    return patterns


def is_template_valid(template):
    """Checks that the template has one [X], one [Y], and extra text."""
    return (template.count("[X]") == 1 and template.count("[Y]") == 1
            and not re.match(r'^[\b\[X\]\b\b\[Y\]\b., ]+$', template))


def clean_template(template):
    template = template.lower()
    # Remove extra spaces and extra brackets from the subject/object and
    # capitalize them.
    template = re.sub('\[+ ?[x] ?\]+', '[X]', template)
    template = re.sub('\[+ ?[y] ?\]+', '[Y]', template)
    # Remove final puntuaction
    template = re.sub(r'[.:。]', '', template)
    # Remove extra spaces
    template = re.sub(r' +', ' ', template)
    template = re.sub(r' $', '', template)
    return template