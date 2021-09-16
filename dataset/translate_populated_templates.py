import collections
import json
import os
import re
from typing import List

from logger_utils import get_logger

from dataset.translate_utils import TRANSLATOR_TO_OBJECT, Translator

LOG = get_logger(__name__)

K_POPULATED_TEMPLATES = 10


def get_k_subject_object_tuples(filename, k_tuples):
    tuples = {}
    added = set()
    with open(filename) as fp:
        for i, line in enumerate(fp):
            line_data = json.loads(line)
            if (line_data["sub_label"] not in added
                    and line_data["obj_label"] not in added):
                tuples[i] = (line_data["sub_label"], line_data["obj_label"])
                added.add(line_data["sub_label"])
                added.add(line_data["obj_label"])
            if len(tuples) == k_tuples:
                break
    return tuples


def get_subject_object_tuples_from_lines(filename, lineas_to_add):
    tuples = {}
    with open(filename) as fp:
        for i, line in enumerate(fp):
            line_data = json.loads(line)
            if i in lineas_to_add:
                tuples[i] = (line_data["sub_label"], line_data["obj_label"])
    return tuples


def get_populated_phrases(template, tuples):
    populated_phrases = []
    for obj, sub in tuples:
        phrase = template.replace("[X]", obj)
        populated_phrases.append(phrase.replace("[Y]", sub))
    return populated_phrases


def get_template_from_common_indexes(common_indexes, string_list_1,
                                     string_list_2):
    str_sequence = []
    skipped = []
    for i_common in range(len(common_indexes)):
        # Some words are skipped in both strings between the last common and the
        # current common index.
        last_common_index_1st_word = - \
            1 if i_common == 0 else common_indexes[i_common-1][0]
        last_common_index_2nd_word = - \
            1 if i_common == 0 else common_indexes[i_common-1][1]
        if (common_indexes[i_common][0] > last_common_index_1st_word + 1 and
                common_indexes[i_common][1] > last_common_index_2nd_word + 1):
            str_sequence.append('[X/Y]')
            skipped_str_1 = ' '.join(
                string_list_1[last_common_index_1st_word +
                              1:common_indexes[i_common][0]])
            skipped_str_2 = ' '.join(
                string_list_2[last_common_index_2nd_word +
                              1:common_indexes[i_common][1]])
            skipped.append((skipped_str_1, skipped_str_2))
        str_sequence.append(string_list_1[common_indexes[i_common][0]])
    if (len(string_list_1) > common_indexes[i_common][0] + 1
            and len(string_list_2) > common_indexes[i_common][1] + 1):
        str_sequence.append('[X/Y]')
        skipped_str_1 = ' '.join(string_list_1[common_indexes[i_common][0] +
                                               1:])
        skipped_str_2 = ' '.join(string_list_2[common_indexes[i_common][1] +
                                               1:])
        skipped.append((skipped_str_1, skipped_str_2))
    return ' '.join(str_sequence), skipped


def longest_common_subsequence(string_list_1, string_list_2):
    m = len(string_list_1)
    n = len(string_list_2)
    # L[i][j] contains the length of the LCS of X[0..i-1] and Y[0..j-1].
    L = [[None] * (n + 1) for i in range(m + 1)]
    sequence = [[None] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
                sequence[i][j] = []
            elif string_list_1[i - 1] == string_list_2[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
                sequence[i][j] = sequence[i - 1][j - 1].copy() + [
                    (i - 1, j - 1)
                ]
            else:
                if L[i - 1][j] > L[i][j - 1]:
                    L[i][j] = L[i - 1][j]
                    sequence[i][j] = sequence[i - 1][j]
                else:
                    L[i][j] = L[i][j - 1]
                    sequence[i][j] = sequence[i][j - 1]
    lcs = sequence[m][n]
    if not lcs:
        return None, None
    return get_template_from_common_indexes(lcs, string_list_1, string_list_2)


def set_cleaned_words(list, index=None):
    output = set()
    for string in list:
        if index is not None:
            string = string[index]
        for word in string.split(' '):
            word = re.sub('[.,:]', '', re.sub('[.,:]', '', word))
            word = re.sub(' +', ' ', word)
            output.add(word.lower())
    return output


def get_object_subject_order(set_first_position, set_second_position,
                             en_tuples, translated_tuples):
    set_first_position = set_cleaned_words(set_first_position)
    set_second_position = set_cleaned_words(set_second_position)
    subj_en = set_cleaned_words(en_tuples, 0)
    obj_en = set_cleaned_words(en_tuples, 1)
    subj_translated = set_cleaned_words(translated_tuples, 0)
    obj_translated = set_cleaned_words(translated_tuples, 1)
    total_subj = len(subj_en) + len(subj_translated)
    total_obj = len(obj_en) + len(obj_translated)
    first_subj = (len(set_first_position.intersection(subj_en)) + len(
        set_first_position.intersection(subj_translated))) / total_subj
    first_obj = (len(set_first_position.intersection(obj_en)) + len(
        set_first_position.intersection(obj_translated))) / total_obj
    second_subj = (len(set_second_position.intersection(subj_en)) + len(
        set_second_position.intersection(subj_translated))) / total_subj
    second_obj = (len(set_second_position.intersection(obj_en)) + len(
        set_second_position.intersection(obj_translated))) / total_obj
    if first_subj > second_subj and first_obj < second_obj:
        return ["[X]", "[Y]"]
    elif first_subj < second_subj and first_obj > second_obj:
        return ["[Y]", "[X]"]
    else:
        LOG.info(
            "Error: couldn't conclude on order of X and Y (first_subj={}, "
            "first_obj={}, second_subj={}, second_obj={}).".format(
                first_subj, first_obj, second_subj, second_obj))
        return None, None


def get_templates_from_populated_translations(translated_phrases: List,
                                              en_tuples: List,
                                              translated_tuples: List):
    """Returns the templates that are present in more than one translation.

    The template is defined as the longest common subsequence of words between
    more than two translations. It assumes that [X] and [Y] are positioned where
    words are skipped (i.e. do not match) in both of the translations. For
    defining which one is [X] and which one [Y] it simply compares the words
    skipped with the tuples passed as arguments.

    Args:
        translated_phrases: list with the translated phrases.
        en_tuples: List containing the tuples (subject, object) used to populate the templates.
        en_tuples: List containing the tuples (subject, object) that are the translation of the tuples in english.
    Returns:
        List of templates where each contains exactly one [X] and one [Y].
    """
    potential_templates = collections.defaultdict(int)
    # Each set contains the phrases for the object/subject found.
    non_overlapping_phrases = [set(), set()]
    for i in range(0, len(translated_phrases)):
        for j in range(i + 1, len(translated_phrases)):
            lcs, non_overlapping = longest_common_subsequence(
                translated_phrases[i].split(' '),
                translated_phrases[j].split(' '))
            if not lcs:
                continue
            if lcs.count("[X/Y]") != 2:
                continue
            for subject_or_object in non_overlapping[0]:
                non_overlapping_phrases[0].add(subject_or_object)
            for subject_or_object in non_overlapping[1]:
                non_overlapping_phrases[1].add(subject_or_object)
            potential_templates[lcs] += 1
    first_str, second_str = get_object_subject_order(*non_overlapping_phrases,
                                                     en_tuples,
                                                     translated_tuples)
    if first_str is None:
        return []
    final_templates = []
    for template, _ in filter(lambda x: x[1] > 1, potential_templates.items()):
        final_tempalte = template.replace("[X/Y]", first_str, 1)
        final_templates.append(final_tempalte.replace("[X/Y]", second_str, 1))
    return final_templates


def translate_populated_templates(templates: List[str], template_key: str,
                                  tuples_folder: str, relation_filename: str,
                                  wiki_lang_to_translator_lang: dict,
                                  translator: Translator) -> dict:
    """Translates each template to all the languages after filling the [X] and [Y].

    Args:
        templates: List of templates to translate.
        template_key: string key to access the actual template in the templates dict.
        tuples_folder: path to the folder containing the objects and subjects in each language.
        relation_filename: filename corresponding to the relation of this
            tuples, used to access the tuples in the tuples folder.
        wiki_lang_to_translator_lang: dict that maps wiki languages to the translator languages.
        translator: Translator to use for the translations.
    Returns:
        A dictionary mapping from languages to a list containing the translated templates.
    """
    translator = TRANSLATOR_TO_OBJECT[translator]
    translated_templates = {}
    for wikiid, translator_id in wiki_lang_to_translator_lang.items():
        LOG.info("Translating {}".format(wikiid))
        translated_templates[wikiid] = set()
        for template in templates:
            LOG.info("Translating template: '{}'".format(
                template[template_key]))
            en_tuples_file = os.path.join(tuples_folder, "en",
                                          relation_filename)
            translated_tuples_file = os.path.join(tuples_folder, wikiid,
                                                  relation_filename)
            if (not os.path.isfile(en_tuples_file)
                    or not os.path.isfile(translated_tuples_file)):
                LOG.info(
                    "There are no tuples files for this relation (Not found: "
                    "{} and {}).".format(en_tuples_file,
                                         translated_tuples_file))
                break
            en_tuples = get_k_subject_object_tuples(en_tuples_file,
                                                    K_POPULATED_TEMPLATES)
            translated_tuples = get_subject_object_tuples_from_lines(
                translated_tuples_file, en_tuples.keys())
            populated_phrases = get_populated_phrases(template[template_key],
                                                      en_tuples.values())
            try:
                translated_phrases = [
                    translator.translate(text,
                                         from_lang="en",
                                         to_lang=translator_id)
                    for text in populated_phrases
                ]
            except Exception as e:
                LOG.info("Exception: {}".format(e))
                break
            final_templates = get_templates_from_populated_translations(
                translated_phrases, en_tuples.values(),
                translated_tuples.values())
            translated_templates[wikiid].update(final_templates)
        if len(translated_templates[wikiid]) == 0:
            LOG.warning(
                "Skipping language '{}', not one translation was succesful!".
                format(wikiid))
            continue
        LOG.info("Successful translations. Received {} templates, created {} "
                 "translations".format(len(templates),
                                       len(translated_templates[wikiid])))
    return translated_templates
