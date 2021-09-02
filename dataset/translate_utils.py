from enum import Enum
import pandas as pd


class Translator(Enum):
    GOOGLE = 'google'
    BING = 'bing'


LANGUAGE_CODES_TO_FILE_HEADER = {
    Translator.GOOGLE: "googletranslate",
    Translator.BING: "BCP_47",
}


def get_wiki_language_mapping(path, translator):
    language_mapping = pd.read_csv(path, sep='\t')
    translator_codes = language_mapping[
        LANGUAGE_CODES_TO_FILE_HEADER[translator]].values
    return pd.Series(language_mapping.wiki.values, translator_codes).to_dict()


def get_wiki_languages(path):
    language_mapping = pd.read_csv(path, sep='\t')
    return language_mapping.wiki.values
