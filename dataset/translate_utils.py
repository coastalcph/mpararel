import requests
import uuid
from enum import Enum
import pandas as pd
from googletrans import Translator
import numpy as np

BING_SUBSCRIPTION_KEY = "7d397449d9454dc995a0f9ab234fe430"
BING_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"


class Translator(Enum):
    GOOGLE = 'google'
    BING = 'bing'


TRANSLATOR_TO_CODES_HEADER = {
    Translator.GOOGLE: "googletranslate",
    Translator.BING: "BCP_47",
}


def translate_with_google(text, from_lang, to_lang):
    translator = Translator()
    result = translator.translate(text, src=from_lang, dest=to_lang)
    return result.text


def translate_with_bing(text, from_lang, to_lang):
    path = '/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = BING_ENDPOINT + path + params
    headers = {
        'Ocp-Apim-Subscription-Key': BING_SUBSCRIPTION_KEY,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{
        'text': text
    }]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]


def translate(translator: Translator, text: str, from_lang: str, to_lang: str) -> str:
    if translator == Translator.GOOGLE:
        return translate_with_google(text, from_lang, to_lang)
    elif translator == Translator.BING:
        return translate_with_bing(text, from_lang, to_lang)
    else:
        raise Exception("Unknown translator: {}, should be one of: {}".format(
            translator, list(Translator)))


def get_wiki_language_mapping(path: str, translator: Translator) -> dict:
    """Returns the language codes mapping between wikipedia and the translator."""
    language_mapping = pd.read_csv(path, sep='\t')
    translator_codes = language_mapping[
        TRANSLATOR_TO_CODES_HEADER[translator]].values
    wiki_to_translator_code = {}
    for wiki_lang, translator_lang in zip(language_mapping.wiki.values, translator_codes):
        # Empty codes are read as NaN by pandas.
        if not isinstance(translator_lang, str) and translator == Translator.GOOGLE:
            # For Google translate we try the wiki language code.
            translator_lang = wiki_lang
        if isinstance(translator_lang, str):
            wiki_to_translator_code[wiki_lang] = translator_lang
    return wiki_to_translator_code


def get_wiki_languages(path: str) -> np.ndarray:
    language_mapping = pd.read_csv(path, sep='\t')
    return language_mapping.wiki.values
