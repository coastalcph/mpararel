import uuid
from enum import Enum

import numpy as np
import pandas as pd
import requests
from easynmt import EasyNMT
from googletrans import Translator as GoogleTranslatorAPI
from logger_utils import get_logger

LOG = get_logger(__name__)


class Translator(Enum):
    GOOGLE = 'google'
    BING = 'bing'
    OPUSMT = 'opus_mt'
    MBART50_EN2MULTILINGUAL = 'mbart50_en2m'
    M2M100_BIG = 'm2m100_big'


class GoogleTranslator():
    LANGUAGE_CODES_HEADER = "googletranslate"

    def translate(self, text, from_lang, to_lang):
        result = GoogleTranslatorAPI().translate(text,
                                                 src=from_lang,
                                                 dest=to_lang)
        return result.text


class BingTranslator():
    LANGUAGE_CODES_HEADER = "bing_BCP_47"
    BING_SUBSCRIPTION_KEY = "7d397449d9454dc995a0f9ab234fe430"
    BING_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"

    def translate(self, text, from_lang, to_lang):
        path = '/translate?api-version=3.0'
        params = '&from={}&to={}'.format(from_lang, to_lang)
        constructed_url = BingTranslator.BING_ENDPOINT + path + params
        headers = {
            'Ocp-Apim-Subscription-Key': BingTranslator.BING_SUBSCRIPTION_KEY,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        body = [{'text': text}]
        request = requests.post(constructed_url, headers=headers, json=body)
        response = request.json()
        return response[0]["translations"][0]["text"]


class EasyNmtTranslator():
    AVAILABLE_MODELS = ['opus-mt', 'mbart50_en2m', 'm2m_100_1.2B']
    model = None

    def __init__(self, model_name: str):
        if EasyNmtTranslator.model:
            if model_name != self.model_name:
                raise Exception(
                    "Attempting to create a different instance of the EasyNMT "
                    "singleton.")
        if model_name not in self.AVAILABLE_MODELS:
            raise Exception(
                "Model name ({}) is wrong, should be one of: {}".format(
                    model_name, self.AVAILABLE_MODELS))
        self.model_name = model_name

    def translate(self, text, from_lang, to_lang):
        if not EasyNmtTranslator.model:
            EasyNmtTranslator.model = EasyNMT(self.model_name)
            LOG.info("{} from EasyNMT was loaded in the device: {}".format(
                self.model_name, EasyNmtTranslator.model.device))
        translations = EasyNmtTranslator.model.translate([text],
                                                         source_lang=from_lang,
                                                         target_lang=to_lang)
        return translations[0]


class OpusMT(EasyNmtTranslator):
    """1200+ 1 to 1 language translation models (each model ~300 MB of size).

    Documentation: https://github.com/Helsinki-NLP/Opus-MT
    """
    LANGUAGE_CODES_HEADER = "opus_mt"

    def __init__(self):
        super().__init__('opus-mt')


class MBart50En2M(EasyNmtTranslator):
    """Multilingual BART english to 50 other languages.

    Documentation: https://github.com/pytorch/fairseq/tree/master/examples/multilingual
    """
    LANGUAGE_CODES_HEADER = "mbart_50"

    def __init__(self):
        super().__init__('mbart50_en2m')


class M2M100_1_2BillionParams(EasyNmtTranslator):
    """Multilingual BART many to many languages (100 different) with 1.2B params.

    Documentation: https://github.com/pytorch/fairseq/tree/master/examples/m2m_100
    """
    LANGUAGE_CODES_HEADER = "m2m_100"

    def __init__(self):
        super().__init__('m2m_100_1.2B')


TRANSLATOR_TO_OBJECT = {
    Translator.GOOGLE: GoogleTranslator(),
    Translator.BING: BingTranslator(),
    Translator.OPUSMT: OpusMT(),
    Translator.MBART50_EN2MULTILINGUAL: MBart50En2M(),
    Translator.M2M100_BIG: M2M100_1_2BillionParams(),
}


def get_wiki_language_mapping(path: str, translator: Translator) -> dict:
    """Returns the language codes mapping between wikipedia and the translator."""
    language_mapping = pd.read_csv(path, sep='\t')
    translator_codes_header = (
        TRANSLATOR_TO_OBJECT[translator].LANGUAGE_CODES_HEADER)
    translator_codes = language_mapping[translator_codes_header].values
    wiki_to_translator_code = {}
    for wiki_lang, translator_lang in zip(language_mapping.wiki.values,
                                          translator_codes):
        # Empty codes are read as NaN by pandas.
        if not isinstance(translator_lang,
                          str) and translator == Translator.GOOGLE:
            # For Google translate we try the wiki language code.
            translator_lang = wiki_lang
        if isinstance(translator_lang, str):
            wiki_to_translator_code[wiki_lang] = translator_lang
    return wiki_to_translator_code


def get_wiki_languages(path: str) -> np.ndarray:
    language_mapping = pd.read_csv(path, sep='\t')
    return language_mapping.wiki.values
