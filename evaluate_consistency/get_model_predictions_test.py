import unittest
from unittest import mock

import numpy as np
import torch

import evaluate_consistency.get_model_predictions as get_model_predictions
from evaluate_consistency.get_model_predictions import TemplateTuple


def dummy_tokenizer(s):
    if s.startswith("C"):
        return [s]
    return [s[:-2], s[-2:]]


class TestGetModelsPredictions(unittest.TestCase):
    def test_data_generator(self):
        tuples = {
            "en": {
                "P103.jsonl": [{
                    "sub_label": "Ottawa",
                    "obj_label": "Canada"
                }, {
                    "sub_label": "Berlin",
                    "obj_label": "Germany"
                }],
                "P101.jsonl": [{
                    "sub_label": "Albert Einstein",
                    "obj_label": "Ulm"
                }]
            },
        }
        templates = {
            "en": {
                "P103.jsonl": ["[X] is the capital of [Y]"],
                "P101.jsonl": ["[X] was born in [Y]", "[Y] is [X]'s hometown"]
            },
        }
        get_tuples = lambda lang, relation: tuples[lang][relation]
        get_candidates = lambda lang, relation: [
            pair["obj_label"] for pair in tuples[lang][relation]
        ]
        get_templates = lambda lang, relation: templates[lang][relation]
        languages = templates.keys()
        relations = templates["en"].keys()
        tokenizer = mock.Mock()
        tokenizer.mask_token = "[MASK]"
        tokenizer.tokenize.side_effect = dummy_tokenizer
        tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: [
            -1 for t in tokens
        ]
        template_tuple_examples = get_model_predictions.GenerateTemplateTupleExamples(
            tokenizer, languages, relations, get_candidates, get_templates,
            get_tuples)
        results = [r for r in template_tuple_examples]
        expected_results = [
            ([
                'Ottawa is the capital of [MASK]',
                'Ottawa is the capital of [MASK] [MASK]'
            ], [{
                'Canada': [-1]
            }, {
                'Germany': [-1, -1]
            }],
             TemplateTuple(language='en',
                           relation='P103.jsonl',
                           template='[X] is the capital of [Y]',
                           subject='Ottawa',
                           object='Canada')),
            ([
                'Berlin is the capital of [MASK]',
                'Berlin is the capital of [MASK] [MASK]'
            ], [{
                'Canada': [-1]
            }, {
                'Germany': [-1, -1]
            }],
             TemplateTuple(language='en',
                           relation='P103.jsonl',
                           template='[X] is the capital of [Y]',
                           subject='Berlin',
                           object='Germany')),
            (['Albert Einstein was born in [MASK] [MASK]'], [{
                'Ulm': [-1, -1]
            }],
             TemplateTuple(language='en',
                           relation='P101.jsonl',
                           template='[X] was born in [Y]',
                           subject='Albert Einstein',
                           object='Ulm')),
            (["[MASK] [MASK] is Albert Einstein's hometown"], [{
                'Ulm': [-1, -1]
            }],
             TemplateTuple(language='en',
                           relation='P101.jsonl',
                           template="[Y] is [X]'s hometown",
                           subject='Albert Einstein',
                           object='Ulm'))
        ]
        self.assertListEqual(results, expected_results)

    def test_data_generator_adding_point(self):
        tuples = {
            "en": {
                "P103.jsonl": [{
                    "sub_label": "Santiago",
                    "obj_label": "Chile"
                }]
            },
        }
        templates = {
            "en": {
                "P103.jsonl": ["[X] is the capital of [Y]"],
            },
        }
        get_tuples = lambda lang, relation: tuples[lang][relation]
        get_candidates = lambda lang, relation: [
            pair["obj_label"] for pair in tuples[lang][relation]
        ]
        get_templates = lambda lang, relation: templates[lang][relation]
        languages = templates.keys()
        relations = templates["en"].keys()
        tokenizer = mock.Mock()
        tokenizer.mask_token = "[MASK]"
        tokenizer.tokenize.side_effect = dummy_tokenizer
        tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: [
            -1 for t in tokens
        ]
        template_tuple_examples = get_model_predictions.GenerateTemplateTupleExamples(
            tokenizer,
            languages,
            relations,
            get_candidates,
            get_templates,
            get_tuples,
            add_at_eos=' .')
        results = [r for r in template_tuple_examples]
        expected_results = [
            ([
                'Santiago is the capital of [MASK] .',
            ], [{
                'Chile': [-1]
            }],
             TemplateTuple(language='en',
                           relation='P103.jsonl',
                           template='[X] is the capital of [Y]',
                           subject='Santiago',
                           object='Chile')),
        ]
        self.assertListEqual(results, expected_results)

    def test_get_masks_indices(self):
        encoded_input = mock.Mock()
        encoded_input.input_ids = torch.tensor([[1, 2, 25, 4, 5],
                                                [25, 1, 0, 0, 0]])
        self.assertTrue(
            torch.all(
                get_model_predictions.get_masks_indices(encoded_input, 25) ==
                torch.tensor([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])))

    def test_get_candidates_probabilities(self):
        logits = torch.zeros(5, 4)  # (num_tokens, vocab_size)
        logits[3, :] = torch.tensor([3, 5, 6, 8])
        masks_indices = torch.tensor([0, 0, 0, 1, 0])
        candidates_to_ids_i = {'Canada': [0], "Chile": [1]}
        candidates_to_prob = get_model_predictions.get_candidates_probabilities(
            logits, masks_indices, candidates_to_ids_i)
        self.assertDictEqual(candidates_to_prob, {"Canada": 3, "Chile": 5})
        logits[2, :] = torch.tensor([3, 5, 6, 8])
        masks_indices = torch.tensor([0, 0, 1, 1, 0])
        candidates_to_prob = get_model_predictions.get_candidates_probabilities(
            logits, masks_indices, {'Germany': [1, 2]})
        self.assertDictEqual(candidates_to_prob, {"Germany": 5.5})

    def test_get_predicted_and_rank_of_correct(self):
        candidates_to_prob = {"Canada": 1, "Germany": 5, "Chile": 3}
        predicted, rank_correct = get_model_predictions.get_predicted_and_rank_of_correct(
            candidates_to_prob, "Chile")
        self.assertEqual(predicted, "Germany")
        self.assertEqual(rank_correct, 1)
        predicted, rank_correct = get_model_predictions.get_predicted_and_rank_of_correct(
            candidates_to_prob, "Canada")
        self.assertEqual(predicted, "Germany")
        self.assertEqual(rank_correct, 2)

    def test_batchify(self):
        batches = get_model_predictions.batchify(np.arange(0, 5),
                                                 np.arange(5, 10), 3)
        np.testing.assert_equal(batches,
                                [[[0, 1, 2], [5, 6, 7]], [[3, 4], [8, 9]]])
        batches = get_model_predictions.batchify(np.arange(0, 5),
                                                 np.arange(5, 10), 6)
        np.testing.assert_equal(batches, [[np.arange(0, 5), np.arange(5, 10)]])

    @mock.patch('json.dump')
    def test_write_predictions(self, json_mock):
        en_p101 = {
            "Ottawa-Canada": ['[X] is the capital of [Y]', "Japan", "1"]
        }
        es_p101 = {
            "Ottawa-Canadá": ["[X] es la capital de [Y]", "Canadá", "0"]
        }
        results = {
            "en": {
                "P101.jsonl": en_p101
            },
            "es": {
                "P101.jsonl": es_p101
            }
        }
        m = mock.mock_open()
        with mock.patch('builtins.open', m), mock.patch('os.makedirs'):
            get_model_predictions.write_predictions(results, "", None)
        m.assert_any_call('en/P101.jsonl', 'w')
        m.assert_any_call('es/P101.jsonl', 'w')
        json_mock.assert_has_calls(
            [mock.call(en_p101, m()),
             mock.call(es_p101, m())])

    @mock.patch('json.dump')
    def test_write_predictions_with_existing_predictions(self, json_dump_mock):
        en_p303 = {
            "Albert Einstein-Ulm": [('[X] was born in [Y]', "Berlin", "1")]
        }
        fr_p101 = {
            "Ottawa-Canadá": ["[X] est la capitale de [Y]", "Canadá", "0"]
        }
        results = {
            "en": {
                "P101.jsonl": {
                    "Ottawa-Canada":
                    [('[X], the capital of [Y]', "Japan", "1")]
                },
                "P303.jsonl": en_p303
            },
            "fr": {
                "P101.jsonl": fr_p101
            }
        }
        m = mock.mock_open()
        with mock.patch('builtins.open', m), mock.patch('os.makedirs'), \
                mock.patch('os.listdir') as mock_listdir, \
                mock.patch('os.path.isfile') as mock_isfile, \
                mock.patch('json.load') as json_load_mock, \
                mock.patch('shutil.copytree') as copy_mock :
            list_dir_dispatch = {
                "existing/en": ["P101.jsonl"],
                "existing": ["es", "en"]
            }
            mock_listdir.side_effect = lambda path: list_dir_dispatch[path]
            mock_isfile.side_effect = lambda path: path.endswith("P101.jsonl")
            json_load_mock.return_value = {
                "Ottawa-Canada":
                [('[X] is the capital of [Y]', 'Germany', '1'),
                 ('The capital of [Y] is [X]', 'Germany', '0')],
                "Berlin-Germany":
                [('[X] is the capital of [Y]', 'Germany', '0')]
            }
            get_model_predictions.write_predictions(results, "output",
                                                    "existing")
        m.assert_any_call('existing/en/P101.jsonl', 'r')
        m.assert_any_call('output/en/P101.jsonl', 'w')
        m.assert_any_call('output/en/P303.jsonl', 'w')
        m.assert_any_call('output/fr/P101.jsonl', 'w')
        json_dump_mock.assert_has_calls([
            mock.call(
                {
                    "Ottawa-Canada":
                    [('[X], the capital of [Y]', "Japan", "1"),
                     ('[X] is the capital of [Y]', 'Germany', '1'),
                     ('The capital of [Y] is [X]', 'Germany', '0')],
                    "Berlin-Germany": [
                        ('[X] is the capital of [Y]', 'Germany', '0')
                    ]
                }, m()),
            mock.call(en_p303, m()),
            mock.call(fr_p101, m())
        ],
                                        any_order=True)
        copy_mock.assert_called_once_with("existing/es", "output/es")


if __name__ == '__main__':
    unittest.main()
