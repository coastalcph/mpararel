import unittest
from unittest import mock

import numpy as np

from evaluate_consistency.run_evaluation import (compute_metrics_by_language,
                                                 compute_relation_metrics,
                                                 filter_predictions)


class TestMyClass(unittest.TestCase):
    def setUp(self):
        self.tuple_to_prediction = {
            "Rubens Barrichello-Brazil":
            [["[X] is [Y] citizen", "Brazil", "0"],
             ["as a citizen of [Y], [X]", "Germany", "5"],
             ["[X], who has a citizenship of [Y]", "Brazil", "0"],
             ["[X] a citizen of [Y]", "Brazil", "0"]],
            "Yves Mirande-France":
            [["[X] is [Y] citizen", "Brazil", "10"],
             ["as a citizen of [Y], [X]", "Germany", "9"],
             ["[X], who has a citizenship of [Y]", "France", "0"],
             ["[X] a citizen of [Y]", "Germany", "5"]],
        }
        self.accuracy = np.average([3 / 4, 1 / 4])
        self.consistency = np.average([3 / (4 * 3 / 2), 1 / (4 * 3 / 2)])
        self.consistency_accuracy = np.average([3 / (4 * 3 / 2), 0])
        self.templates = [
            "[X] is [Y] citizen", "as a citizen of [Y], [X]",
            "[X], who has a citizenship of [Y]", "[X] a citizen of [Y]"
        ]

    def test_compute_relation_metrics(self):
        metrics = compute_relation_metrics(self.tuple_to_prediction.items())
        self.assertEqual(metrics["accuracy"], self.accuracy)
        self.assertEqual(metrics["consistency"], self.consistency)
        self.assertEqual(metrics["accuracy-consistency"],
                         self.consistency_accuracy)
        self.assertEqual(metrics["mlama-accuracy"], 0)
        metrics = compute_relation_metrics(self.tuple_to_prediction.items(),
                                           "[X] a citizen of [Y]")
        self.assertEqual(metrics["mlama-accuracy"], 0.5)

    def test_compute_metrics_by_language(self):
        mpararel = {
            "en": {
                "P101.jsonl": set(self.templates),
                "P102.jsonl": set(self.templates)
            },
            "es": {
                "P101.jsonl": set(self.templates[:2])
            }
        }
        m = mock.mock_open()
        with mock.patch('builtins.open', m), \
                mock.patch('json.load') as json_load_mock:
            json_load_mock.return_value = self.tuple_to_prediction
            (language_to_avg_metrics, language_to_std_metrics,
             stats_by_language) = compute_metrics_by_language(
                 mpararel, "", {})
        m.assert_any_call('en/P101.jsonl', 'r')
        m.assert_any_call('en/P102.jsonl', 'r')
        m.assert_any_call('es/P101.jsonl', 'r')
        # The macro average would divide by 2 so the accuracy would be the same
        # as in one relation.
        self.assertEqual(language_to_avg_metrics["en"]["accuracy"],
                         self.accuracy)
        self.assertEqual(language_to_avg_metrics["es"]["accuracy"], 0.25)
        self.assertEqual(language_to_std_metrics["en"]["accuracy"], 0.0)
        self.assertEqual(language_to_std_metrics["es"]["accuracy"], 0.0)
        self.assertSetEqual(
            set(list(language_to_avg_metrics.items())[0][1].keys()),
            set([
                "accuracy", "consistency", "accuracy-consistency",
                "mlama-accuracy"
            ]))
        self.assertDictEqual(
            stats_by_language, {
                "en": {
                    "removed_repeated_subjects":
                    0,
                    "total_phrases":
                    len(self.templates) * 2 *
                    len(self.tuple_to_prediction.keys())
                },
                "es": {
                    "removed_repeated_subjects":
                    0,
                    "total_phrases":
                    len(self.templates[:2]) *
                    len(self.tuple_to_prediction.keys())
                },
            })

    def test_compute_metrics_by_language_std(self):
        mpararel = {
            "en": {
                "P101.jsonl": set(self.templates[:2]),  # 0.25
                "P102.jsonl": set(self.templates)  # 0.5
            },
        }
        m = mock.mock_open()
        with mock.patch('builtins.open', m), \
                mock.patch('json.load') as json_load_mock:
            json_load_mock.return_value = self.tuple_to_prediction
            (language_to_avg_metrics, language_to_std_metrics,
             _) = compute_metrics_by_language(mpararel, "", {})
        m.assert_any_call('en/P101.jsonl', 'r')
        m.assert_any_call('en/P102.jsonl', 'r')
        # The macro average would divide by 2 so the accuracy would be the same
        # as in one relation.
        self.assertEqual(language_to_avg_metrics["en"]["accuracy"],
                         np.average([0.25, 0.5]))
        self.assertEqual(language_to_std_metrics["en"]["accuracy"],
                         np.std([0.25, 0.5]))

    def test_filter_predictions_repeated_subjects(self):
        tuple_to_prediction = self.tuple_to_prediction
        tuple_to_prediction["Rubens Barrichello-Chile"] = [[
            "[X] is [Y] citizen", "Chile", "0"
        ]]
        with mock.patch('wandb.run'):
            filtered_tuple_to_prediction, stats = filter_predictions(
                set(self.templates), tuple_to_prediction.items(), True)
        self.assertListEqual(
            filtered_tuple_to_prediction,
            [("Yves Mirande-France",
              self.tuple_to_prediction["Yves Mirande-France"])])
        self.assertEqual(stats["removed_repeated_subjects"], 2)
        self.assertEqual(stats["total_phrases"], 4)


if __name__ == '__main__':
    unittest.main()
