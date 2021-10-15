import unittest
from unittest import mock
from evaluate_consistency.run_evaluation import compute_relation_metrics
import numpy as np


class TestMyClass(unittest.TestCase):
    def test_compute_relation_metrics(self):
        tuple_to_prediction = {
            "Rubens Barrichello-Brazil":
            [["[X] is [Y] citizen", "Brazil", "0"],
             ["as a citizen of [Y], [X]", "Germany", "5"],
             ["[X], who has a citizenship of [Y]", "Brazil", "0"],
             ["[X] a citizen of [Y]", "Brazil", "0"]],
            "Yves Mirande-France":
            [["[X] is [Y] citizen", "Brazil", "10"],
             ["as a citizen of [Y], [X]", "France", "0"],
             ["[X], who has a citizenship of [Y]", "Germany", "9"],
             ["[X] a citizen of [Y]", "Germany", "5"]],
        }
        metrics = compute_relation_metrics(tuple_to_prediction)
        self.assertEqual(metrics["accuracy"], np.average([3 / 4, 1 / 4]))
        self.assertEqual(metrics["consistency"],
                         np.average([3 / (4 * 3 / 2), 1 / (4 * 3 / 2)]))
        self.assertEqual(metrics["accuracy-consistency"],
                         np.average([3 / (4 * 3 / 2), 0]))


if __name__ == '__main__':
    unittest.main()