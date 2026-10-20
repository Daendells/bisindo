"""The page's numbers come from summarize() and threshold_stats(); pin their rules down.

Run: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_site_data import LETTERS, summarize, threshold_stats  # noqa: E402


def items(*rows):
    """(truth, prediction or None, confidence) for each photo; every letter appears at least once."""
    return [(L, L, 0.9) for L in LETTERS if L not in {r[0] for r in rows}] + list(rows)


class Summarize(unittest.TestCase):
    def test_a_photo_without_a_box_counts_as_wrong(self):
        s = summarize(items(("A", None, 0.0)))
        self.assertEqual((s["n"], s["correct"]), (26, 25))
        self.assertEqual(s["per_letter"]["A"], {"n": 1, "ok": 0, "confused": {"no detection": 1}})

    def test_confusions_are_counted_per_true_letter(self):
        s = summarize(items(("M", "N", 0.8), ("M", "N", 0.7), ("N", "M", 0.6), ("N", "N", 0.9)))
        self.assertEqual(s["per_letter"]["M"], {"n": 2, "ok": 0, "confused": {"N": 2}})
        self.assertEqual(s["per_letter"]["N"], {"n": 2, "ok": 1, "confused": {"M": 1}})
        self.assertAlmostEqual(s["top1"], 25 / 28, places=6)

    def test_macro_f1_is_the_plain_mean_over_letters(self):
        s = summarize(items(("A", "B", 0.9)))  # A: P 0, R 0 -> F1 0; B: P 1/2, R 1 -> F1 2/3
        self.assertAlmostEqual(s["macro_f1"], round((24 + 2 / 3) / 26, 4), places=4)


class Threshold(unittest.TestCase):
    def test_precision_and_coverage_only_count_photos_above_the_threshold(self):
        t = threshold_stats([("A", "A", 0.95), ("B", "C", 0.65), ("C", "C", 0.59), ("D", None, 0.0)])
        self.assertEqual(t["coverage"], 0.5)
        self.assertEqual(t["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
