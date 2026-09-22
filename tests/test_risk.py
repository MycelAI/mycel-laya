"""Selective-risk regression tests on synthetic independent predictions, CPU only."""
import copy
from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.risk import binomial_upper_bound, evaluate_risk_policy, select_risk_policy  # noqa: E402


def samples(prefix, size=300, languages=("en", "de"), error_every=None, score=0.95):
    return [{"id": "%s:%s:%s" % (prefix, lang, i), "group_id": "%s:family:%s" % (prefix, i),
             "language": lang, "probabilities": [score, 1 - score],
             "target": 1 if error_every and i % error_every == 0 else 0}
            for lang in languages for i in range(size)]


class BinomialTests(unittest.TestCase):
    def test_closed_form_boundaries_and_published_reference(self):
        self.assertEqual(binomial_upper_bound(0, 0), 1)
        self.assertEqual(binomial_upper_bound(100, 100), 1)
        self.assertAlmostEqual(binomial_upper_bound(0, 100), 1 - 0.05 ** 0.01, places=14)
        self.assertAlmostEqual(binomial_upper_bound(1, 2), math.sqrt(0.95), places=14)
        # SciPy's published exact 95% two-sided interval for 7/50 uses alpha=.025
        # in its upper tail: docs.scipy.org/.../BinomTestResult.proportion_ci.html
        self.assertAlmostEqual(binomial_upper_bound(7, 50, 0.025), 0.26739600249700846, places=12)

    def test_exact_small_sample_coverage(self):
        # Enumerate the binomial sampling distribution, rather than simulating a
        # conveniently passing seed. The probability of undercoverage is <= alpha.
        for n in (1, 2, 5, 10, 25):
            for alpha in (0.01, 0.05, 0.2):
                bounds = [binomial_upper_bound(k, n, alpha) for k in range(n + 1)]
                self.assertEqual(bounds, sorted(bounds))
                for step in range(1, 100):
                    p = step / 100
                    missed = math.fsum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                                       for k in range(n + 1) if bounds[k] < p)
                    self.assertLessEqual(missed, alpha + 1e-12)

    def test_invalid_counts_and_alpha(self):
        for k, n, alpha in [(True, 1, .05), (1.0, 2, .05), (-1, 2, .05), (2, 1, .05),
                            (0, -1, .05), (0, 1, 0), (0, 1, 1), (0, 1, float("nan"))]:
            with self.subTest(k=k, n=n, alpha=alpha), self.assertRaises(ValueError):
                binomial_upper_bound(k, n, alpha)

    def test_large_sample_tail_against_decimal_probability_recurrence(self):
        # Independent arithmetic checks log-gamma cancellation and extreme tails.
        with localcontext() as context:
            context.prec = 60
            for k, n, alpha in [(100, 5000, .001), (990, 1000, .05), (1, 10000, 1e-8)]:
                bound = binomial_upper_bound(k, n, alpha)
                p = Decimal(str(bound))
                q = 1 - p
                term = q ** n
                cdf = term
                for j in range(1, k + 1):
                    term *= Decimal(n - j + 1) / j * p / q
                    cdf += term
                self.assertAlmostEqual(float(cdf) / alpha, 1.0, places=8)


class SelectionTests(unittest.TestCase):
    def test_correct_confident_population_can_pass_and_roundtrip_json(self):
        import json
        rows = samples("policy")
        before = copy.deepcopy(rows)
        selected = select_risk_policy(rows, required_languages=["de", "en"], thresholds=[.5, .9, .99])
        self.assertTrue(selected["passed"])
        self.assertEqual(selected["tail_probability"], .05 / (2 * 2 * 3))
        self.assertEqual(selected["by_language"]["en"]["selected"]["threshold"], .9)
        self.assertEqual(rows, before)
        self.assertEqual(selected, select_risk_policy(reversed(rows), required_languages=["en", "de"],
                                                     thresholds=[.99, .9, .5]))
        tested = evaluate_risk_policy(samples("test"), json.loads(json.dumps(selected)))
        self.assertTrue(tested["passed"])
        self.assertEqual(tested["tail_probability"], .05 / (2 * 2))
        self.assertGreater(tested["by_language"]["en"]["error_upper"], 0)
        self.assertLess(tested["by_language"]["en"]["coverage_lower"], 1)

    def test_overconfident_errors_do_not_pass(self):
        selected = select_risk_policy(samples("wrong", error_every=2, score=1), required_languages=["en", "de"])
        self.assertFalse(selected["passed"])
        self.assertTrue(all(entry["selected"] is None for entry in selected["by_language"].values()))

    def test_selection_excludes_uncertain_errors_while_retaining_coverage(self):
        rows = samples("mixed", size=1000, languages=("en",))
        for i, row in enumerate(rows):
            if i >= 700:
                row["probabilities"] = [.6, .4]
                row["target"] = 1
            elif i % 100 == 0:
                row["target"] = 1
        selected = select_risk_policy(rows, required_languages=["en"], thresholds=[.5, .9, .99])
        chosen = selected["by_language"]["en"]["selected"]
        self.assertTrue(selected["passed"])
        self.assertEqual(chosen["threshold"], .9)
        self.assertEqual(chosen["accepted"], 700)
        self.assertEqual(chosen["errors"], 7)
        self.assertGreater(chosen["error_upper"], .01)
        self.assertGreaterEqual(chosen["coverage_lower"], .5)

    def test_more_threshold_comparisons_require_more_evidence(self):
        rows = samples("limited", size=80, languages=("en",))
        single = select_risk_policy(rows, required_languages=["en"], thresholds=[.9])
        multiple = select_risk_policy(rows, required_languages=["en"], thresholds=[.8, .9, .95])
        self.assertTrue(single["passed"])
        self.assertFalse(multiple["passed"])

    def test_no_evidence_and_too_few_examples_cannot_pass(self):
        for rows in ([], samples("small", size=5), samples("missing", languages=("en",))):
            with self.subTest(size=len(rows)):
                selected = select_risk_policy(rows, required_languages=["en", "de"])
                self.assertFalse(selected["passed"])
                with self.assertRaises(ValueError):
                    evaluate_risk_policy(samples("test"), selected)

    def test_defer_everything_fails_coverage_and_has_no_error_estimate(self):
        selected = select_risk_policy(samples("defer", score=.5), required_languages=["en", "de"], thresholds=[.9])
        result = selected["by_language"]["en"]["candidates"][0]
        self.assertFalse(selected["passed"])
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["coverage_lower"], 0)
        self.assertEqual(result["error_upper"], 1)
        self.assertIsNone(result["error_rate"])

    def test_language_failure_cannot_be_hidden_by_pooling(self):
        rows = samples("good", size=1000, languages=("en",)) + samples("bad", size=100, languages=("de",), error_every=2)
        selected = select_risk_policy(rows, required_languages=["en", "de"], thresholds=[.9])
        self.assertFalse(selected["passed"])
        self.assertIsNotNone(selected["by_language"]["en"]["selected"])
        self.assertIsNone(selected["by_language"]["de"]["selected"])

    def test_duplicates_cannot_inflate_evidence(self):
        rows = samples("repeat", size=1, languages=("en",))
        duplicate = {**rows[0], "id": "different-id"}
        for extra in (rows[0], duplicate):
            with self.assertRaisesRegex(ValueError, "independent group"):
                select_risk_policy(rows + [extra], required_languages=["en"])

    def test_bad_predictions_and_protocols_are_rejected(self):
        for fields in ({"probabilities": [1.2, -.2]}, {"probabilities": [.9, .9]},
                       {"probabilities": [float("nan"), 0]}, {"probabilities": [True, 0]},
                       {"probabilities": [1]}, {"target": True}, {"target": 2},
                       {"target": "0"}, {"group_id": ""}, {"language": "fr"}):
            row = {**samples("invalid", size=1, languages=("en",))[0], **fields}
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                select_risk_policy([row], required_languages=["en"])
        for settings in ({"required_languages": []}, {"required_languages": ["en", "en"]},
                         {"min_coverage": 0}, {"max_error": 1}, {"alpha": 1},
                         {"thresholds": []}, {"thresholds": [.9, .9]}, {"thresholds": [float("inf")]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                select_risk_policy([], **{"required_languages": ["en"], **settings})


class IndependentTestTests(unittest.TestCase):
    def setUp(self):
        self.selection = select_risk_policy(samples("policy"), required_languages=["en", "de"], thresholds=[.9])

    def test_selection_examples_and_families_cannot_be_reused(self):
        for field, value in [("id", "policy:en:0"), ("group_id", "policy:family:0")]:
            rows = samples("test")
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reuses"):
                evaluate_risk_policy(rows, self.selection)

    def test_locked_policy_can_fail_fresh_population(self):
        for rows in ([], samples("test", score=.6), samples("test", error_every=3)):
            with self.subTest(size=len(rows)):
                self.assertFalse(evaluate_risk_policy(rows, self.selection)["passed"])

    def test_option_count_and_incomplete_selection_are_rejected(self):
        rows = samples("test")
        rows[0]["probabilities"] = [.9, .05, .05]
        with self.assertRaisesRegex(ValueError, "fixed option schema"):
            evaluate_risk_policy(rows, self.selection)
        for field, value in [("schema_version", True), ("stage", "final_test"), ("group_hashes", []),
                             ("id_hashes", None), ("n_options", True), ("by_language", {})]:
            invalid = {**self.selection, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_risk_policy(samples("test"), invalid)


if __name__ == "__main__":
    unittest.main()
