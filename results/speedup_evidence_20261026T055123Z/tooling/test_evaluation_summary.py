import csv
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_summary import gate, reserve, score, summarize


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = {"experiment_id": "test", "stage_gates": {"P2": "passed"},
                         "budget_caps": {"P5_episodes": 2, "P6_episodes": 2, "P4_rollouts": 24,
                                         "infrastructure_retries_total": 4, "retries_per_original": 1},
                         "splits": {k: [{"task": "t1", "episode_index": 1}] for k in ("P5", "P6")}}
        (self.root / "execution_manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temp.cleanup()

    def take(self, **kwargs):
        defaults = dict(stage="P5", task="t1", episode_index=1, arm="A")
        return reserve(self.root, **(defaults | kwargs))

    def test_no_refund_no_duplicate(self):
        self.take()
        with self.assertRaisesRegex(ValueError, "already reserved"):
            self.take()
        self.take(arm="B")
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.take()

    def test_retry_has_one_shared_extra_allowance(self):
        original = json.loads(self.take().read_text())["reservation_id"]
        with self.assertRaises(ValueError):
            self.take(retry_of=original)
        self.take(retry_of=original, reason="saved HTTP transport failure log")
        with self.assertRaisesRegex(ValueError, "retry allowance"):
            self.take(retry_of=original, reason="repeat")
        with self.assertRaisesRegex(ValueError, "change"):
            self.take(arm="B", retry_of=original, reason="different arm")

    def test_holdout_and_unlisted_reset_rejected(self):
        with self.assertRaisesRegex(ValueError, "passing P5"):
            self.take(stage="P6")
        with self.assertRaisesRegex(ValueError, "preregistered"):
            self.take(episode_index=7)

    def test_threshold_requires_complete_finite_service(self):
        for elapsed in (None, float("nan"), float("inf"), -1, True, 300.001):
            self.assertFalse(score(True, elapsed))
        self.assertTrue(score(True, 300))
        self.assertFalse(score(False, 100))

    def test_missing_pairs_not_converted_into_observed_failures(self):
        rows = [dict(task="t1", episode_index=1, arm=a, status="not_started") for a in "AB"]
        result = gate(rows, expected_pairs=1, mechanism_passed=True)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["completed_pairs"], 0)

    def test_two_success_gain_and_no_raw_regression_gate(self):
        rows = []
        for i in range(9):
            for arm in "AB":
                rows.append(dict(task="t1", episode_index=i, arm=arm, status="success",
                                 raw_libero_success=True, time_to_raw_success_s=250,
                                 service_elapsed_s=290 if arm == "B" and i < 2 else 400))
        self.assertEqual(gate(rows, mechanism_passed=True)["status"], "continue")
        rows[1]["raw_libero_success"] = False
        self.assertEqual(gate(rows, mechanism_passed=True)["status"], "freeze")

    def test_infrastructure_original_retained_not_replaced_by_retry(self):
        rows = [dict(task="t1", episode_index=1, arm=a, status="preparation_failed",
                     raw_libero_success=False, service_elapsed_s=30) for a in "AB"]
        rows.append(dict(task="t1", episode_index=1, arm="B", retry_of="trial_001",
                         status="success", raw_libero_success=True,
                         time_to_raw_success_s=50, service_elapsed_s=100))
        result = gate(rows, expected_pairs=1, mechanism_passed=True)
        self.assertEqual(result["completed_pairs"], 1)
        self.assertEqual(result["counts"]["B"]["S300"], 0)

    def test_missing_measurements_remain_unknown(self):
        rows = [dict(task="t1", episode_index=1, arm=a, status="success") for a in "AB"]
        result = gate(rows, expected_pairs=1, mechanism_passed=True)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["completed_pairs"], 0)

    def test_out_of_window_common_success_cannot_qualify_speed_branch(self):
        rows = [dict(task="t1", episode_index=i, arm=a, status="success",
                     raw_libero_success=True, time_to_raw_success_s=901,
                     service_elapsed_s=1500 if a == "A" else 1000)
                for i in range(9) for a in "AB"]
        result = gate(rows, mechanism_passed=True)
        self.assertEqual(result["common_success_pairs"], 0)
        self.assertEqual(result["status"], "freeze")

    def test_summary_ignores_claim_receipt_and_reads_separate_infrastructure_evidence(self):
        reserved = self.take()
        reserved.with_suffix('.claim.json').write_text(json.dumps({'session_dir': '/test'}))
        session = self.root / 'evaluations' / 'trial_001'
        session.mkdir(parents=True)
        (session / 'evidence_result.json').write_text(json.dumps({
            'reservation_id': 'trial_001', 'status': 'preparation_error',
            'raw_libero_success': False, 'service_elapsed_s': 7.5}))
        (session / 'infrastructure_annotation.json').write_text(json.dumps({
            'reservation_id': 'trial_001', 'infrastructure_failure': True,
            'infrastructure_evidence': 'prepare.log', 'reason': 'empty isolated credential store'}))
        summarize(self.root)
        sensitivity = json.loads((self.root / 'sensitivity_analysis.json').read_text())
        self.assertEqual(sensitivity['P5']['excluded_conditions'], [['t1', 1]])
        self.assertEqual(sensitivity['P5']['retained_complete_pairs'], 0)
        with (self.root / 'paired_results.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['reason'], 'empty isolated credential store')

    def test_online_stop_explains_unstarted_without_overwriting_terminal_failure(self):
        self.take()
        session = self.root / 'evaluations' / 'trial_001'
        session.mkdir(parents=True)
        receipt = session / 'evidence_result.json'
        receipt.write_text(json.dumps({
            'reservation_id': 'trial_001', 'status': 'preparation_error',
            'raw_libero_success': False, 'service_elapsed_s': 7.5}))
        summarize(self.root)  # Optional stop receipt is absent; terminal status is the fallback.
        with (self.root / 'paired_results.csv').open() as stream:
            self.assertEqual(list(csv.DictReader(stream))[0]['reason'], 'preparation_error')

        original = json.loads(receipt.read_text()) | {'reason': 'original transport failure'}
        receipt.write_text(json.dumps(original))
        receipt_bytes = receipt.read_bytes()
        for stop, expected in [
                ({'reason': 'cumulative budget exhausted', 'decision': 'freeze'},
                 'cumulative budget exhausted'),
                ({'decision': 'freeze after repeated defect'}, 'freeze after repeated defect')]:
            with self.subTest(stop=stop):
                (self.root / 'online_stop.json').write_text(json.dumps(stop))
                summarize(self.root)
                with (self.root / 'paired_results.csv').open() as stream:
                    terminal, unstarted = list(csv.DictReader(stream))
                self.assertEqual(terminal['reason'], 'original transport failure')
                self.assertEqual(terminal['status'], 'preparation_error')
                self.assertEqual(terminal['raw_libero_success'], 'False')
                self.assertEqual(terminal['service_elapsed_s'], '7.5')
                self.assertEqual(unstarted['status'], 'not_started')
                self.assertEqual(unstarted['reason'], expected)
                for key in ('raw_libero_success', 'service_elapsed_s', 'S300'):
                    self.assertEqual(unstarted[key], '')
                with (self.root / 'heldout_results.csv').open() as stream:
                    heldout = list(csv.DictReader(stream))
                self.assertTrue(all(row['reason'] == 'P5 not passed; holdout remains sealed'
                                    for row in heldout))
                self.assertEqual(receipt.read_bytes(), receipt_bytes)


if __name__ == "__main__":
    unittest.main()
