import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / 'run_unified_denoisekt_bayes.py'
SPEC = importlib.util.spec_from_file_location('unified_bayes_runner', RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class UnifiedBayesProtocolTest(unittest.TestCase):
    def test_merge_json_object_retains_other_dataset_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.json'
            path.write_text(
                json.dumps({'assist2009': {'completed_folds': 5}}),
                encoding='utf-8',
            )
            merged = RUNNER.merge_json_object(
                path, {'peiyou': {'completed_folds': 5}}
            )
            self.assertEqual(set(merged), {'assist2009', 'peiyou'})
            self.assertEqual(
                set(json.loads(path.read_text(encoding='utf-8'))),
                {'assist2009', 'peiyou'},
            )

    def test_pykt_search_explicitly_disables_all_test_evaluation(self):
        self.assertEqual(RUNNER.pykt_evaluation_flags('search'), (0, 0))
        self.assertEqual(RUNNER.pykt_evaluation_flags('final'), (0, 0))

    def test_matra_search_invokes_validation_only_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'runtime': {'python': '/usr/bin/python3'}}
            command = RUNNER.matra_command(
                config, Path(directory), fold=3, phase='search'
            )
            self.assertIn('--validation_only', command)
            self.assertNotIn(
                '--validation_only',
                RUNNER.matra_command(config, Path(directory), fold=3, phase='final'),
            )


if __name__ == '__main__':
    unittest.main()
