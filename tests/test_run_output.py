import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_output
from preprocessing.cctv import main as convert_main


class RunOutputTests(unittest.TestCase):
    def test_runs_are_unique_and_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'output'
            with patch.object(run_output, 'OUTPUT_ROOT', root):
                first = run_output.create_run_dir('train/model')
                second = run_output.create_run_dir('train/model')
            self.assertNotEqual(first, second)
            for run in (first, second):
                self.assertTrue(run.is_dir())
                self.assertEqual(run.parent, root)

    def test_artifact_destinations_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(run_output.artifact_path(root, 'logs/train.log'),
                             root.resolve() / 'logs/train.log')
            for destination in ('../escaped.json', str(root / 'absolute.json')):
                with self.assertRaises(ValueError):
                    run_output.artifact_path(root, destination)

    def test_conversion_writes_each_run_under_output(self):
        result = {'labels': ['fi', 'no'], 'database': {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'output'
            with patch.object(run_output, 'OUTPUT_ROOT', root), patch(
                'sys.argv', ['cctv_json.py', '--annotations', 'input.json',
                             '--video-dir', 'videos']
            ), patch('preprocessing.cctv.convert_cctv_dataset', return_value=result) as convert:
                first = convert_main()
                second = convert_main()
            self.assertNotEqual(first, second)
            for run in (first, second):
                self.assertEqual(json.loads((run / 'cctv_jpg.json').read_text()), result)
                self.assertEqual(run.parent, root)
            self.assertEqual(convert.call_args.args[2], second.resolve() / 'cctv_jpg')


if __name__ == '__main__':
    unittest.main()
