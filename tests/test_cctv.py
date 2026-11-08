import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from preprocessing.cctv import extract_clip


class ExtractionTests(unittest.TestCase):
    def run_extraction(self, directory, start, end, frames, write_ok=True):
        capture = SimpleNamespace(
            isOpened=lambda: True,
            read=lambda: next(frames, (False, None)),
            release=lambda: released.append(True),
        )
        released, written = [], []

        def write(path, image):
            written.append(image)
            if write_ok:
                Path(path).touch()
            return write_ok

        cv2 = SimpleNamespace(VideoCapture=lambda _: capture,
                              resize=lambda image, _: image, imwrite=write)
        with patch.dict(sys.modules, {"cv2": cv2}):
            count = extract_clip("video.mpeg", directory, start, end)
        self.assertEqual(released, [True])
        return count, written

    def test_frame_alignment_and_shorter_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            count, written = self.run_extraction(
                directory, 3, 6, iter((True, i) for i in range(10)))
            self.assertEqual((count, written), (3, [3, 4, 5]))
            count, written = self.run_extraction(
                directory, 3, 4, iter((True, i) for i in range(10)))
            self.assertEqual((count, written), (1, [3]))
            self.assertEqual(len(list(Path(directory).glob('image_*.jpg'))), 1)
            self.assertEqual((Path(directory) / 'n_frames').read_text(), '1')

    def test_failed_read_is_not_resized(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_extraction(directory, 0, 5, iter([])), (0, []))

    def test_failed_write_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                self.run_extraction(directory, 0, 1, iter([(True, 0)]), False)


if __name__ == '__main__':
    unittest.main()
