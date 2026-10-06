import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from export_session import export_session


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.laps = []
        for number in range(1, 7):
            name = f'lap_{number:02}.csv'
            duration = 100 + number
            samples = [dict(lap_time_s=i * duration / 2000, position_normalized=i / 2000,
                            speed_kmh=200, throttle=1, brake=0, tc=0, abs=0) for i in range(2000)]
            with (self.directory / name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
            self.laps.append(dict(file=name, game_lap_number=number, lap_time_ms=duration * 1000,
                                  valid=True, complete=True, contains_pit=False))
        self.metadata()

    def metadata(self):
        (self.directory / 'session.json').write_text(json.dumps(dict(car='porsche', track='monza', laps=self.laps)))

    def test_default_selection_and_original_bytes_and_manifest_hashes(self):
        original = (self.directory / 'session.json').read_bytes()
        (self.directory / 'analysis.json').write_text('{"stale": true}')
        with zipfile.ZipFile(export_session(self.directory)) as archive:
            manifest = json.loads(archive.read('export_manifest.json'))
            self.assertEqual({x['file'] for x in manifest['selected_laps']},
                             {'lap_01.csv', 'lap_02.csv', 'lap_04.csv', 'lap_06.csv'})
            self.assertEqual(archive.read('session.json'), original)
            self.assertEqual(json.loads(archive.read('analysis.json'))['reference_file'], 'lap_01.csv')
            for item in manifest['files']:
                self.assertEqual(item['sha256'], hashlib.sha256(archive.read(item['file'])).hexdigest())
        self.assertEqual((self.directory / 'session.json').read_bytes(), original)

    def test_manual_selection_and_custom_output(self):
        destination = self.directory / 'packages' / 'selected.zip'
        self.assertEqual(export_session(self.directory, laps=['lap_05.csv', 'lap_05.csv'], output=destination), destination.resolve())
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual({n for n in archive.namelist() if n.endswith('.csv')}, {'lap_01.csv', 'lap_05.csv'})

    def test_all_mode_includes_partial_and_invalid_but_not_active_csv(self):
        self.laps[2]['valid'] = False
        self.metadata()
        (self.directory / 'lap_07.partial.csv').write_text('partial')
        (self.directory / 'lap_08.recording.csv').write_text('active')
        with zipfile.ZipFile(export_session(self.directory, all_csv=True)) as archive:
            self.assertIn('lap_07.partial.csv', archive.namelist())
            self.assertIn('lap_03.csv', archive.namelist())
            self.assertNotIn('lap_08.recording.csv', archive.namelist())
            self.assertEqual(len([n for n in archive.namelist() if n.endswith('.csv')]), 7)

    def test_missing_selected_csv_does_not_replace_existing_package(self):
        destination = export_session(self.directory)
        original = destination.read_bytes()
        (self.directory / 'lap_05.csv').unlink()
        with self.assertRaises(ValueError):
            export_session(self.directory, laps=['lap_05.csv'])
        self.assertEqual(destination.read_bytes(), original)
        with self.assertRaises(ValueError):
            export_session(self.directory, laps=['../secret.csv'])

    def test_fallback_includes_game_best_and_usable_reference(self):
        (self.directory / 'lap_01.csv').write_text('corrupt')
        with zipfile.ZipFile(export_session(self.directory, laps=['lap_06.csv'])) as archive:
            manifest = json.loads(archive.read('export_manifest.json'))
            self.assertTrue(manifest['reference_fallback'])
            self.assertEqual(manifest['reference_file'], 'lap_02.csv')
            self.assertTrue({'lap_01.csv', 'lap_02.csv', 'lap_06.csv'}.issubset(archive.namelist()))


if __name__ == '__main__':
    unittest.main()
