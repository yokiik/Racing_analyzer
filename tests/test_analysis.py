import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_session import LapTrace, NUMERIC, build_analysis, metrics, write_analysis
from acc_reader import Graphics, Physics
from recorder import Recorder, telemetry_sample


def rows(extra=0, offset=0):
    result = []
    for i in range(2000):
        p = i / 2000 + offset
        if p > 1: break
        t = 100 * p + extra * min(1, max(0, (p - .1) / .1))
        speed = 250 - 180 * max(0, 1 - abs(p - .15) / .05)
        result.append(dict(lap_time_s=t, position_normalized=p, speed_kmh=speed,
                           brake=1 if .115 <= p < .14 else 0,
                           throttle=0 if .11 <= p < .16 else .3 if .16 <= p < .18 else 1,
                           tc=1 if .17 <= p < .185 else 0,
                           abs=.5 if .115 <= p < .14 else 0))
    return result


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.infos = []

    def add(self, filename, samples, ms=100000, valid=True, complete=True):
        with (self.directory / filename).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(samples[0]))
            w.writeheader()
            w.writerows(samples)
        self.infos.append(dict(file=filename, game_lap_number=len(self.infos) + 1,
                               lap_time_ms=ms, valid=valid, complete=complete, contains_pit=False))
        self.metadata()

    def metadata(self, track='monza'):
        (self.directory / 'session.json').write_text(json.dumps(
            dict(car='porsche_992_gt3_r', track=track, laps=self.infos)))

    def test_known_corner_metrics_and_intervention_durations(self):
        zone = next(z for z in metrics(LapTrace(rows(), 100000)) if z['name'] == 'Rettifilo')
        self.assertEqual(zone['time_s'], 10)
        self.assertEqual(zone['entry_speed_kmh'], 250)
        self.assertEqual(zone['min_speed_kmh'], 70)
        self.assertEqual(zone['exit_speed_kmh'], 250)
        self.assertEqual(zone['brake_start']['lap_time_s'], 11.5)
        self.assertEqual(zone['first_throttle']['lap_time_s'], 16)
        self.assertEqual(zone['full_throttle']['lap_time_s'], 18)
        self.assertEqual(zone['tc_active_s'], 1.5)
        self.assertEqual(zone['abs_active_s'], 2.5)

    def test_aligned_delta_conserves_total_and_excludes_invalid_reference(self):
        self.add('best.csv', rows(), 100000)
        self.add('slower.csv', rows(8, offset=.00013), 108000)
        self.add('invalid_faster.csv', rows(-5), 95000, valid=False)
        self.add('partial.csv', rows()[:200], complete=False)
        report = write_analysis(self.directory)
        self.assertEqual(report['reference_file'], 'best.csv')
        slower = next(l for l in report['laps'] if l['file'] == 'slower.csv')
        self.assertEqual(slower['delta_s'], 8)
        rettifilo = next(z for z in slower['zones'] if z['name'] == 'Rettifilo')
        # Two sampling grids straddle a pace-change boundary differently.
        # Linear alignment must keep the local error below one sample interval.
        self.assertLess(abs(rettifilo['delta_s'] - 8), .02)
        self.assertAlmostEqual(sum(z['delta_s'] for z in slower['zones']), 8, places=3)
        self.assertEqual(rettifilo['reference_metrics']['abs_active_s'], 2.5)
        self.assertEqual(report['excluded'][0]['file'], 'partial.csv')
        self.assertIn('лучший:', (self.directory / 'report.html').read_text())

    def test_absent_events_are_null_and_noise_is_not_a_pedal_event(self):
        samples = rows()
        for r in samples:
            r['brake'] = 0
            r['throttle'] = 1
        # A single brake spike cannot pass the sustained event requirement.
        samples[230]['brake'] = 1
        zone = metrics(LapTrace(samples, 100000))[1]
        self.assertIsNone(zone['brake_start'])
        self.assertIsNone(zone['first_throttle'])
        self.assertIsNone(zone['full_throttle'])
        self.assertEqual(zone['full_throttle_status'], 'already_full_no_lift')

    def test_full_throttle_before_minimum_speed_is_preserved(self):
        samples = rows()
        for r in samples:
            if .13 <= r['position_normalized'] < .18:
                r['throttle'] = 1
        zone = metrics(LapTrace(samples, 100000))[1]
        self.assertEqual(zone['full_throttle']['lap_time_s'], 13)
        self.assertLess(zone['full_throttle']['position_normalized'], zone['min_speed_position'])

    def test_bad_trace_and_unsupported_track_are_reported(self):
        samples = rows()
        samples[200]['position_normalized'] = .01
        self.add('bad.csv', samples)
        report = build_analysis(self.directory)
        self.assertEqual(report['status'], 'reference_telemetry_unavailable')
        self.assertEqual(report['reference_file'], 'bad.csv')
        self.assertIn('backwards', report['excluded'][0]['reason'])
        self.metadata('spa')
        self.assertEqual(build_analysis(self.directory)['status'], 'unsupported_track_or_demo')
        with self.assertRaises(ValueError):
            LapTrace(rows()[500:], 100000)

    def test_recorder_builds_report_after_finish_and_keeps_recording(self):
        recorder = Recorder(self.directory)
        self.addCleanup(recorder.close)
        g = Graphics(status=2, session=0, iCurrentTime=99950,
                     normalizedCarPosition=.9995, isValidLap=1, packet_id=1)
        p = Physics(speed=250, packet_id=1)
        recorder.accept('porsche_992_gt3_r', 'monza', g, telemetry_sample(p, g))
        for packet, sample in enumerate(rows(), 2):
            g.completedLaps = 1
            g.packet_id = p.packet_id = packet
            g.iCurrentTime = round(sample['lap_time_s'] * 1000)
            g.normalizedCarPosition = sample['position_normalized']
            g.iLastTime = 100000
            p.speed, p.throttle, p.brake = sample['speed_kmh'], sample['throttle'], sample['brake']
            p.tc, p.abs = sample['tc'], sample['abs']
            recorder.accept('porsche_992_gt3_r', 'monza', g, telemetry_sample(p, g))
        g.completedLaps = 2
        g.iCurrentTime = 0
        g.normalizedCarPosition = 0
        g.packet_id = p.packet_id = 3000
        recorder.accept('porsche_992_gt3_r', 'monza', g, telemetry_sample(p, g))
        self.assertIsNotNone(recorder.file)
        report = json.loads((recorder.directory / 'analysis.json').read_text())
        self.assertEqual(report['reference_file'], 'lap_02.csv')
        self.assertTrue((recorder.directory / 'report.html').exists())
        recorder.close()
        report = json.loads((recorder.directory / 'analysis.json').read_text())
        self.assertTrue(any(x['file'] == 'lap_03.partial.csv' for x in report['excluded']))

    def test_true_fastest_reference_is_not_replaced_by_slower_usable_lap(self):
        fast = rows()
        fast[200]['position_normalized'] = .01
        self.add('lap_03.csv', fast, 112100)
        slow = rows()
        for r in slow:
            r['lap_time_s'] *= 1.12327
        self.add('lap_04.csv', slow, 112327)
        report = write_analysis(self.directory)
        self.assertEqual(report['reference_file'], 'lap_03.csv')
        self.assertEqual(report['reference_lap_time_s'], 112.1)
        self.assertEqual(report['status'], 'reference_telemetry_unavailable')
        self.assertEqual(report['laps'][0]['delta_s'], .227)
        self.assertNotIn('delta_s', report['laps'][0]['zones'][0])
        self.assertIn('более медленный круг не подставляется', (self.directory / 'report.html').read_text())

    def test_fastest_full_valid_lap_with_reported_times(self):
        for filename, seconds, valid, complete in (
                ('lap_04.csv', 112.327, True, True), ('lap_03.csv', 112.100, True, True),
                ('invalid.csv', 110, False, True), ('partial.csv', 109, True, False)):
            samples = rows()
            for r in samples:
                r['lap_time_s'] *= seconds / 100
            self.add(filename, samples, round(seconds * 1000), valid=valid, complete=complete)
        report = build_analysis(self.directory)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['reference_file'], 'lap_03.csv')
        self.assertEqual(report['reference_lap_time_s'], 112.1)

    def test_throttle_steps_steering_and_split_activity(self):
        samples = rows()
        for r in samples:
            p = r['position_normalized']
            if .16 <= p < .18:
                r['throttle'] = .05 + (p - .16) * 46.5
            r['steer'] = .2 if p < .166 else .64 if p < .18 else .35
            if p == .14:
                r['steer'] = -.8
            r['tc'] = 1 if .12 <= p < .13 or .17 <= p < .185 else 0
            r['abs'] = 1 if .115 <= p < .14 or .155 <= p < .16 else 0
        zone = metrics(LapTrace(samples, 100000))[1]
        progression = zone['throttle_progression']
        self.assertEqual([progression['steps'][str(p)]['lap_time_s'] for p in (5,25,50,75,98)],
                         [16, 16.45, 17, 17.55, 18])
        self.assertEqual(progression['first_to_full_s'], 2)
        self.assertEqual(progression['5_to_50_s'], 1)
        self.assertEqual(zone['tc_before_min_s'], 1)
        self.assertEqual(zone['tc_after_min_s'], 1.5)
        self.assertEqual(zone['abs_before_min_s'], 2.5)
        self.assertEqual(zone['abs_after_min_s'], .5)
        self.assertEqual(zone['tc_active_s'], zone['tc_before_min_s'] + zone['tc_after_min_s'])
        self.assertEqual(zone['steering']['max_abs']['input_pct'], 80)
        self.assertEqual(zone['steering']['at_first_throttle']['input_pct'], 20)
        self.assertEqual(zone['steering']['at_50_throttle']['input_pct'], 64)
        self.assertEqual(zone['steering']['at_full_throttle']['input_pct'], 35)

    def test_two_throttle_openings_are_not_joined(self):
        samples = rows()
        for r in samples:
            if .169 <= r['position_normalized'] < .175:
                r['throttle'] = 0
        zone = metrics(LapTrace(samples, 100000))[1]
        self.assertIsNone(zone['throttle_progression']['steps']['98'])
        self.assertIsNone(zone['throttle_progression']['first_to_full_s'])
        self.assertIsNotNone(zone['full_throttle'])

    def test_track_map_and_labels_from_actual_coordinate_channels(self):
        import math
        samples = rows()
        for r in samples:
            p = r['position_normalized']
            r['x'], r['z'] = 100 * math.cos(2 * math.pi * p), 200 * math.sin(2 * math.pi * p)
            r['steer'] = .2
        self.add('map.csv', samples)
        report = write_analysis(self.directory)
        self.assertEqual(report['track_map']['source_file'], 'map.csv')
        self.assertEqual(len(report['track_map']['markers']), 7)
        text = (self.directory / 'report.html').read_text()
        self.assertIn('<svg', text)
        for label in ('T1–T2', 'T4–T5', 'T6', 'T7', 'T8–T10', 'T11'):
            self.assertIn(label, text)


if __name__ == '__main__':
    unittest.main()
