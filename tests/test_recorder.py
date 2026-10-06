import csv
import json
import tempfile
import unittest
from pathlib import Path

from acc_reader import Graphics, Physics
from recorder import Recorder, telemetry_sample


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recorder = Recorder(self.temp.name)
        self.addCleanup(self.recorder.close)
        self.packet = 0

    def feed(self, lap, ms, last=90000, valid=1, status=2, pit=0, car="porsche"):
        self.packet += 1
        graphics = Graphics(status=status, session=0, completedLaps=lap - 1,
                            iCurrentTime=ms, iLastTime=last, isValidLap=valid,
                            packet_id=self.packet, isInPitLane=pit)
        sample = telemetry_sample(Physics(packet_id=self.packet, speed=184, gear=5), graphics)
        self.recorder.accept(car, "monza", graphics, sample)
        return graphics, sample

    def full_lap(self, lap=2, valid=1, pit=0, last=90000):
        # Fast synthetic replay of 20Hz lap-time samples; no wall-clock waiting.
        for ms in range(50, 90000, 50):
            self.feed(lap, ms, valid=valid, pit=pit)
        self.feed(lap + 1, 50, last=last)

    def metadata(self):
        return json.loads((self.recorder.directory / "session.json").read_text())

    def test_partial_full_best_and_stop(self):
        self.feed(1, 89000)
        self.full_lap()
        directory = self.recorder.directory
        self.recorder.close()
        data = self.metadata()
        self.assertEqual([lap["complete"] for lap in data["laps"]], [False, True, False])
        self.assertEqual(data["laps"][1]["lap_time_ms"], 90000)
        self.assertEqual(data["best_lap"]["file"], "lap_02.csv")
        self.assertEqual((directory / "best_lap.csv").read_bytes(),
                         (directory / "lap_02.csv").read_bytes())
        with (directory / "lap_02.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1799)
        self.assertEqual(rows[0]["lap_number"], "2")
        self.assertEqual(rows[0]["gear"], "4")

    def test_invalid_and_pit_not_best(self):
        self.feed(1, 89000)
        self.full_lap(valid=0)
        self.assertIsNone(self.metadata()["best_lap"])
        self.full_lap(lap=3, pit=1)
        self.assertIsNone(self.metadata()["best_lap"])

    def test_pause_duplicates_and_replay(self):
        g, sample = self.feed(1, 30000)
        self.recorder.accept("porsche", "monza", g, sample)
        self.assertEqual(self.recorder.rows, 1)
        self.feed(1, 30000, status=3)
        self.feed(1, 30050)
        self.assertEqual(self.recorder.rows, 2)
        self.feed(1, 30100, status=1)
        self.assertIsNone(self.recorder.file)
        self.assertEqual(self.metadata()["laps"][0]["reason"], "replay")

    def test_timing_reset_and_track_car_change(self):
        self.feed(3, 30000)
        first = self.recorder.directory
        self.feed(3, 0)
        self.assertNotEqual(first, self.recorder.directory)
        data = json.loads((first / "session.json").read_text())
        self.assertFalse(data["laps"][0]["complete"])
        second = self.recorder.directory
        self.feed(3, 50, car="ferrari")
        self.assertNotEqual(second, self.recorder.directory)

    def test_skipped_samples_prevent_best(self):
        self.feed(1, 89000)
        self.feed(2, 50)
        self.feed(2, 89000)
        self.feed(3, 50)
        self.assertFalse(self.metadata()["laps"][1]["complete"])
        self.assertIsNone(self.metadata()["best_lap"])

    def test_player_coordinates_use_id_lookup(self):
        g = Graphics(activeCars=2, playerCarID=42)
        g.carID[0], g.carID[1] = 10, 42
        g.carCoordinates[1][:] = [1, 2, 3]
        sample = telemetry_sample(Physics(), g)
        self.assertEqual([sample[k] for k in ("x", "y", "z")], [1, 2, 3])

    def test_best_retains_faster_lap(self):
        self.feed(1, 89000)
        self.full_lap()
        self.full_lap(lap=3, last=90500)
        self.assertEqual(self.metadata()["best_lap"]["file"], "lap_02.csv")
        for ms in range(100, 88000, 50):
            self.feed(4, ms)
        self.feed(5, 50, last=88000)
        self.assertEqual(self.metadata()["best_lap"]["file"], "lap_04.csv")

    def test_disconnect_preserves_partial_and_starts_new_session(self):
        self.feed(2, 40000)
        directory = self.recorder.directory
        self.recorder.reset()
        self.assertTrue((directory / "lap_01.partial.csv").exists())
        self.feed(2, 41000)
        self.assertNotEqual(directory, self.recorder.directory)

    def test_acc_timer_resets_before_lap_counter(self):
        self.feed(5, 113520)
        self.feed(6, 52, last=113522)
        directory = self.recorder.directory
        for ms in range(102, 113383, 50):
            self.feed(6, ms)
        # Actual ACC transition from the user's Monza recording: timer is
        # already 7ms, but lap count and normalized position still belong to lap 6.
        self.recorder.previous["position_normalized"] = 0.9996869
        g = Graphics(status=2, session=0, completedLaps=5, iCurrentTime=7,
                     iLastTime=113425, normalizedCarPosition=1, isValidLap=1,
                     packet_id=100000)
        self.recorder.accept("porsche", "monza", g,
                             telemetry_sample(Physics(packet_id=100000), g))
        self.assertEqual(self.recorder.directory, directory)
        self.assertEqual(self.recorder.previous["lap_number"], 6)
        self.feed(7, 65, last=113425)
        data = self.metadata()
        self.assertEqual(len(data["laps"]), 2)
        self.assertTrue(data["laps"][1]["complete"])
        self.assertEqual(data["laps"][1]["lap_time_ms"], 113425)
        self.assertEqual(data["best_lap"]["file"], "lap_02.csv")


if __name__ == "__main__":
    unittest.main()
