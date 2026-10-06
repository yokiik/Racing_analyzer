"""CSV recording and lap boundaries, independent of Windows shared memory."""

import csv
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


WHEELS = ("fl", "fr", "rl", "rr")
FIELDS = [
    "elapsed_s", "lap_time_s", "lap_number", "position_normalized",
    "speed_kmh", "throttle", "brake", "steer", "gear", "rpm", "fuel",
    "tc", "abs", "yaw_rad", "pitch_rad", "roll_rad", "is_valid_lap",
    "in_pit", "in_pit_lane", "sector", "physics_packet", "graphics_packet",
    "x", "y", "z", "velocity_x", "velocity_y", "velocity_z",
    "acc_g_x", "acc_g_y", "acc_g_z",
    "angular_velocity_local_x", "angular_velocity_local_y", "angular_velocity_local_z",
] + [f"{name}_{wheel}" for name in
     ("wheel_slip", "tyre_pressure", "tyre_core_temp_c", "brake_temp_c")
     for wheel in WHEELS]


def telemetry_sample(physics, graphics):
    """Keep raw TC/ABS values, not setup levels or an invented event count."""
    sample = {
        "lap_time_s": graphics.iCurrentTime / 1000,
        "lap_number": graphics.completedLaps + 1,
        "position_normalized": graphics.normalizedCarPosition,
        "speed_kmh": physics.speed, "throttle": physics.throttle,
        "brake": physics.brake, "steer": physics.steer, "gear": physics.gear - 1,
        "rpm": physics.rpm, "fuel": physics.fuel, "tc": physics.tc,
        "abs": physics.abs, "yaw_rad": physics.heading,
        "pitch_rad": physics.pitch, "roll_rad": physics.roll,
        "is_valid_lap": graphics.isValidLap, "in_pit": graphics.isInPit,
        "in_pit_lane": graphics.isInPitLane, "sector": graphics.currentSectorIndex,
        "physics_packet": physics.packet_id, "graphics_packet": graphics.packet_id,
        "x": "", "y": "", "z": "",
    }
    for i in range(max(0, min(graphics.activeCars, 60))):
        if graphics.carID[i] == graphics.playerCarID:
            sample.update(zip(("x", "y", "z"), graphics.carCoordinates[i]))
            break
    for prefix, values in [("velocity", physics.velocity), ("acc_g", physics.accG),
                           ("angular_velocity_local", physics.localAngularVel)]:
        sample.update({f"{prefix}_{axis}": values[i] for i, axis in enumerate("xyz")})
    for prefix, values in [("wheel_slip", physics.wheelSlip),
                           ("tyre_pressure", physics.wheelPressure),
                           ("tyre_core_temp_c", physics.TyreCoreTemp),
                           ("brake_temp_c", physics.brakeTemp)]:
        sample.update({f"{prefix}_{wheel}": values[i] for i, wheel in enumerate(WHEELS)})
    return sample


class Recorder:
    def __init__(self, output, hz=20, demo=False):
        self.output = Path(output).resolve()
        self.hz = hz
        self.demo = demo
        self.directory = None
        self.identity = None
        self.file = None
        self.previous = None
        self.best_ms = None
        self.message = "Waiting for session"

    def _metadata(self):
        temporary = self.directory / "session.json.tmp"
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(self.directory / "session.json")

    def _session(self, identity):
        self.identity = identity
        self.output.mkdir(parents=True, exist_ok=True)
        track = re.sub(r"[^a-zA-Z0-9_-]", "_", identity[1])[:60] or "unknown"
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + track
        self.directory = self.output / name
        self.directory.mkdir()
        self.started = time.monotonic()
        self.best_ms = None
        self.metadata = {
            "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
            "car": identity[0], "track": identity[1], "session_type": identity[2],
            "session_index": identity[3], "polling_hz": self.hz, "demo": self.demo,
            "laps": [], "best_lap": None,
            "notes": "Independent physics/graphics snapshots; start/finish sampled at polling rate. "
                     "TC/ABS are raw physics channels. Gear -1=R, 0=N. Wheel order FL/FR/RL/RR. "
                     "Validity is aggregated from observed samples; no inferred final-line validity.",
        }
        self._metadata()
        print(f"\nRecording to: {self.directory}")

    def _start_lap(self, sample, boundary):
        self.lap_number = sample["lap_number"]
        self.path = self.directory / f"lap_{len(self.metadata['laps']) + 1:02d}.recording.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.rows = 0
        self.full_start = boundary and sample["lap_time_s"] <= 0.25
        self.valid = True
        self.pit = False
        self.gap = False
        self.last_flush = time.monotonic()

    def _finish(self, crossed=False, lap_ms=None, reason="interrupted"):
        if self.file is None:
            return
        self.file.close()
        self.file = None
        complete = crossed and self.full_start and not self.gap and self.rows > 1
        suffix = "" if complete else ".partial"
        path = self.path.with_name(self.path.name.replace(".recording.csv", suffix + ".csv"))
        self.path.replace(path)
        info = {"file": path.name, "game_lap_number": self.lap_number,
                "complete": complete, "valid": self.valid, "contains_pit": self.pit,
                "samples": self.rows, "lap_time_ms": lap_ms if crossed else None,
                "reason": "finished" if complete else reason if not crossed else
                          "missing_start_or_samples"}
        self.metadata["laps"].append(info)
        if complete and self.valid and not self.pit and lap_ms and lap_ms > 0:
            if self.best_ms is None or lap_ms < self.best_ms:
                self.best_ms = lap_ms
                shutil.copyfile(path, self.directory / "best_lap.csv")
                self.metadata["best_lap"] = {"file": path.name, "lap_time_ms": lap_ms}
        self._metadata()
        print(f"\nSaved {path.name}: {self.rows} samples" +
              (f", {lap_ms / 1000:.3f}s" if crossed and lap_ms else ""))

    def reset(self, reason="disconnected"):
        self._finish(reason=reason)
        self.identity = None
        self.previous = None
        self.message = "Waiting for session"

    def accept(self, car, track, graphics, sample):
        if graphics.status == 3:  # pause: leave lap open, no samples
            return
        if graphics.status != 2:
            self.reset("replay" if graphics.status == 1 else "session_ended")
            return
        if not car or not track:
            return
        identity = (car, track, graphics.session, graphics.sessionIndex)
        if identity != self.identity:
            self.reset("session_changed")
            self._session(identity)
        previous = self.previous
        if previous is not None:
            delta_laps = sample["lap_number"] - previous["lap_number"]
            # ACC can reset the timer one graphics update before completedLaps.
            # Ignore that transitional snapshot and wait for the counter, keeping
            # the last pre-finish sample for gap detection and the old lap intact.
            if (delta_laps == 0 and previous["lap_time_s"] > 1
                    and previous["position_normalized"] > 0.98
                    and sample["lap_time_s"] <= 0.25
                    and (sample["position_normalized"] > 0.98
                         or sample["position_normalized"] < 0.02)):
                return
            # Return to garage, restart, teleport or skipped laps: never join them.
            if delta_laps < 0 or delta_laps > 1 or (
                delta_laps == 0 and sample["lap_time_s"] < previous["lap_time_s"] - 0.5
            ):
                self.reset("timing_reset")
                self._session(identity)
                previous = None
            elif delta_laps == 1:
                if graphics.iLastTime / 1000 - previous["lap_time_s"] > 1:
                    self.gap = True
                self._finish(crossed=True, lap_ms=graphics.iLastTime)
                self._start_lap(sample, boundary=True)
            elif (sample["physics_packet"], sample["graphics_packet"]) == (
                previous["physics_packet"], previous["graphics_packet"]
            ):
                return
            elif sample["lap_time_s"] - previous["lap_time_s"] > 1:
                self.gap = True
        if self.file is None:
            # A program started mid-lap must label that lap partial.
            self._start_lap(sample, boundary=False)
        self.valid = self.valid and bool(sample["is_valid_lap"])
        self.pit = self.pit or bool(sample["in_pit"] or sample["in_pit_lane"])
        row = dict(sample, elapsed_s=round(time.monotonic() - self.started, 6))
        self.writer.writerow(row)
        self.rows += 1
        self.previous = sample.copy()
        if time.monotonic() - self.last_flush >= 1:
            self.file.flush()
            self.last_flush = time.monotonic()
        self.message = f"REC lap {self.lap_number} ({self.rows} samples)"

    def close(self):
        self.reset("program_stopped")
