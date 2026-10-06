"""ACC Coach: live telemetry and automatic per-lap CSV recording."""

import argparse
import ctypes as c
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from recorder import Recorder, telemetry_sample


# Prefixes through brake temperatures and lap validity. Windows wchar_t is always 2 bytes,
# unlike ctypes.c_wchar on macOS. ACC uses 4-byte structure alignment.
class Physics(c.LittleEndianStructure):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [
        ("packet_id", c.c_int32),
        ("throttle", c.c_float),
        ("brake", c.c_float),
        ("fuel", c.c_float),
        ("gear", c.c_int32),
        ("rpm", c.c_int32),
        ("steer", c.c_float),
        ("speed", c.c_float),
        ("velocity", (c.c_float * 3)),
        ("accG", (c.c_float * 3)),
        ("wheelSlip", (c.c_float * 4)),
        ("wheelLoad", (c.c_float * 4)),
        ("wheelPressure", (c.c_float * 4)),
        ("wheelAngularSpeed", (c.c_float * 4)),
        ("tyreWear", (c.c_float * 4)),
        ("tyreDirtyLevel", (c.c_float * 4)),
        ("TyreCoreTemp", (c.c_float * 4)),
        ("camberRAD", (c.c_float * 4)),
        ("suspensionTravel", (c.c_float * 4)),
        ("drs", c.c_float),
        ("tc", c.c_float),
        ("heading", c.c_float),
        ("pitch", c.c_float),
        ("roll", c.c_float),
        ("cgHeight", c.c_float),
        ("carDamage", (c.c_float * 5)),
        ("numberOfTyresOut", c.c_int32),
        ("pitLimiterOn", c.c_int32),
        ("abs", c.c_float),
        ("kersCharge", c.c_float),
        ("kersInput", c.c_float),
        ("autoshifterOn", c.c_int32),
        ("rideHeight", (c.c_float * 2)),
        ("turboBoost", c.c_float),
        ("ballast", c.c_float),
        ("airDensity", c.c_float),
        ("airTemp", c.c_float),
        ("roadTemp", c.c_float),
        ("localAngularVel", (c.c_float * 3)),
        ("finalFF", c.c_float),
        ("perfomanceMeter", c.c_float),
        ("engineBrake", c.c_int32),
        ("ersRecoveryLevel", c.c_int32),
        ("ersPowerLevel", c.c_int32),
        ("ersHeatCharging", c.c_int32),
        ("ersIsCharging", c.c_int32),
        ("kersCurrentKJ", c.c_float),
        ("drsAvailable", c.c_int32),
        ("drsEnabled", c.c_int32),
        ("brakeTemp", (c.c_float * 4)),
    ]


class Graphics(c.LittleEndianStructure):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [
        ("packet_id", c.c_int32),
        ("status", c.c_int32),
        ("session", c.c_int32),
        ("currentTime", (c.c_uint16 * 15)),
        ("lastTime", (c.c_uint16 * 15)),
        ("bestTime", (c.c_uint16 * 15)),
        ("split", (c.c_uint16 * 15)),
        ("completedLaps", c.c_int32),
        ("position", c.c_int32),
        ("iCurrentTime", c.c_int32),
        ("iLastTime", c.c_int32),
        ("iBestTime", c.c_int32),
        ("sessionTimeLeft", c.c_float),
        ("distanceTraveled", c.c_float),
        ("isInPit", c.c_int32),
        ("currentSectorIndex", c.c_int32),
        ("lastSectorTime", c.c_int32),
        ("numberOfLaps", c.c_int32),
        ("tyreCompound", (c.c_uint16 * 33)),
        ("replayTimeMultiplier", c.c_float),
        ("normalizedCarPosition", c.c_float),
        ("activeCars", c.c_int32),
        ("carCoordinates", ((c.c_float * 3) * 60)),
        ("carID", (c.c_int32 * 60)),
        ("playerCarID", c.c_int32),
        ("penaltyTime", c.c_float),
        ("flag", c.c_int32),
        ("penalty", c.c_int32),
        ("idealLineOn", c.c_int32),
        ("isInPitLane", c.c_int32),
        ("surfaceGrip", c.c_float),
        ("mandatoryPitDone", c.c_int32),
        ("windSpeed", c.c_float),
        ("windDirection", c.c_float),
        ("isSetupMenuVisible", c.c_int32),
        ("mainDisplayIndex", c.c_int32),
        ("secondaryDisplyIndex", c.c_int32),
        ("TC", c.c_int32),
        ("TCCUT", c.c_int32),
        ("EngineMap", c.c_int32),
        ("ABS", c.c_int32),
        ("fuelXLap", c.c_float),
        ("rainLights", c.c_int32),
        ("flashingLights", c.c_int32),
        ("lightsStage", c.c_int32),
        ("exhaustTemperature", c.c_float),
        ("wiperLV", c.c_int32),
        ("driverStintTotalTimeLeft", c.c_int32),
        ("driverStintTimeLeft", c.c_int32),
        ("rainTyres", c.c_int32),
        ("sessionIndex", c.c_int32),
        ("usedFuel", c.c_float),
        ("deltaLapTime", (c.c_uint16 * 15)),
        ("iDeltaLapTime", c.c_int32),
        ("estimatedLapTime", (c.c_uint16 * 15)),
        ("iEstimatedLapTime", c.c_int32),
        ("isDeltaPositive", c.c_int32),
        ("iSplit", c.c_int32),
        ("isValidLap", c.c_int32),
    ]


class Static(c.LittleEndianStructure):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [
        ("sm_version", c.c_uint16 * 15),
        ("game_version", c.c_uint16 * 15),
        ("number_of_sessions", c.c_int32), ("num_cars", c.c_int32),
        ("car", c.c_uint16 * 33), ("track", c.c_uint16 * 33),
    ]


def decode_text(value):
    return bytes(value).decode("utf-16-le", errors="replace").split("\0", 1)[0]


def gear_label(raw):
    return "R" if raw == 0 else "N" if raw == 1 else str(raw - 1)


def format_telemetry(car, track, physics):
    return (
        f"{car} | {track} | {physics.speed:.0f} km/h | "
        f"throttle {physics.throttle:.0%} | brake {physics.brake:.0%} | "
        f"gear {gear_label(physics.gear)} | RPM {physics.rpm}"
    )


class SharedPage:
    """OpenFileMapping never creates a fake page when ACC is absent."""

    def __init__(self, name, layout):
        self.layout = layout
        self.api = c.WinDLL("kernel32", use_last_error=True)
        self.api.OpenFileMappingW.argtypes = [c.c_uint32, c.c_int, c.c_wchar_p]
        self.api.OpenFileMappingW.restype = c.c_void_p
        self.api.MapViewOfFile.argtypes = [c.c_void_p, c.c_uint32, c.c_uint32,
                                           c.c_uint32, c.c_size_t]
        self.api.MapViewOfFile.restype = c.c_void_p
        self.api.UnmapViewOfFile.argtypes = [c.c_void_p]
        self.api.UnmapViewOfFile.restype = c.c_int
        self.api.CloseHandle.argtypes = [c.c_void_p]
        self.api.CloseHandle.restype = c.c_int
        self.handle = self.api.OpenFileMappingW(4, False, name)  # FILE_MAP_READ
        if not self.handle:
            raise c.WinError(c.get_last_error())
        self.view = self.api.MapViewOfFile(self.handle, 4, 0, 0, c.sizeof(layout))
        if not self.view:
            error = c.get_last_error()
            self.api.CloseHandle(self.handle)
            raise c.WinError(error)

    def read(self):
        # Require two identical copies: avoid displaying a partially updated page.
        for _ in range(5):
            first = c.string_at(self.view, c.sizeof(self.layout))
            second = c.string_at(self.view, c.sizeof(self.layout))
            if first == second:
                return self.layout.from_buffer_copy(second)
        raise RuntimeError("Shared memory is changing; retry next sample")

    def close(self):
        self.api.UnmapViewOfFile(self.view)
        self.api.CloseHandle(self.handle)


def run_live(hz, recorder):
    previous_width = 0
    while True:
        try:
            with ExitStack() as stack:
                pages = []
                for name, layout in [("physics", Physics), ("graphics", Graphics),
                                     ("static", Static)]:
                    page = SharedPage("Local\\acpmf_" + name, layout)
                    stack.callback(page.close)
                    pages.append(page)
                physics_page, graphics_page, static_page = pages
                last_packet = None
                last_change = time.monotonic()
                while True:
                    try:
                        graphics = graphics_page.read()
                        physics = physics_page.read()
                        static = static_page.read()
                        if graphics_page.read().packet_id != graphics.packet_id:
                            time.sleep(1 / hz)
                            continue
                    except RuntimeError:
                        time.sleep(1 / hz)
                        continue
                    if graphics.packet_id != last_packet:
                        last_packet = graphics.packet_id
                        last_change = time.monotonic()
                    if graphics.status != 3 and time.monotonic() - last_change > 5:
                        print("\nACC stopped updating. Reconnecting...")
                        recorder.reset("connection_stale")
                        break
                    recorder.accept(decode_text(static.car), decode_text(static.track),
                                    graphics, telemetry_sample(physics, graphics))
                    if graphics.status == 2:
                        line = format_telemetry(decode_text(static.car),
                                                decode_text(static.track), physics)
                        line += " | " + recorder.message
                    else:
                        line = {0: "ACC: waiting for a session", 1: "ACC: replay",
                                3: "ACC: paused"}.get(graphics.status, "ACC: unknown status")
                    print("\r" + line.ljust(previous_width), end="", flush=True)
                    previous_width = len(line)
                    time.sleep(1 / hz)
        except OSError as error:
            recorder.reset("connection_error")
            if getattr(error, "winerror", None) != 2:
                raise
            print("\rWaiting for ACC. Start the game and enter a session.", flush=True)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Show synthetic data on any OS")
    parser.add_argument("--hz", type=int, choices=range(20, 51), default=20,
                        metavar="20..50", help="Polling frequency (default: 20)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "sessions",
                        help="Recording folder (default: sessions beside the script)")
    args = parser.parse_args()
    if not args.demo and sys.platform != "win32":
        parser.exit(1, "ACC shared memory requires Windows. Use --demo on this computer.\n")
    recorder = Recorder(args.output, args.hz, demo=args.demo)
    try:
        if args.demo:
            print("DEMO: synthetic data, no connection to ACC")
            start = time.monotonic()
            while True:
                t = time.monotonic() - start
                lap_t = t % 8
                graphics = Graphics(status=2, session=3, completedLaps=int(t // 8),
                                    iCurrentTime=int(lap_t * 1000), iLastTime=8000,
                                    normalizedCarPosition=lap_t / 8, isValidLap=1,
                                    packet_id=int(t * args.hz))
                physics = Physics(packet_id=int(t * args.hz), speed=184 + 20 * math.sin(t),
                                  throttle=(1 + math.sin(t)) / 2,
                                  brake=0, gear=5, rpm=6500)
                recorder.accept("porsche_991ii_gt3_r", "demo_monza", graphics,
                                telemetry_sample(physics, graphics))
                print("\r" + format_telemetry("porsche_991ii_gt3_r", "demo_monza", physics)
                      + " | " + recorder.message,
                      end="", flush=True)
                time.sleep(1 / args.hz)
        else:
            run_live(args.hz, recorder)
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as error:
        parser.exit(1, f"Cannot read or save ACC telemetry: {error}\n")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
