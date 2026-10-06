"""First ACC Coach step: read existing Windows shared memory, read-only."""

import argparse
import ctypes as c
import math
import sys
import time
from contextlib import ExitStack


# Only the prefixes needed for this step. Windows wchar_t is always 2 bytes,
# unlike ctypes.c_wchar on macOS. ACC uses 4-byte structure alignment.
class Physics(c.LittleEndianStructure):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [
        ("packet_id", c.c_int32), ("throttle", c.c_float),
        ("brake", c.c_float), ("fuel", c.c_float),
        ("gear", c.c_int32), ("rpm", c.c_int32),
        ("steer", c.c_float), ("speed", c.c_float),
    ]


class Graphics(c.LittleEndianStructure):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [
        ("packet_id", c.c_int32), ("status", c.c_int32),
        ("session", c.c_int32),
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


def run_live(hz):
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
                    except RuntimeError:
                        time.sleep(1 / hz)
                        continue
                    if graphics.packet_id != last_packet:
                        last_packet = graphics.packet_id
                        last_change = time.monotonic()
                    if time.monotonic() - last_change > 5:
                        print("\nACC stopped updating. Reconnecting...")
                        break
                    if graphics.status == 2:
                        line = format_telemetry(decode_text(static.car),
                                                decode_text(static.track), physics)
                    else:
                        line = {0: "ACC: waiting for a session", 1: "ACC: replay",
                                3: "ACC: paused"}.get(graphics.status, "ACC: unknown status")
                    print("\r" + line.ljust(previous_width), end="", flush=True)
                    previous_width = len(line)
                    time.sleep(1 / hz)
        except OSError as error:
            if getattr(error, "winerror", None) != 2:
                raise
            print("\rWaiting for ACC. Start the game and enter a session.", flush=True)
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Show synthetic data on any OS")
    parser.add_argument("--hz", type=int, choices=range(20, 51), default=20,
                        metavar="20..50", help="Polling frequency (default: 20)")
    args = parser.parse_args()
    if not args.demo and sys.platform != "win32":
        parser.exit(1, "ACC shared memory requires Windows. Use --demo on this computer.\n")
    try:
        if args.demo:
            print("DEMO: synthetic data, no connection to ACC")
            start = time.monotonic()
            while True:
                t = time.monotonic() - start
                physics = Physics(speed=184 + 20 * math.sin(t),
                                  throttle=(1 + math.sin(t)) / 2,
                                  brake=0, gear=5, rpm=6500)
                print("\r" + format_telemetry("porsche_991ii_gt3_r", "monza", physics),
                      end="", flush=True)
                time.sleep(1 / args.hz)
        else:
            run_live(args.hz)
    except KeyboardInterrupt:
        print("\nStopped.")
    except OSError as error:
        parser.exit(1, f"Cannot read ACC shared memory: {error}\n")


if __name__ == "__main__":
    main()
