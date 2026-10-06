import ctypes
import struct
import unittest

from acc_reader import Graphics, Physics, Static, decode_text, format_telemetry, gear_label


class ReaderTests(unittest.TestCase):
    def test_physics_binary_layout(self):
        raw = struct.pack("<ifffiiff", 42, 0.73, 0.18, 25, 5, 6500, -0.1, 184)
        physics = Physics.from_buffer_copy(raw + bytes(ctypes.sizeof(Physics) - len(raw)))
        self.assertEqual(Physics.speed.offset, 28)
        self.assertEqual(physics.packet_id, 42)
        self.assertAlmostEqual(physics.throttle, 0.73, places=6)
        self.assertEqual(format_telemetry("Porsche", "Monza", physics),
                         "Porsche | Monza | 184 km/h | throttle 73% | brake 18% | gear 4 | RPM 6500")

    def test_windows_utf16_offsets(self):
        raw = bytearray(200)
        raw[68:68 + 14] = "Porsche".encode("utf-16-le")
        raw[134:134 + 10] = "monza".encode("utf-16-le")
        data = Static.from_buffer_copy(raw)
        self.assertEqual(Static.car.offset, 68)
        self.assertEqual(Static.track.offset, 134)
        self.assertEqual(decode_text(data.car), "Porsche")
        self.assertEqual(decode_text(data.track), "monza")

    def test_gears(self):
        self.assertEqual([gear_label(n) for n in range(4)], ["R", "N", "1", "2"])

    def test_extended_binary_channels(self):
        raw = bytearray(568)
        struct.pack_into("<f", raw, 204, 0.4)
        struct.pack_into("<f", raw, 252, 0.6)
        struct.pack_into("<4f", raw, 348, 400, 401, 402, 403)
        struct.pack_into("<f", raw, 564, 0.55)
        physics = Physics.from_buffer_copy(raw)
        self.assertAlmostEqual(physics.tc, 0.4, places=6)
        self.assertAlmostEqual(physics.abs, 0.6, places=6)
        self.assertEqual(list(physics.brakeTemp), [400, 401, 402, 403])
        self.assertAlmostEqual(physics.brakeBias, 0.55, places=6)
        raw = bytearray(1412)
        struct.pack_into("<i", raw, 132, 4)
        struct.pack_into("<ii", raw, 140, 1234, 90000)
        struct.pack_into("<f", raw, 248, 0.25)
        struct.pack_into("<i", raw, 1216, 42)
        struct.pack_into("<i", raw, 1320, 2)
        struct.pack_into("<i", raw, 1408, 1)
        graphics = Graphics.from_buffer_copy(raw)
        self.assertEqual(graphics.completedLaps, 4)
        self.assertEqual(graphics.iCurrentTime, 1234)
        self.assertEqual(graphics.iLastTime, 90000)
        self.assertEqual(graphics.normalizedCarPosition, 0.25)
        self.assertEqual(graphics.playerCarID, 42)
        self.assertEqual(graphics.sessionIndex, 2)
        self.assertEqual(graphics.isValidLap, 1)


if __name__ == "__main__":
    unittest.main()
