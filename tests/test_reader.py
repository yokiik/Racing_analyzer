import ctypes
import struct
import unittest

from acc_reader import Physics, Static, decode_text, format_telemetry, gear_label


class ReaderTests(unittest.TestCase):
    def test_physics_binary_layout(self):
        raw = struct.pack("<ifffiiff", 42, 0.73, 0.18, 25, 5, 6500, -0.1, 184)
        physics = Physics.from_buffer_copy(raw)
        self.assertEqual(ctypes.sizeof(Physics), 32)
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


if __name__ == "__main__":
    unittest.main()
