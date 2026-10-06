import unittest

from setup_report import build_report, leaves


class SetupReportTests(unittest.TestCase):
    def report(self, **sections):
        return build_report(dict(carName='porsche_992_gt3_r', **sections))

    def field(self, report, path):
        return next(f for f in report['fields'] if f['path'] == path)

    def test_porsche_index_conversions(self):
        report = self.report(
            basicSetup={'tyres': {'tyrePressure': [57, 57, 44, 45]},
                        'alignment': {'camber': [0, 0, 2, 2], 'casterLF': 7},
                        'electronics': {'eCUMap': 7}},
            advancedSetup={'mechanicalBalance': {'brakeBias': 29, 'wheelRate': [4, 4, 5, 5]},
                           'aeroBalance': {'rideHeight': [0, 11, 11, 18]},
                           'drivetrain': {'preload': 0}})
        self.assertEqual(self.field(report, 'basicSetup.tyres.tyrePressure[0]')['value'], 26)
        self.assertEqual(self.field(report, 'basicSetup.alignment.camber[2]')['value'], -3.3)
        self.assertEqual(self.field(report, 'basicSetup.alignment.casterLF')['value'], 7.8)
        self.assertEqual(self.field(report, 'basicSetup.electronics.eCUMap')['value'], 8)
        self.assertEqual(self.field(report, 'advancedSetup.mechanicalBalance.brakeBias')['value'], 48.8)
        self.assertEqual(self.field(report, 'advancedSetup.mechanicalBalance.wheelRate[2]')['value'], 187000)
        self.assertEqual(self.field(report, 'advancedSetup.aeroBalance.rideHeight[2]')['value'], 66)
        self.assertEqual(self.field(report, 'advancedSetup.aeroBalance.rideHeight[1]')['interpretation'], 'raw_only')
        self.assertEqual(self.field(report, 'advancedSetup.drivetrain.preload')['value'], 20)

    def test_pit_strategy_and_unknown_fields_are_not_lost(self):
        setup = {'carName': 'porsche_992_gt3_r', 'basicSetup': {'strategy': {
            'pitStrategy': [{'fuelToAdd': 20, 'rearBrakePadCompound': 2,
                             'tyres': {'tyrePressure': [57, 57, 44, 45]}}]}},
                 'future': {'control': [7, 8]}, 'empty': []}
        report = build_report(setup)
        self.assertEqual(len(report['fields']), len(list(leaves(setup))))
        self.assertEqual(self.field(report, 'basicSetup.strategy.pitStrategy[0].rearBrakePadCompound')['value'], 3)
        self.assertEqual(self.field(report, 'basicSetup.strategy.pitStrategy[0].tyres.tyrePressure[0]')['value'], 26)
        self.assertEqual(self.field(report, 'future.control[1]')['raw'], 8)

    def test_other_car_and_invalid_indices_are_raw_only(self):
        report = build_report({'carName': 'other_car', 'advancedSetup': {
            'mechanicalBalance': {'brakeBias': 29}}})
        self.assertEqual(report['decoded_count'], 0)
        report = self.report(advancedSetup={'mechanicalBalance': {'wheelRate': [999, -1]}})
        self.assertEqual(report['decoded_count'], 0)
        self.assertEqual(self.field(report, 'advancedSetup.mechanicalBalance.wheelRate[0]')['raw'], 999)


if __name__ == '__main__':
    unittest.main()
