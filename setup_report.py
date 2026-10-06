"""Inventory every ACC setup field; reference decoding for Porsche 992 GT3 R."""

import argparse
import json
import re
from pathlib import Path

REFERENCE = "https://github.com/RiddleTime/Race-Element/tree/55121bbaf163768813ffc1dc7d622c2b4056be8e/Race_Element.Data.ACC/SetupParser"
WHEELS = ("FL", "FR", "RL", "RR")
# Numeric lookup data describes game controls, not simulation physics.
CASTER = [6.5, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7, 7.8, 8, 8.2, 8.4, 8.6, 8.8, 9,
          9.2, 9.4, 9.6, 9.8, 10, 10.2, 10.4, 10.6, 10.8, 11, 11.2, 11.4,
          11.6, 11.8, 12, 12.2, 12.4]
SPRINGS = ([100500, 110000, 114000, 119000, 127000, 137000, 141500, 146000, 155000, 173500],
           [137000, 149500, 156000, 162000, 174500, 187000, 193000, 199500, 212000, 237000])

# Group, Russian label, decoding rule. All unrecognized fields are retained.
CATALOG = {}
def section(prefix, group, fields):
    for key, label, rule in fields:
        CATALOG[prefix + '.' + key] = (group, label, rule)

section('basicSetup.tyres', 'Шины', [
    ('tyreCompound', 'Состав шин', 'compound'), ('tyrePressure', 'Холодное давление', 'pressure')])
section('basicSetup.alignment', 'Геометрия', [
    ('camber', 'Развал', 'camber'), ('toe', 'Схождение', 'toe'),
    ('staticCamber', 'Внутренний staticCamber', None),
    ('toeOutLinear', 'Внутренний toeOutLinear', None),
    ('casterLF', 'Кастер слева', 'caster'), ('casterRF', 'Кастер справа', 'caster'),
    ('steerRatio', 'Передаточное отношение руля', 'steering')])
section('basicSetup.electronics', 'Электроника', [
    ('tC1', 'TC1', 'direct'), ('tC2', 'TC2', 'direct'), ('abs', 'ABS', 'direct'),
    ('eCUMap', 'Карта ECU', 'one_based'), ('fuelMix', 'Fuel mix: внутренний индекс', None),
    ('telemetryLaps', 'Круги встроенной телеметрии', 'direct')])
STRATEGY = [
    ('fuel', 'Топливо по файлу: исходное значение', None), ('nPitStops', 'Число пит-стопов', 'direct'),
    ('tyreSet', 'Комплект шин: внутренний индекс', None),
    ('frontBrakePadCompound', 'Передние колодки', 'one_based'),
    ('rearBrakePadCompound', 'Задние колодки', 'one_based'),
    ('fuelPerLap', 'Расход по файлу: исходное значение', None), ('fuelToAdd', 'Дозаправка: исходное значение', None)]
section('basicSetup.strategy', 'Стратегия', STRATEGY)
section('basicSetup.strategy.pitStrategy[]', 'Пит-стратегия', STRATEGY)
section('basicSetup.strategy.pitStrategy[].tyres', 'Пит-стратегия', [
    ('tyreCompound', 'Состав шин на пит-стопе', 'compound'),
    ('tyrePressure', 'Давление шин на пит-стопе', 'pressure')])
section('advancedSetup.mechanicalBalance', 'Механика и тормоза', [
    ('aRBFront', 'Передний стабилизатор', 'direct'), ('aRBRear', 'Задний стабилизатор', 'direct'),
    ('wheelRate', 'Жёсткость пружины', 'spring'),
    ('bumpStopRateUp', 'Жёсткость отбойника', 'bumpstop'),
    ('bumpStopRateDn', 'Внутренний bumpStopRateDn', None),
    ('bumpStopWindow', 'Зазор отбойника', 'direct'),
    ('brakeTorque', 'Мощность тормозов', 'brake_power'), ('brakeBias', 'Баланс тормозов', 'bias')])
section('advancedSetup.dampers', 'Амортизаторы', [
    ('bumpSlow', 'Медленное сжатие', 'direct'), ('bumpFast', 'Быстрое сжатие', 'direct'),
    ('reboundSlow', 'Медленный отбой', 'direct'), ('reboundFast', 'Быстрый отбой', 'direct')])
section('advancedSetup.aeroBalance', 'Аэродинамика', [
    ('rideHeight', 'Клиренс', 'height'), ('rodLength', 'Внутренний rodLength', None),
    ('splitter', 'Сплиттер', 'direct'), ('rearWing', 'Заднее крыло', 'direct'),
    ('brakeDuct', 'Тормозной воздуховод', 'direct')])
section('advancedSetup.drivetrain', 'Трансмиссия', [('preload', 'Преднатяг дифференциала', 'preload')])
CATALOG.update({'carName': ('Метаданные', 'Машина', None),
                'trackBopType': ('Метаданные', 'Тип BoP: внутренний код', None)})


def leaves(value, path=''):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from leaves(child, f'{path}.{key}' if path else key)
    elif isinstance(value, list) and value:
        for i, child in enumerate(value):
            yield from leaves(child, f'{path}[{i}]')
    else:
        yield path, value


def decode(rule, raw, index):
    if rule == 'compound' and type(raw) is int and raw in (0, 1):
        return ('dry' if raw == 0 else 'wet'), None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError('not numeric')
    if type(raw) is not int or raw < 0:
        raise ValueError('not a non-negative control index')
    if rule == 'direct': return raw, 'setting'
    if rule == 'one_based': return raw + 1, 'setting'
    if rule == 'pressure' and raw <= 147: return round(20.3 + raw / 10, 1), 'psi'
    if rule == 'toe' and raw <= 80: return round(-0.4 + raw / 100, 2), 'deg'
    if rule == 'camber' and index is not None and index < 4 and raw <= 25:
        return round((-4 if index < 2 else -3.5) + raw / 10, 1), 'deg'
    if rule == 'caster' and raw < len(CASTER): return CASTER[raw], 'deg'
    if rule == 'steering' and raw <= 6: return 11 + raw, 'ratio'
    if rule == 'spring' and index is not None and index < 4:
        return SPRINGS[index // 2][raw], 'N/m'
    if rule == 'bumpstop' and raw <= 22: return 300 + 100 * raw, 'reference scale'
    if rule == 'brake_power' and raw <= 20: return 80 + raw, '%'
    if rule == 'bias' and raw <= 105: return round(43 + raw / 5, 1), '%'
    if rule == 'height' and index in (0, 2) and raw <= (32 if index == 0 else 35):
        return (53 if index == 0 else 55) + raw, 'mm'
    if rule == 'preload' and raw <= 28: return 20 + raw * 10, 'Nm'
    raise ValueError('unverified index or conversion')


def build_report(setup):
    if not isinstance(setup, dict):
        raise ValueError('ACC setup JSON must contain an object')
    supported = setup.get('carName') == 'porsche_992_gt3_r'
    result = {'car': setup.get('carName'), 'reference': REFERENCE,
              'decoding_profile': 'porsche_992_gt3_r' if supported else None,
              'verified_against_current_game_ui': False,
              'notes': 'Saved file declared by user; active setup is not detected. '
                       'Conversions are community reference mappings, not validated in the current game UI. '
                       'Raw/internal fields are not assumed to be adjustable controls.', 'fields': []}
    for path, raw in leaves(setup):
        normalized = re.sub(r'\[\d+\]', '[]', path)
        key = normalized[:-2] if normalized.endswith('[]') else normalized
        group, label, rule = CATALOG.get(key, ('Неизвестные поля', path, None))
        match = re.search(r'\[(\d+)\]$', path)
        index = int(match[1]) if match else None
        entry = {'path': path, 'group': group, 'label': label, 'raw': raw,
                 'interpretation': 'raw_only'}
        if index is not None:
            if key == 'advancedSetup.aeroBalance.brakeDuct' and index < 2:
                entry['position'] = ('front', 'rear')[index]
            elif rule == 'height':
                entry['position'] = {0: 'front', 2: 'rear'}.get(index, 'internal')
            elif key in CATALOG and index < 4 and key not in (
                    'basicSetup.strategy.pitStrategy[]',):
                entry['position'] = WHEELS[index]
        if supported and rule:
            try:
                entry['value'], entry['unit'] = decode(rule, raw, index)
                entry['interpretation'] = 'reference_decoded'
            except (ValueError, IndexError):
                pass
        result['fields'].append(entry)
    result['field_count'] = len(result['fields'])
    result['decoded_count'] = sum(f['interpretation'] == 'reference_decoded' for f in result['fields'])
    return result


def format_report(report):
    lines = ['СЕТАП ACC — ' + str(report['car']),
             'Расшифровка по справочному парсеру; с текущим интерфейсом игры не сверена.',
             'Файл выбран пользователем. Его загрузка в игре автоматически не проверяется.']
    groups = dict.fromkeys(f['group'] for f in report['fields'])
    for group in groups:
        lines.extend(['', group.upper()])
        for field in report['fields']:
            if field['group'] != group:
                continue
            label = field['label'] + (' / ' + field['position'] if 'position' in field else '')
            if field['interpretation'] == 'reference_decoded':
                value = f"{field['value']} {field['unit'] or ''}".strip()
                lines.append(f"{label}: {value} (JSON: {field['raw']})")
            else:
                lines.append(f"{label}: {field['raw']} (исходное значение; не расшифровано)")
            lines.append('  ' + field['path'])
    lines.extend(['', 'Источник справочных преобразований: ' + report['reference']])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('setup', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = build_report(json.loads(args.setup.read_text(encoding='utf-8-sig')))
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(text, encoding='utf-8')
        else:
            print(text)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot read setup: {error}\n')


if __name__ == '__main__':
    main()
