"""Distance-aligned Monza corner analysis for recorded ACC sessions. Stdlib only."""

import argparse
import bisect
import csv
import html
import json
import math
from pathlib import Path

# Working zones calibrated against the user's 2026-10-06 ACC Monza recording.
# Fixed comparison boundaries, not surveyed kerb/apex coordinates.
MONZA = [
    ('Start straight', 0, .10, False), ('Rettifilo', .10, .20, True),
    ('Curva Grande', .20, .32, True), ('Roggia', .32, .41, True),
    ('Lesmo 1', .41, .47, True), ('Lesmo 2', .47, .53, True),
    ('Serraglio straight', .53, .63, False), ('Ascari', .63, .75, True),
    ('Back straight', .75, .85, False), ('Parabolica', .85, .97, True),
    ('Finish straight', .97, 1, False),
]
THRESHOLDS = {'brake': .05, 'first_throttle': .05, 'full_throttle': .98,
              'pedal_hold_s': .10, 'full_throttle_hold_s': .20,
              'tc_abs_active': 0, 'max_gap_s': 1, 'throttle_steps': [.05, .25, .50, .75, .98]}
CORNER_NUMBERS = {'Rettifilo': 'T1–T2', 'Curva Grande': 'T3', 'Roggia': 'T4–T5',
                  'Lesmo 1': 'T6', 'Lesmo 2': 'T7', 'Ascari': 'T8–T10', 'Parabolica': 'T11'}
OPTIONAL = ('steer', 'x', 'z')
NUMERIC = ('lap_time_s', 'position_normalized', 'speed_kmh', 'throttle', 'brake', 'tc', 'abs')


class LapTrace:
    def __init__(self, rows, lap_ms):
        self.rows = []
        for raw in rows:
            row = {k: float(raw[k]) for k in NUMERIC}
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError('non-finite telemetry')
            for key in OPTIONAL:
                try:
                    value = float(raw.get(key, ''))
                    row[key] = value if math.isfinite(value) else None
                except (ValueError, TypeError):
                    row[key] = None
            t, p = row['lap_time_s'], row['position_normalized']
            if not 0 <= p <= 1 or t < 0 or row['speed_kmh'] < 0:
                raise ValueError('invalid time, position or speed')
            if not all(0 <= row[k] <= 1 for k in ('brake', 'throttle')):
                raise ValueError('pedal outside 0..1')
            if self.rows:
                prev = self.rows[-1]
                if t < prev['lap_time_s'] or p < prev['position_normalized'] - .0001:
                    raise ValueError('lap time or track position moves backwards')
                if t - prev['lap_time_s'] > THRESHOLDS['max_gap_s']:
                    raise ValueError('more than one second of missing telemetry')
                # Duplicate graph timings/positions do not add fake durations.
                if t == prev['lap_time_s'] or p <= prev['position_normalized']:
                    continue
            self.rows.append(row)
        if len(self.rows) < 3:
            raise ValueError('not enough distinct samples')
        first, last = self.rows[0], self.rows[-1]
        self.lap_s = lap_ms / 1000
        if (first['lap_time_s'] > .25 or first['position_normalized'] > .005
                or last['position_normalized'] < .995
                or not 0 <= self.lap_s - last['lap_time_s'] <= 1):
            raise ValueError('start or finish coverage is incomplete')
        # Timing at the finish is known from ACC. Boundary speed is a nearest
        # sample estimate at the very short missing start/finish tails.
        if first['position_normalized'] > 0:
            self.rows.insert(0, dict(first, position_normalized=0, lap_time_s=0))
        elif first['lap_time_s'] != 0:
            raise ValueError('position zero does not correspond to time zero')
        if last['position_normalized'] < 1:
            self.rows.append(dict(last, position_normalized=1, lap_time_s=self.lap_s))
        elif abs(last['lap_time_s'] - self.lap_s) > .25:
            raise ValueError('finish timing is inconsistent')
        else:
            self.rows[-1] = dict(last, lap_time_s=self.lap_s)
        self.positions = [r['position_normalized'] for r in self.rows]

    def at(self, position):
        i = bisect.bisect_left(self.positions, position)
        if i < len(self.rows) and self.positions[i] == position:
            return dict(self.rows[i])
        left, right = self.rows[i - 1], self.rows[i]
        ratio = (position - self.positions[i - 1]) / (self.positions[i] - self.positions[i - 1])
        result = {k: left[k] + ratio * (right[k] - left[k]) for k in NUMERIC}
        for key in OPTIONAL:
            result[key] = (left[key] + ratio * (right[key] - left[key])
                           if left[key] is not None and right[key] is not None else None)
        # Signals use zero-order hold; interpolating them invents interventions.
        for k in ('tc', 'abs', 'brake', 'throttle'):
            result[k] = left[k]
        result['position_normalized'] = position
        return result

    def zone(self, start, end):
        return [self.at(start)] + [r for r in self.rows if start < r['position_normalized'] < end] + [self.at(end)]


def event(rows, key, threshold, hold, after, ignore_active_at_start=False):
    for i, row in enumerate(rows):
        if row['lap_time_s'] < after or row[key] < threshold:
            continue
        if i == 0 and ignore_active_at_start:
            continue
        if i and rows[i - 1][key] >= threshold:
            continue
        j = i
        while j + 1 < len(rows) and rows[j][key] >= threshold and rows[j]['lap_time_s'] - row['lap_time_s'] < hold:
            j += 1
        if rows[j][key] >= threshold and rows[j]['lap_time_s'] - row['lap_time_s'] >= hold - 1e-9:
            return {'lap_time_s': round(row['lap_time_s'], 4),
                    'position_normalized': round(row['position_normalized'], 6)}
    return None


def active_duration(rows, key):
    return round(sum(b['lap_time_s'] - a['lap_time_s'] for a, b in zip(rows, rows[1:])
                     if a[key] > THRESHOLDS['tc_abs_active']), 4)


def phase_duration(rows, key, start, end):
    return round(sum(max(0, min(b['lap_time_s'], end) - max(a['lap_time_s'], start))
                     for a, b in zip(rows, rows[1:]) if a[key] > 0), 4)


def steering_value(raw):
    return {'raw': raw, 'input_pct': round(abs(raw) * 100, 2)
            if raw is not None and abs(raw) <= 1 else None}


def opening_progression(rows, first):
    result = {'basis': 'first_sustained_opening_in_zone', 'steps': {},
              'first_to_full_s': None, '5_to_50_s': None, '50_to_98_s': None}
    for percent in (5, 25, 50, 75, 98):
        result['steps'][str(percent)] = None
    if first is None:
        return result
    start = next(i for i, r in enumerate(rows) if r['lap_time_s'] >= first['lap_time_s'] - .0001)
    end = next((i for i in range(start + 1, len(rows)) if rows[i]['throttle'] < .05), len(rows))
    opening = rows[start:end]
    result['steps']['5'] = dict(first)
    for percent in (25, 50, 75, 98):
        result['steps'][str(percent)] = event(opening, 'throttle', percent / 100,
                                             .20 if percent == 98 else .10,
                                             opening[0]['lap_time_s'])
    for a, b, key in ((5, 98, 'first_to_full_s'), (5, 50, '5_to_50_s'), (50, 98, '50_to_98_s')):
        left, right = result['steps'][str(a)], result['steps'][str(b)]
        if left and right:
            result[key] = round(right['lap_time_s'] - left['lap_time_s'], 4)
    return result


def track_map(traces, reference, laps):
    ordered = ([reference] if reference else []) + [l for l in laps if l is not reference]
    for lap in ordered:
        trace = traces[lap['file']]
        points = [{'position': r['position_normalized'], 'x': r['x'], 'z': r['z']}
                  for r in trace.rows if r['x'] is not None and r['z'] is not None]
        if len(points) < 3 or max(p['x'] for p in points) - min(p['x'] for p in points) < 1:
            continue
        stride = max(1, len(points) // 350)
        markers = []
        for zone in lap['zones']:
            if not zone['is_corner']:
                continue
            position = .26 if zone['name'] == 'Curva Grande' else zone['min_speed_position']
            nearest = min(points, key=lambda p: abs(p['position'] - position))
            markers.append(dict(nearest, label=zone['label']))
        return {'source_file': lap['file'], 'coordinates': 'recorded_player_x_z',
                'points': points[::stride] + [points[-1]], 'markers': markers}
    return None


def metrics(trace):
    result = []
    for name, start, end, corner in MONZA:
        rows = trace.zone(start, end)
        slowest = min(rows, key=lambda r: r['speed_kmh'])
        brake = (None if rows[0]['brake'] >= .05 else
                 event(rows, 'brake', .05, .10, rows[0]['lap_time_s'], True))
        # First reopening after a sampled throttle lift, not the throttle carried
        # into the corner from the preceding straight.
        lift = next((r['lap_time_s'] for r in rows if r['throttle'] < .05), None)
        first_gas = event(rows, 'throttle', .05, .10, lift) if lift is not None else None
        full_gas = event(rows, 'throttle', .98, .20,
                         lift if lift is not None else rows[0]['lap_time_s'], True)
        if full_gas is None and rows[0]['throttle'] >= .98 and lift is None:
            full_status = 'already_full_no_lift'
        else:
            full_status = 'detected' if full_gas else 'not_detected'
        progression = opening_progression(rows, first_gas)
        def steer_at(moment):
            if moment is None:
                return steering_value(None)
            nearest = min(rows, key=lambda r: abs(r['lap_time_s'] - moment['lap_time_s']))
            return steering_value(nearest['steer'])
        steering_samples = [abs(r['steer']) for r in rows if r['steer'] is not None]
        steering = {'max_abs': steering_value(max(steering_samples) if steering_samples else None),
                    'at_first_throttle': steer_at(first_gas),
                    'at_50_throttle': steer_at(progression['steps']['50']),
                    'at_full_throttle': steer_at(full_gas)}
        split = slowest['lap_time_s']
        result.append({
            'name': name, 'is_corner': corner, 'start_position': start, 'end_position': end,
            'label': name + (' — ' + CORNER_NUMBERS[name] if corner else ''),
            'corner_numbers': CORNER_NUMBERS.get(name),
            'entry_lap_time_s': round(rows[0]['lap_time_s'], 4),
            'exit_lap_time_s': round(rows[-1]['lap_time_s'], 4),
            'time_s': round(rows[-1]['lap_time_s'] - rows[0]['lap_time_s'], 4),
            'entry_speed_kmh': round(rows[0]['speed_kmh'], 2),
            'min_speed_kmh': round(slowest['speed_kmh'], 2),
            'min_speed_position': round(slowest['position_normalized'], 6),
            'exit_speed_kmh': round(rows[-1]['speed_kmh'], 2),
            'brake_start': brake, 'brake_status': 'active_at_entry' if rows[0]['brake'] >= .05 else
            ('detected' if brake else 'not_detected'),
            'first_throttle': first_gas, 'first_throttle_status': 'detected' if first_gas else
            ('no_lift' if lift is None else 'not_detected'),
            'full_throttle': full_gas, 'full_throttle_status': full_status,
            'tc_active_s': active_duration(rows, 'tc'), 'abs_active_s': active_duration(rows, 'abs'),
            'phase_split_lap_time_s': round(split, 4),
            'tc_before_min_s': phase_duration(rows, 'tc', rows[0]['lap_time_s'], split),
            'tc_after_min_s': phase_duration(rows, 'tc', split, rows[-1]['lap_time_s']),
            'abs_before_min_s': phase_duration(rows, 'abs', rows[0]['lap_time_s'], split),
            'abs_after_min_s': phase_duration(rows, 'abs', split, rows[-1]['lap_time_s']),
            'throttle_progression': progression, 'steering': steering,
        })
    return result


def load_trace(directory, info):
    path = directory / info['file']
    if path.name != info['file'] or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('lap file must be inside the session directory')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return LapTrace(list(csv.DictReader(f)), info['lap_time_ms'])


def build_analysis(directory):
    directory = Path(directory)
    session = json.loads((directory / 'session.json').read_text(encoding='utf-8-sig'))
    report = {'schema_version': 2, 'car': session['car'], 'track': session['track'],
              'profile': 'monza_working_zones_v1', 'profile_status': 'approximate_calibrated_zones',
              'thresholds': THRESHOLDS, 'reference_file': None, 'laps': [], 'excluded': [],
              'notes': ['Zone boundaries are fixed comparison windows, not surveyed corner entry/exit.',
                        'Delta = this zone time minus best valid lap zone time; positive is slower.',
                        'TC/ABS duration = raw physics signal > 0, held until the next sample.',
                        'No inferred driving mistake or cause. Best complete valid lap is selected before telemetry checks.',
                        'First/full throttle require rising edges and sustained thresholds; absent events are null.',
                        'Straight sections are included so zone deltas account for whole-lap delta.']}
    if session.get('recovered'):
        report['notes'].append('This session contains derived copies recovered from the previous recorder boundary bug.')
    if session['track'] != 'monza' or session.get('demo'):
        report['profile'] = None
        report['status'] = 'unsupported_track_or_demo'
        return report
    # Select from authoritative lap metadata BEFORE telemetry quality filtering.
    # Missing/poor telemetry must not silently substitute a slower lap.
    candidates = [i for i in session['laps'] if i.get('complete') is True
                  and i.get('valid') is True
                  and isinstance(i.get('lap_time_ms'), (int, float))
                  and math.isfinite(i['lap_time_ms']) and i['lap_time_ms'] > 0]
    chosen = min(candidates, key=lambda i: i['lap_time_ms']) if candidates else None
    report['reference_candidates'] = [{'file': i['file'], 'lap_time_ms': i['lap_time_ms']} for i in candidates]
    traces = {}
    for info in session['laps']:
        if not info['complete'] or not info.get('lap_time_ms'):
            report['excluded'].append({'file': info['file'], 'reason': 'partial_or_unknown_time'})
            continue
        try:
            trace = load_trace(directory, info)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report['excluded'].append({'file': info['file'], 'reason': str(error)})
            continue
        traces[info['file']] = trace
        report['laps'].append({'file': info['file'], 'game_lap_number': info['game_lap_number'],
                               'lap_time_s': trace.lap_s, 'valid': info['valid'],
                               'contains_pit': info['contains_pit'], 'setup': info.get('setup'),
                               'zones': metrics(trace)})
    if chosen is None:
        report['status'] = 'no_valid_reference'
        return report
    report['reference_file'] = chosen['file']
    report['reference_lap_time_s'] = chosen['lap_time_ms'] / 1000
    reference = next((lap for lap in report['laps'] if lap['file'] == chosen['file']), None)
    report['status'] = 'ok' if reference else 'reference_telemetry_unavailable'
    report['track_map'] = track_map(traces, reference, report['laps'])
    for lap in report['laps']:
        lap['delta_s'] = round(lap['lap_time_s'] - report['reference_lap_time_s'], 4)
        if reference is None:
            continue
        for zone, best in zip(lap['zones'], reference['zones']):
            zone['reference_time_s'] = best['time_s']
            zone['delta_s'] = round(zone['time_s'] - best['time_s'], 4)
            zone['reference_metrics'] = {key: best[key] for key in (
                'entry_speed_kmh', 'min_speed_kmh', 'exit_speed_kmh', 'brake_start',
                'first_throttle', 'full_throttle', 'tc_active_s', 'abs_active_s',
                'throttle_progression', 'tc_before_min_s', 'tc_after_min_s',
                'abs_before_min_s', 'abs_after_min_s', 'steering')}
    return report


def display_event(value):
    return '—' if value is None else f"{value['lap_time_s']:.3f} с / {value['position_normalized']:.2%}"


def render_track_map(data):
    points = data['points']
    xmin, xmax = min(p['x'] for p in points), max(p['x'] for p in points)
    zmin, zmax = min(p['z'] for p in points), max(p['z'] for p in points)
    scale = min(580 / max(1, xmax - xmin), 410 / max(1, zmax - zmin))
    def xy(p):
        return 25 + (p['x'] - xmin) * scale, 435 - (p['z'] - zmin) * scale
    path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in map(xy, points + points[:1]))
    out = ['<h2>Схема трассы и повороты</h2>',
           '<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:center">',
           '<svg role="img" aria-label="Monza: отмеченные зоны поворотов" viewBox="0 0 640 470" '
           'style="max-width:600px;width:100%;flex:1 1 280px">',
           f'<polyline points="{path}" fill="none" stroke="#b6c0cc" stroke-width="4"/>']
    for i, marker in enumerate(data['markers'], 1):
        x, y = xy(marker)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="#11161d" stroke="#a6d4ff"/>')
        out.append(f'<text x="{x:.1f}" y="{y + 6:.1f}" text-anchor="middle" fill="#e8edf3" font-size="20">{i}</text>')
    out.extend(['</svg><ol style="flex:1 1 240px;line-height:2">'])
    out.extend(f'<li>{html.escape(m["label"])}</li>' for m in data['markers'])
    out.extend(['</ol></div>', f'<p>Траектория по координатам X/Z файла {html.escape(data["source_file"])}. '
                'Маркеры — минимум скорости в зонах; для Curva Grande — середина дуги. Это схема записанной траектории.</p>'])
    return ''.join(out)


def render_extended(lap):
    out = ['<h3>Газ, фазы TC/ABS и руль</h3><div class="scroll"><table><tr><th>Поворот</th>',
           '<th>5%</th><th>25%</th><th>50%</th><th>75%</th><th>98%</th><th>5→98%, с</th>',
           '<th>TC до / после min, с</th><th>ABS до / после min, с</th>',
           '<th>Руль max / первый газ / 50% / полный газ</th></tr>']
    def number(value):
        return '—' if value is None else f'{value:.3f}'
    def steer(value):
        if value['input_pct'] is not None:
            return f"{value['input_pct']:.1f}%"
        return '—' if value['raw'] is None else f"raw {value['raw']:.3f}"
    def steering(z):
        return ' / '.join(steer(z['steering'][key]) for key in
                          ('max_abs', 'at_first_throttle', 'at_50_throttle', 'at_full_throttle'))
    for zone in lap['zones']:
        if not zone['is_corner']:
            continue
        best = zone.get('reference_metrics')
        def paired(value, ref):
            return value + (f'<br><small>лучший: {ref}</small>' if best else '')
        out.append(f'<tr><td>{html.escape(zone["label"])}</td>')
        for percent in ('5', '25', '50', '75', '98'):
            moment = zone['throttle_progression']['steps'][percent]
            ref = best['throttle_progression']['steps'][percent] if best else None
            out.append('<td>' + paired(number(moment['lap_time_s'] if moment else None),
                                      number(ref['lap_time_s'] if ref else None)) + '</td>')
        out.append('<td>' + paired(number(zone['throttle_progression']['first_to_full_s']),
                                  number(best['throttle_progression']['first_to_full_s']) if best else '') + '</td>')
        for signal in ('tc', 'abs'):
            phase = lambda z: f"{z[signal + '_before_min_s']:.3f} / {z[signal + '_after_min_s']:.3f}"
            out.append('<td>' + paired(phase(zone), phase(best) if best else '') + '</td>')
        out.append('<td>' + paired(steering(zone), steering(best) if best else '') + '</td></tr>')
    out.append('</table></div><small>Моменты газа — секунды от старта круга. 5→98% — длительность '
               'одного открытия; «—» означает, что эта ступень не наблюдалась устойчиво в данном открытии. '
               'Полный газ в основной таблице может относиться к следующему открытию после повторного отпускания.</small>')
    return ''.join(out)


def render_html(report):
    esc = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>ACC Coach — повороты</title>',
             '<style>body{font:16px system-ui;background:#11161d;color:#e8edf3;margin:32px;max-width:1500px}'
             'table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #35404b}'
             'small,p{color:#b6c0cc}summary{cursor:pointer;margin:22px 0;font-size:20px}a{color:#a6d4ff}.scroll{overflow:auto}</style>',
             '<h1>ACC Coach — анализ Monza</h1>', f"<p>{esc(report['car'])} · {esc(report['track'])}</p>",
             '<p>Предварительные зоны по позиции на трассе. Скорости входа и выхода измерены на границах зон. '
             'Δ — разница с лучшим полным валидным кругом этой сессии: плюс — потеря, минус — выигрыш.</p>',
             '<p>Тормоз ≥5% и первый газ ≥5% держатся ≥0.10 с; полный газ ≥98% держится ≥0.20 с '
             'после отпускания газа (если оно было). TC/ABS — время ненулевого исходного сигнала. '
             '«—» означает, что подходящее событие не найдено; это не ноль секунд. Время педалей — от старта круга, '
             'позиция — доля трассы. Прямые учитываются отдельно.</p>']
    if report['reference_file']:
        parts.append(f"<p>Лучший: {esc(report['reference_file'])} · {report['reference_lap_time_s']:.3f} с</p>")
        if report['status'] == 'reference_telemetry_unavailable':
            parts.append('<p>Телеметрия настоящего лучшего круга не прошла проверку или отсутствует. '
                         'Дельты поворотов недоступны; более медленный круг не подставляется. Причина — в списке исключений.</p>')
    else:
        parts.append('<p>Пока нет подходящего лучшего круга. Сравнение появится после полного валидного круга Monza.</p>')
    if report.get('track_map'):
        parts.append(render_track_map(report['track_map']))
    parts.append('<p>Новые показатели: ступени газа относятся к первому устойчивому открытию в зоне, '
                 'до следующего отпускания ниже 5%. TC/ABS разбиты по времени минимальной скорости. '
                 'Руль — исходный сигнал ACC и его модуль в % ввода (при диапазоне −1…1), не угол колёс в градусах.</p>')
    for lap in report['laps']:
        label = 'валидный' if lap['valid'] else 'НЕВАЛИДНЫЙ'
        if lap['contains_pit']: label += ', с боксами'
        total_delta = f" · Δ {lap['delta_s']:+.3f} с" if 'delta_s' in lap else ''
        parts.append(f"<details open><summary>{esc(lap['file'])} · круг {lap['game_lap_number']} · {lap['lap_time_s']:.3f} с{total_delta} · {label}</summary>")
        parts.append('<div class="scroll"><table><tr><th>Участок</th><th>Время / Δ, с</th><th>Вход / min / выход, км/ч</th>'
                     '<th>Тормоз: время / позиция</th><th>Первый газ</th><th>Полный газ</th><th>TC / ABS, с</th></tr>')
        for zone in lap['zones']:
            delta = f"{zone['delta_s']:+.3f}" if 'delta_s' in zone else '—'
            best = zone.get('reference_metrics')
            def cell(current, reference):
                return current + (f'<br><small>лучший: {reference}</small>' if best else '')
            speed = lambda z: f"{z['entry_speed_kmh']:.1f} / {z['min_speed_kmh']:.1f} / {z['exit_speed_kmh']:.1f}"
            activity = lambda z: f"{z['tc_active_s']:.3f} / {z['abs_active_s']:.3f}"
            parts.append(f"<tr><td>{esc(zone['label'])}</td><td>{zone['time_s']:.3f} / {delta}</td>"
                         f"<td>{cell(speed(zone), speed(best) if best else '')}</td>"
                         f"<td>{cell(display_event(zone['brake_start']), display_event(best['brake_start']) if best else '')}</td>"
                         f"<td>{cell(display_event(zone['first_throttle']), display_event(best['first_throttle']) if best else '')}</td>"
                         f"<td>{cell(display_event(zone['full_throttle']), display_event(best['full_throttle']) if best else '')}</td>"
                         f"<td>{cell(activity(zone), activity(best) if best else '')}</td></tr>")
        parts.append('</table></div>')
        parts.append(render_extended(lap))
        parts.append('<small>Если тормоз уже нажат на входе, его начало находится до зоны. '
                     'На повороте без отпускания газа нового момента открытия нет. Подробные статусы и значения '
                     'лучшего круга доступны в analysis.json.</small></details>')
    if report['excluded']:
        parts.append('<details><summary>Не включённые файлы</summary><ul>')
        parts += [f"<li>{esc(item['file'])}: {esc(item['reason'])}</li>" for item in report['excluded']]
        parts.append('</ul></details>')
    parts.append('</html>')
    return ''.join(parts)


def write_analysis(directory):
    directory = Path(directory)
    report = build_analysis(directory)
    outputs = {'analysis.json': json.dumps(report, ensure_ascii=False, indent=2),
               'report.html': render_html(report)}
    for name, content in outputs.items():
        temporary = directory / (name + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(directory / name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path, help='Directory containing session.json and lap CSV files')
    args = parser.parse_args()
    try:
        report = write_analysis(args.session)
        print(f"Report: {(args.session / 'report.html').resolve()}")
        print(f"Status: {report['status']}; analyzed laps: {len(report['laps'])}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Cannot analyze session: {error}\n')


if __name__ == '__main__':
    main()
