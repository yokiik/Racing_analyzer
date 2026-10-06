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
              'tc_abs_active': 0, 'max_gap_s': 1}
NUMERIC = ('lap_time_s', 'position_normalized', 'speed_kmh', 'throttle', 'brake', 'tc', 'abs')


class LapTrace:
    def __init__(self, rows, lap_ms):
        self.rows = []
        for raw in rows:
            row = {k: float(raw[k]) for k in NUMERIC}
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError('non-finite telemetry')
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
        result.append({
            'name': name, 'is_corner': corner, 'start_position': start, 'end_position': end,
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
    report = {'schema_version': 1, 'car': session['car'], 'track': session['track'],
              'profile': 'monza_working_zones_v1', 'profile_status': 'approximate_calibrated_zones',
              'thresholds': THRESHOLDS, 'reference_file': None, 'laps': [], 'excluded': [],
              'notes': ['Zone boundaries are fixed comparison windows, not surveyed corner entry/exit.',
                        'Delta = this zone time minus best valid lap zone time; positive is slower.',
                        'TC/ABS duration = raw physics signal > 0, held until the next sample.',
                        'No inferred driving mistake or cause. Report is rebuilt against the current session best.',
                        'First/full throttle require rising edges and sustained thresholds; absent events are null.',
                        'Straight sections are included so zone deltas account for whole-lap delta.']}
    if session.get('recovered'):
        report['notes'].append('This session contains derived copies recovered from the previous recorder boundary bug.')
    if session['track'] != 'monza' or session.get('demo'):
        report['profile'] = None
        report['status'] = 'unsupported_track_or_demo'
        return report
    for info in session['laps']:
        if not info['complete'] or not info.get('lap_time_ms'):
            report['excluded'].append({'file': info['file'], 'reason': 'partial_or_unknown_time'})
            continue
        try:
            trace = load_trace(directory, info)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report['excluded'].append({'file': info['file'], 'reason': str(error)})
            continue
        report['laps'].append({'file': info['file'], 'game_lap_number': info['game_lap_number'],
                               'lap_time_s': trace.lap_s, 'valid': info['valid'],
                               'contains_pit': info['contains_pit'], 'setup': info.get('setup'),
                               'zones': metrics(trace)})
    eligible = [lap for lap in report['laps'] if lap['valid'] and not lap['contains_pit']]
    if not eligible:
        report['status'] = 'no_valid_reference'
        return report
    reference = min(eligible, key=lambda lap: lap['lap_time_s'])
    report['reference_file'] = reference['file']
    report['reference_lap_time_s'] = reference['lap_time_s']
    report['status'] = 'ok'
    for lap in report['laps']:
        lap['delta_s'] = round(lap['lap_time_s'] - reference['lap_time_s'], 4)
        for zone, best in zip(lap['zones'], reference['zones']):
            zone['reference_time_s'] = best['time_s']
            zone['delta_s'] = round(zone['time_s'] - best['time_s'], 4)
            zone['reference_metrics'] = {key: best[key] for key in (
                'entry_speed_kmh', 'min_speed_kmh', 'exit_speed_kmh', 'brake_start',
                'first_throttle', 'full_throttle', 'tc_active_s', 'abs_active_s')}
    return report


def display_event(value):
    return '—' if value is None else f"{value['lap_time_s']:.3f} с / {value['position_normalized']:.2%}"


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
    else:
        parts.append('<p>Пока нет подходящего лучшего круга. Сравнение появится после полного валидного круга Monza.</p>')
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
            parts.append(f"<tr><td>{esc(zone['name'])}</td><td>{zone['time_s']:.3f} / {delta}</td>"
                         f"<td>{cell(speed(zone), speed(best) if best else '')}</td>"
                         f"<td>{cell(display_event(zone['brake_start']), display_event(best['brake_start']) if best else '')}</td>"
                         f"<td>{cell(display_event(zone['first_throttle']), display_event(best['first_throttle']) if best else '')}</td>"
                         f"<td>{cell(display_event(zone['full_throttle']), display_event(best['full_throttle']) if best else '')}</td>"
                         f"<td>{cell(activity(zone), activity(best) if best else '')}</td></tr>")
        parts.append('</table></div><small>Если тормоз уже нажат на входе, его начало находится до зоны. '
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
