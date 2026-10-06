"""Build a local ZIP for AI analysis; nothing is uploaded."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from analyze_session import write_analysis


def session_file(directory, name):
    path = directory / name
    if not isinstance(name, str) or Path(name).name != name or not path.resolve().is_relative_to(directory):
        raise ValueError(f'Unsafe session filename: {name}')
    if not path.is_file():
        raise ValueError(f'Missing session file: {name}')
    return path


def export_session(directory, all_csv=False, laps=None, output=None):
    directory = Path(directory).resolve()
    if all_csv and laps:
        raise ValueError('Use either all CSV or selected laps')
    # Always refresh analysis so a stale reference cannot enter the package.
    report = write_analysis(directory)
    session = json.loads((directory / 'session.json').read_text(encoding='utf-8-sig'))
    infos = {lap['file']: lap for lap in session['laps']}
    selected = []
    reasons = {}

    def select(name, reason):
        if name and name not in selected:
            session_file(directory, name)
            selected.append(name)
            reasons[name] = reason

    if all_csv:
        for path in sorted(directory.glob('*.csv')):
            if not path.name.endswith('.recording.csv'):
                select(path.name, 'all_csv')
    else:
        game_best = report.get('game_best_file')
        if not game_best:
            candidates = [lap for lap in session['laps'] if lap.get('complete') is True
                          and lap.get('valid') is True and lap.get('contains_pit') is False
                          and isinstance(lap.get('lap_time_ms'), (int, float)) and lap['lap_time_ms'] > 0]
            game_best = min(candidates, key=lambda lap: lap['lap_time_ms'])['file'] if candidates else None
        select(game_best, 'game_best')
        if report.get('status') == 'ok':
            select(report.get('reference_file'), 'analysis_reference')
        if laps:
            for name in laps:
                if name not in infos:
                    raise ValueError(f'Lap is not listed in session.json: {name}')
                select(name, 'user_selected_comparison')
        else:
            eligible = sorted((lap for lap in session['laps'] if lap.get('complete') is True
                               and lap.get('valid') is True and lap.get('contains_pit') is False
                               and lap['file'] not in selected
                               and lap['file'] in {x['file'] for x in report.get('laps', [])}),
                              key=lambda lap: lap['lap_time_ms'])
            if eligible:
                for index, reason in ((0, 'next_fastest'), (len(eligible) // 2, 'median_comparison'),
                                      (len(eligible) - 1, 'slowest_comparison')):
                    select(eligible[index]['file'], reason)
    if not selected:
        raise ValueError('No finalized CSV available for this export; record a full lap or use --all-csv')
    destination = Path(output).resolve() if output else directory / ('ai_series.zip' if all_csv else 'ai_analysis.zip')
    if destination.suffix.lower() != '.zip':
        raise ValueError('Export destination must end with .zip')
    files = ['session.json', 'analysis.json'] + selected
    # Saved setup improves interpretation when the user associated one with this session.
    files += [name for name in ('setup.json', 'setup_summary.json', 'setup_summary.txt') if (directory / name).is_file()]
    payloads = {name: session_file(directory, name).read_bytes() for name in files}
    manifest = {'schema_version': 1, 'mode': 'all_csv' if all_csv else 'selected',
                'reference_file': report.get('reference_file'), 'game_best_file': report.get('game_best_file'),
                'reference_fallback': report.get('reference_fallback', False),
                'analysis_status': report['status'], 'selected_laps': [
                    {'file': name, 'selection_reason': reasons[name], 'metadata': infos.get(name)} for name in selected],
                'files': [{'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in payloads.items()],
                'notes': ['CSV values are original raw recordings. Session metadata describes all laps, including CSV not selected.',
                          'All-CSV mode includes partial/invalid laps and best_lap.csv if present; excludes active .recording.csv.',
                          'Use game_valid/telemetry_valid and contains_pit when comparing laps. Nothing was uploaded.']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.zip.tmp', delete=False) as temp:
            temp_path = Path(temp.name)
        with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in payloads.items():
                archive.writestr(name, data)
            archive.writestr('export_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        temp_path.replace(destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all-csv', action='store_true', help='Include all finalized CSV, including partial and invalid laps')
    group.add_argument('--laps', nargs='+', help='Comparison filenames, e.g. lap_03.csv lap_09.csv; best/reference are always included')
    parser.add_argument('--output', type=Path, help='Custom ZIP destination')
    args = parser.parse_args()
    try:
        path = export_session(args.session, args.all_csv, args.laps, args.output)
        print(f'AI analysis package: {path}')
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Cannot export session: {error}\n')


if __name__ == '__main__':
    main()
