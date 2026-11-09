"""Verify the archive's byte integrity without running any scientific calculations.

Run from any location: python documentation/literature_informed/check_package.py
"""
from pathlib import Path
import csv, hashlib, sys
root = Path(__file__).resolve().parents[2]
manifest = Path(__file__).with_name('PACKAGE_SHA256.csv')
bad = []
missing = []
with manifest.open(newline='', encoding='utf-8') as fh:
    rows = list(csv.DictReader(fh))
for row in rows:
    file = root.joinpath(*row['repository_path'].split('/'))
    if not file.is_file():
        missing.append(row['repository_path'])
    elif hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
        bad.append(row['repository_path'])
print(f'Checked {len(rows)} archived files: {len(missing)} missing, {len(bad)} modified')
for p in missing[:25]: print('MISSING', p)
for p in bad[:25]: print('MODIFIED', p)
sys.exit(1 if missing or bad else 0)
