from pathlib import Path
import argparse, json, sys, unicodedata
root=Path(__file__).resolve().parents[1]
norm=lambda s:unicodedata.normalize('NFC',s)
manifest=json.loads((root/'docs/data_manifest.json').read_text())
parser=argparse.ArgumentParser(description='Check the original local acquisition manifest (includes archive duplicates).')
parser.add_argument('--data-dir',type=Path,default=root/'data')
args=parser.parse_args()
actual={norm(str(p.relative_to(args.data_dir))):p for p in args.data_dir.rglob('*.csv')}
fail=[]
for r in manifest:
 p=actual.get(norm(r['path']))
 if p is None:fail.append(r['path']+': missing')
 elif p.stat().st_size!=r['bytes']:fail.append(r['path']+': size mismatch')
print('Checked',len(manifest),'expected CSV files')
for msg in fail:print(msg)
if fail:sys.exit(1)
print('PASS: paths and byte sizes match. This is not a content hash check.')
