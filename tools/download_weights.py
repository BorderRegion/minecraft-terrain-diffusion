"""Download the expanded-v2 release EMA pair and verify every byte's SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://github.com/BorderRegion/minecraft-terrain-diffusion/releases/download/heightfield-expanded-v2-20261103/'


def checksum(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'weights')
    args = parser.parse_args()
    manifest_path = ROOT / 'WEIGHTS.json'
    manifest = json.loads(manifest_path.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for row in manifest['weights']:
        name = row['file']
        if Path(name).name != name:
            raise ValueError('Unexpected asset path')
        target = args.out / name
        if target.exists():
            if target.stat().st_size != row['bytes'] or checksum(target) != row['sha256']:
                raise ValueError('Existing file differs; move it aside before downloading: ' + str(target))
            print('Already verified: ' + name, flush=True)
            continue
        temporary = target.with_suffix('.pt.download')
        print('Downloading: ' + name, flush=True)
        try:
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as source, temporary.open('wb') as dest:
                shutil.copyfileobj(source, dest, 1024**2)
            if temporary.stat().st_size != row['bytes'] or checksum(temporary) != row['sha256']:
                raise ValueError('Downloaded asset integrity mismatch: ' + name)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print('Verified: ' + name, flush=True)
    if args.out.resolve() != manifest_path.parent.resolve():
        shutil.copyfile(manifest_path, args.out / 'WEIGHTS.json')


if __name__ == '__main__':
    main()
