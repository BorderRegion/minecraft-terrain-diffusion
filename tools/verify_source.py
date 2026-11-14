"""Verify archived release source and screenshots without importing PyTorch."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / 'provenance/SOURCE_FILES.json').read_text())['files']
    full = json.loads((root / 'provenance/MANIFEST.json').read_text())['files']
    images = [r for r in full if r['path'].startswith('evidence/') and r['path'].endswith('.png')]
    for row in source + images:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Path escapes repository')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Integrity mismatch: ' + row['path'])
    files = list(root.rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
    print(json.dumps({'verified_original_source_files': len(source),
                      'verified_original_screenshots': len(images),
                      'python_syntax_checked': len(files),
                      'model_weights_included': False,
                      'generation_tested_by_this_check': False}, indent=2))


if __name__ == '__main__':
    main()
