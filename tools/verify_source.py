"""Validate source provenance, screenshots and syntax without importing PyTorch."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    runtime = json.loads((root / 'provenance/ORIGINAL_RUNTIME_MANIFEST.json').read_text())
    optional = {'sample.py', 'sample_global_depth_encoding.py', 'sample_conditioned_global.py'}
    for row in runtime['files']:
        data = (root / 'source' / row['name']).read_bytes()
        if row['name'] in optional:
            data = data.replace(b'try:\n    import torch_musa\nexcept ImportError:\n    torch_musa = None\n', b'import torch_musa\n')
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Source differs beyond declared optional MUSA import: ' + row['name'])
    delivery = json.loads((root / 'provenance/ORIGINAL_DELIVERY.json').read_text())
    images = delivery['formal_screenshots'] + delivery['additional_focus_screenshots']
    for row in images:
        if hashlib.sha256((root / 'evidence' / row['file']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Screenshot integrity mismatch')
    files = [root / 'infer.py'] + list((root / 'source').glob('*.py')) + list((root / 'tools').glob('*.py'))
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
    print(json.dumps({'verified_runtime_files': len(runtime['files']),
                      'optional_musa_import_changes_only': True,
                      'verified_actual_game_screenshots': len(images),
                      'python_syntax_checked': len(files)}, indent=2))


if __name__ == '__main__':
    main()
