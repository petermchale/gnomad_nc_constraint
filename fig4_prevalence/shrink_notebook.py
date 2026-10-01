"""Shrink an executed notebook's outputs without losing any result.

    python shrink_notebook.py NOTEBOOK.ipynb            # report only
    python shrink_notebook.py NOTEBOOK.ipynb --write    # shrink in place

Reports where the bytes are, cell by cell, and each distinct warning in stderr with its
count, then removes or shrinks only what carries
no result:

  * stderr streams (warnings and progress noise; stdout, where every table prints, is kept)
  * the text/html copy of a displayed object that also has a text/plain copy
    (a DataFrame preview; the plain-text preview survives)
  * PNG figures, re-encoded with a 256-colour palette and optimized, and scaled down
    to MAX_WIDTH pixels if wider (a line or bar plot loses nothing visible)

--write first copies the original to the system temp directory and says where.
"""
import argparse
import base64
import io
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

MAX_WIDTH = 1000


def size(obj):
    return len(json.dumps(obj))


def report(nb, title, top=10):
    rows = sorted(
        ((size(c.get('outputs', [])), i, len(c.get('outputs', [])), ''.join(c['source']).strip().splitlines()[0][:60])
         for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and ''.join(c['source']).strip()),
        reverse=True,
    )
    print(f'{title}: notebook {size(nb)/1e6:.1f} MB, outputs {sum(r[0] for r in rows)/1e6:.1f} MB; largest cells:')
    for s, i, n, head in rows[:top]:
        print(f'  cell {i:3d}  {s/1e6:6.2f} MB  {n:4d} outputs  | {head}')


def report_warnings(nb, top=10):
    # What the stderr about to be discarded actually says: each distinct warning, counted
    counts = Counter()
    for c in nb['cells']:
        for o in c.get('outputs', []):
            if o['output_type'] == 'stream' and o.get('name') == 'stderr':
                text = o['text'] if isinstance(o['text'], str) else ''.join(o['text'])
                counts.update(re.sub(r'^\S*/', '', line.strip()) for line in text.splitlines() if 'Warning' in line)
    if not counts:
        print('\nno warnings in stderr')
        return
    print(f'\n{sum(counts.values())} warning lines in stderr, {len(counts)} distinct; most frequent:')
    for line, n in counts.most_common(top):
        print(f'  {n:7d} x  {line[:200]}')


def shrink_png(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGBA')
    method = Image.Quantize.FASTOCTREE if img.mode == 'RGBA' else Image.Quantize.MEDIANCUT
    out = io.BytesIO()
    img.quantize(colors=256, method=method).save(out, format='PNG', optimize=True)
    new = base64.b64encode(out.getvalue()).decode()
    return new if len(new) < len(b64) else b64


def shrink(nb):
    counts = {'stderr': 0, 'html': 0, 'png': 0}
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        kept = []
        for o in c.get('outputs', []):
            if o['output_type'] == 'stream' and o.get('name') == 'stderr':
                counts['stderr'] += 1
                continue
            data = o.get('data', {})
            if 'text/html' in data and 'text/plain' in data:
                del data['text/html']
                counts['html'] += 1
            if 'image/png' in data:
                png = data['image/png']
                data['image/png'] = shrink_png(png if isinstance(png, str) else ''.join(png))
                counts['png'] += 1
            kept.append(o)
        c['outputs'] = kept
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--write', action='store_true', help='shrink in place (default: report only)')
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text())
    report(nb, 'before')
    report_warnings(nb)
    counts = shrink(nb)
    print(f"\nremoved {counts['stderr']} stderr outputs and {counts['html']} HTML previews; "
          f"re-encoded {counts['png']} figures\n")
    report(nb, 'after')

    if args.write:
        backup = Path(tempfile.gettempdir()) / f'{args.notebook.name}.orig'
        shutil.copy2(args.notebook, backup)
        args.notebook.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
        print(f'\nwritten in place; original copied to {backup}')
    else:
        print('\nreport only; rerun with --write to shrink in place')


if __name__ == '__main__':
    main()
