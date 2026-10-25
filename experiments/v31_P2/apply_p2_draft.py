"""Apply the replacements of paper/drafts/p2.tex to a copy of paper/main.tex (mechanical merge).

The draft is a list of blocks:

    %R-BEGIN <id> <one-line description>
    %OLD| <old text, line 1>          (exact text of main.tex; lines joined with newlines; must occur exactly once)
    %OLD| <old text, line 2>
    %NEW-BEGIN
    <new LaTeX, verbatim>
    %NEW-END
    %R-END <id>

    python experiments/v31_P2/apply_p2_draft.py --main paper/main.tex --draft paper/drafts/p2.tex --out <copy.tex>
    python experiments/v31_P2/apply_p2_draft.py --check            # only check that every OLD text occurs once

The script never writes paper/main.tex: --out must be a different path.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))


def parse(draft_text):
    blocks, cur, mode = [], None, None
    for line in draft_text.split('\n'):
        if line.startswith('%R-BEGIN '):
            assert cur is None, f'nested block at {line!r}'
            parts = line[len('%R-BEGIN '):].split(' ', 1)
            cur = dict(id=parts[0], desc=parts[1] if len(parts) > 1 else '', old=[], new=[])
            mode = 'head'
        elif line.startswith('%R-END'):
            assert cur is not None and mode == 'after', f'unbalanced block end at {line!r}'
            blocks.append(dict(cur, old='\n'.join(cur['old']), new='\n'.join(cur['new'])))
            cur, mode = None, None
        elif cur is not None and mode == 'head' and line.startswith('%OLD| '):
            cur['old'].append(line[len('%OLD| '):])
        elif cur is not None and mode == 'head' and line.startswith('%OLD|'):
            cur['old'].append(line[len('%OLD|'):])              # an empty old line
        elif cur is not None and line.strip() == '%NEW-BEGIN':
            mode = 'new'
        elif cur is not None and line.strip() == '%NEW-END':
            mode = 'after'
        elif cur is not None and mode == 'new':
            cur['new'].append(line)
    assert cur is None, 'unterminated block'
    ids = [b['id'] for b in blocks]
    assert len(ids) == len(set(ids)), 'duplicate block ids'
    return blocks


def apply(main_text, blocks, strict=True):
    out, report = main_text, []
    for b in blocks:
        k = out.count(b['old'])
        report.append((b['id'], k, b['desc']))
        if k != 1:
            if strict:
                raise SystemExit(f"block {b['id']}: old text occurs {k} times ({b['desc']})")
            continue
        out = out.replace(b['old'], b['new'])
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', default=os.path.join(ROOT, 'paper', 'main.tex'))
    ap.add_argument('--draft', default=os.path.join(ROOT, 'paper', 'drafts', 'p2.tex'))
    ap.add_argument('--out')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    main_text = open(a.main).read()
    blocks = parse(open(a.draft).read())
    out, report = apply(main_text, blocks, strict=not a.check)
    for bid, k, desc in report:
        print(f'{bid:5s} occurs {k}x  {desc}')
    bad = [r for r in report if r[1] != 1]
    print(f'{len(blocks)} blocks, {len(bad)} not found exactly once')
    if a.check:
        sys.exit(1 if bad else 0)
    assert a.out and os.path.abspath(a.out) != os.path.abspath(a.main), '--out must be a copy, never main.tex'
    with open(a.out, 'w') as fh:
        fh.write(out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
