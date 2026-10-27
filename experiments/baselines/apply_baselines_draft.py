"""Splice paper/drafts/baselines.tex into a COPY of paper/main.tex (never into main.tex itself).

The draft consists of blocks
    %B-BEGIN <id> <description>
    %AFTER| <anchor>        (insert the new text after the line that contains this exact text; it must occur once)
  or
    %OLD| <old text>        (one or more lines: replace this exact text, which must occur once, by the new text)
    %NEW-BEGIN
    ...new text...
    %NEW-END
    %B-END <id>

    python experiments/baselines/apply_baselines_draft.py --check
    python experiments/baselines/apply_baselines_draft.py --out $SCRATCH/main_baselines.tex
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
MAIN = os.path.join(ROOT, 'paper', 'main.tex')
DRAFT = os.path.join(ROOT, 'paper', 'drafts', 'baselines.tex')


def blocks(text):
    out = []
    for m in re.finditer(r'^%B-BEGIN (\S+)(.*?)\n(.*?)^%B-END \1\s*$', text, flags=re.S | re.M):
        bid, body = m.group(1), m.group(3)
        after = [ln[len('%AFTER| '):] for ln in body.splitlines() if ln.startswith('%AFTER| ')]
        old = [ln[len('%OLD| '):] if ln.startswith('%OLD| ') else '' for ln in body.splitlines()
               if ln.startswith('%OLD|')]
        nm = re.search(r'^%NEW-BEGIN\n(.*?)^%NEW-END', body, flags=re.S | re.M)
        assert nm, bid
        out.append(dict(id=bid, after=after[0] if after else None, old='\n'.join(old) if old else None,
                        new=nm.group(1)))
    return out


def apply(main, bl):
    for b in bl:
        if b['old'] is not None:
            assert main.count(b['old']) == 1, (b['id'], main.count(b['old']))
            main = main.replace(b['old'], b['new'].rstrip('\n'))
        else:
            lines = main.split('\n')
            hits = [i for i, ln in enumerate(lines) if b['after'] in ln]
            assert len(hits) == 1, (b['id'], len(hits))
            i = hits[0]
            lines[i + 1:i + 1] = [''] + b['new'].rstrip('\n').split('\n')
            main = '\n'.join(lines)
    return main


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--draft', default=DRAFT, help='draft file (default paper/drafts/baselines.tex; also '
                                                   'paper/drafts/fullrecord.tex)')
    ap.add_argument('--main', default=MAIN, help='the text to splice into (default paper/main.tex, read only)')
    a = ap.parse_args()
    main = open(a.main).read()
    bl = blocks(open(a.draft).read())
    merged = apply(main, bl)
    print(f'{len(bl)} blocks apply cleanly')
    if a.out:
        assert os.path.abspath(a.out) != os.path.abspath(MAIN)
        with open(a.out, 'w') as fh:
            fh.write(merged)
        print('wrote', a.out)


if __name__ == '__main__':
    main_()
