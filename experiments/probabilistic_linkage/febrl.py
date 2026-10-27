"""FEBRL4 (Christen's FEBRL generator; 5,000 original records and 5,000 corrupted duplicates, one per original) as an
external check of the key schemes' precision and recall.  The CSV files ship with the `recordlinkage` Python package
(recordlinkage/datasets/febrl/dataset4a.csv, dataset4b.csv); pass their directory, or the script looks for an
installed copy.  The data are not copied into the repo.

FEBRL has no sex field: it is set to the constant 0, so the keys that use it degrade to keys without it.
"""
from __future__ import annotations

import csv
import glob
import os

import numpy as np

from generator import People


def find_dir():
    for pat in ('/usr/local/lib/python3*/dist-packages/recordlinkage/datasets/febrl',
                '/usr/lib/python3*/dist-packages/recordlinkage/datasets/febrl',
                os.path.expanduser('~/.local/lib/python3*/site-packages/recordlinkage/datasets/febrl')):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def _load(path):
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh, skipinitialspace=True))
    head = [h.strip() for h in rows[0]]
    recs = [dict(zip(head, [c.strip() for c in r])) for r in rows[1:] if r]
    n = len(recs)
    first = np.empty(n, object); last = np.empty(n, object); post = np.empty(n, object)
    year = np.full(n, -1); month = np.full(n, -1); day = np.full(n, -1)
    pid = np.empty(n, int)
    for k, r in enumerate(recs):
        first[k] = r['given_name']; last[k] = r['surname']; post[k] = r['postcode']
        d = r['date_of_birth']
        if len(d) == 8 and d.isdigit():
            y, m, dd = int(d[:4]), int(d[4:6]), int(d[6:])
            if 1 <= m <= 12 and 1 <= dd <= 31:
                year[k], month[k], day[k] = y, m, dd
        pid[k] = int(r['rec_id'].split('-')[1])
    return People(first, last, np.zeros(n, int), year, month, day, post, pid)


def load(dirpath=None):
    d = dirpath or find_dir()
    if d is None:
        raise FileNotFoundError('FEBRL4 CSVs not found (pip install recordlinkage, or pass the directory)')
    A = _load(os.path.join(d, 'dataset4a.csv'))       # originals: O's register
    B = _load(os.path.join(d, 'dataset4b.csv'))       # duplicates: R's cohort
    pos = {p: t for t, p in enumerate(A.pid)}
    true_pos = np.array([pos.get(p, -1) for p in B.pid])
    return A, B, true_pos
