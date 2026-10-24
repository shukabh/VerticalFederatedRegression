"""Run R and O as two separate OS processes connected by a socket; collect each party's local results.

The only thing the two processes share is the socket, over which every message is an explicit frame
(wire.py).  Each party's inputs are handed to its own process at start; the results each process returns to
the caller (transcripts, R's estimates, O's timings) are local outputs, not protocol messages.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import traceback


def _target(role, sock, args, q):
    try:
        sys.setrecursionlimit(10000)
        opts = args[-1]
        if opts.get('omp_threads'):                     # before the native module is loaded
            os.environ['OMP_NUM_THREADS'] = str(opts['omp_threads'])
        if role == 'R':
            from .party_r import run_R as fn
        else:
            from .party_o import run_O as fn
        res = fn(sock, *args)
        q.put((role, 'ok', res))
    except BaseException as e:  # noqa
        q.put((role, 'error', traceback.format_exc()))
    finally:
        try:
            sock.close()
        except Exception:
            pass


def run(ag, cp, data_R, data_O, opts_R=None, opts_O=None, timeout=3600):
    ctx = mp.get_context('spawn')
    sR, sO = socket.socketpair()
    q = ctx.Queue()
    pR = ctx.Process(target=_target, args=('R', sR, (ag, cp, data_R, opts_R or {}), q))
    pO = ctx.Process(target=_target, args=('O', sO, (ag, cp, data_O, opts_O or {}), q))
    pR.start()
    pO.start()
    sR.close()
    sO.close()
    out, errs = {}, {}
    import queue as _queue
    import time as _time
    t_end = _time.time() + timeout
    while len(out) + len(errs) < 2 and not errs:
        try:
            role, status, res = q.get(timeout=2.0)
        except _queue.Empty:
            dead = [r for r, p in (('R', pR), ('O', pO)) if not p.is_alive() and r not in out]
            if dead or _time.time() > t_end:
                try:                                   # a result may still be in flight
                    role, status, res = q.get(timeout=5.0)
                except _queue.Empty:
                    errs['/'.join(dead) or 'timeout'] = f'process exited without a result (exit codes ' \
                        f'R={pR.exitcode}, O={pO.exitcode})'
                    break
            else:
                continue
        if status == 'ok':
            out[role] = res
        else:
            errs[role] = res
    for p in (pR, pO):
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    if errs:
        raise RuntimeError('party failed:\n' + '\n'.join(f'[{k}] {v}' for k, v in errs.items()))
    return out['R'], out['O']
