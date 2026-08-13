#!/usr/bin/env python3
"""Fetch zarr boxes from the Vesuvius open-data bucket, with an on-disk cache.

Mirrors the read path of audit_submission/code/eval_surface_pred.py, but pulls
chunks over HTTP instead of from a local store.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import urllib.request

BASE = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
CACHE = Path(os.environ.get("CHUNK_CACHE",
                            Path(__file__).resolve().parent / "cache"))

LO = "PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
PRED = ("PHerc0139/representations/predictions/surfaces/"
        "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr")

SCROLLS = {
    "0139": dict(
        lo=LO, pred=PRED,
        labels=os.environ.get("LABELS_0139", "labels0139_L1.zarr"),
        origin=(1728, 576, 480), shape=(1248, 2304, 2208),
        lo_l1=(10487, 3311, 3311),
        gone_pts="results/pass6_gone_pts.npy"),
    "1203": dict(
        lo="PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr",
        pred=("PHerc1203/representations/predictions/surfaces/"
              "20250820131727-surface-20260413222639-surface-m7-L0-th0.2"
              ".zarr"),
        labels=os.environ.get("LABELS_1203", "labels1203_L1.zarr"),
        origin=(3936, 0, 0), shape=(2016, 3456, 3456),
        lo_l1=(9489, 3422, 3422),
        gone_pts=None),          # no per-arc list published for this scroll
}

BLOSC = numcodecs.Blosc()


def _get(key, dest, retries=3):
    """Returns 'ok', 'absent' (the store genuinely has no such chunk, so it
    reads as fill_value), or 'failed'.

    Absent and failed must stay separate. Both would land in the array as
    zeros, and zeros in the prediction volume read as 'the model marked
    nothing here', which is the finding this code exists to measure. A dropped
    request would therefore manufacture missed sheets, and only ever in the
    direction that flatters the result.
    """
    if dest.is_file() and dest.stat().st_size > 0:
        return "ok"
    if dest.with_suffix(".absent").is_file():
        return "absent"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{key}"
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                if r.status != 200:
                    last = f"status {r.status}"
                    continue
                body = r.read()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(body)
            os.replace(tmp, dest)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                dest.with_suffix(".absent").write_bytes(b"")
                return "absent"
            last = f"http {e.code}"
        except Exception as e:
            last = repr(e)
    print(f"    FAILED {key}: {last}", flush=True)
    return "failed"


def fetch_chunks(prefix, level, keys, par=16):
    """keys: list of (cz, cy, cx). Returns (got, absent). Raises if any chunk
    could not be settled either way."""
    got, absent, failed = set(), set(), []
    def one(k):
        cz, cy, cx = k
        d = CACHE / prefix.replace("/", "_") / str(level) / f"{cz}_{cy}_{cx}"
        return k, _get(f"{prefix}/{level}/{cz}/{cy}/{cx}", d)
    with ThreadPoolExecutor(par) as ex:
        for k, st in ex.map(one, keys):
            if st == "ok":
                got.add(k)
            elif st == "absent":
                absent.add(k)
            else:
                failed.append(k)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(keys)} chunks could not be fetched and "
            f"could not be confirmed absent, e.g. {failed[:3]}. Refusing to "
            f"treat them as empty: in the prediction volume that would read "
            f"as the model having marked nothing there.")
    return got, absent


def read_box(prefix, level, ch, compressed, z0, z1, y0, y1, x0, x1,
             par=16, verbose=False):
    """Read [z0:z1, y0:y1, x0:x1] as uint8, fetching what it needs."""
    keys = [(cz, cy, cx)
            for cz in range(z0 // ch, (z1 - 1) // ch + 1)
            for cy in range(y0 // ch, (y1 - 1) // ch + 1)
            for cx in range(x0 // ch, (x1 - 1) // ch + 1)]
    if verbose:
        print(f"  {prefix.split('/')[-1][:40]} L{level}: {len(keys)} chunks",
              flush=True)
    got, absent = fetch_chunks(prefix, level, keys, par)
    if verbose:
        print(f"  got {len(got)}, absent-by-404 {len(absent)}, "
              f"of {len(keys)}", flush=True)
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    exp = ch * ch * ch
    for (cz, cy, cx) in keys:
        d = CACHE / prefix.replace("/", "_") / str(level) / f"{cz}_{cy}_{cx}"
        if (cz, cy, cx) not in got:
            continue
        raw = d.read_bytes()
        if compressed:
            buf = BLOSC.decode(raw)
        else:
            buf = raw
        if len(buf) != exp:
            raise RuntimeError(f"chunk {cz}/{cy}/{cx} decoded to {len(buf)} "
                               f"bytes, expected {exp}")
        a = np.frombuffer(buf, np.uint8).reshape(ch, ch, ch)
        az0, ay0, ax0 = cz * ch, cy * ch, cx * ch
        sz0, sy0, sx0 = max(z0, az0), max(y0, ay0), max(x0, ax0)
        sz1, sy1, sx1 = min(z1, az0 + ch), min(y1, ay0 + ch), min(x1, ax0 + ch)
        if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
            continue
        out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
            a[sz0 - az0:sz1 - az0, sy0 - ay0:sy1 - ay0, sx0 - ax0:sx1 - ax0]
    return out


def read_labels(z0, z1, y0, y1, x0, x1, scroll="0139"):
    """Local packaged truth labels (L1 grid), in *local* label indices."""
    root = Path(SCROLLS[scroll]["labels"])
    ch = 128
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    for cz in range(z0 // ch, (z1 - 1) // ch + 1):
        for cy in range(y0 // ch, (y1 - 1) // ch + 1):
            for cx in range(x0 // ch, (x1 - 1) // ch + 1):
                f = root / str(cz) / str(cy) / str(cx)
                if not f.is_file():
                    continue
                buf = BLOSC.decode(f.read_bytes())
                a = np.frombuffer(buf, np.uint8).reshape(ch, ch, ch)
                az0, ay0, ax0 = cz * ch, cy * ch, cx * ch
                sz0, sy0, sx0 = max(z0, az0), max(y0, ay0), max(x0, ax0)
                sz1 = min(z1, az0 + ch)
                sy1 = min(y1, ay0 + ch)
                sx1 = min(x1, ax0 + ch)
                if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
                    continue
                out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0,
                    sx0 - x0:sx1 - x0] = \
                    a[sz0 - az0:sz1 - az0, sy0 - ay0:sy1 - ay0,
                      sx0 - ax0:sx1 - ax0]
    return out


if __name__ == "__main__":
    # smoke test: one pred chunk and one lo chunk must decode to exactly ch^3
    p = read_box(PRED, 0, 192, True, 4112, 4114, 2880, 3072, 4800, 4992,
                 verbose=True)
    print("pred box", p.shape, "positives", int((p > 0).sum()))
    l = read_box(LO, 1, 128, False, 2056, 2058, 1408, 1536, 2432, 2560,
                 verbose=True)
    print("lo box", l.shape, "mean", float(l.mean()))
    lb = read_labels(2056 - 1728, 2058 - 1728, 800, 928, 1900, 2028)
    print("labels box", lb.shape, "material px",
          int(((lb & 2) > 0).sum()), "centerline px", int(((lb & 4) > 0).sum()))
