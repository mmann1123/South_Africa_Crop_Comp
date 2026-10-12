"""SHA-256 manifest of published inputs/outputs, to prove this experiment changed none of them.

  python guard.py snapshot results/guard_before.txt
  python guard.py compare  results/guard_before.txt results/guard_after.txt   (exit 1 on any change)
"""
import hashlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WATCH = [
    "data/merged_dl_train.parquet",
    "data/merged_dl_test.parquet",
    "out_of_sample",                       # predictions_*.csv and scoring_results/
    "deep_learn/src/models",
    "experiments/field_reduction/models",
    "experiments/field_reduction/results",
    "deep_learn/src/config.py",
    "experiments/field_reduction/models_arch.py",
    "experiments/field_reduction/subsample.py",
    "experiments/field_reduction/experiment_config.py",
]
SKIP_DIRS = {"__pycache__"}


def files():
    for w in WATCH:
        p = os.path.join(REPO, w)
        if os.path.isfile(p):
            yield p
        else:
            for root, dirs, fs in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for f in sorted(fs):
                    yield os.path.join(root, f)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def snapshot(out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        for p in files():
            fh.write(f"{sha(p)}  {os.path.relpath(p, REPO)}\n")
    print(f"wrote {out}")


def compare(a, b):
    A = dict(reversed(l.rstrip("\n").split("  ", 1)) for l in open(a))
    B = dict(reversed(l.rstrip("\n").split("  ", 1)) for l in open(b))
    changed = sorted(k for k in A.keys() & B.keys() if A[k] != B[k])
    added, removed = sorted(B.keys() - A.keys()), sorted(A.keys() - B.keys())
    for tag, ks in [("CHANGED", changed), ("ADDED", added), ("REMOVED", removed)]:
        for k in ks:
            print(f"{tag}: {k}")
    ok = not (changed or added or removed)
    print(f"guard: {len(A)} files checked, {'no changes' if ok else 'CHANGES FOUND'}")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        snapshot(sys.argv[2])
    else:
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
