"""Guard for this experiment: ../guard.py's published files plus the completed baselines.

  python guard_sparsity.py snapshot results/guard_before.txt
  python guard_sparsity.py compare  results/guard_before.txt results/guard_after.txt
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import guard  # noqa: E402

guard.WATCH = guard.WATCH + [
    "experiments/recent_baselines/models/minirocket_field",
    "experiments/recent_baselines/models/minirocket_pixel",
    "experiments/recent_baselines/models/timesnet_field",
    "experiments/recent_baselines/models/timesnet_pixel",
    "experiments/recent_baselines/results",
]

if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        guard.snapshot(sys.argv[2])
    else:
        sys.exit(0 if guard.compare(sys.argv[2], sys.argv[3]) else 1)
