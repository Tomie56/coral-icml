#!/usr/bin/env python3
"""
create_train_val_test_split.py

Create question-grouped train/val/test splits for CORAL probe rows.

The activation collectors store one row per (question, option) with ids like:
  "<dataset>_<subject>_<split>_<idx>_opt<j>"

We must split at the QUESTION level so that all options of a question stay in
the same partition (to avoid leakage).

Outputs (row indices into probe_data.npz arrays):
  - train_row_indices.npy
  - val_row_indices.npy
  - test_row_indices.npy

Default ratios match the paper's MMLU split: 60/20/20.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np


def _qid_from_row_id(row_id: str) -> str:
    s = str(row_id)
    return s.rsplit("_opt", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_npz", type=str, required=True, help="Path to probe_data.npz")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write *.npy indices")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.60)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument(
        "--require_complete_questions",
        action="store_true",
        help="If set, drop any question that doesn't have all options (e.g. not exactly 4 rows).",
    )
    args = ap.parse_args()

    fracs = np.array([args.train_frac, args.val_frac, args.test_frac], dtype=np.float64)
    if np.any(fracs < 0):
        raise ValueError("Fractions must be non-negative.")
    if not np.isclose(fracs.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Fractions must sum to 1.0, got {fracs.sum():.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.probe_npz, allow_pickle=True)
    if "ids" not in data:
        raise KeyError("probe_data.npz missing required key 'ids'")

    ids = data["ids"].astype(str)
    qids = np.array([_qid_from_row_id(x) for x in ids], dtype=object)

    # Build question -> row indices
    q_to_rows: Dict[str, List[int]] = {}
    for i, q in enumerate(qids):
        q_to_rows.setdefault(q, []).append(int(i))

    questions = list(q_to_rows.keys())
    if args.require_complete_questions:
        # Heuristic: keep questions with a constant number of options.
        # For MMLU this is exactly 4.
        lens = np.array([len(q_to_rows[q]) for q in questions], dtype=int)
        most_common = int(np.bincount(lens).argmax()) if len(lens) else 0
        questions = [q for q in questions if len(q_to_rows[q]) == most_common]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(questions)

    n_q = len(questions)
    n_train = int(np.floor(n_q * args.train_frac))
    n_val = int(np.floor(n_q * args.val_frac))
    # remainder to test
    n_test = n_q - n_train - n_val

    q_train = questions[:n_train]
    q_val = questions[n_train : n_train + n_val]
    q_test = questions[n_train + n_val :]

    def rows_for(q_list: List[str]) -> np.ndarray:
        rows = []
        for q in q_list:
            rows.extend(q_to_rows[q])
        rows = np.array(sorted(rows), dtype=np.int64)
        return rows

    train_rows = rows_for(q_train)
    val_rows = rows_for(q_val)
    test_rows = rows_for(q_test)

    np.save(out_dir / "train_row_indices.npy", train_rows)
    np.save(out_dir / "val_row_indices.npy", val_rows)
    np.save(out_dir / "test_row_indices.npy", test_rows)

    print("Wrote splits to:", out_dir)
    print(f"Questions: total={n_q} train={len(q_train)} val={len(q_val)} test={len(q_test)}")
    print(f"Rows:      total={len(ids)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")


if __name__ == "__main__":
    main()

