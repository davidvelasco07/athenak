#!/usr/bin/env python3
"""Summarize JSONL benchmark output. Python standard library only."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if line.startswith("{")]
        modes = {row["mode"]: row for row in records
                 if row.get("type") == "benchmark"}
        if not modes:
            continue
        dense, forced, sparse = (modes[key] for key in ("dense", "forced", "sparse"))
        rows.append(dict(
            case=path.stem,
            nonzero_fraction=sparse["nonzero_fraction"],
            dense_seconds=dense["seconds_median"],
            forced_seconds=forced["seconds_median"],
            sparse_seconds=sparse["seconds_median"],
            sparse_speedup=dense["seconds_median"] / sparse["seconds_median"],
            forced_overhead=forced["seconds_median"] / dense["seconds_median"],
            reserved_fraction=(sparse["reserved_payload_bytes"]
                               + sparse["device_metadata_bytes"])
                              / dense["dense_payload_bytes"],
        ))
    for row in rows:
        print(f"{row['case']:14} f={row['nonzero_fraction']:.4f} "
              f"speedup={row['sparse_speedup']:.3f} "
              f"forced/dense={row['forced_overhead']:.3f} "
              f"reserved/dense={row['reserved_fraction']:.4f}")
    if args.csv and rows:
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
