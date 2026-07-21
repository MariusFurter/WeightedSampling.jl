#!/usr/bin/env python3
"""Parse `RESULT,<framework>,key1=val1,key2=val2,...` lines out of a raw
benchmark log (as produced by run_grid.sh, one line per framework per grid
point) into a tidy long-format CSV: framework,T,N,metric,value.

Usage: parse_results.py <raw_log> <csv_out>
"""
import csv
import sys


def main(raw_log, csv_out):
    rows = []
    with open(raw_log) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("RESULT,"):
                continue
            parts = line.split(",")
            framework = parts[1]
            kv = {}
            for item in parts[2:]:
                if "=" not in item:
                    continue
                k, v = item.split("=", 1)
                kv[k] = v
            t = kv.pop("T", "")
            n = kv.pop("N", "")
            for metric, value in kv.items():
                rows.append((framework, t, n, metric, value))

    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["framework", "T", "N", "metric", "value"])
        w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <raw_log> <csv_out>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
