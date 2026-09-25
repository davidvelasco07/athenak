#!/usr/bin/env python3
"""Bounded regression for the N64 isothermal PPM+MOOD velocity runaway.

Run a PROBLEM=gmtf binary, or check an existing run containing stdout.log and
GMTF_mood.log. The original density-only detector stalls before t=0.1 despite
returning exit code zero on its cycle limit. No third-party Python packages needed.
"""
import argparse
import math
from pathlib import Path
import re
import subprocess
import tempfile


def check_run(directory):
    directory = Path(directory)
    log = (directory / "stdout.log").read_text()
    final = re.findall(r"^time=(\S+) cycle=(\d+)", log, re.MULTILINE)
    if (not final or not math.isfinite(float(final[-1][0]))
            or float(final[-1][0]) < 0.1 - 1.0e-7):
        raise AssertionError("N64 failed to reach t=0.1 before the cycle limit")
    samples = re.findall(r"elapsed=\S+ cycle=\d+ time=\S+ dt=(\S+)", log)
    if (not samples or any(not math.isfinite(float(dt)) for dt in samples)
            or any(float(dt) < 1.0e-6 for dt in samples[:-1])):
        raise AssertionError("N64 timestep collapsed below 1e-6")
    lines = (directory / "GMTF_mood.log").read_text().splitlines()
    rows = [[int(x) for x in line.split()] for line in lines
            if line.strip() and not line.startswith("#")]
    if not rows or any(len(row) != 9 for row in rows):
        raise AssertionError("Missing or unrecognized event counters")
    if any(any(row[1:6]) for row in rows):
        raise AssertionError("N64 required EOS floors, velocity ceilings, or C2P repairs")
    if sum(row[8] for row in rows) == 0:
        raise AssertionError("MOOD detector never fired")
    return f"PASS: t={final[-1][0]}, cycles={final[-1][1]}, no EOS repairs"


def run(exe, directory):
    root = Path(__file__).resolve().parents[1]
    args = [str(Path(exe).resolve()), "-i",
            str(root / "inputs/tests/gmtf_mood.athinput"),
            "time/nlim=400", "time/tlim=0.1", "time/ndiag=1",
            "output2/dt=0", "output4/dt=0", "output5/dt=0"]
    with (directory / "stdout.log").open("w") as out:
        subprocess.run(args, cwd=directory, stdout=out,
                       stderr=subprocess.STDOUT, check=True, timeout=600)
    print(check_run(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--athena", type=Path, help="Serial PROBLEM=gmtf executable")
    group.add_argument("--check-output", type=Path, help="Check an existing run")
    options = parser.parse_args()
    if options.check_output:
        print(check_run(options.check_output))
    else:
        with tempfile.TemporaryDirectory(prefix="gmtf-mood-") as tmp:
            run(options.athena, Path(tmp))
