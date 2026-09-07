#!/usr/bin/env python3
"""Analyze validation results and emit tables + pass/fail gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

VAL = Path(__file__).resolve().parents[1]
RESULTS = VAL / "results"
MANIFEST = VAL / "manifest.yaml"
VENDOR = VAL / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import yaml

PHYSICS_FROM_CASE = {
    "nr_hydro": "hydro",
    "nr_mhd": "mhd",
    "gr_hydro": "grhydro",
    "gr_mhd": "grmhd",
    "sod": "hydro",
    "shu_osher": "hydro",
    "rj2a": "mhd",
    "bw": "mhd",
    "mb2_gr": "grhydro",
    "mub1_gr": "grmhd",
    "implode_hydro": "hydro",
    "orszag_tang": "mhd",
    "ringing_128": "mhd",
    "blast_grmhd_minkowski": "grmhd",
}

HYDRO_WAVE_NAMES = {
    "0": "left-going acoustic",
    "1": "entropy/contact",
    "2": "transverse shear y",
    "3": "transverse shear z",
    "4": "right-going acoustic",
}
MHD_WAVE_NAMES = {
    "0": "left-going fast magnetosonic",
    "1": "left-going Alfvén",
    "2": "left-going slow magnetosonic",
    "3": "entropy/contact",
    "4": "right-going slow magnetosonic",
    "5": "right-going Alfvén",
    "6": "right-going fast magnetosonic",
}


def _physics(r):
    return r.get("physics") or PHYSICS_FROM_CASE.get(r.get("case", ""), r.get("case"))


def _dim(r):
    return r.get("dim") or "1d"


def _wave_name(physics, wave):
    names = MHD_WAVE_NAMES if physics in ("mhd", "grmhd") else HYDRO_WAVE_NAMES
    return names.get(str(wave), f"wave {wave}")


def load_summaries():
    rows = []
    # Read only canonical suite trees.  Auxiliary comparison/sync directories under
    # results/ may contain stale copies with identical outdir keys.
    for suite in ("linear", "shocks", "stress", "tolerance"):
        root = RESULTS / suite
        if not root.exists():
            continue
        for p in root.rglob("summary.json"):
            try:
                rows.append(json.loads(p.read_text()))
            except Exception:
                pass
    # Aggregate summary_*.json files intentionally repeat the per-run summaries and
    # may be stale after a partial rerun, so they are not analysis inputs.
    # Deduplicate canonical rows independently of machine-specific output paths.
    by = {}
    for r in rows:
        key = (
            r.get("suite"),
            r.get("case"),
            r.get("physics"),
            r.get("dim"),
            r.get("scheme"),
            str(r.get("wave")),
            r.get("rtol"),
            r.get("nx1"),
            r.get("nx2"),
            r.get("nx3"),
        )
        by.setdefault(key, r)
    return list(by.values())


def linear_gates(rows, allowed=None):
    findings = []
    by = defaultdict(dict)
    for r in rows:
        if r.get("suite") != "linear":
            continue
        key = (r["case"], _physics(r), _dim(r), r["scheme"], str(r["wave"]))
        if allowed is not None and (r["case"], _dim(r), str(r["wave"])) not in allowed:
            continue
        by[key][r["nx1"]] = r
    for (case, physics, dim, scheme, wave), dens in sorted(by.items()):
        ns = sorted(dens)
        if not ns:
            continue
        l1s = [dens[n].get("l1_rms") for n in ns]
        moods = [dens[n].get("nmood_total") for n in ns]
        orders = []
        for i in range(1, len(ns)):
            if l1s[i - 1] and l1s[i] and l1s[i - 1] > 0:
                orders.append(float(np.log2(l1s[i - 1] / l1s[i])))
        zero_ok = True
        if scheme == "ppm_fb":
            zero_ok = all((m == 0 or m is None) for m in moods)
        # Prefer clean rc=0; still plot families that wrote L1 then crashed on teardown
        completed = all(dens[n].get("returncode") == 0 for n in ns)
        usable = any(v is not None for v in l1s)
        findings.append(
            {
                "case": case,
                "physics": physics,
                "dim": dim,
                "scheme": scheme,
                "wave": wave,
                "wave_name": _wave_name(physics, wave),
                "resolutions": ns,
                "l1": l1s,
                "orders": orders,
                "nmood": moods,
                "zero_demotions_pass": zero_ok if scheme == "ppm_fb" else None,
                "completed": completed,
                "usable": usable,
            }
        )
    return findings


def shock_overview(rows):
    out = []
    for r in rows:
        if r.get("suite") != "shocks":
            continue
        out.append(
            {
                "case": r["case"],
                "physics": _physics(r),
                "dim": _dim(r),
                "scheme": r["scheme"],
                "nx1": r["nx1"],
                "rc": r.get("returncode"),
                "nmood": r.get("nmood_total"),
                "wall_s": r.get("wall_s"),
            }
        )
    return sorted(out, key=lambda x: (x["physics"], x["case"], x["scheme"], x["nx1"]))


def stress_overview(rows, allowed_cases=None):
    out = []
    for r in rows:
        if r.get("suite") != "stress_smoke":
            continue
        if allowed_cases is not None and r.get("case") not in allowed_cases:
            continue
        out.append(
            {
                "case": r.get("case"),
                "physics": _physics(r),
                "dim": _dim(r),
                "scheme": r.get("scheme"),
                "returncode": r.get("returncode"),
                "nmood_total": r.get("nmood_total"),
                "wall_s": r.get("wall_s"),
                # The Method section states that rc=0 is not evidence of success, so the
                # table must not report rc alone: carry the gate the prose promises.
                "health": _verdict(r),
                "floors": ((r.get("health") or {}).get("floors_total")),
            }
        )
    return out


def _verdict(r):
    h = r.get("health") or {}
    if r.get("returncode"):
        return "rc!=0"
    if h.get("final_finite") is False:
        return "non-finite"
    if h.get("dt_collapse"):
        return "dt collapse"
    if h.get("implausible") is True:
        return "implausible"
    f = h.get("floors_total") or 0
    return "clean" if f == 0 else "floored"


def tolerance_overview(rows):
    out = []
    for r in rows:
        if r.get("suite") != "tolerance":
            continue
        out.append(
            {
                "case": r.get("case"),
                "physics": _physics(r),
                "dim": _dim(r),
                "scheme": r.get("scheme"),
                "rtol": r.get("rtol"),
                "nx1": r.get("nx1"),
                "returncode": r.get("returncode"),
                "nmood_total": r.get("nmood_total"),
                "wall_s": r.get("wall_s"),
                "plot_field": r.get("plot_field"),
            }
        )
    return sorted(
        out,
        key=lambda x: (str(x.get("case") or ""), -float(x.get("rtol") or 0.0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = load_summaries()
    manif = yaml.safe_load(MANIFEST.read_text()) or {}
    allowed_linear = set()
    for case in (manif.get("linear_waves") or {}).get("cases") or []:
        for dim in (manif.get("linear_waves") or {}).get("dimensions") or {}:
            for wave in case.get(f"waves_{dim}", case.get("waves", ["0"])):
                allowed_linear.add((case["id"], dim, str(wave)))
    allowed_stress = {
        c["id"] for c in ((manif.get("stress") or {}).get("local_smoke") or [])
    }
    linear = linear_gates(rows, allowed_linear)
    shocks = shock_overview(rows)
    stress = stress_overview(rows, allowed_stress)
    tolerance = tolerance_overview(rows)
    report = {
        "n_records": (
            sum(len(f.get("resolutions") or []) for f in linear)
            + len(shocks)
            + len(stress)
            + len(tolerance)
        ),
        "linear": linear,
        "shocks": shocks,
        "stress": stress,
        "tolerance": tolerance,
    }
    lin = report["linear"]
    ppm_fb = [f for f in lin if f["scheme"] == "ppm_fb"]
    report["gates"] = {
        "ppm_fb_linear_zero_demotions": all(f["zero_demotions_pass"] for f in ppm_fb)
        if ppm_fb
        else None,
        "ppm_fb_linear_completed": all(f["completed"] for f in ppm_fb) if ppm_fb else None,
        "all_completed": all(r.get("returncode", 1) == 0 for r in rows) if rows else None,
    }
    text = json.dumps(report, indent=2)
    print(text[:4000])
    if args.write:
        # Guard against analysing a partial tree.  analyze.py walks the per-run
        # directories, while summary_<suite>.json is the record of what was actually
        # run -- so a checkout that has the summaries synced from a cluster but only a
        # handful of local run dirs silently produces an analysis over the local subset.
        # That happened: 40 records over 10 cases were reported where the run was 133
        # over 19, and the resulting report showed rc!=0 for runs that had succeeded.
        for _suite in ("stress", "tolerance", "linear", "shocks"):
            _sf = RESULTS / f"summary_{_suite}.json"
            if not _sf.exists():
                continue
            try:
                _n_summary = len(json.loads(_sf.read_text()))
            except Exception:
                continue
            _n_dirs = sum(1 for _ in (RESULTS / _suite).rglob("summary.json")) \
                if (RESULTS / _suite).is_dir() else 0
            if _n_dirs < _n_summary:
                print(f"WARNING: {_suite}: {_n_dirs} run dirs present but "
                      f"summary_{_suite}.json records {_n_summary} runs -- this "
                      f"analysis covers only what is on disk.")
        out = RESULTS / "analysis.json"
        out.write_text(text)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
