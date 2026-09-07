#!/usr/bin/env python3
"""Manifest-driven MOOD fallback validation runner.

Usage:
  python validation/fallback/scripts/run_suite.py --athena path/to/athena --suite linear
  python validation/fallback/scripts/run_suite.py --athena path/to/athena --suite shocks
  python validation/fallback/scripts/run_suite.py --athena path/to/athena --suite all --dry-run
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VAL = Path(__file__).resolve().parents[1]
MANIFEST = VAL / "manifest.yaml"
RESULTS = VAL / "results"
VENDOR = VAL / "vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

try:
    import yaml
except ImportError:
    yaml = None


class _StrictLoader(yaml.SafeLoader if yaml else object):
    """SafeLoader that refuses duplicate mapping keys.

    PyYAML silently keeps the LAST of two identical keys.  A mis-indented case block
    once produced a second `cluster:` under `stress:`, which parsed without complaint
    and dropped every case in the first one -- the run then quietly covered fewer cases
    than the manifest appears to describe.  Fail loudly instead.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_manifest():
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with open(MANIFEST) as f:
        return yaml.load(f, Loader=_StrictLoader)


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def scheme_overrides(manif, scheme_name, soe, use_uct=False, linear=False,
                     emf_name=None):
    s = manif["schemes"][scheme_name]
    d = manif["defaults"]
    ng = s["nghost_mhd_uct"] if (soe == "mhd" and use_uct) else s["nghost_hydro"]
    args = [
        f"time/integrator={s['integrator']}",
        f"{soe}/reconstruct={s['reconstruct']}",
        f"mesh/nghost={ng}",
    ]
    # FOFC is the non-MOOD schemes' a-posteriori safety net and is the counterpart of
    # MOOD (the two are mutually exclusive in the code, and unlimited PPM *requires*
    # mood=true).  Giving ppm_fb MOOD while leaving PLM/WENOZ/TENO bare is not a like-
    # for-like robustness comparison, so every non-MOOD scheme gets fofc unless the
    # manifest says otherwise.  It is a no-op wherever no floor is hit.
    if not s["mood"] and s.get("fofc", True):
        args.append(f"{soe}/fofc=true")

    if soe == "hydro":
        args.append(f"hydro/mood={'true' if s['mood'] else 'false'}")
        if s["mood"]:
            args += [
                f"hydro/mood_nad_scale={d['mood_nad_scale']}",
                f"hydro/mood_rtol={d['mood_rtol']}",
                f"hydro/mood_sed={'true' if d['mood_sed'] else 'false'}",
            ]
            if linear and "mood_nad_v" in d:
                nad_v = d['mood_nad_v']
                if nad_v is False:
                    nad_v = 'off'
                args.append(f"hydro/mood_nad_v={nad_v}")
    else:
        # NR MHD uses UCT-HLLD (high-order CT + HO face→cell B). GR/SR stay ct_contact.
        args.append(f"mhd/mood={'true' if s['mood'] else 'false'}")
        if use_uct:
            # uct_hlld needs rsolver=hlld, which SR/GR reject outright, so relativistic
            # cases that opt into UCT get uct_hll.  A case may name the variant itself.
            args.append(f"mhd/emf={emf_name or 'uct_hlld'}")
        else:
            args.append("mhd/emf=ct_contact")
        if s["mood"]:
            args += [
                f"mhd/mood_nad_scale={d['mood_nad_scale']}",
                f"mhd/mood_rtol={d['mood_rtol']}",
                f"mhd/mood_sed={'true' if d['mood_sed'] else 'false'}",
            ]
            # mood_nad_v=off is a LINEAR-WAVE concession only: at amplitude 1e-6 the
            # velocity channel falsely demotes the Alfven/entropy families.  Applying
            # it everywhere cripples the detector on nonlinear problems -- measured on
            # the Minkowski blast, turning it off raised EOS floor hits from 5,439 to
            # 101,757 and left visible mottling through the outer shell.  Nonlinear
            # suites use the code default (comps).
            if linear and "mood_nad_v" in d:
                # YAML may parse bare `off` as False
                nad_v = d['mood_nad_v']
                if nad_v is False:
                    nad_v = 'off'
                args.append(f"mhd/mood_nad_v={nad_v}")
            if "mood_nad_b" in d:
                args.append(f"mhd/mood_nad_b={d['mood_nad_b']}")
    return args


def case_emf(case):
    """The emf a case asks for, or None to let the automatic choice stand."""
    return (case or {}).get("emf")


def mhd_use_uct(scheme_name, case=None, general_rel=False, special_rel=False):
    """UCT for all NR MHD schemes (plm/wenoz/teno/ppm_fb).

    SR/GR default to ct_contact but may opt in with `emf: uct_hll` on the case.  That
    used to be impossible: the relativistic solvers fed the UCT composition the spatial
    four-velocity instead of the coordinate transport velocity v^i = u^i/u^0, which
    drove the whole domain to the density floor.  Fixed in the four relativistic
    rsolvers; uct_hlld stays Newtonian-only because it requires rsolver=hlld.
    """
    del scheme_name  # all NR MHD schemes use UCT
    if case is not None:
        general_rel = bool(case.get("general_rel", False))
        special_rel = bool(case.get("special_rel", False))
        emf = case_emf(case)
        if emf:
            return emf != "ct_contact"
    return not (general_rel or special_rel)


def apply_athinput_overrides(src: Path, dst: Path, overrides):
    """Write a copy of src with block/key=value overrides applied in-file.

    AthenaK command-line overrides can only *modify* keys that already exist in
    the input file; they cannot introduce new keys such as hydro/mood.  So the
    suite materializes a per-run athinput that contains every override.
    """
    text = Path(src).read_text()
    lines = text.splitlines(keepends=True)
    blocks = {}
    order = []
    current = None
    preamble = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("<") and s.endswith(">") and not s.startswith("</"):
            current = s[1:-1].strip()
            if current not in blocks:
                blocks[current] = []
                order.append(current)
            continue
        if current is None:
            preamble.append(line)
        else:
            blocks[current].append(i)

    ovmap = {}
    delete_keys = {}  # block -> set(keys)  value "__DELETE__" removes a key
    for ov in overrides:
        if "=" not in ov or "/" not in ov.split("=", 1)[0]:
            raise ValueError(f"override must be block/key=value, got {ov!r}")
        left, value = ov.split("=", 1)
        block, key = left.split("/", 1)
        if value == "__DELETE__":
            delete_keys.setdefault(block, set()).add(key)
            continue
        ovmap.setdefault(block, {})[key] = value

    out_lines = list(preamble)
    written = set()
    for bname in order:
        out_lines.append(f"<{bname}>\n")
        written.add(bname)
        body_idxs = blocks.get(bname, [])
        seen_keys = set()
        for i in body_idxs:
            line = lines[i]
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith("<")
                and "=" in stripped
            ):
                before = stripped.split("=", 1)[0].strip()
                key = before.split()[0] if before else ""
                if key and bname in delete_keys and key in delete_keys[bname]:
                    seen_keys.add(key)
                    continue
                if key and bname in ovmap and key in ovmap[bname]:
                    comment = ""
                    if "#" in line:
                        comment = "  #" + line.split("#", 1)[1].rstrip("\n")
                    out_lines.append(f"{key} = {ovmap[bname][key]}{comment}\n")
                    seen_keys.add(key)
                    continue
            out_lines.append(line)
        if bname in ovmap:
            for key, value in ovmap[bname].items():
                if key not in seen_keys:
                    out_lines.append(f"{key} = {value}\n")
                    seen_keys.add(key)
        out_lines.append("\n")

    for bname, kvs in ovmap.items():
        if bname in written:
            continue
        out_lines.append(f"<{bname}>\n")
        for key, value in kvs.items():
            out_lines.append(f"{key} = {value}\n")
        out_lines.append("\n")

    Path(dst).write_text("".join(out_lines))


def run_one(athena, input_file, outdir, basename, overrides, dry_run=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_input = outdir / "run.athinput"
    all_ov = list(overrides) + [f"job/basename={basename}"]
    apply_athinput_overrides(Path(input_file), run_input, all_ov)
    cmd = [str(athena), "-i", str(run_input), "-d", str(outdir)]
    meta = {
        "command": cmd,
        "overrides": all_ov,
        "base_input": str(input_file),
        "cwd": str(ROOT),
        "commit": git_rev(),
        "host": platform.node(),
        "platform": platform.platform(),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "provenance.json").write_text(json.dumps(meta, indent=2))
    if dry_run:
        print("DRY:", " ".join(cmd))
        return 0, 0.0
    log = outdir / "run.log"
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta["wall_s"] = wall
    meta["returncode"] = proc.returncode
    (outdir / "provenance.json").write_text(json.dumps(meta, indent=2))
    return proc.returncode, wall


def parse_nmood(log_path: Path) -> int:
    """Sum mood demotions from eventlog if present; else 0."""
    total = 0
    for p in Path(log_path).parent.glob("*.log"):
        if p.name == "run.log":
            continue
        try:
            lines = p.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # cycle ... fofc mood  (mood is last column when present)
            if len(parts) >= 9:
                try:
                    total += int(parts[-1])
                except ValueError:
                    pass
    return total


def parse_l1_from_errs(run_dir: Path):
    errs = list(run_dir.glob("*-errs.dat"))
    if not errs:
        return None
    for line in errs[0].read_text().splitlines()[::-1]:
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split()
        if len(cols) >= 5:
            return float(cols[4])
    return None


def _waves_for_dim(case, dim):
    key = f"waves_{dim}"
    if key in case:
        return [str(w) for w in case[key]]
    # Back-compat with older manifests that used a single "waves" list
    return [str(w) for w in case.get("waves", ["0"])]


def _mesh_for_dim(dim_cfg, res):
    """Return (nx1,nx2,nx3, mb1,mb2,mb3) for a linear-wave dimension."""
    nx1 = int(res)
    f2 = float(dim_cfg.get("nx2_factor", 0) or 0)
    f3 = float(dim_cfg.get("nx3_factor", 0) or 0)
    nx2 = max(1, int(round(nx1 * f2))) if f2 > 0 else 1
    nx3 = max(1, int(round(nx1 * f3))) if f3 > 0 else 1
    # Keep MeshBlocks as single-block for local CPU runs
    return nx1, nx2, nx3, nx1, nx2, nx3


def run_linear(
    manif,
    athena,
    dry_run,
    schemes,
    case_ids=None,
    dims_filter=None,
    mood_rtol=None,
    mood_atol=None,
    force=False,
):
    summary = []
    dims = manif["linear_waves"].get("dimensions") or {
        "1d": {
            "resolutions": manif["linear_waves"].get("resolutions", [32, 64, 128, 256]),
            "along_x1": True,
            "nx2_factor": 0,
            "nx3_factor": 0,
        }
    }
    for case in manif["linear_waves"]["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        soe = case["soe"]
        physics = case.get("physics") or case["id"]
        for dim, dim_cfg in dims.items():
            if dims_filter and dim not in dims_filter:
                continue
            waves = _waves_for_dim(case, dim)
            for scheme in schemes:
                for wv in waves:
                    for res in dim_cfg["resolutions"]:
                        # Prefer new dim-aware names; also accept legacy 1d dirs
                        name = f"{case['id']}_{scheme}_{dim}_w{wv}_N{res}"
                        outdir = RESULTS / "linear" / name
                        legacy = RESULTS / "linear" / f"{case['id']}_{scheme}_w{wv}_N{res}"
                        if force and outdir.exists():
                            shutil.rmtree(outdir)
                        if dim == "1d" and not force and not outdir.exists() and legacy.exists():
                            # Reuse prior 1D results (same physics)
                            if (legacy / "summary.json").exists() and not dry_run:
                                with open(legacy / "summary.json") as f:
                                    rec = json.load(f)
                                rec["dim"] = "1d"
                                rec["physics"] = physics
                                rec["outdir"] = str(legacy)
                                summary.append(rec)
                                print(f"reuse legacy {legacy.name}")
                                continue
                        if (outdir / "summary.json").exists() and not dry_run:
                            with open(outdir / "summary.json") as f:
                                previous = json.load(f)
                            if previous.get("returncode") == 0:
                                print(f"skip existing {name}")
                                summary.append(previous)
                                continue
                            print(f"rerun incomplete {name}")
                            shutil.rmtree(outdir)
                        ov = scheme_overrides(
                            manif, scheme, soe,
                            use_uct=(soe == "mhd" and mhd_use_uct(scheme, case)),
                            linear=True,
                        )
                        if scheme == "ppm_fb" and mood_rtol is not None:
                            ov.append(f"{soe}/mood_rtol={mood_rtol:.17g}")
                        if scheme == "ppm_fb" and mood_atol is not None:
                            ov.append(f"{soe}/mood_atol={mood_atol:.17g}")
                        # Newtonian zero-speed characteristic families need a nonzero
                        # background advection speed for a finite wave period. Hydro
                        # flags 1--3 are entropy/transverse-shear modes; MHD flag 3 is
                        # the entropy/contact mode.
                        advected = (
                            (soe == "hydro" and str(wv) in ("1", "2", "3"))
                            or (soe == "mhd" and str(wv) == "3")
                        )
                        vx0 = "1.0" if advected else "0.0"
                        if case["general_rel"] or case["special_rel"]:
                            vx0 = None
                        nx1, nx2, nx3, mb1, mb2, mb3 = _mesh_for_dim(dim_cfg, res)
                        along = "true" if dim_cfg.get("along_x1", dim == "1d") else "false"
                        rsolver = (
                            manif["defaults"]["gr_rsolver"]
                            if case["general_rel"]
                            else (
                                "hlld"
                                if (soe == "mhd" and mhd_use_uct(scheme, case))
                                else (
                                    manif["defaults"]["mhd_rsolver"]
                                    if soe == "mhd"
                                    else manif["defaults"]["hydro_rsolver"]
                                )
                            )
                        )
                        ov += [
                            f"time/cfl_number={manif['defaults']['cfl']}",
                            f"time/tlim={manif['linear_waves']['tlim_periods']}",
                            "time/ndiag=1000",
                            f"mesh/nx1={nx1}",
                            f"mesh/nx2={nx2}",
                            f"mesh/nx3={nx3}",
                            f"meshblock/nx1={mb1}",
                            f"meshblock/nx2={mb2}",
                            f"meshblock/nx3={mb3}",
                            "mesh_refinement/refinement=none",
                            f"{soe}/rsolver={rsolver}",
                            f"problem/along_x1={along}",
                            "problem/along_x2=false",
                            "problem/along_x3=false",
                            f"problem/amp={manif['linear_waves']['amp']}",
                            f"problem/wave_flag={wv}",
                            f"coord/special_rel={'true' if case['special_rel'] else 'false'}",
                            f"coord/general_rel={'true' if case['general_rel'] else 'false'}",
                            "output1/file_type=log",
                            "output1/dcycle=1",
                        ]
                        if vx0 is not None:
                            ov.append(f"problem/vx0={vx0}")
                        rc, wall = run_one(
                            athena,
                            ROOT / case["input"],
                            outdir,
                            name,
                            ov,
                            dry_run=dry_run,
                        )
                        rec = {
                            "suite": "linear",
                            "case": case["id"],
                            "physics": physics,
                            "dim": dim,
                            "scheme": scheme,
                            "wave": wv,
                            "nx1": res,
                            "returncode": rc,
                            "wall_s": wall,
                            "l1_rms": None if dry_run else parse_l1_from_errs(outdir),
                            "nmood_total": None if dry_run else parse_nmood(outdir / "run.log"),
                            "mood_rtol": (
                                mood_rtol
                                if scheme == "ppm_fb" and mood_rtol is not None
                                else manif["defaults"]["mood_rtol"]
                            ),
                            "mood_atol": (
                                mood_atol
                                if scheme == "ppm_fb" and mood_atol is not None
                                else manif["defaults"].get("mood_atol", 0.0)
                            ),
                            "outdir": str(outdir),
                        }
                        if not dry_run:
                            (outdir / "summary.json").write_text(json.dumps(rec, indent=2))
                        summary.append(rec)
                        print(
                            f"{name}: rc={rc} L1={rec['l1_rms']} "
                            f"nmood={rec['nmood_total']} ({wall:.1f}s)"
                        )
    return summary


def run_shocks(manif, athena, dry_run, schemes):
    summary = []
    for case in manif["shocks"]["cases"]:
        soe = case["soe"]
        for scheme in schemes:
            for res in manif["shocks"]["resolutions"]:
                name = f"{case['id']}_{scheme}_N{res}"
                outdir = RESULTS / "shocks" / name
                if (outdir / "summary.json").exists() and not dry_run:
                    print(f"skip existing {name}")
                    with open(outdir / "summary.json") as f:
                        summary.append(json.load(f))
                    continue
                ov = scheme_overrides(
                    manif, scheme, soe,
                    use_uct=(soe == "mhd" and mhd_use_uct(scheme, case)),
                )
                rsolver = case.get("rsolver")
                if rsolver is None:
                    rsolver = (
                        manif["defaults"]["gr_rsolver"]
                        if case["general_rel"]
                        else (
                            manif["defaults"]["mhd_rsolver"]
                            if soe == "mhd"
                            else manif["defaults"]["hydro_rsolver"]
                        )
                    )
                # UCT-HLLD requires hlld
                if soe == "mhd" and mhd_use_uct(scheme, case):
                    rsolver = "hlld"
                ov += [
                    f"time/cfl_number={manif['defaults']['cfl']}",
                    f"time/tlim={case['tlim']}",
                    "time/nlim=-1",
                    "time/ndiag=100",
                    f"mesh/nx1={res}",
                    "mesh/nx2=1",
                    "mesh/nx3=1",
                    f"meshblock/nx1={min(res, 256)}",
                    "meshblock/nx2=1",
                    "meshblock/nx3=1",
                    "mesh_refinement/refinement=none",
                    f"{soe}/rsolver={rsolver}",
                    f"coord/special_rel={'true' if case['special_rel'] else 'false'}",
                    f"coord/general_rel={'true' if case['general_rel'] else 'false'}",
                    "output1/file_type=tab",
                    f"output1/variable={soe}_w",
                    # Dump IC and final state (dt = tlim → one dump at tlim)
                    f"output1/dt={case['tlim']}",
                    # Do NOT touch slice_x2/slice_x3 here.  Each shock deck already
                    # ships the right value for its own domain (0.0 for the
                    # [-0.5,0.5] boxes, 0.5 for shu_osher's [0,1]).  Overriding to a
                    # hard-coded 0.5 put the slice ON the boundary for five of six
                    # decks and aborted the whole suite; DELETING the keys instead
                    # makes AthenaK's tab writer emit a 12-column header over
                    # 8 columns of data, which athena_read then cannot parse.
                    "output2/file_type=log",
                    "output2/dcycle=1",
                ]
                rc, wall = run_one(
                    athena, ROOT / case["input"], outdir, name, ov, dry_run=dry_run
                )
                rec = {
                    "suite": "shocks",
                    "case": case["id"],
                    "scheme": scheme,
                    "nx1": res,
                    "returncode": rc,
                    "wall_s": wall,
                    "nmood_total": None if dry_run else parse_nmood(outdir / "run.log"),
                    "outdir": str(outdir),
                }
                if not dry_run:
                    (outdir / "summary.json").write_text(json.dumps(rec, indent=2))
                summary.append(rec)
                print(f"{name}: rc={rc} nmood={rec['nmood_total']} ({wall:.1f}s)")
    return summary


def parse_health(outdir: Path) -> dict:
    """Physical-health gate for a completed run.

    `returncode == 0` is NOT evidence of success: a run whose state has been wiped
    can still print "Terminating on time limit".  This collects the evidence that
    actually discriminates -- EOS floor/failure counts from the eventlog, and the
    timestep history from run.log (a collapse to ~0 followed by a jump is the
    signature of a destroyed state).
    """
    out = {
        "eos_dfloor": 0, "eos_efloor": 0, "eos_tfloor": 0,
        "eos_vceil": 0, "eos_fail": 0,
        "dt_min": None, "dt_max": None, "dt_collapse": False,
        "healthy": None,
    }
    outdir = Path(outdir)

    # --- EOS floor / failure counters from the eventlog -------------------------------
    for p_ in outdir.glob("*.log"):
        if p_.name == "run.log":
            continue
        try:
            lines = p_.read_text().splitlines()
        except OSError:
            continue
        cols = None
        for line in lines:
            if line.startswith("#"):
                cols = line.lstrip("#").split()
                continue
            if not line.strip() or cols is None:
                continue
            parts = line.split()
            if len(parts) != len(cols):
                continue
            for name, val in zip(cols, parts):
                if name in out:
                    try:
                        out[name] += int(val)
                    except ValueError:
                        pass

    # --- timestep history -------------------------------------------------------------
    dts = []
    log = outdir / "run.log"
    if log.exists():
        for line in log.read_text().splitlines():
            m = re.search(r"dt=([0-9.eE+-]+)", line)
            if m:
                try:
                    dts.append(float(m.group(1)))
                except ValueError:
                    pass
    if dts:
        out["dt_min"] = min(dts)
        out["dt_max"] = max(dts)
        # a healthy run varies dt by O(10); a wiped state drops it by many decades
        out["dt_collapse"] = bool(out["dt_min"] <= 0.0 or
                                  out["dt_min"] < 1.0e-6 * out["dt_max"])

    # --- is the FINAL state actually finite? ----------------------------------------
    # dt_collapse and eos_fail do not catch everything: a run can carry NaN cells to
    # the end without the timestep ever collapsing, and it still returns 0.  Check the
    # last dump's actual field payloads (big-endian float32 after each SCALARS block).
    out["final_finite"] = None
    try:
        import numpy as _np
        dumps = sorted(outdir.rglob("*.vtk"))
        if dumps:
            raw = dumps[-1].read_bytes()
            finite = True
            pos = 0
            while True:
                i = raw.find(b"SCALARS ", pos)
                if i < 0:
                    break
                k = raw.find(b"LOOKUP_TABLE default\n", i)
                if k < 0:
                    break
                start = k + len(b"LOOKUP_TABLE default\n")
                nxt = raw.find(b"SCALARS ", start)
                end = nxt if nxt > 0 else len(raw)
                n = (end - start) // 4
                if n:
                    arr = _np.frombuffer(raw[start:start + 4 * n], dtype=">f4")
                    if not _np.isfinite(arr).all():
                        finite = False
                        break
                pos = start + 4 * n
            out["final_finite"] = finite
    except Exception:
        pass

    # --- is the final state PLAUSIBLE, not merely finite? --------------------------
    # Finiteness is not enough either.  Measured on the beta=5e-11 MHD jet, a
    # mood_max_revs=3 run finished "finite" with rho_max = 1.2e12 and |By|/B0 = 7e19
    # -- eight and fifteen orders of magnitude off the reference -- and would have
    # passed a finite-only gate.  Flag a final state whose density spans an
    # implausible dynamic range for a shock problem.
    out["dyn_range"] = None
    out["implausible"] = None
    try:
        import numpy as _np
        dumps = sorted(outdir.rglob("*.vtk"))
        if dumps:
            raw = dumps[-1].read_bytes()
            i = raw.find(b"SCALARS dens")
            if i >= 0:
                k = raw.find(b"LOOKUP_TABLE default\n", i)
                start = k + len(b"LOOKUP_TABLE default\n")
                nxt = raw.find(b"SCALARS ", start)
                end = nxt if nxt > 0 else len(raw)
                n = (end - start) // 4
                a = _np.frombuffer(raw[start:start + 4 * n], dtype=">f4")
                a = a[_np.isfinite(a) & (a > 0)]
                if a.size:
                    rng = float(a.max() / a.min())
                    out["dyn_range"] = rng
                    out["implausible"] = bool(rng > 1.0e8)
    except Exception:
        pass

    floors = sum(out[k] for k in
                 ("eos_dfloor", "eos_efloor", "eos_tfloor", "eos_vceil", "eos_fail"))
    out["floors_total"] = floors
    out["healthy"] = ((not out["dt_collapse"])
                      and out["eos_fail"] == 0
                      and out["final_finite"] is not False
                      and out["implausible"] is not True)
    return out


def run_stress_smoke(manif, athena, dry_run, schemes, case_ids=None, force=False):
    summary = []
    default_athena = Path(athena)
    for case in manif["stress"]["local_smoke"]:
        if case_ids and case["id"] not in case_ids:
            continue
        soe = case["soe"]
        physics = case.get("physics") or case["id"]
        dim = case.get("dim") or "2d"
        case_athena = Path(ROOT / case["binary"]) if case.get("binary") else default_athena
        if not case_athena.is_absolute():
            case_athena = ROOT / case_athena
        case_schemes = case.get("schemes") or schemes
        for scheme in case_schemes:
            name = f"{case['id']}_{scheme}_smoke"
            outdir = RESULTS / "stress" / name
            if force and outdir.exists():
                shutil.rmtree(outdir)
            if (outdir / "summary.json").exists() and not dry_run:
                print(f"skip existing {name}")
                with open(outdir / "summary.json") as f:
                    summary.append(json.load(f))
                continue
            nx1 = int(case.get("nx1", case.get("nx", 64)))
            nx2 = int(case.get("nx2", case.get("nx", nx1)))
            ov = scheme_overrides(
                manif, scheme, soe,
                use_uct=(soe == "mhd" and mhd_use_uct(scheme, case)),
                emf_name=case_emf(case),
            )
            use_uct = soe == "mhd" and mhd_use_uct(scheme, case)
            # Riemann solver.  The relativistic test comes FIRST and is not conditioned
            # on UCT: GR/SR MHD must never get the Newtonian hlld (rejected outright,
            # "rsolver 'hlld' not implemented for GR dynamics"), and a relativistic case
            # may now opt into emf=uct_hll, which would otherwise select hlld here.
            if soe == "mhd":
                if case.get("general_rel") or case.get("special_rel"):
                    rsolver = manif["defaults"]["gr_rsolver"]
                elif use_uct:
                    rsolver = "hlld"
                else:
                    rsolver = manif["defaults"]["mhd_rsolver"]
            else:
                rsolver = manif["defaults"]["hydro_rsolver"]
            cfl = case.get("cfl", manif["defaults"]["cfl"])
            # Generous cycle cap rather than -1.  A scheme that collapses its
            # timestep otherwise spins for the whole wall clock and never records a
            # result: measured, unprotected TENO on the beta=0.02 blast froze at
            # dt=0 and reached cycle 823,400 at t=0.169 of 0.2.  The cap is far
            # above any healthy run here (the most is ~14k cycles, current_sheet at
            # 1024^2), so it truncates only runs that have already failed -- and the
            # health gate flags them either way.
            nlim = case.get("nlim", 200000)
            ov += [
                f"time/cfl_number={cfl}",
                f"time/tlim={case['tlim']}",
                "time/ndiag=100",
                f"mesh/nx1={nx1}",
                f"mesh/nx2={nx2}",
                "mesh/nx3=1",
                f"meshblock/nx1={nx1}",
                f"meshblock/nx2={nx2}",
                "meshblock/nx3=1",
                "mesh_refinement/refinement=none",
                f"{soe}/rsolver={rsolver}",
                "output1/file_type=vtk",
                f"output1/variable={soe}_w" if soe == "hydro" else "output1/variable=mhd_w_bcc",
                # A case may ask for intermediate dumps (output_dt) when the
                # comparison is a time series rather than a single final frame --
                # e.g. the RR22 Kelvin-Helmholtz, which the paper shows at
                # t = 5, 8, 12, 20.  Default stays "first and last only".
                f"output1/dt={case.get('output_dt', case['tlim'])}",
                # Drop any inherited 1D slice keys so 2D dumps stay 2D
                "output1/slice_x1=__DELETE__",
                "output1/slice_x2=__DELETE__",
                "output1/slice_x3=__DELETE__",
                "output2/file_type=log",
                "output2/dcycle=1",
            ]
            for extra in case.get("overrides", []):
                ov.append(extra)
            for extra in (case.get("scheme_overrides") or {}).get(scheme, []):
                ov.append(extra)
            # Cycle cap LAST.  It has to come after the case overrides, which carry
            # time/nlim=-1: appended before them it is simply overwritten, and a
            # collapsed run then spins for the whole wall clock.  Measured the wrong
            # way round -- unprotected TENO on the beta=0.02 blast reached cycle
            # 4,179,900 and timed out a 2 h job with the cap "applied".
            ov.append(f"time/nlim={nlim}")
            if case["id"] == "blast_grmhd_minkowski":
                ov += [
                    "coord/general_rel=true",
                    "coord/minkowski=true",
                    f"mhd/rsolver={manif['defaults']['gr_rsolver']}",
                ]
            if not case_athena.exists():
                raise FileNotFoundError(
                    f"Athena binary for {case['id']} not found: {case_athena}\n"
                    "Run validation/fallback/scripts/build_2d_problem_binaries.sh"
                )
            rc, wall = run_one(
                case_athena, ROOT / case["input"], outdir, name, ov, dry_run=dry_run
            )
            rec = {
                "suite": "stress_smoke",
                "case": case["id"],
                "physics": physics,
                "dim": dim,
                "scheme": scheme,
                "nx1": nx1,
                "nx2": nx2,
                "returncode": rc,
                "wall_s": wall,
                "nmood_total": None if dry_run else parse_nmood(outdir / "run.log"),
                "health": None if dry_run else parse_health(outdir),
                "outdir": str(outdir),
                "binary": str(case_athena),
            }
            if not dry_run:
                (outdir / "summary.json").write_text(json.dumps(rec, indent=2))
            summary.append(rec)
            print(f"{name}: rc={rc} nmood={rec['nmood_total']} ({wall:.1f}s)")
    return summary


def run_tolerance(manif, athena, dry_run, case_ids=None, force=False):
    """Sweep relaxed-DMP rtol for each configured 2D stress problem."""
    raw = manif.get("tolerance") or {}
    cases = raw.get("cases")
    if cases is None:
        cases = [raw] if raw.get("id") else []
    cases = list(cases)
    # `include_stress` sweeps rtol on stress cases without restating their setup:
    # the config is taken from stress.local_smoke so grid, tlim, solver and
    # overrides cannot drift between the two suites.
    inc = raw.get("include_stress") or []
    if inc:
        stress_by_id = {c["id"]: c for c in
                        (manif.get("stress") or {}).get("local_smoke") or []}
        have = {c["id"] for c in cases}
        for cid in inc:
            if cid in have or cid not in stress_by_id:
                continue
            src = dict(stress_by_id[cid])
            src["rtols"] = raw.get("default_rtols", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
            cases.append(src)
    default_athena = Path(athena)
    summary = []
    for cfg in cases:
        if case_ids and cfg["id"] not in case_ids:
            continue
        soe = cfg["soe"]
        nx1 = int(cfg.get("nx1", cfg.get("nx", 200)))
        nx2 = int(cfg.get("nx2", cfg.get("nx", nx1)))
        case_athena = Path(ROOT / cfg["binary"]) if cfg.get("binary") else default_athena
        if not case_athena.is_absolute():
            case_athena = ROOT / case_athena
        if not dry_run and not case_athena.exists():
            raise FileNotFoundError(
                f"Athena binary for {cfg['id']} not found: {case_athena}"
            )
        use_uct = soe == "mhd" and mhd_use_uct("ppm_fb", cfg)
        # Relativistic FIRST, and not conditioned on UCT -- same rule as
        # run_stress_smoke: GR/SR MHD must never get the Newtonian hlld.
        if soe == "mhd":
            if cfg.get("general_rel") or cfg.get("special_rel"):
                rsolver = manif["defaults"]["gr_rsolver"]
            elif use_uct:
                rsolver = "hlld"
            else:
                rsolver = manif["defaults"]["mhd_rsolver"]
        else:
            rsolver = manif["defaults"]["hydro_rsolver"]
        vtk_var = f"{soe}_w" if soe == "hydro" else "mhd_w_bcc"
        for rtol in cfg["rtols"]:
            rtol = float(rtol)
            tag = f"{rtol:.0e}"
            name = f"{cfg['id']}_ppm_fb_rtol_{tag}"
            outdir = RESULTS / "tolerance" / name
            if force and outdir.exists():
                shutil.rmtree(outdir)
            if (outdir / "summary.json").exists() and not dry_run:
                print(f"skip existing {name}")
                summary.append(json.loads((outdir / "summary.json").read_text()))
                continue
            ov = scheme_overrides(manif, "ppm_fb", soe, use_uct=use_uct,
                                  emf_name=case_emf(cfg))
            ov += [
                f"{soe}/mood_rtol={rtol:.17g}",
                f"time/cfl_number={cfg.get('cfl', manif['defaults']['cfl'])}",
                f"time/tlim={cfg['tlim']}",
                "time/nlim=-1",
                "time/ndiag=100",
                f"mesh/nx1={nx1}",
                f"mesh/nx2={nx2}",
                "mesh/nx3=1",
                f"meshblock/nx1={nx1}",
                f"meshblock/nx2={nx2}",
                "meshblock/nx3=1",
                "mesh_refinement/refinement=none",
                f"{soe}/rsolver={rsolver}",
                "output1/file_type=vtk",
                f"output1/variable={vtk_var}",
                f"output1/dt={cfg['tlim']}",
                "output1/slice_x1=__DELETE__",
                "output1/slice_x2=__DELETE__",
                "output1/slice_x3=__DELETE__",
                "output2/file_type=log",
                "output2/dcycle=1",
            ]
            ov.extend(cfg.get("overrides", []))
            # Cap the cycle count LAST, after the case overrides.  A stress case
            # carries time/nlim=-1, and at a loose tolerance the fallback
            # under-protects badly enough that the timestep collapses -- which is a
            # result worth recording, but uncapped it grinds for the whole wall
            # clock (observed: jet at rtol=1e-2 reached cycle 519,300 and t=15.9 of
            # 200).  The health gate flags the collapse either way.
            ov.append(f"time/nlim={cfg.get('tol_nlim', 120000)}")
            rc, wall = run_one(
                case_athena, ROOT / cfg["input"], outdir, name, ov, dry_run=dry_run
            )
            rec = {
                "suite": "tolerance",
                "case": cfg["id"],
                "physics": cfg.get("physics", soe),
                "dim": cfg.get("dim", "2d"),
                "scheme": "ppm_fb",
                "rtol": rtol,
                "nx1": nx1,
                "nx2": nx2,
                "returncode": rc,
                "wall_s": wall,
                "nmood_total": None if dry_run else parse_nmood(outdir / "run.log"),
                "outdir": str(outdir),
                "plot_field": cfg.get("plot_field", "dens"),
            }
            if not dry_run:
                (outdir / "summary.json").write_text(json.dumps(rec, indent=2))
            summary.append(rec)
            print(
                f"{name}: rc={rc} nmood={rec['nmood_total']} "
                f"({wall:.1f}s)"
            )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athena", required=True, help="Path to athena binary")
    ap.add_argument(
        "--suite",
        choices=["linear", "shocks", "stress", "tolerance", "all"],
        default="all",
    )
    ap.add_argument("--schemes", default="ppm_fb,plm,wenoz,teno")
    ap.add_argument(
        "--cases",
        default="",
        help="Comma-separated case ids for linear/stress/tolerance (default: all)",
    )
    ap.add_argument(
        "--dims",
        default="",
        help="Comma-separated linear-wave dimensions (default: all)",
    )
    ap.add_argument("--mood-rtol", type=float, default=None)
    ap.add_argument("--mood-atol", type=float, default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Delete and rerun selected linear/stress/tolerance outputs",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    athena = Path(args.athena)
    if not args.dry_run and not athena.exists():
        raise SystemExit(f"athena binary not found: {athena}")
    manif = load_manifest()
    schemes = [s.strip() for s in args.schemes.split(",")]
    case_ids = {s.strip() for s in args.cases.split(",") if s.strip()}
    dims_filter = {s.strip() for s in args.dims.split(",") if s.strip()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_sum = []
    if args.suite in ("linear", "all"):
        all_sum += run_linear(
            manif,
            athena,
            args.dry_run,
            schemes,
            case_ids=case_ids,
            dims_filter=dims_filter,
            mood_rtol=args.mood_rtol,
            mood_atol=args.mood_atol,
            force=args.force,
        )
    if args.suite in ("shocks", "all"):
        all_sum += run_shocks(manif, athena, args.dry_run, schemes)
    if args.suite in ("stress", "all"):
        all_sum += run_stress_smoke(
            manif,
            athena,
            args.dry_run,
            schemes,
            case_ids=case_ids,
            force=args.force,
        )
    if args.suite in ("tolerance", "all"):
        all_sum += run_tolerance(
            manif, athena, args.dry_run, case_ids=case_ids, force=args.force
        )
    out = RESULTS / f"summary_{args.suite}.json"
    out.write_text(json.dumps(all_sum, indent=2))
    print(f"Wrote {out} ({len(all_sum)} records)")


if __name__ == "__main__":
    main()
