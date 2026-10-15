"""Build the four-revision OT report from a saved result checkpoint."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def floor_interval(r):
    lo,hi=r.get('first_floor_time_interval',[None,None])
    if lo is None:return 'Not recorded'
    return f'{lo:.7f}–{hi:.7f}' if hi is not None else f'≥ {lo:.7f}'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    a=ap.parse_args();out=a.report
    d=json.loads((out/'results.json').read_text());meta=json.loads((out/'checkpoint.json').read_text())
    old=json.loads(a.baseline.read_text())['reports']['runs']
    config=d['config'];runs=d['runs']
    assert config['rtol']==1.e-8 and config['max_revisions']==4
    orders=list(map(int,config['orders'].split(',')));res=list(map(int,config['resolutions'].split(',')))
    by={(r['order'],r['nx']):r for r in runs};prior={(r['order'],r['nx']):r for r in old['runs']}
    active=set(meta.get('active_cases',[]));clean=[r for r in runs if r.get('clean')]
    floors=[r for r in runs if r.get('first_floor_cycle') is not None]
    stamp=datetime.fromisoformat(meta['captured_at_utc']).astimezone(ZoneInfo('America/New_York')).strftime('%B %d, %Y at %H:%M:%S %Z')
    expected=len(orders)*len(res)
    def status(order,n):
        r=by.get((order,n))
        if r is None:return 'Running' if f'ot_fv{order}_n{n}' in active else 'Pending'
        if r.get('clean'):return 'Clean to t = 1'
        if r.get('first_floor_cycle') is not None:return 'Energy floor' if r.get('energy_floors',0) else 'Floor or conversion failure'
        return 'Incomplete or failed'
    fig,ax=plt.subplots(figsize=(12,4.5),constrained_layout=True)
    palette={'Clean to t = 1':'#d7eadf','Energy floor':'#f4d7d7','Running':'#d7e7f7','Pending':'#eff1f3'}
    for j,o in enumerate(orders):
        for i,n in enumerate(res):
            s=status(o,n);r=by.get((o,n));label=s
            if r and r.get('first_floor_cycle') is not None:
                lo,hi=r.get('first_floor_time_interval',[None,None])
                label+='\n'+(f't ≈ {(lo+hi)/2:.3f}' if hi is not None else f't ≥ {lo:.3f}')
            ax.add_patch(Rectangle((i-.48,j-.45),.96,.9,facecolor=palette.get(s,'#f2e4cc'),edgecolor='white'))
            ax.text(i,j,label,ha='center',va='center',fontsize=9)
    ax.set(xlim=(-.5,len(res)-.5),ylim=(len(orders)-.5,-.5))
    ax.set_xticks(range(len(res)),[f'{n}²' for n in res]);ax.set_yticks(range(len(orders)),[f'FV{o}' for o in orders])
    ax.tick_params(length=0);plt.setp(ax.spines.values(),visible=False)
    ax.set_title(f'Orszag–Tang: four revisions, local rcfl tolerance 1e−8\n{len(runs)}/{expected} cases completed; checkpoint {stamp}',fontsize=12,pad=15)
    fig.savefig(out/'status.png',dpi=180);plt.close(fig)
    matrix=['| Resolution | FV4 | FV6 | FV8 |','|---|---|---|---|']
    for n in res:
        cells=[]
        for o in orders:
            r=by.get((o,n));s=status(o,n)
            if r and r.get('first_floor_cycle') is not None:s+=f' ({floor_interval(r)})'
            cells.append(s)
        matrix.append(f'| {n}² | '+' | '.join(cells)+' |')
    metrics=['| Case | Steps | First floor cycle | Energy floors | Last diagnostic time | Last saved field time |',
             '|---|---:|---:|---:|---:|---:|']
    rows=[]
    for r in runs:
        metrics.append(f"| FV{r['order']} {r['nx']}² | {r.get('cycles')} | {r.get('first_floor_cycle') or '—'} | {r.get('energy_floors',0)} | {r.get('final_time_stdout',r.get('last_diagnostic_time',0)):.7f} | {r.get('final_time',0):.7f} |")
        rows.append({k:r.get(k) for k in ('name','order','nx','clean','cycles','first_floor_cycle','energy_floors','density_floors','c2p_failures','density_min','pressure_min','max_abs_divb','demotions_per_cell_stage')})
        rows[-1].update(first_floor_time_interval=floor_interval(r) if r.get('first_floor_cycle') else '',
                       relative_mass_drift=r.get('conservation_max_drift',{}).get('mass'),
                       relative_energy_drift=r.get('conservation_max_drift',{}).get('tot-E'))
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    comparison=['| Case | Two revisions and 1e-4 | Four revisions and 1e-8 |','|---|---|---|']
    for r in runs:
        p=prior.get((r['order'],r['nx']))
        if p:
            previous='Clean to t = 1' if p['clean'] else f'Energy floor ({floor_interval(p)})'
            current=status(r['order'],r['nx'])
            if r.get('first_floor_cycle') is not None:current+=f' ({floor_interval(r)})'
            comparison.append(f"| FV{r['order']} {r['nx']}² | {previous} | {current} |")
    maxclean=lambda key:max((r.get('conservation_max_drift',{}).get(key,0) for r in clean),default=0)
    maxdiv=max((r.get('max_abs_divb',0)*np.sqrt(4*np.pi)/r['nx'] for r in runs),default=0)
    dc=max((s.get('tier_fractions',{}).get('2',0) for r in runs for s in r.get('snapshots',[])),default=0)
    recovered=[f"FV{r['order']} at {r['nx']}²" for r in clean if (r['order'],r['nx']) in prior and not prior[r['order'],r['nx']]['clean']]
    regressed=[f"FV{r['order']} at {r['nx']}²" for r in floors if (r['order'],r['nx']) in prior and prior[r['order'],r['nx']]['clean']]
    text=f'''# Orszag Tang robustness with four revisions and tolerance 1e-8

Draft checkpoint: **{stamp}**. **{len(runs)} of {expected} cases completed:** {len(clean)} clean to t = 1,
{len(floors)} with actual floor events, and {expected-len(runs)} awaiting completed results.
The ongoing sweep covers FV4/FV6/FV8 through 4096². EOS counters include ghost
cells. A separate diagnostic reproduction confirms that the FV4 512² events
occur in physical cells and expose a PAD inconsistency.

![Completed and pending cases](status.png)

## Configuration and scope

Newtonian ideal MHD, gamma = 5/3, stock Orszag–Tang initial data, periodic unit
square [-0.5, 0.5]². HLLD fluxes, UCT-HLLD EMFs, RK3 and CFL = 0.25.
The frozen path exchanges immutable high-order data before fallback revisions.
Four revisions use 7/7/8 ghost cells for FV4/FV6/FV8, with 64² meshblocks.
The local detector uses `mood_nad_scale=rcfl`, **`mood_rtol=1e-8`**,
`mood_atol=0`, `mood_eps0=1e-12` and SED enabled. NAD checks density, energy,
velocity components and magnetic components with a face-average magnetic reference.
`mood_edge=flag`, `mood_touched_only=false`; density and pressure floors are 1e-12.

Apollo GPU 1, NVIDIA A100-PCIE-40GB, single GPU without MPI. Numerical source base:
`903b6b9606a440ab2476aa49c17775e74e9ba7d9`. The executable is unchanged from the
earlier sweep; SHA256: `{d['executable_sha256']}`.
Evolution and history use double precision; saved fields use single precision.
History cadence is 0.01; fields are saved every 0.25 and at normal completion.

## Results at this checkpoint

{chr(10).join(matrix)}

Clean means reaching t = 1 with finite, positive saved states and zero actual EOS
floors or conversion failures. Energy-floor times above are diagnostic brackets
around the first affected cycle. Ordinary PLM/DC demotions and PAD rejections
are counted separately from actual floor corrections.

{chr(10).join(metrics)}

The harness stops an active case after detecting an actual floor or conversion
failure, polling every second. Return code -15 for these cases is an intentional
stop, not a solver crash. Floor totals depend on the delay before that stop;
first-floor cycle and time are better comparison measures. The last saved field
can precede the floor event, so positive snapshot minima do not establish positivity
through the stop time. Pending and running cases contribute no completed statistics.

## Comparison with the earlier sweep

{chr(10).join(comparison)}

Completed cases that now reach t = 1 after earlier floor events: {', '.join(recovered) or 'none'}.
Completed cases that now encounter floors after earlier clean results: {', '.join(regressed) or 'none'}.
Both revision count and tolerance changed, and FV4/FV6 require larger halos.
These comparisons do not isolate the effect of tolerance or revision count.
The interrupted four-revision 1e-4 sweep is a separate partial dataset and is
excluded from this report. This study measures robustness, not formal accuracy order.

## Confirmed detector inconsistency

For this Newtonian configuration, NAD tests density, internal energy density,
all three velocity components and all three magnetic components. Bounds use the
stage-input 3 by 3 neighborhood, with local rcfl relaxation and smooth-extrema
exemptions. NAD applies only at the base tier and demotes to PLM; PAD and nonfinite
checks can demote PLM to donor cell.

The code currently forms candidate magnetic fields by averaging the CT-updated
faces for PAD, while production primitive conversion uses the unlimited FV
face-to-cell interpolant. PAD and production can therefore subtract different
magnetic energies from the same total energy. This inconsistency needs to be
resolved before treating the current PAD check as validation of the accepted
state. In an instrumented FV4 512² reproduction, all 35 observed energy-floor
events are in physical cells still at the base tier. At the first event, the
same conserved state gives internal energy `+3.2027924e-3` with the face average
and `-1.0961833e-5` with the production FV4 interpolant. The original first-floor
cycle, 5263, is reproduced with zero PAD rejections. This is a physical failure,
not a ghost-counter artifact. See the [audit and raw evidence](pad_audit/README.md).

Other cases still require physical-cell classification: their aggregate EOS
counters include ghosts. The main sweep retains its original executable, so
these results describe the implementation with this defect. The numerical
correction has passed the [focused GPU checks](../pad_correction/README.md);
none of the sweep data above uses it.

The two-revision, 1e-4 FV4 512² control also passes with seven ghosts. All twenty
saved field datasets match the earlier five-ghost run bitwise. The halo increase
alone therefore does not explain the regression. The completed fixed-halo controls
are clean at 1e-4 with either two or four revisions. At 1e-8, two revisions count
their first floor at t = 0.8400127–0.8411955 and four at t = 0.8834502–0.8844739.
The tighter tolerance triggers the counted-floor regression for both revision
counts; four revisions delay onset at that tolerance. The audit also records a
separate source-level relativistic velocity NAD issue.

## Conservation and fallback diagnostics

Across the {len(clean)} clean completed cases, maximum relative mass drift is
{maxclean('mass'):.3e} and maximum relative total-energy drift is {maxclean('tot-E'):.3e}.
For all completed saved snapshots, `dx * max(abs(div B)) / B0` is at most
{maxdiv:.3e}, with `B0 = 1/sqrt(4*pi)`. The largest saved DC fraction is {dc:.3e};
sparse snapshots do not resolve transient DC use between outputs.
Failed cases' sampled histories can precede floor onset or include subsequent
floor corrections. Their drift is not a floor-free conservation measurement.

![Prefix diagnostics for completed cases](figures/robustness.png)

Demotion rates count events divided by `3 * N² * completed_steps`; both tier
transitions can count. These rates are not unique cell fractions, and failed
cases cover shorter intervals. Timing fields are not performance benchmarks:
the first case overlapped the end of the superseded sweep.

## Density and fallback at the final time

![Density at t equals 1](figures/t1/density.png)

![Fallback levels at t equals 1](figures/t1/fallback.png)

Only actual t = 1 states are shown. Grey panels distinguish absent final output
after a floor stop from cases without completed results. Earlier snapshots are
not substituted for final states. Fallback levels represent the last RK stage.

The [density](figures/density.png) and [fallback](figures/fallback.png) maps near
t = 0.5 provide an earlier common-time comparison; actual times are panel titles.

## Data and reproduction

- `results.json`: frozen checkpoint with histories, event counters and snapshot metrics.
- `checkpoint.json`: capture time, active cases and remote data location.
- `summary.csv`: metrics for completed cases only.
- `inputs/`: exact completed-case decks, logs, histories and result records.
- The [earlier report](../README.md) retains the two-revision study and controls.

From an Apollo environment with the modules in the earlier report:

```sh
python3 validation/frozen_emf_halo/ot_resolution/run_study.py --source . \\
  --exe ot-resolution/athena-ot --output ot-resolution/four-revision-rtol1e-8 \\
  --resolutions 64,128,256,512,1024,2048,4096 --orders 4,6,8 \\
  --revisions 4 --rtol 1e-8 --timeout 172800 --pre-pad-fix-halo --resume
```

Raw binary fields and incremental live results remain on Apollo at
`{meta['study_path']}`.
This report freezes the checkpoint above; later completions require a refresh.
The home filesystem hit its user quota while starting FV6 at 2048². Sixteen valid
completed records were preserved, the aggregate index was reconstructed, and the
remaining cases resumed on user-owned scratch. The interrupted case was retained
separately and is not classified as a numerical failure. The original home study
path remains a symlink; `quota_recovery.json` records the relocation. Result files
now use atomic replacement to preserve the last complete record after write errors.
'''
    (out/'README.md').write_text(text)
    print(f'REPORT_READY {len(runs)}/{expected}: {len(clean)} clean, {len(floors)} floor cases')


if __name__=='__main__':main()
