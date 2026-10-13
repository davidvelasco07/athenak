#!/usr/bin/env python3
"""Generate hydrostatic well-balanced test decks.  usage: mkdeck.py out.athinput k=v ..."""
import sys
P = dict(basename='hse', nx1=64, nx2=64, mbx1=0, mbx2=0, x1min=0.0, x1max=1.0,
         x2min=0.0, x2max=1.0, ix1='user', ox1='user', ix2='periodic', ox2='periodic',
         ng=3, integrator='rk2', cfl=0.4, tlim=1.0, recon='plm', rsolver='hllc',
         wb='true', mood='false', fofc='false', gdir=1, gval=-1.0, g1=0.0, g2=0.0,
         fluid='hydro', emf='ct_contact', mood_edge='auto', nad_b='comps', b1=0.0, b2=0.0, b3=0.0, beta=1.0, iprob=1, amp=0.0, hst_dt=0.01, bin_dt=0.25, nlim=-1, extra='', tail='')
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1)
    P[k] = v
if P['mbx1'] in (0, '0'): P['mbx1'] = P['nx1']
if P['mbx2'] in (0, '0'): P['mbx2'] = P['nx2']
grav = (f"const_accel_dir = {P['gdir']}\nconst_accel_val = {P['gval']}" if int(P['gdir']) != 0
        else f"const_accel_dir = 0\nconst_accel_x1 = {P['g1']}\nconst_accel_x2 = {P['g2']}")
deck = f"""<job>
basename = {P['basename']}

<mesh>
nghost = {P['ng']}
nx1 = {P['nx1']}
x1min = {P['x1min']}
x1max = {P['x1max']}
ix1_bc = {P['ix1']}
ox1_bc = {P['ox1']}
nx2 = {P['nx2']}
x2min = {P['x2min']}
x2max = {P['x2max']}
ix2_bc = {P['ix2']}
ox2_bc = {P['ox2']}
nx3 = 1
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic

<meshblock>
nx1 = {P['mbx1']}
nx2 = {P['mbx2']}
nx3 = 1

<time>
evolution = dynamic
integrator = {P['integrator']}
cfl_number = {P['cfl']}
nlim = {P['nlim']}
tlim = {P['tlim']}
ndiag = 100

<{P['fluid']}>
eos = ideal
gamma = 1.6666666666666667
reconstruct = {P['recon']}
rsolver = {P['rsolver']}
well_balanced = {P['wb']}
mood = {P['mood']}
fofc = {P['fofc']}
{"emf = " + P['emf'] + "\nmood_edge = " + P['mood_edge'] + "\nmood_nad_b = " + P['nad_b'] if P['fluid'] == 'mhd' else ""}

<{P['fluid']}_srcterms>
const_accel = true
{grav}

<problem>
pgen_name = hydrostatic
iprob = {P['iprob']}
pulse_amp = {P['amp']}
b1 = {P['b1']}
b2 = {P['b2']}
b3 = {P['b3']}
beta = {P['beta']}
{P['extra']}

<output1>
file_type = hst
dt = {P['hst_dt']}

<output2>
file_type = bin
variable = {P['fluid']}_u
dt = {P['bin_dt']}

<output3>
file_type = bin
variable = {P['fluid']}_w
dt = {P['bin_dt']}
"""
# perturbation output (W - W_eq in double) for the exact gate; WB-off runs get the same
# equilibrium through wb_diagnostics
deck = deck.replace("well_balanced = false", "well_balanced = false\nwb_diagnostics = true")
deck += f"""
<output9>
file_type = bin
variable = {P['fluid']}_wpert
dt = {P['bin_dt']}
"""
deck += P['tail']
open(sys.argv[1], 'w').write(deck.replace('\\n', '\n'))
