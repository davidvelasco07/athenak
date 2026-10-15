"""Run fixed-parameter Orszag-Tang robustness tests with frozen MOOD on a GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np


def input_deck(order, nx, block, tlim, solver='hlld', frozen=True, nghost=None, revisions=2, rtol=1.e-4):
    ng=max(order,revisions+3) if frozen else (order//2+1)*(revisions+1)
    if nghost is not None:ng=nghost
    sections={
      'job':{'basename':'ot'},
      'mesh':{'nghost':ng},'meshblock':{},
      'time':{'evolution':'dynamic','integrator':'rk3','cfl_number':0.25,
              'nlim':-1,'tlim':tlim,'ndiag':10},
      'mhd':{'eos':'ideal','reconstruct':f'fv{order}','rsolver':solver,
             'emf':'uct_hlld' if solver=='hlld' else 'uct_hll','gamma':5/3,
             'mood':'true','mood_detect':'true','mood_frozen_emf':str(frozen).lower(),
             'mood_max_revs':revisions,'mood_nad_scale':'rcfl','mood_rtol':rtol,
             'mood_atol':0.,'mood_eps0':1.e-12,'mood_sed':'true',
             'mood_nad_energy':'true','mood_nad_v':'comps','mood_nad_b':'comps',
             'mood_nad_bcc':'avg','mood_edge':'flag','mood_touched_only':'false',
             'dfloor':1.e-12,'pfloor':1.e-12},
      'problem':{'pgen_name':'orszag_tang'},
      'output1':{'file_type':'log','dcycle':1},
      'output2':{'file_type':'hst','dt':0.01,'data_format':'%.17e'}}
    for dim in (1,2,3):
        sections['mesh'].update({f'nx{dim}':nx if dim<3 else 1,
             f'x{dim}min':-.5,f'x{dim}max':.5,
             f'ix{dim}_bc':'periodic',f'ox{dim}_bc':'periodic'})
        sections['meshblock'][f'nx{dim}']=min(block,nx) if dim<3 else 1
    for i,var in enumerate(('mhd_u_bcc','mhd_w','mhd_fb_level','mhd_divb'),3):
        sections[f'output{i}']={'file_type':'bin','variable':var,'dt':.25}
    return ''.join(f'<{b}>\n'+''.join(f'{k} = {v}\n' for k,v in vals.items())+'\n'
                   for b,vals in sections.items())


def assemble(path, reader):
    d=reader.read_binary(str(path))
    sx,sy,sz=d['nx1_out_mb'],d['nx2_out_mb'],d['nx3_out_mb']
    arrays={}
    for v in d['var_names']:
        a=np.empty((d['Nx3'],d['Nx2'],d['Nx1']),dtype=np.float64)
        for m,loc in enumerate(d['mb_logical']):
            x,y,z=map(int,loc[:3])
            a[z*sz:(z+1)*sz,y*sy:(y+1)*sy,x*sx:(x+1)*sx]=d['mb_data'][v][m]
        arrays[v]=a
    return d,arrays


def analyze(run, nx, tlim, reader):
    result={'snapshots':[]}
    stdout=(run/'stdout.txt').read_text(errors='replace')
    times=re.findall(r'^time=(\S+) cycle=(\d+)',stdout,re.M)
    if times:result.update(final_time_stdout=float(times[-1][0]),cycles=int(times[-1][1]))
    diagnostics=re.findall(r'cycle=(\d+) time=(\S+) dt=(\S+)',stdout)
    if diagnostics:
        result['last_diagnostic_time']=float(diagnostics[-1][1])
        result['dt_min']=min(float(v[2]) for v in diagnostics)
        if not times:result['cycles']=int(diagnostics[-1][0])
    timer=re.search(r'cpu time used\s*=\s*(\S+)',stdout)
    if timer:result['solver_seconds']=float(timer[1])
    logfile=run/'ot.log'
    if logfile.exists():
        rows=np.loadtxt(logfile,ndmin=2)
        if rows.size:
            for i,key in [(1,'density_floors'),(2,'energy_floors'),(3,'temperature_floors'),
                          (4,'velocity_ceilings'),(5,'c2p_failures'),(7,'pad_rejections'),(8,'demotions')]:
                result[key]=int(rows[:,i].sum())
            result['cycles']=max(result.get('cycles',0),int(rows[:,0].max()))
            result['cycles_with_demotion']=int(np.count_nonzero(rows[:,8]))
            result['first_floor_cycle']=next((int(r[0]) for r in rows if r[1:6].sum()>0),None)
            if result['first_floor_cycle'] is not None and diagnostics:
                c=result['first_floor_cycle']
                before=[float(v[1]) for v in diagnostics if int(v[0])<=c]
                after=[float(v[1]) for v in diagnostics if int(v[0])>c]
                result['first_floor_time_interval']=[max(before,default=0.),min(after,default=None)]
            if result.get('cycles',0):
                result['demotions_per_cell_stage']=result['demotions']/(3*nx*nx*result['cycles'])
    history=next(run.glob('*.hst'),None)
    if history:
        lines=history.read_text().splitlines()
        names=re.findall(r'\[\d+\]=(\S+)',next(s for s in lines if '[1]=' in s))
        h=np.loadtxt(history,ndmin=2)
        result['history_finite']=bool(np.isfinite(h).all())
        result['history_last_time']=float(h[-1,0])
        result['conservation_max_drift']={}
        for name in ('mass','1-mom','2-mom','3-mom','tot-E'):
            i=names.index(name)
            # Relative drift for nonzero mass/energy; absolute drift for zero momenta.
            denom=abs(h[0,i]) if name in ('mass','tot-E') else 1.
            result['conservation_max_drift'][name]=float(np.max(abs(h[:,i]-h[0,i]))/denom)
        result['history']={'labels':names,'values':h.tolist()}
    groups={}
    result['snapshot_read_errors']={}
    for f in sorted((run/'bin').glob('*.bin')):
        try:d,a=assemble(f,reader)
        except Exception as exc:
            result['snapshot_read_errors'][f.name]=repr(exc);continue
        t=float(d['time']);s=groups.setdefault(round(t,10),{'time':t,'finite':True})
        s['finite']=s['finite'] and all(bool(np.isfinite(v).all()) for v in a.values())
        if '.mhd_w.' in f.name:
            s['density_min']=float(a['dens'].min());s['density_max']=float(a['dens'].max())
            s['pressure_min']=float((2/3)*a['eint'].min())
            s['pressure_max']=float((2/3)*a['eint'].max())
        elif '.mhd_fb_level.' in f.name:
            levels=next(iter(a.values()))
            s['tier_fractions']={str(i):float(np.count_nonzero(levels==i)/levels.size) for i in (0,1,2)}
        elif '.mhd_divb.' in f.name:
            s['max_abs_divb']=max(float(np.max(abs(v))) for v in a.values())
            s['mean_abs_divb']=max(float(np.mean(abs(v))) for v in a.values())
    result['snapshots']=[groups[t] for t in sorted(groups)]
    result['final_time']=max(groups,default=0.)
    result['reached_tlim']=abs(result['final_time']-tlim)<1.e-8
    result['finite']=bool(result.get('history_finite',False) and not result['snapshot_read_errors'] and all(s['finite'] for s in groups.values()))
    result['positive']=bool(groups and all(s.get('density_min',-1)>0 and s.get('pressure_min',-1)>0 for s in groups.values()))
    result['clean']=bool(result.get('code') in (None,0) and result['reached_tlim'] and result['finite'] and result['positive'] and
                         all(result.get(k,0)==0 for k in ('density_floors','energy_floors','temperature_floors','velocity_ceilings','c2p_failures')))
    if groups:
        result['density_min']=min(s.get('density_min',float('inf')) for s in groups.values())
        result['pressure_min']=min(s.get('pressure_min',float('inf')) for s in groups.values())
        result['max_abs_divb']=max(s.get('max_abs_divb',0.) for s in groups.values())
    return result


def execute(exe,run,timeout,continue_after_floor=False):
    reason=None
    with (run/'stdout.txt').open('w') as out:
        proc=subprocess.Popen([str(exe),'-i','input.athinput'],cwd=run,
                              stdout=out,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while proc.poll() is None:
            try:proc.wait(timeout=1.)
            except subprocess.TimeoutExpired:pass
            if proc.poll() is not None:break
            if time.monotonic()-start>timeout:reason='wall_timeout'
            log=run/'ot.log'
            if not continue_after_floor and log.exists():
                for line in log.read_text().splitlines():
                    if line.startswith('#'):continue
                    values=line.split()
                    if len(values)==9 and any(int(v)>0 for v in values[1:6]):
                        reason='stopped_after_floor_or_conversion_failure';break
            if reason:
                proc.terminate()
                try:proc.wait(timeout=10.)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
    return proc.returncode,reason


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resolutions',default='64,128,256,512,1024')
    ap.add_argument('--orders',default='4,6,8');ap.add_argument('--block',type=int,default=64)
    ap.add_argument('--revisions',type=int,default=2)
    ap.add_argument('--rtol',type=float,default=1.e-4)
    ap.add_argument('--nghost',type=int,help='Override ghost count for a halo control')
    ap.add_argument('--tlim',type=float,default=1.);ap.add_argument('--timeout',type=float,default=7200.)
    ap.add_argument('--solver',choices=['hlld','hlle'],default='hlld')
    ap.add_argument('--continue-after-floor',action='store_true')
    ap.add_argument('--legacy',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();a.source=a.source.resolve();a.exe=a.exe.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.source/'vis/python'));import bin_convert as reader
    report={'host':platform.node(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
       'executable_sha256':hashlib.sha256(a.exe.read_bytes()).hexdigest(),
       'config':{'resolutions':a.resolutions,'orders':a.orders,'block':a.block,'nghost_override':a.nghost,'tlim':a.tlim,
                 'solver':a.solver,'frozen':not a.legacy,'cfl':.25,'nad':'rcfl','rtol':a.rtol,'max_revisions':a.revisions,'continue_after_floor':a.continue_after_floor},'runs':[]}
    for nx in map(int,a.resolutions.split(',')):
        for order in map(int,a.orders.split(',')):
            name=f'ot_fv{order}_n{nx}';d=a.output/name;d.mkdir(exist_ok=True)
            resultfile=d/'result.json'
            if a.resume and resultfile.exists():r=json.loads(resultfile.read_text())
            else:
                (d/'input.athinput').write_text(input_deck(order,nx,a.block,a.tlim,a.solver,not a.legacy,a.nghost,a.revisions,a.rtol))
                start=time.monotonic()
                code,reason=execute(a.exe,d,a.timeout,a.continue_after_floor)
                r={'name':name,'order':order,'nx':nx,'code':code,'stop_reason':reason,
                   'wall_seconds':time.monotonic()-start}
                try:r.update(analyze(d,nx,a.tlim,reader))
                except Exception as e:r.update(clean=False,analysis_error=repr(e))
                r['clean']=r.get('clean',False) and code==0
                r=finite_json(r)
                resultfile.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
            report['runs'].append(r)
            (a.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in r.items() if k not in ('history','snapshots')}),flush=True)
    print('STUDY_COMPLETE',sum(bool(r.get('clean')) for r in report['runs']),'/',len(report['runs']),flush=True)

if __name__=='__main__':main()
