"""Frozen-candidate NR/SR/GR hydro/MHD correctness and fallback-active GPU timing."""
from pathlib import Path
import argparse, json, re, subprocess, time, sys, hashlib, statistics, os, platform
import numpy as np

CASES=[r+'_'+m for r in ('nr','sr','gr') for m in ('hydro','mhd')]
def deck(case,order,nx,block,steps,frozen=True,ng=None,outputs=True,dim=2):
    regime,mod=case.split('_');rel=regime!='nr';is_mhd=mod=='mhd'
    if ng is None:ng=max(order if is_mhd else order//2+1,5) if frozen else (2*order+2 if is_mhd else order//2+3)
    parts={'job':{'basename':'test'},'mesh':{'nghost':ng},'meshblock':{},
      'time':{'evolution':'dynamic','integrator':'rk3','cfl_number':.2,'nlim':steps,'tlim':100,'ndiag':steps},
      'coord':{'special_rel':str(regime=='sr').lower(),'general_rel':str(regime=='gr').lower(),'minkowski':'true'},
      mod:{'eos':'ideal','reconstruct':f'fv{order}','rsolver':'hlle','gamma':4/3 if rel else 5/3,
           'mood':'true','mood_detect':'true','mood_max_revs':2,'mood_nad_scale':'rcfl',
           'mood_rtol':1.e-4,'mood_sed':'true','dfloor':1.e-12,'pfloor':1.e-12,
           'mood_frozen_emf' if is_mhd else 'mood_frozen_flux':str(frozen).lower()},
      'problem':{'inner_radius':.8,'outer_radius':1.,'pn_amb':.005 if rel else .1,
                 'dn_amb':.0001 if rel else 1.,'pi_amb':.005 if rel else .1,
                 'di_amb':.0001 if rel else 1.,'prat':200 if rel else 100,
                 'drat':100 if rel else 1,'b_amb':.1,'b_angle':35},
      'output1':{'file_type':'log','dcycle':1},
      'output2':{'file_type':'hst','dt':100,'data_format':'%.17e'}}
    for d in (1,2,3):
        parts['mesh'].update({f'nx{d}':nx if d<=dim else 1,f'x{d}min':-6,f'x{d}max':6,
                            f'ix{d}_bc':'periodic',f'ox{d}_bc':'periodic'})
        parts['meshblock'][f'nx{d}']=block if d<=dim else 1
    if is_mhd:parts[mod].update(emf='uct_hll',mood_edge='flag',mood_nad_bcc='avg',mood_touched_only='false',mood_nad_v='off')
    if outputs:
        variables=[mod+'_u_bcc' if is_mhd else 'hydro_u',mod+'_w',mod+'_fb_level']
        if is_mhd:variables+=['mhd_divb']
        for i,v in enumerate(variables,3):parts['output'+str(i)]={'file_type':'bin','variable':v,'dt':100}
    return ''.join('<'+b+'>\n'+''.join(f'{k} = {v}\n' for k,v in vals.items())+'\n' for b,vals in parts.items())

def fields(path):
    d=bc.read_binary(str(path));sx,sy,sz=d['nx1_out_mb'],d['nx2_out_mb'],d['nx3_out_mb']
    out={}
    for v in d['var_names']:
        g=np.empty((d['Nx3'],d['Nx2'],d['Nx1']))
        for m,loc in enumerate(d['mb_logical']):
            x,y,z=map(int,loc[:3]);g[z*sz:(z+1)*sz,y*sy:(y+1)*sy,x*sx:(x+1)*sx]=d['mb_data'][v][m]
        out[v]=g
    return d,out

def metrics(path):
    result={}
    log=path/'test.log'
    if log.exists():
        rows=np.loadtxt(log,comments='#',ndmin=2)
        result.update(demotions=int(rows[:,8].sum()),density_floors=int(rows[:,1].sum()),
                      energy_floors=int(rows[:,2].sum()),c2p_failures=int(rows[:,5].sum()))
    else:result['demotions']=0
    for f in sorted((path/'bin').glob('*.00001.bin')):
        d,a=fields(f);result['time']=d['time'];assert all(np.isfinite(v).all() for v in a.values()),f
        if '.mhd_w.' in f.name or '.hydro_w.' in f.name:
            result['primitive_min']={k:float(v.min()) for k,v in a.items()}
            assert list(a.values())[0].min()>0 and list(a.values())[4].min()>0,result
        if '_fb_level.' in f.name:result['dc_cells']=sum(int(np.count_nonzero(v==2)) for v in a.values())
        if '.mhd_divb.' in f.name:result['max_divb']=max(float(np.max(np.abs(v))) for v in a.values())
    # History sums preserve full precision independent of the binary output format.
    hist=next(path.glob('*.hst'),path/'test.hst')
    if hist.exists():
        lines=hist.read_text().splitlines();data=np.loadtxt(hist,ndmin=2)
        assert np.isfinite(data).all(),hist
        names=re.findall(r'\[\d+\]=(\S+)',next(x for x in lines if '[1]=' in x))
        result['conservation_drift']={k:float(abs(data[-1,i]-data[0,i])/max(1,abs(data[0,i]))) for i,k in enumerate(names) if k in ('mass','1-mom','2-mom','3-mom','tot-E')}
    return result

def compare(a,b):
    result={}
    for f in sorted((a/'bin').glob('*.00001.bin')):
        da,A=fields(f);db,B=fields(b/'bin'/f.name)
        result[f.name]={'time_difference':abs(da['time']-db['time']),
           'max_abs':max(float(np.max(np.abs(A[k]-B[k]))) for k in A),
           'normalized_max':max(float(np.max(abs(A[k]-B[k]))/max(1,float(np.max(abs(A[k]))))) for k in A),
           'different':sum(int(np.count_nonzero(A[k]!=B[k])) for k in A)}
    return result

def main():
    global bc
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--exe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--phase',choices=['correctness','benchmark'],default='correctness')
    ap.add_argument('--cases',default=','.join(CASES));ap.add_argument('--orders',default='4,6,8')
    ap.add_argument('--dim',type=int,choices=[2,3],default=2)
    ap.add_argument('--block',type=int,default=64)
    ap.add_argument('--nx',type=int,default=64);ap.add_argument('--steps',type=int,default=16)
    ap.add_argument('--require-dc',action='store_true')
    ap.add_argument('--mpi',type=int,default=1)
    ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();a.source=a.source.resolve();a.exe=a.exe.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.source/'vis/python'));import bin_convert as bc
    report={'phase':a.phase,'host':platform.node(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'nx':a.nx,'dim':a.dim,'block':a.block,'steps':a.steps,'repeats':a.repeats,'executable_sha256':hashlib.sha256(a.exe.read_bytes()).hexdigest(),'runs':[],'comparisons':{},'timings':{}}
    def save(): (a.output/'results.json').write_text(json.dumps(report,indent=2))
    def run(name,s,info):
        out=a.output/name;out.mkdir(exist_ok=True);(out/'input.athinput').write_text(s)
        start=time.monotonic()
        ranks=a.mpi if not name.endswith('_reference') else 1
        cmd=(['mpiexec','-n',str(ranks)] if ranks>1 else [])+[str(a.exe),'-i','input.athinput']
        if not (a.resume and (out/'result.json').exists()):
            with (out/'stdout.txt').open('w') as f:
                try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
                except subprocess.TimeoutExpired:code=-999
            text=(out/'stdout.txt').read_text();m=re.search(r'cpu time used\s*=\s*(\S+)',text)
            r=dict(name=name,code=code,mpi_ranks=ranks,wall_seconds=time.monotonic()-start,solver_seconds=float(m[1]) if m else None,**info)
            if code==0:
                r.update(metrics(out))
                cycles=re.findall(r'^time=(\S+) cycle=(\d+)',text,re.M)
                if cycles:r.update(final_time=float(cycles[-1][0]),cycles=int(cycles[-1][1]))
                if r['solver_seconds']:r['million_zonecycles_per_second']=a.nx**a.dim*a.steps/r['solver_seconds']/1.e6
            else:r['tail']=text[-1800:]
            (out/'result.json').write_text(json.dumps(r,indent=2))
        else:r=json.loads((out/'result.json').read_text())
        report['runs'].append(r);save();print(json.dumps(r),flush=True)
        assert r['code']==0 and r['solver_seconds'] is not None,r
        assert r.get('cycles')==a.steps,r
        assert r['demotions']>0,('fallback did not activate',r)
        if a.require_dc:assert r.get('dc_cells',0)>0,('DC did not activate',r)
        assert r.get('c2p_failures',0)==0,r
        assert r.get('density_floors',0)==0 and r.get('energy_floors',0)==0,r
        return r
    for case in a.cases.split(','):
        for order in map(int,a.orders.split(',')):
            base=f'{case}_fv{order}'
            if a.phase=='correctness':
                for variant,blocks,ng in [('reference',1,20),('minimum',2,max(order if case.endswith('_mhd') else order//2+1,5))]:
                    run(base+'_'+variant,deck(case,order,a.nx,a.nx//blocks,a.steps,ng=ng,dim=a.dim),dict(case=case,order=order,frozen=True,ng=ng))
                c=compare(a.output/(base+'_reference'),a.output/(base+'_minimum'))
                report['comparisons'][base]=c;save()
                assert c and all(v['time_difference']<1.e-12 and v['normalized_max']<2.e-6 for v in c.values()),(base,c)
            else:
                times={}
                # Alternate paths to reduce thermal/clock drift bias; all runs demote.
                for repeat in range(a.repeats+1):
                    for label,frozen in [('new',True),('legacy',False)]:
                        name=f'{base}_{label}_{repeat}'
                        r=run(name,deck(case,order,a.nx,a.block,a.steps,frozen=frozen,outputs=False,dim=a.dim),dict(case=case,order=order,frozen=frozen,repeat=repeat,warmup=repeat==0))
                        if repeat:times.setdefault(label,[]).append(r['solver_seconds'])
                report['timings'][base]={k:statistics.median(v) for k,v in times.items()}
                report['timings'][base]['samples_seconds']=times
                report['timings'][base]['new_over_legacy']=statistics.median(times['new'])/statistics.median(times['legacy']);save()
    print('MATRIX_PASS',flush=True)
if __name__=='__main__':main()
