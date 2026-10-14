from pathlib import Path
import json,subprocess,re,time,sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from halo_test import assemble
W=Path(__file__).resolve().parent;EXE=W/'build/src/athena';RUNS=W/'physics'
TEMPLATE=(W/'runs/fv8_r2_min/input.athinput').read_text()
def setparam(s,block,key,value):
    a=s.index('<'+block+'>');e=s.find('\n<',a+1);e=len(s) if e<0 else e
    part=s[a:e];pattern=rf'(?m)^{re.escape(key)}\s*=.*$'
    if re.search(pattern,part):part=re.sub(pattern,f'{key} = {value}',part)
    else:part+='\n'+f'{key} = {value}'+'\n'
    return s[:a]+part+s[e:]
def deck(order=8,ng=8,nx=64,blocks=2,case='ot',tlim=.25,solver='hlld',steps=-1,default=True):
    s=TEMPLATE
    vals={'mesh':{'nx1':nx,'nx2':nx,'nghost':ng},'meshblock':{'nx1':nx//blocks,'nx2':nx//blocks},'time':{'nlim':steps,'tlim':tlim,'ndiag':50},'mhd':{'reconstruct':f'fv{order}','mood_nad_scale':'relative','mood_test_dump':'false','rsolver':solver,'emf':'uct'},'output4':{'data_format':'%.17e'}}
    for b,ps in vals.items():
        for k,v in ps.items():s=setparam(s,b,k,v)
    if default:s=re.sub(r'(?m)^mood_frozen_emf =.*\n','',s)
    s=re.sub(r'(?m)^mood_owner_audit =.*\n','',s)
    ps='pgen_name = orszag_tang\n'
    if case=='wave':ps='''pgen_name = linear_wave
wave_flag = 0
amp = 1.e-6
dens = 1.0
pgas = 0.6
bx0 = 1.0
by0 = 1.4142135623730951
bz0 = 0.5
along_x1 = true
'''
    if case=='hydro':
        s=s.replace('<mhd>','<hydro>').replace('mhd_u_bcc','hydro_u').replace('mhd_fb_level','hydro_fb_level').replace('variable = mhd_divb','variable = hydro_w')
        s=setparam(s,'hydro','rsolver','hllc');s=setparam(s,'hydro','mood_nad_scale','relative')
        ps='''pgen_name = shock_tube
shock_dir = 1
xshock = 0.0
dl = 1.0
ul = 0.0
vl = 0.0
wl = 0.0
pl = 1.0
dr = 0.125
ur = 0.0
vr = 0.0
wr = 0.0
pr = 0.1
'''
    a=s.index('<problem>');e=s.index('<output1>',a);s=s[:a]+'<problem>\n'+ps+s[e:]
    if case!='hydro':s+='\n<output5>\nfile_type = bin\nvariable = mhd_w\ndt = 100\n'
    return s

def execute(name,s,exe=EXE,mpi=1,extra=None):
    p=RUNS/name;p.mkdir(parents=True,exist_ok=True);(p/'input.athinput').write_text(s)
    cmd=(['mpiexec','-n',str(mpi)] if mpi>1 else [])+[str(exe),'-i','input.athinput']+(extra or [])
    t=time.monotonic()
    with (p/'run.log').open('w') as out:
        try:code=subprocess.run(cmd,cwd=p,stdout=out,stderr=subprocess.STDOUT,timeout=300).returncode
        except subprocess.TimeoutExpired:code=-999
    r=dict(name=name,code=code,seconds=time.monotonic()-t,cmd=cmd)
    (p/'result.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
    return r

def metrics(name):
    p=RUNS/name
    files=sorted((p/'bin').glob('*mhd_u_bcc.*.bin'))
    if not files:files=sorted((p/'bin').glob('*hydro_u.*.bin'))
    assert len(files)>=2,p
    di,A=assemble(files[0]);df,B=assemble(files[-1]);nv=list(A)[:5]
    drift={v:abs(float(np.mean(B[v])-np.mean(A[v])))/max(1.,float(np.mean(np.abs(A[v])))) for v in nv}
    vals=list(B.values());m={'time':df['time'],'finite':all(np.isfinite(v).all() for v in vals),'conservation_drift':drift,'density_min':float(vals[0].min())}
    div=list((p/'bin').glob('*mhd_divb.00001.bin'))
    if div:m['divb_max']=max(float(np.max(np.abs(v))) for v in assemble(div[0])[1].values())
    prim=list((p/'bin').glob('*mhd_w.00001.bin'))
    if prim:m['primitive_min']={k:float(v.min()) for k,v in assemble(prim[0])[1].items()}
    if (p/'halo-errors.dat').exists():m['errors']=(p/'halo-errors.dat').read_text()
    return m

def compare(a,b):
    diffs={}
    for f in (RUNS/a/'bin').glob('*.bin'):
        d,A=assemble(f);e,B=assemble(RUNS/b/'bin'/f.name)
        diffs[f.name]={'max_abs':max(float(np.max(np.abs(A[k]-B[k]))) for k in A),'time_equal':d['time']==e['time']}
    return diffs

def main():
    jobs=[];pairs=[]
    for order in (4,6,8):
        for label,blocks,ng in [('ref',1,14),('min',2,max(order,5))]:
            jobs.append((f'ot_fv{order}_{label}',deck(order=order,ng=ng,blocks=blocks)))
        pairs.append((f'ot_fv{order}_ref',f'ot_fv{order}_min'))
        for nx in (32,64):jobs.append((f'wave_fv{order}_n{nx}',deck(order=order,ng=max(order,5),nx=nx,blocks=2,case='wave',tlim=1.0)))
    for solver in ('llf','hlle','lhlld'):
        for label,blocks,ng in [('ref',1,14),('min',2,8)]:
            jobs.append((f'ot_{solver}_{label}',deck(ng=ng,blocks=blocks,solver=solver,steps=8,tlim=1)))
        pairs.append((f'ot_{solver}_ref',f'ot_{solver}_min'))
    for label,blocks in [('ref',1),('min',2)]:jobs.append((f'hydro_{label}',deck(order=4,ng=8,blocks=blocks,case='hydro',tlim=.05)))
    pairs.append(('hydro_ref','hydro_min'))
    with ThreadPoolExecutor(max_workers=3) as pool:runs=list(pool.map(lambda x:execute(*x),jobs))
    r={'runs':runs,'metrics':{},'comparisons':{}}
    for a in runs:
        if a['code']==0:r['metrics'][a['name']]=metrics(a['name'])
    for a,b in pairs:
        if all(x['code']==0 for x in runs if x['name'] in (a,b)):r['comparisons'][a+' vs '+b]=compare(a,b)
    (W/'physics-results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r['metrics'],indent=2))
if __name__=='__main__':main()
