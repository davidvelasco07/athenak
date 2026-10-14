"""Run and compare the isolated frozen-EMF solver; no production files are changed."""
from pathlib import Path
import json, subprocess, sys, time
import numpy as np

W=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'source/vis/python'))
import bin_convert as bc
EXE=W/'build/src/athena'
RUNS=W/'runs'

def run(name, order=8, ng=12, revisions=2, blocks=2, dim=2, steps=1,
        frozen=True, halo=None, seed=-1, dump=True, mpi=1, amp=.1, drat=2, owner=True):
    p=RUNS/name; p.mkdir(parents=True,exist_ok=True)
    nx=64 if dim==2 else 32; nz=1 if dim==2 else nx
    args=dict(order=order,ng=ng,revisions=revisions,blocks=blocks,dim=dim,
              steps=steps,frozen=frozen,halo=halo,seed=seed,dump=dump,mpi=mpi,amp=amp,drat=drat,owner=owner)
    (p/'parameters.json').write_text(json.dumps(args,indent=2))
    deck=f'''<job>
basename = halo
<mesh>
nghost = {ng}
nx1 = {nx}
nx2 = {nx}
nx3 = {nz}
x1min = -0.5
x1max = 0.5
x2min = -0.5
x2max = 0.5
x3min = -0.5
x3max = 0.5
ix1_bc = periodic
ox1_bc = periodic
ix2_bc = periodic
ox2_bc = periodic
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = {nx//blocks}
nx2 = {nx//blocks}
nx3 = {1 if dim==2 else nx//blocks}
<time>
evolution = dynamic
integrator = rk3
cfl_number = 0.25
nlim = {steps}
tlim = {0.0001 if steps==1 else 1.0}
ndiag = 1
<mhd>
eos = ideal
reconstruct = fv{order}
rsolver = hlld
emf = uct_hlld
gamma = 1.6666666666666667
mood = true
mood_edge = flag
mood_frozen_emf = {str(frozen).lower()}
mood_test_halo = {-1 if halo is None else halo}
mood_test_seed = {seed}
mood_test_dump = {str(dump).lower()}
mood_max_revs = {revisions}
mood_nad_scale = relative
mood_rtol = 1.e-5
mood_sed = true
mood_touched_only = false
<problem>
rad = 0.3
amp = {amp}
drat = {drat}
iprob = {1 if dim==2 else 4}
<output1>
file_type = bin
variable = mhd_u_bcc
dt = 100
<output2>
file_type = bin
variable = mhd_fb_level
dt = 100
<output3>
file_type = bin
variable = mhd_divb
dt = 100
<output4>
file_type = hst
dt = 100
'''
    (p/'input.athinput').write_text(deck)
    cmd=(["mpiexec","-n",str(mpi)] if mpi>1 else [])+[str(EXE),'-i','input.athinput']
    t=time.monotonic()
    with (p/'run.log').open('w') as log:
        try: code=subprocess.run(cmd,cwd=p,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
        except subprocess.TimeoutExpired: code=-999
    result=dict(name=name,code=code,seconds=time.monotonic()-t,**args)
    if code: result['tail']=(p/'run.log').read_text()[-2000:]
    (p/'run_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

def assemble(path):
    d=bc.read_binary(str(path)); nx,ny,nz=d['Nx1'],d['Nx2'],d['Nx3']
    sx,sy,sz=d['nx1_out_mb'],d['nx2_out_mb'],d['nx3_out_mb']
    out={}
    for v in d['var_names']:
        G=np.empty((nz,ny,nx),dtype=np.float64)
        for m,loc in enumerate(d['mb_logical']):
            x,y,z=map(int,loc[:3])
            G[z*sz:(z+1)*sz,y*sy:(y+1)*sy,x*sx:(x+1)*sx]=d['mb_data'][v][m]
        out[v]=G
    return d,out

def compare(ref,other,audits=True):
    a=RUNS/ref;b=RUNS/other
    result={'reference':ref,'test':other,'fields':{},'audits':{}}
    for f in sorted((a/'bin').glob('*.bin')):
        g=b/'bin'/f.name
        if not g.exists(): result['fields'][f.name]={'missing':True};continue
        da,A=assemble(f);db,B=assemble(g)
        result['fields'][f.name]={
            'time_equal':da['time']==db['time'],
            'finite':all(np.isfinite(v).all() for v in B.values()),
            'max_abs':max(float(np.max(np.abs(A[v]-B[v]))) for v in A),
            'different':sum(int(np.count_nonzero(A[v]!=B[v])) for v in A)}
    if audits:
        pars=json.loads((a/'parameters.json').read_text())
        nx=64 if pars['dim']==2 else 32;ny=nx;nz=1 if pars['dim']==2 else nx
        for f in sorted(a.glob('audit_*_rank0.dat')):
            key=f.name.split('_rank')[0]
            aa=np.concatenate([np.fromfile(x,dtype=np.float64).reshape(-1,13) for x in a.glob(key+'_rank*.dat')])
            # Compare the SAME global coordinates, including the reference block's
            # physical-boundary ghosts. Periodically wrapping ghosts onto interiors
            # can introduce unrelated round-off from initialization/stage storage.
            lo=aa[:,:3].min(axis=0).astype(int);hi=aa[:,:3].max(axis=0).astype(int)
            shape=hi-lo+1
            def flat(rows):
                xyz=rows[:,:3].astype(int)-lo
                return (xyz[:,2]*shape[1]+xyz[:,1])*shape[0]+xyz[:,0]
            table=np.full((int(np.prod(shape)),9),np.nan);table[flat(aa)]=aa[:,3:12]
            files=list(b.glob(key+'_rank*.dat'))
            if not files: result['audits'][key]={'missing':True};continue
            bb=np.concatenate([np.fromfile(x,dtype=np.float64).reshape(-1,13) for x in files])
            ids=flat(bb)
            delta=np.abs(table[ids]-bb[:,3:12]);interior=bb[:,12]==1
            result['audits'][key]={
                'max_abs':float(delta.max()),
                'interior_max_abs':float(delta[interior].max()),
                'different':int(np.count_nonzero(delta)),
                'flag_differences':int(np.count_nonzero(delta[:,-1])),
                'interior_flag_differences':int(np.count_nonzero(delta[interior,-1]))}
    result['fields_exact']=bool(result['fields']) and all(
        not r.get('missing') and r['time_equal'] and r['finite'] and r['different']==0
        for r in result['fields'].values())
    result['audits_exact']=bool(result['audits']) and all(
        not r.get('missing') and r['different']==0 for r in result['audits'].values())
    return result

if __name__=='__main__':
    if sys.argv[1]=='run':
        run(sys.argv[2],**json.loads(sys.argv[3]))
    elif sys.argv[1]=='compare':
        print(json.dumps(compare(sys.argv[2],sys.argv[3]),indent=2))
