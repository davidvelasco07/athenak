from physics_suite import *
rs=[];cs={}
# Unchanged relative-mode arithmetic: compare against the pre-removal binary.
for case in (['ot','hydro'] if (W/'athena-pre-local').exists() else []):
    s=deck(order=4,ng=8,case=case,tlim=.005,steps=8,default=False)
    for label,exe in [('before',W/'athena-pre-local'),('after',EXE)]:
        r=execute(case+'_'+label,s,exe=exe);rs.append(r);assert r['code']==0
    cs[case]=compare(case+'_before',case+'_after')
    assert all(x['max_abs']==0 and x['time_equal'] for x in cs[case].values()),cs[case]
# Removed options reject loudly for both physics modules.
for mod,case in [('mhd','ot'),('hydro','hydro')]:
    for mode in ('gcfl','grange','gdu'):
        s=setparam(deck(case=case),mod,'mood_nad_scale',mode)
        r=execute(f'reject_{mod}_{mode}',s);rs.append(r)
        assert r['code']!=0 and 'is unsupported' in (RUNS/r['name']/'run.log').read_text()
# Explicit unsupported frozen method and too-small bounds reject.
for name,s in [('small_ng',deck(order=8,ng=7)),('nonperiodic',setparam(deck(default=False),'mesh','ix1_bc','outflow'))]:
    r=execute('reject_'+name,s);rs.append(r);assert r['code']!=0
# Continue a checkpoint: comparison uses final dump only (restart starts mid-run).
s=deck(order=8,ng=8,steps=8,tlim=1)
s+='\n<output6>\nfile_type = rst\ndcycle = 4\n'
r=execute('restart_full',s);rs.append(r);assert r['code']==0
files=sorted((RUNS/'restart_full/rst').glob('*.rst'));assert files,files
# Read headers to find cycle-4 checkpoint via restart stdout; first noninitial checkpoint.
print('CHECKPOINTS',files,flush=True)
p=RUNS/'restart_resume';p.mkdir(parents=True,exist_ok=True)
cmd=[str(EXE),'-r',str(files[1]),'time/nlim=8']
with (p/'run.log').open('w') as out:code=subprocess.run(cmd,cwd=p,stdout=out,stderr=subprocess.STDOUT,timeout=60).returncode
assert code==0,(p/'run.log').read_text()
restart={}
for f in (p/'bin').glob('*.bin'):
    g=RUNS/'restart_full/bin'/f.name
    if not g.exists():continue
    a,A=assemble(f);b,B=assemble(g)
    if a['time']==b['time']:
        restart[f.name]=max(float(np.max(np.abs(A[k]-B[k]))) for k in A)
assert restart and max(restart.values())==0,restart
(W/'control-results.json').write_text(json.dumps({'runs':rs,'relative_equivalence':cs,'restart':restart},indent=2))
