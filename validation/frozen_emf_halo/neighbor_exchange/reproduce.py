"""Reproduce neighbor-exchange/local-NAD validation in a fresh audited build."""
from pathlib import Path
import argparse,subprocess,shutil,tarfile,sys,json

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--ref',default='HEAD')
    ap.add_argument('--kokkos',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=8)
    a=ap.parse_args();dst=a.destination.resolve();here=Path(__file__).resolve().parent
    if dst.exists():ap.error('destination must not already exist')
    src=dst/'source';src.mkdir(parents=True)
    with (dst/'source.tar').open('wb') as out:
        subprocess.run(['git','-C',str(a.repo.resolve()),'archive',a.ref],stdout=out,check=True)
    with tarfile.open(dst/'source.tar') as tf:tf.extractall(src,filter='data')
    (src/'kokkos').rmdir();(src/'kokkos').symlink_to(a.kokkos.resolve(),target_is_directory=True)
    subprocess.run(['git','apply',str(here/'audit.patch')],cwd=src,check=True)
    for name in ['halo_test.py','run_max_sweep.py','physics_suite.py','accuracy_comparison.py','controls.py','benchmark.py','local_rcfl.py']:
        shutil.copy2(here/name,dst/name)
    with (dst/'configure.log').open('w') as log:
        subprocess.run(['cmake','-S',str(src),'-B',str(dst/'build'),'-DCMAKE_BUILD_TYPE=Release','-DAthena_ENABLE_MPI=ON','-DPROBLEM=fluids/field_loop','-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON'],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (dst/'build.log').open('w') as log:
        subprocess.run(['cmake','--build',str(dst/'build'),f'-j{a.jobs}'],stdout=log,stderr=subprocess.STDOUT,check=True)
    def py(name):subprocess.run([sys.executable,str(dst/name)],cwd=dst,check=True)
    py('run_max_sweep.py')
    sys.path.insert(0,str(dst));import halo_test as h
    result=json.loads((dst/'max-sweep-results.json').read_text())
    for name,ref,order,dim,blocks in [('fv8_mpi4','fv8_r2_ref',8,2,2),('fv8_3d_mpi4','fv8_3d_ref',8,3,2),('fv4_smallblocks','fv4_r2_ref',4,2,8),('fv8_smallblocks','fv8_r2_ref',8,2,8)]:
        r=h.run(name,order=order,ng=max(order,5),dim=dim,blocks=blocks,mpi=4)
        assert r['code']==0,r
        c=h.compare(ref,name);assert c['fields_exact'] and c['audits_exact'],c
        result['runs'].append(r);result['comparisons'].append(c)
    (dst/'local-halo-results.json').write_text(json.dumps(result,indent=2))
    py('physics_suite.py');py('accuracy_comparison.py');py('controls.py');py('benchmark.py');py('local_rcfl.py')
    import physics_suite as p
    r=p.execute('ot_fv8_mpi4',p.deck(order=8,ng=8),mpi=4);assert r['code']==0
    c=p.compare('ot_fv8_ref','ot_fv8_mpi4');assert all(v['max_abs']==0 and v['time_equal'] for v in c.values()),c
    (dst/'mpi-physics-results.json').write_text(json.dumps({'run':r,'comparison':c,'metrics':p.metrics('ot_fv8_mpi4')},indent=2))
    print('Validation completed:',dst)

if __name__=='__main__':main()
