"""Build and run the 50-case owner-exchange maximum-halo audit."""
from pathlib import Path
import argparse,json,subprocess,shutil,tarfile,sys

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--kokkos',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=8)
    a=ap.parse_args();dst=a.destination.resolve();here=Path(__file__).resolve().parent
    if dst.exists(): ap.error('destination must not already exist')
    src=dst/'source';src.mkdir(parents=True)
    with (dst/'baseline.tar').open('wb') as out:
        subprocess.run(['git','-C',str(a.repo.resolve()),'archive','2d251bb09d67fe76ac97d419041c33dc906db2ca'],stdout=out,check=True)
    with tarfile.open(dst/'baseline.tar') as tf:tf.extractall(src,filter='data')
    (src/'kokkos').rmdir();(src/'kokkos').symlink_to(a.kokkos.resolve(),target_is_directory=True)
    for patch in [here.parent/'frozen-emf-prototype.patch',here.parent/'halo-audit-instrumentation.patch',here/'owner-exchange-audit.patch']:
        subprocess.run(['git','apply',str(patch)],cwd=src,check=True)
    for name in ['halo_test.py','run_max_sweep.py']:shutil.copy2(here/name,dst/name)
    with (dst/'configure.log').open('w') as log:
        subprocess.run(['cmake','-S',str(src),'-B',str(dst/'build'),'-DCMAKE_BUILD_TYPE=Release','-DAthena_ENABLE_MPI=ON','-DPROBLEM=fluids/field_loop','-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON'],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (dst/'build.log').open('w') as log:
        subprocess.run(['cmake','--build',str(dst/'build'),f'-j{a.jobs}'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(dst/'run_max_sweep.py')],cwd=dst,check=True)
    sys.path.insert(0,str(dst));import halo_test as h
    results=json.loads((dst/'max-sweep-results.json').read_text())
    for name,ref,dim in [('fv8_mpi4','fv8_r2_ref',2),('fv8_3d_mpi4','fv8_3d_ref',3)]:
        run=h.run(name,order=8,ng=8,dim=dim,mpi=4);assert run['code']==0,run
        c=h.compare(ref,name);assert c['fields_exact'] and c['audits_exact'],c
        results['runs'].append(run);results['comparisons'].append(c)
    (dst/'max-halo-results.json').write_text(json.dumps(results,indent=2))
    print('Passed 50 runs / 28 positive and negative-control comparisons:',dst/'max-halo-results.json')

if __name__=='__main__':main()
