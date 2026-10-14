"""Reproduce the frozen-EMF halo experiment from its recorded baseline commit."""
from pathlib import Path
import argparse,subprocess,shutil,tarfile,sys

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True,help='AthenaK repository containing commit 2d251bb09')
    ap.add_argument('--kokkos',type=Path,required=True,help='Existing Kokkos source directory')
    ap.add_argument('--destination',type=Path,required=True,help='New experiment directory')
    ap.add_argument('--jobs',type=int,default=8)
    args=ap.parse_args()
    dst=args.destination.resolve()
    if dst.exists(): ap.error('destination must not already exist')
    dst.mkdir(parents=True);src=dst/'source';src.mkdir()
    here=Path(__file__).resolve().parent
    with (dst/'baseline.tar').open('wb') as out:
        subprocess.run(['git','-C',str(args.repo.resolve()),'archive',
            '2d251bb09d67fe76ac97d419041c33dc906db2ca'],stdout=out,check=True)
    with tarfile.open(dst/'baseline.tar') as tf: tf.extractall(src,filter='data')
    if (src/'kokkos').exists(): (src/'kokkos').rmdir()
    (src/'kokkos').symlink_to(args.kokkos.resolve(),target_is_directory=True)
    scripts=['make_prototype.py','add_audit.py','add_chain.py','halo_test.py',
             'run_sweep.py','run_extended.py','collect_results.py']
    for name in scripts: shutil.copy2(here/name,dst/name)
    def py(name,*extra):
        subprocess.run([sys.executable,str(dst/name),*extra],cwd=dst,check=True)
    for name in scripts[:3]: py(name)
    with (dst/'configure.log').open('w') as log:
        subprocess.run(['cmake','-S',str(src),'-B',str(dst/'build'),
          '-DCMAKE_BUILD_TYPE=Release','-DAthena_ENABLE_MPI=ON',
          '-DPROBLEM=fluids/field_loop','-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON'],
          stdout=log,stderr=subprocess.STDOUT,check=True)
    with (dst/'build.log').open('w') as log:
        subprocess.run(['cmake','--build',str(dst/'build'),f'-j{args.jobs}'],
          stdout=log,stderr=subprocess.STDOUT,check=True)
    py('run_sweep.py')
    extras=[
      ('fv8_seed','{"order":8,"ng":10,"blocks":2,"seed":2080}'),
      ('fv8_3d_ref','{"order":8,"ng":14,"blocks":1,"dim":3}'),
      ('fv8_3d_ng10','{"order":8,"ng":10,"blocks":2,"dim":3}'),
      ('fv8_mpi4','{"order":8,"ng":10,"blocks":2,"mpi":4}'),
      ('fv8_3d_mpi4','{"order":8,"ng":10,"blocks":2,"dim":3,"mpi":4}')]
    for name,params in extras: py('halo_test.py','run',name,params)
    py('run_extended.py')
    py('collect_results.py')
    print('Results:',dst/'halo-test-results.json')

if __name__=='__main__': main()
