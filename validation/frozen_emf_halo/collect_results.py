from pathlib import Path
import json,hashlib,sys
import numpy as np
from halo_test import W,RUNS,compare,assemble

def collect():
    sweep=json.loads((W/'sweep.json').read_text())
    ext=json.loads((W/'extended.json').read_text())
    extra=[compare('fv8_3d_ref','fv8_3d_ng10'),compare('fv8_ref','fv8_mpi4'),
           compare('fv8_3d_ref','fv8_3d_mpi4')]
    p=RUNS/'fv8_seed'
    a=np.fromfile(p/'audit_c0_s1_r1_rank0.dat').reshape(-1,13)
    b=np.fromfile(p/'audit_c0_s1_r2_rank0.dat').reshape(-1,13)
    a=a[a[:,12]==1];b=b[b[:,12]==1]
    a=a[np.lexsort((a[:,0],a[:,1],a[:,2]))];b=b[np.lexsort((b[:,0],b[:,1],b[:,2]))]
    mask=np.any(a[:,3:11]!=b[:,3:11],axis=1)
    radius=np.max(np.abs(b[:,:3]-[32,32,0]),axis=1)
    support=dict(changed_cells=int(mask.sum()),max_chebyshev_cell_distance=int(radius[mask].max()),
                 changed_outside_one_cell=int(np.count_nonzero(mask&(radius>1))),
                 coordinates=b[mask,:3].astype(int).tolist())
    divb={}
    for order in (4,6,8):
        _,d=assemble(RUNS/f'long_fv{order}_min/bin/halo.mhd_divb.00001.bin')
        divb[f'fv{order}']=max(float(np.abs(a).max()) for a in d.values())
    runs=[json.loads(f.read_text()) for f in sorted(RUNS.glob('*/run_result.json'))]
    modified=['src/mhd/mhd.hpp','src/mhd/mhd.cpp','src/mhd/mhd_mood.cpp',
              'src/mhd/mhd_tasks.cpp','src/outputs/binary.cpp']
    result=dict(
        baseline_commit='2d251bb09d67fe76ac97d419041c33dc906db2ca',
        scope='Isolated CPU prototype; uniform periodic Newtonian ideal MHD, UCT-HLLD, frozen complete tier EMFs',
        build=dict(mpi=True,kokkos_bounds_check=True,build_type='Release',problem='fluids/field_loop'),
        source_sha256={f:hashlib.sha256((W/'source'/f).read_bytes()).hexdigest() for f in modified},
        binary_sha256=hashlib.sha256((W/'build/src/athena').read_bytes()).hexdigest(),
        derived_bound='initial detection halo = revisions; nghost >= FV order + revisions',
        two_revision_ghost_counts={'fv4':6,'fv6':8,'fv8':10},
        single_repair_support=support,long_run_max_abs_divb=divb,
        run_count=len(runs),runs=runs,
        comparisons=sweep['comparisons']+ext['comparisons']+extra,
        limitations=[
            'Prototype caches full PLM/DC tiers and does not measure production performance.',
            'No SMR/AMR, GR/SR, well-balanced, diffusion, GPU, or nonperiodic-boundary certification.',
            'One fewer ghost cell changed outer-halo candidates; ordinary field-loop final fields still matched.',
            'The deliberately adversarial detector demonstrates failure of one fewer detection-halo layer.',
            'Candidate comparisons use identical global coordinates, including physical-boundary ghost copies.',
            'Reference and test use the same frozen-EMF method, not the old recomposition method.'
        ])
    return result

if __name__=='__main__':
    dest=Path(sys.argv[1]) if len(sys.argv)>1 else W/'halo-test-results.json'
    dest.write_text(json.dumps(collect(),indent=2)+'\n')
    print(dest)
