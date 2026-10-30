from pathlib import Path
import argparse,subprocess,shlex,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--output',required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[2];B=Path(a.build).resolve();T=Path(a.output).resolve();T.mkdir(parents=True,exist_ok=True)
s=(R/'src/mhd/mhd_mood.cpp').read_text().split('\ntemplate <MHD_RSolver rsolver_method_>\nvoid MHD::MOODLoop')[0]+'\n} // namespace mhd\n'+Path(__file__).with_name('candidate_test_main.inc').read_text()
(T/'test_gr_candidate.cpp').write_text(s)
flags={}
for line in (B/'src/CMakeFiles/athena.dir/flags.make').read_text().splitlines():
 if ' = ' in line:
  k,v=line.split(' = ',1);flags[k]=shlex.split(v)
cmd=['/usr/bin/c++','-O2','-std=c++17']+flags['CXX_DEFINES']+['-I'+str(R/'src/mhd')]+flags['CXX_INCLUDES']+[str(T/'test_gr_candidate.cpp')]+[str(B/f'kokkos/{n}/src/libkokkos{n}.a') for n in ['core','containers','simd']]+['-o',str(T/'test_gr_candidate')]
(T/'compile_command.json').write_text(json.dumps(cmd,indent=2));r=subprocess.run(cmd,capture_output=True,text=True);(T/'compile.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
r=subprocess.run([str(T/'test_gr_candidate')],capture_output=True,text=True);(T/'test.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
v=json.loads(r.stdout);v['mhd_source_sha256']=hashlib.sha256((R/'src/mhd/mhd_mood.cpp').read_bytes()).hexdigest();(T/'unit_validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
