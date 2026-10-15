"""Relink the current fully built CUDA core with the unmodified OT generator."""
import argparse,hashlib,json,re,shlex,subprocess
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args()
root=args.root.resolve()
src=root/'source';base=root/'build-prod';out=root/'build-ot/src';out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base/'src/athena')=='5e659ff9f3bd1d5c79c0868944c1f17d8b3c71963601b77d9d39fcaab3d05520'
uses={}
for p in (src/'src').rglob('*'):
 if p.suffix in ('.cpp','.hpp') and not p.name.startswith('._'):
  tokens=sorted(set(re.findall(r'\b(?:PROBLEM_GENERATOR|USER_PROBLEM_ENABLED)\b',p.read_text())))
  if tokens:uses[str(p.relative_to(src))]=tokens
assert uses=={'src/pgen/pgen.cpp':['USER_PROBLEM_ENABLED'],'src/utils/show_config.cpp':['PROBLEM_GENERATOR']},uses
config=(base/'config.hpp').read_text();problem='../../validation/frozen_emf_halo/ot_resolution/ot_pgen'
new=re.sub(r'^#define PROBLEM_GENERATOR .*$', '#define PROBLEM_GENERATOR "'+problem+'"',config,flags=re.M)
normalize=lambda s:re.sub(r'^#define PROBLEM_GENERATOR .*$', '',s,flags=re.M)
assert normalize(config)==normalize(new) and '#define USER_PROBLEM_ENABLED 1' in new
(out/'config.hpp').write_text(new)
flags=(base/'src/CMakeFiles/athena.dir/flags.make').read_text();args=[]
for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):args+=shlex.split(re.search(r'^'+key+r' = (.*)$',flags,re.M)[1])
commands=[]
with (root/'build-ot.log').open('w') as log:
 for source,obj in [(src/'validation/frozen_emf_halo/ot_resolution/ot_pgen.cpp',out/'problem.cpp.o'),(src/'src/utils/show_config.cpp',out/'show_config.cpp.o'),(src/'src/pgen/pgen.cpp',out/'pgen.cpp.o')]:
  cmd=[str(src/'kokkos/bin/nvcc_wrapper'),'-I'+str(out),*args,'-c',str(source),'-o',str(obj)];commands.append(cmd)
  subprocess.run(cmd,cwd=base/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
 link=shlex.split((base/'src/CMakeFiles/athena.dir/link.txt').read_text());cmd=[];core={};removed=[]
 for arg in link:
  if arg=='CMakeFiles/athena.dir/pgen/pgen.cpp.o':arg=str(out/'pgen.cpp.o')
  elif arg=='CMakeFiles/athena.dir/utils/show_config.cpp.o':arg=str(out/'show_config.cpp.o')
  elif arg=='CMakeFiles/athena.dir/pgen/fluids/blast.cpp.o':removed.append(arg);continue
  elif arg=='athena':arg=str(out/'athena')
  elif arg.endswith('.o'):core[arg]=sha(base/'src'/arg)
  cmd.append(arg)
 assert len(removed)==1
 cmd.insert(cmd.index('-o'),str(out/'problem.cpp.o'));commands.append(cmd)
 subprocess.run(cmd,cwd=base/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
 conf=subprocess.run([str(out/'athena'),'-c'],capture_output=True,text=True,check=True).stdout
 assert 'ot_pgen' in conf;log.write(conf)
(root/'ot-build.json').write_text(json.dumps({'source_commit':'bdef110c34fecdefebcd0973945be92440a8dbc9','executable_sha256':sha(out/'athena'),'base_executable_sha256':sha(base/'src/athena'),'commands':commands,'macro_uses':uses,'shared_core_sha256':core,'source_sha256':{str(p.relative_to(src)):sha(p) for p in (src/'src').rglob('*') if p.suffix in ('.cpp','.hpp') and not p.name.startswith('._')},'config':conf},indent=2)+'\n')
print('OT_BUILD_READY',sha(out/'athena'),flush=True)
