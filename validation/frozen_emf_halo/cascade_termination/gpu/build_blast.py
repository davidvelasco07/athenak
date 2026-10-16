"""Derive one blast selector from the current freshly built CUDA numerical core."""
from pathlib import Path
import hashlib,json,re,shlex,subprocess
root=Path(__file__).resolve().parent;src=root/'source';base=root/'build';out=root/'build-blast/src';out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
uses={}
for p in (src/'src').rglob('*'):
 if p.suffix in ('.cpp','.hpp') and not p.name.startswith('._'):
  tokens=sorted(set(re.findall(r'\b(?:PROBLEM_GENERATOR|USER_PROBLEM_ENABLED)\b',p.read_text())))
  if tokens:uses[str(p.relative_to(src))]=tokens
assert uses=={'src/pgen/pgen.cpp':['USER_PROBLEM_ENABLED'],'src/utils/show_config.cpp':['PROBLEM_GENERATOR']},uses
config=(base/'config.hpp').read_text();assert '#define USER_PROBLEM_ENABLED 0' in config
new=re.sub(r'^#define PROBLEM_GENERATOR .*$', '#define PROBLEM_GENERATOR "fluids/blast"',config,flags=re.M).replace('#define USER_PROBLEM_ENABLED 0','#define USER_PROBLEM_ENABLED 1')
normalize=lambda s:re.sub(r'^#define (PROBLEM_GENERATOR|USER_PROBLEM_ENABLED) .*$', '',s,flags=re.M)
assert normalize(config)==normalize(new);(out/'config.hpp').write_text(new)
flags=(base/'src/CMakeFiles/athena.dir/flags.make').read_text();args=[]
for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):args+=shlex.split(re.search(r'^'+key+r' = (.*)$',flags,re.M)[1])
commands=[]
with (root/'build-blast.log').open('w') as log:
 for source,obj in [(src/'src/pgen/fluids/blast.cpp',out/'problem.o'),(src/'src/pgen/pgen.cpp',out/'pgen.o'),(src/'src/utils/show_config.cpp',out/'show_config.o')]:
  cmd=[str(src/'kokkos/bin/nvcc_wrapper'),'-I'+str(out),*args,'-c',str(source),'-o',str(obj)];commands.append(cmd);subprocess.run(cmd,cwd=base/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
 link=shlex.split((base/'src/CMakeFiles/athena.dir/link.txt').read_text());cmd=[];core={}
 for arg in link:
  if arg=='CMakeFiles/athena.dir/pgen/pgen.cpp.o':arg=str(out/'pgen.o')
  elif arg=='CMakeFiles/athena.dir/utils/show_config.cpp.o':arg=str(out/'show_config.o')
  elif arg=='athena':arg=str(out/'athena')
  elif arg.endswith('.o'):core[arg]=sha(base/'src'/arg)
  cmd.append(arg)
 cmd.insert(cmd.index('-o'),str(out/'problem.o'));commands.append(cmd);subprocess.run(cmd,cwd=base/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
 conf=subprocess.run([str(out/'athena'),'-c'],capture_output=True,text=True,check=True).stdout;assert 'fluids/blast' in conf
(root/'blast-build.json').write_text(json.dumps({'base_binary_sha256':sha(base/'src/athena'),'binary_sha256':sha(out/'athena'),'macro_uses':uses,'core_sha256':core,'commands':commands},indent=2)+'\n')
print('BLAST_VARIANT_READY',flush=True)
