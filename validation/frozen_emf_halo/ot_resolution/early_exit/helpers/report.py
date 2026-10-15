"""Create self-contained scientific maps and a checkpoint report for this OT sweep."""
import argparse,csv,gzip,html,json,math,os,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap,LogNorm
from run_study import assemble,write_json
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);ap.add_argument('--output',type=Path);a=ap.parse_args()
root=a.root.resolve();study=root/'ot-resolution-early-stop';out=a.output.resolve() if a.output else study/'report';out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'source/vis/python'));import bin_convert as reader
report=json.loads((study/'results.json').read_text());runs=report['runs'];orders=(4,6,8);res=(64,128,256,512,1024,2048,4096)
stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC');data={};metrics=[];cmap=ListedColormap(['#edf0f2','#e9b44c','#b82e36']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
for r in runs:
 run=study/r['name'];cache=run/'plot-t1.npz'
 if r.get('reached_tlim') and not cache.exists():
  fields={}
  for kind in ('mhd_w','mhd_u_bcc','mhd_fb_level'):
   files=sorted((run/'bin').glob(f'*.{kind}.*.bin'))
   if not files:continue
   meta,arrays=assemble(files[-1],reader)
   if not np.isclose(meta['time'],1.,atol=1.e-10,rtol=0):raise ValueError(f'{files[-1]} is not t=1')
   if kind=='mhd_w':fields.update(density=arrays['dens'][0],pressure=(2/3)*arrays['eint'][0])
   elif kind=='mhd_u_bcc':fields['magnetic']=np.sqrt(sum(arrays['bcc'+str(i)][0]**2 for i in (1,2,3)))
   else:fields['fallback']=next(iter(arrays.values()))[0]
   del arrays
  # Maps are displayed at <=1024 pixels per axis; metrics always use full arrays.
  stride=max(1,r['nx']//1024)
  np.savez_compressed(cache,**{k:v[::stride,::stride].astype('float32') for k,v in fields.items()})
 if cache.exists():
  with np.load(cache) as z:data[(r['order'],r['nx'])]={k:z[k] for k in z.files}
 diag=r.get('revision_diagnostics',{});final=(r.get('snapshots') or [{}])[-1]
 rec={k:r.get(k) for k in ('name','order','nx','nghost','clean','reached_tlim','final_time','cycles','solver_seconds','wall_seconds','energy_floors','density_floors','temperature_floors','c2p_failures','density_min','pressure_min','max_abs_divb','demotions_per_cell_stage')}
 rec.update(mean_revisions=diag.get('mean_revisions'),unresolved_stages=diag.get('statuses',{}).get('revision_limit',0),rk_stages=diag.get('stages'),relative_mass_drift=r.get('conservation_max_drift',{}).get('mass'),relative_energy_drift=r.get('conservation_max_drift',{}).get('tot-E'),final_plm_fraction=final.get('tier_fractions',{}).get('1'),final_dc_fraction=final.get('tier_fractions',{}).get('2'))
 metrics.append(rec)
 target=out/'inputs'/r['name'];target.mkdir(parents=True,exist_ok=True)
 for name in ('input.athinput','result.json','stdout.txt','ot.log'):
  if (run/name).exists():
   if name.endswith('.txt') or name.endswith('.log'):
    with (run/name).open('rb') as inp,gzip.open(target/(name+'.gz'),'wb') as dest:shutil.copyfileobj(inp,dest)
    if (target/name).exists():(target/name).unlink()
   else:shutil.copy2(run/name,target/name)
 for p in run.glob('*.hst'):shutil.copy2(p,target/p.name)
write_json(out/'results.json',report);shutil.copy2(study/'build-provenance.json',out/'build-provenance.json')
if metrics:
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');w.writeheader();w.writerows(metrics)
for field,label,palette in [('density','Density','magma'),('pressure','Pressure','inferno'),('magnetic','Magnetic-field magnitude','viridis'),('fallback','Last RK-stage tier','')]:
 fig,axes=plt.subplots(3,7,figsize=(19,8.5),layout='constrained',squeeze=False)
 available=[d[field] for d in data.values() if field in d]
 lo=min((float(v.min()) for v in available),default=0);hi=max((float(v.max()) for v in available),default=1)
 im=None
 for row,order in enumerate(orders):
  for col,nx in enumerate(res):
   ax=axes[row,col];rr=next((r for r in runs if r['order']==order and r['nx']==nx),None);d=data.get((order,nx))
   title=f'FV{order}, {nx}²';ax.set_title(title,fontsize=10,color='#b3262b' if rr and not rr.get('clean') else '#222')
   if d and field in d:
    kw={'cmap':cmap,'norm':norm} if field=='fallback' else {'cmap':palette,'vmin':lo,'vmax':hi}
    im=ax.imshow(d[field],origin='lower',extent=[-.5,.5,-.5,.5],interpolation='nearest',**kw)
   else:
    ax.set_facecolor('#edf0f2');ax.set_aspect('equal');ax.text(.5,.5,'Pending' if not rr else 'No t=1 output',ha='center',va='center',transform=ax.transAxes,color='#666',fontsize=10)
   ax.set_xticks([]);ax.set_yticks([])
 if im is not None:
  cb=fig.colorbar(im,ax=axes,shrink=.8,label=label)
  if field=='fallback':cb.set_ticks([0,1,2],labels=['Base','PLM','DC'])
 fig.suptitle(f'Orszag–Tang at t=1 · {label}\nRmax=4, rcfl rtol=1e-8, early exit, ng=3/4/5 · {len(runs)}/21 complete · {stamp}',fontsize=14)
 fig.savefig(out/f'{field}.png',dpi=155);plt.close(fig)
 # A larger 3-order view for the highest resolution with all three outputs.
 common=[n for n in res if all((o,n) in data for o in orders)]
 if common:
  n=max(common);fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
  vals=[data[(o,n)][field] for o in orders];low=min(float(v.min()) for v in vals);high=max(float(v.max()) for v in vals)
  for ax,o,v in zip(axes,orders,vals):
   kw={'cmap':cmap,'norm':norm} if field=='fallback' else {'cmap':palette,'vmin':low,'vmax':high}
   im=ax.imshow(v,origin='lower',extent=[-.5,.5,-.5,.5],interpolation='nearest',**kw);ax.set_title(f'FV{o}, {n}²');ax.set_xticks([]);ax.set_yticks([])
  cb=fig.colorbar(im,ax=axes,shrink=.85,label=label)
  if field=='fallback':cb.set_ticks([0,1,2],labels=['Base','PLM','DC'])
  fig.suptitle(f'Orszag–Tang · t=1 · {label}');fig.savefig(out/f'latest_{field}.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for o,color in zip(orders,('#2466a8','#e58724','#218a61')):
 rows=[r for r in metrics if r['order']==o];x=[r['nx'] for r in rows]
 ys=[[r.get('relative_energy_drift') for r in rows],[r.get('mean_revisions') for r in rows],[r.get('unresolved_stages',0)/r['rk_stages'] if r.get('rk_stages') else np.nan for r in rows],[r.get('energy_floors',0) for r in rows],[r.get('final_plm_fraction') for r in rows],[r.get('max_abs_divb',0)/(r['nx']/np.sqrt(4*np.pi)) for r in rows]]
 for ax,y in zip(axes.flat,ys):
  ax.plot(x,y,'o-',color=color,label=f'FV{o}');ax.set_xscale('log',base=2);ax.set_xticks(res,[str(n) for n in res],rotation=30);ax.grid(alpha=.2)
for ax,title in zip(axes.flat,('Maximum relative energy drift','Mean revisions per RK stage','Fraction of stages unresolved at cap','Raw energy-floor applications (includes ghosts)','PLM cell fraction at t=1, last RK stage',r'Maximum $\Delta x |\nabla\cdot B|/B_0$')):ax.set_title(title,fontsize=10);ax.set_xlabel('Cells per direction')
axes[0,0].set_yscale('log');axes[1,0].set_yscale('symlog',linthresh=1);axes[1,2].set_yscale('log');axes[0,1].set_ylim(0,4.1);axes[0,0].legend(frameon=False)
fig.suptitle('OT fallback robustness · four revisions maximum · rcfl rtol=1e-8');fig.savefig(out/'robustness.png',dpi=160);plt.close(fig)
clean=sum(r.get('clean',False) for r in runs);complete=sum(r.get('reached_tlim',False) for r in runs)
lines=['# OT fallback resolution study: early termination','',f'Checkpoint: **{stamp}**. **{len(runs)}/21** cases recorded; **{complete}** reached t=1; **{clean}** reached t=1 with finite, positive saved states and no EOS repairs or conversion failures.','',
'Numerical source: `bdef110c34fecdefebcd0973945be92440a8dbc9` (early-exit implementation in `a22d8e7ed`). Apollo A100 GPUs 0 and 1, one independent case per GPU, one MPI rank per case. Shared numerical objects come from the fresh full production CUDA build; only the OT selector/dispatcher/config display are recompiled. Exact source/object/binary hashes and commands are in `build-provenance.json`.','',
'FV4/FV6/FV8, 64–4096², 64² blocks, 3/4/5 ghosts; frozen high-order F/E, integer cascade exchange, magnetic volume-average detection; RK3, CFL=0.25, HLLD/UCT-HLLD, four revisions maximum, local rcfl NAD at rtol=1e-8, SED enabled. Density/internal energy/velocity components/magnetic components are detected. Floors are 1e-12. All inputs and raw logs are preserved.','',
'Runs continue after EOS repairs to obtain t=1 maps. A red map title marks a completed case that did not meet the zero-repair criterion. Raw floor counters include ghost cells; they do not establish the number of repaired owned cells. A stage marked `revision_limit` still violates the detector at the cap; zero EOS repairs does not imply every NAD check converged. Fallback maps show only the last RK stage, not cumulative activity. Magnetic maps show the magnitude of magnetic cell averages derived from staggered faces. Snapshot fields are float32; evolution/history are double. Maps above 1024² are displayed by sampling every N/1024-th pixel; diagnostics use the full arrays. This is a robustness study, not a formal-order convergence measurement.','',
'| N | FV4 | FV6 | FV8 |','|---|---|---|---|']
for n in res:
 cells=[]
 for o in orders:
  r=next((r for r in runs if r['order']==o and r['nx']==n),None)
  if not r:cells.append('Pending');continue
  label='No EOS repairs' if r.get('clean') else 'EOS repairs' if r.get('reached_tlim') else f"Incomplete (t={r.get('final_time',0):.3g})"
  diag=r.get('revision_diagnostics',{});cells.append(f"{label}; mean R={diag.get('mean_revisions',float('nan')):.2f}; E floors={r.get('energy_floors',0)}")
 lines.append('| '+str(n)+'² | '+' | '.join(cells)+' |')
lines+=['','## Figures','','![Density at t=1](density.png)','','![Fallback tiers at t=1](fallback.png)','','![Robustness diagnostics](robustness.png)','','[Pressure](pressure.png) · [Magnetic magnitude](magnetic.png) · [Metrics CSV](summary.csv) · [Full results](results.json)']
(out/'README.md').write_text('\n'.join(lines)+'\n')
body=f'<h1>Orszag–Tang fallback resolution study</h1><p>{stamp} · {len(runs)}/21 recorded · {clean} without EOS repairs</p><p>Four revisions maximum; rcfl rtol=1e-8; early termination; 3/4/5 ghosts. Red titles indicate repairs or incomplete cases. Grey panels are pending. Raw EOS counters include ghost cells. Fallback maps show the last RK stage.</p>'
for field in ('density','pressure','magnetic','fallback','robustness'):body+=f'<h2>{html.escape(field.capitalize())}</h2><img src="{field}.png">'
body+='<p><a href="summary.csv">CSV metrics</a> · <a href="results.json">Full JSON</a> · <a href="README.md">Configuration and interpretation</a></p>'
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>OT fallback resolution study</title><style>body{font:16px system-ui;margin:35px auto;max-width:1500px;background:#fafafa;color:#222}img{width:100%;background:white}p{max-width:1050px;line-height:1.5}</style>'+body)
print('REPORT_READY',len(runs),'cases',clean,'zero-repair',flush=True)
