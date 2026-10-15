"""Plot OT robustness diagnostics and final density/fallback maps from a completed sweep."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from run_study import assemble


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--snapshot-time',type=float,choices=[0.,.25,.5,.75,1.],default=.5)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.source.resolve()/'vis/python'));import bin_convert as reader
    d=json.loads((a.runs/'results.json').read_text());runs=d['runs']
    orders=sorted({r['order'] for r in runs});res=sorted({r['nx'] for r in runs})
    colors={4:'#2466a8',6:'#e58724',8:'#218a61'}
    fig,axes=plt.subplots(2,3,figsize=(12,7.3),constrained_layout=True)
    for order in orders:
        rows=sorted([r for r in runs if r['order']==order],key=lambda r:r['nx'])
        x=[r['nx'] for r in rows]
        values=[
          [r.get('density_min',np.nan) for r in rows],
          [r.get('pressure_min',np.nan) for r in rows],
          [r.get('demotions_per_cell_stage',np.nan) for r in rows],
          [r.get('conservation_max_drift',{}).get('tot-E',np.nan) for r in rows],
          [r.get('max_abs_divb',np.nan)/(r['nx']/np.sqrt(4*np.pi)) for r in rows],
          [max((s.get('tier_fractions',{}).get('2',0) for s in r.get('snapshots',[])),default=np.nan) for r in rows]]
        for ax,y in zip(axes.flat,values):
            ax.plot(x,y,'o-',color=colors[order],label=f'FV{order}')
            for rr,yy in zip(rows,y):
                if not rr.get('clean') and yy is not None and np.isfinite(yy):
                    ax.scatter(rr['nx'],yy,marker='x',color='#b3262b',s=70,zorder=5)
            ax.set_xscale('log',base=2);ax.set_xticks(res,[str(n) for n in res]);ax.grid(alpha=.2)
    titles=['Minimum density in saved snapshots','Minimum pressure in saved snapshots',
            'Demotion events per cell per RK stage','Maximum relative total-energy drift',
            r'Maximum $\Delta x\,|\nabla\cdot B|/B_0$', 'Peak DC fraction in saved snapshots']
    for i,(ax,title) in enumerate(zip(axes.flat,titles)):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Cells per direction')
        if i in (0,1,3,4):ax.set_yscale('log')
    dcmax=max((s.get('tier_fractions',{}).get('2',0) for r in runs for s in r.get('snapshots',[])),default=0)
    axes[1,2].set_ylim(0,max(.01,1.1*dcmax))
    if dcmax==0:axes[1,2].text(.04,.85,'No DC cells in saved snapshots',transform=axes[1,2].transAxes,fontsize=9)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Orszag–Tang robustness with frozen MOOD: HLLD / UCT-HLLD, target t = 1\nRed crosses mark failed or incomplete runs',fontsize=13)
    fig.savefig(a.output/'robustness.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(len(orders),len(res),figsize=(3*len(res),3*len(orders)),squeeze=False,constrained_layout=True)
    flagsfig,flagsaxes=plt.subplots(len(orders),len(res),figsize=(3*len(res),3*len(orders)),squeeze=False,constrained_layout=True)
    cmap=ListedColormap(['#edf0f2','#e9b44c','#b82e36']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    fieldlist=[]
    for row,order in enumerate(orders):
        for col,nx in enumerate(res):
            r=next((r for r in runs if r['order']==order and r['nx']==nx),None)
            ax,fx=axes[row,col],flagsaxes[row,col]
            if r is None:
                ax.axis('off');fx.axis('off');continue
            run=a.runs/r['name']
            files=sorted((run/'bin').glob('*.mhd_w.*.bin'))
            index=int(round(a.snapshot_time/.25))
            if len(files)<=index:
                for b in (ax,fx):
                    b.set_facecolor('#edf0f2')
                    b.set_title(f'FV{order}, {nx}²',fontsize=9)
                    b.text(.5,.5,f'No t = {a.snapshot_time:g} output\nStopped after energy-floor detection',ha='center',va='center',transform=b.transAxes,fontsize=8,color='#853237')
                    b.set_xticks([]);b.set_yticks([]);b.set_aspect('equal')
                continue
            meta,fields=assemble(files[index],reader);rho=fields['dens'][0]
            if a.snapshot_time==1. and not np.isclose(meta['time'],1.,rtol=0,atol=1.e-10):
                raise ValueError(f"Expected t=1 output: {files[index]} has t={meta['time']}")
            fieldlist.append((ax,rho))
            fs=sorted((run/'bin').glob('*.mhd_fb_level.*.bin'))
            if fs:
                fm,levels=assemble(fs[index],reader)
                if not np.isclose(fm['time'],meta['time'],rtol=0,atol=1.e-10):
                    raise ValueError(f'Fallback and density times differ in {run}')
                fim=fx.imshow(next(iter(levels.values()))[0],origin='lower',extent=[-.5,.5,-.5,.5],cmap=cmap,norm=norm,interpolation='nearest')
            title=f"FV{order}, {nx}², t={meta['time']:.3f}"
            ax.set_title(title,fontsize=9,color='black' if r.get('clean') else '#b3262b');fx.set_title(title,fontsize=9,color='black' if r.get('clean') else '#b3262b')
            for b in (ax,fx):b.set_xticks([]);b.set_yticks([])
    vmin=min(float(np.min(v[np.isfinite(v)])) for _,v in fieldlist if np.isfinite(v).any());vmax=max(float(np.max(v[np.isfinite(v)])) for _,v in fieldlist if np.isfinite(v).any())
    for ax,rho in fieldlist:
        im=ax.imshow(rho,origin='lower',extent=[-.5,.5,-.5,.5],cmap='magma',vmin=vmin,vmax=vmax,interpolation='nearest')
    if fieldlist:fig.colorbar(im,ax=axes,label='Density',shrink=.8)
    flagsfig.colorbar(fim,ax=flagsaxes,ticks=[0,1,2],label='Last RK stage: 0 = base, 1 = PLM, 2 = DC',shrink=.8)
    caption=('at t = 1; grey panels have no final output' if a.snapshot_time==1.
             else f'near t = {a.snapshot_time:g}; red titles mark runs that required floors before t = 1')
    fig.suptitle(f'Density {caption}')
    flagsfig.suptitle(f'Fallback levels {caption}')
    fig.savefig(a.output/'density.png',dpi=160);flagsfig.savefig(a.output/'fallback.png',dpi=160)
    plt.close(fig);plt.close(flagsfig)
    rows=[]
    for r in runs:
        final=r.get('snapshots',[{}])[-1]
        common=min(r.get('snapshots',[{}]),key=lambda s:abs(s.get('time',0)-a.snapshot_time))
        rows.append({'order':r['order'],'nx':r['nx'],'clean':r.get('clean',False),
          'final_time':r.get('final_time'),'cycles':r.get('cycles'),'solver_seconds':r.get('solver_seconds'),
          'density_floors':r.get('density_floors'),'energy_floors':r.get('energy_floors'),
          'c2p_failures':r.get('c2p_failures'),'density_min':r.get('density_min'),'pressure_min':r.get('pressure_min'),
          'relative_mass_drift':r.get('conservation_max_drift',{}).get('mass'),
          'relative_energy_drift':r.get('conservation_max_drift',{}).get('tot-E'),
          'max_abs_divb':r.get('max_abs_divb'),'demotions_per_cell_stage':r.get('demotions_per_cell_stage'),
          'final_plm_fraction':final.get('tier_fractions',{}).get('1'),
          'final_dc_fraction':final.get('tier_fractions',{}).get('2'),
          'common_snapshot_time':common.get('time'),
          'common_plm_fraction':common.get('tier_fractions',{}).get('1'),
          'common_dc_fraction':common.get('tier_fractions',{}).get('2')})
    with (a.output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('PLOTS_COMPLETE',flush=True)

if __name__=='__main__':main()
