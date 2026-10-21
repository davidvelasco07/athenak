#!/usr/bin/env python3
"""Render actual pilot frame dumps; requires numpy, matplotlib and ffmpeg."""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    meta = json.loads(path.read_text())
    nx, ny, nz = meta['block_cells']
    blocks = np.asarray(meta['blocks'])
    if nz != 1:
        raise ValueError('This renderer displays 2D runs')
    nf = meta['fields']
    values = np.fromfile(path.with_suffix('.bin'), dtype='<f4')
    values = values.reshape(nf, len(blocks), ny, nx)
    bx, by, _ = blocks.max(axis=0) + 1
    grid = np.zeros((nf, by*ny, bx*nx))
    mask = np.zeros((by, bx))
    allocated = np.asarray(meta['allocated']).reshape(len(blocks), -1)
    for b, (x, y, _) in enumerate(blocks):
        grid[:, y*ny:(y+1)*ny, x*nx:(x+1)*nx] = values[:, b]
        mask[y, x] = allocated[b].sum()
    return meta, grid, mask


def paths(root, name):
    return sorted((root/name/'frames').glob('frame_*.json'),
                  key=lambda p: int(p.stem.split('_')[-1]))


def render(root, out, kind):
    hydro = kind == 'tracers'
    left = paths(root, 'hydro_native' if hydro else 'beam_dense')
    right = paths(root, 'hydro_sparse' if hydro else 'beam_sparse')
    if len(left) != len(right) or len(left) < 2:
        raise ValueError('Missing or mismatched frame sequences')
    vmax = max(load(p)[1].max() for p in left)
    extent = [0, 1, 0, 1 if hydro else 0.5]
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': '#101827',
                         'axes.facecolor': '#101827', 'text.color': '#e9eef7',
                         'axes.labelcolor': '#d8e2f1', 'xtick.color': '#b3c5dc',
                         'ytick.color': '#b3c5dc', 'axes.edgecolor': '#59718e'})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8), dpi=100,
                             gridspec_kw={'height_ratios': [1.6, 1]})
    fig.subplots_adjust(left=.065, right=.9, top=.83, bottom=.14, hspace=.43, wspace=.23)
    title = 'Passive tracers: identical transport, different storage' if hydro else \
            'SC beam: identical radiation, fewer stored block–direction pairs'
    fig.text(.065, .95, title, fontsize=18, weight='bold')
    fig.text(.065, .91, 'Actual GPU simulation frames • synchronized physical time' if hydro else
             'Actual GPU simulation frames • quasi-static solver iterations, not photon travel',
             fontsize=11, color='#9cb6d7')
    colors = np.array([[.2,.75,1], [1,.5,.2], [.4,1,.5], [.8,.4,1]])
    images, masks = [], []
    for c, label in enumerate(['Native dense', 'Sparse']):
        ax = axes[0,c]
        image = ax.imshow(np.zeros((8,8,3)) if hydro else np.zeros((8,8)),
                          origin='lower', extent=extent, vmin=0, vmax=vmax, cmap='magma')
        images.append(image)
        ax.set_title(label + (' • tracer mass fractions' if hydro else ' • mean intensity J'),
                     color='#e9eef7', pad=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        mask = axes[1,c].imshow(np.zeros((8,8)), origin='lower', extent=extent,
                                vmin=0, vmax=4 if hydro else 84, cmap='viridis',
                                interpolation='nearest', aspect='auto')
        masks.append(mask)
        axes[1,c].set_xlabel('x'); axes[1,c].set_ylabel('y')
    cax = fig.add_axes([.925,.18,.014,.23])
    cb = fig.colorbar(masks[1], cax=cax)
    cb.set_label('Stored labels / block' if hydro else 'Sparse directions / block (dense: 84)')
    if not hydro:
        masks[1].set_clim(0,2)
        cb.set_ticks([0,1,2])
    if not hydro:
        cax2 = fig.add_axes([.925,.51,.014,.27])
        cb2 = fig.colorbar(images[1], cax=cax2); cb2.set_label('J (same scale)')
    status = fig.text(.065,.865,'',fontsize=12,color='#e9eef7')
    check = fig.text(.065,.065,'',fontsize=11,color='#b9e3cc')
    fig.text(.065,.025, 'Colors distinguish four provenance labels. Diffusion and faint numerical tails can keep blocks allocated.' if hydro else
             'Support follows the interpolation stencil; diffuse numerical wings require more blocks than a geometric ray.',
             fontsize=10,color='#9cb6d7')
    out.mkdir(parents=True,exist_ok=True)
    target = out/('passive-tracers.mp4' if hydro else 'sc-beam.mp4')
    fps = 8 if hydro else 3
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s','1280x800','-r',str(fps),'-i','-',
               '-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    max_error = 0.0
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        for n, (a,b) in enumerate(zip(left,right)):
            ma, ga, aa = load(a); mb, gb, ab = load(b)
            key = 'cycle' if hydro else 'iteration'
            assert ma[key] == mb[key]
            if hydro: assert abs(ma['time']-mb['time']) < 1e-12
            error = float(np.max(np.abs(ga-gb))); max_error = max(max_error,error)
            for c,(grid,mask) in enumerate([(ga,aa),(gb,ab)]):
                display = np.clip(np.einsum('fyx,fc->yxc',grid/vmax,colors),0,1) if hydro else grid[0]
                images[c].set_data(display); masks[c].set_data(mask)
                axes[1,c].set_title(f'Storage map • {int(mask.sum())} allocated pairs',
                                    color='#e9eef7',pad=8)
            status.set_text(f"Physical time {ma['time']:.4f} • cycle {ma['cycle']}" if hydro else
                            f"Convergence sweep {ma['iteration']} • same boundary injection in both runs")
            check.set_text(f'Maximum frame difference: {error:.2e}   |   Playback speed is illustrative; GPU timings are measured separately.')
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())[:,:,:3].tobytes()
            encoder.stdin.write(pixels)
            if n == len(left)-1:
                fig.savefig(out/('passive-tracers.png' if hydro else 'sc-beam.png'),dpi=100)
                for _ in range(2*fps): encoder.stdin.write(pixels)
        encoder.stdin.close()
        if encoder.wait(): raise RuntimeError('ffmpeg failed')
    plt.close(fig)
    record = {'source':str(root),'frames':len(left),'fps':fps,
              'max_frame_difference_float32':max_error,
              'final_coordinate':ma[key], 'video':target.name}
    (out/(kind+'-movie.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    for kind in ['tracers','beam']: render(args.frames,args.output,kind)
