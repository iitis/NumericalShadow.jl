import numpy as np
import pylab as pl
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


def plot_stuff(fname):
    data = np.load(fname)
    xedges = data["x_edges"]
    yedges = data["y_edges"]
    hist = data["hist"].T
    evs = data["evs"]
    nr = data["nr"]
    hist = np.asarray(hist, dtype=np.float64)
    xhist = hist.sum(axis=0)
    yhist = hist.sum(axis=1)
    hist[hist < 1] = None
    fig = pl.figure(figsize=(6, 6))

    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.2, hspace=0.12)

    ax = fig.add_subplot(gs[1, 0])
    pc = ax.pcolormesh(xedges, yedges, hist)
    p = pl.Polygon(nr, facecolor="none", 
              edgecolor='red', lw=2)
    ax.add_patch(p)
    ax.scatter(evs.real, evs.imag, color="k", marker="*", s=60)
    # ax.set_xlim(xedges[0], xedges[-1])
    # ax.set_ylim(yedges[0], yedges[-1])
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histx.plot(xedges[:-1], xhist)
    ax_histx.set_yticks([])
    # ax_histx.set_xticks([])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histy.plot(yhist, yedges[:-1])
    ax_histy.set_xticks([])
    # ax_histy.set_yticks([])
    fig.suptitle(f'{fname.stem}', fontsize=16)
    fig.savefig(f"{fname.parent / fname.stem}.png")


glob = list(Path("results/").glob("*.npz"))
with Pool(20) as p:
    r = list(tqdm(p.imap(plot_stuff, glob), total=len(glob)))