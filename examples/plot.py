import numpy as np
import pylab as pl
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


def plot_stuff(fname):
    data = np.load(fname)
    xedges = data["x_edges"]
    yedges = data["y_edges"]
    hist = data["hist"]
    # hist[hist < 1] = None
    fig = pl.figure(figsize=(6, 6))

    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    pc = ax.pcolormesh(xedges, yedges, hist)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histx.plot(xedges[:-1], hist.sum(axis=0))
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histy.plot(hist.sum(axis=1), yedges[:-1])
    fig.savefig(f"{fname.parent / fname.stem}.png", density=600)


glob = list(Path("results/").glob("*.npz"))
with Pool(20) as p:
    r = list(tqdm(p.imap(plot_stuff, glob), total=len(glob)))