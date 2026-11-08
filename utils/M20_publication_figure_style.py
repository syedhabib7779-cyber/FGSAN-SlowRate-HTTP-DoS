# ===================== CELL (NEW) — M20: publication figure style (run once, before figure cells) =====================
# Makes every subsequently-generated figure submission-ready: large labels, high DPI, vector (PDF+SVG) export,
# and a colorblind/grayscale-safe palette with distinct line styles and markers so panels read in grayscale.
import matplotlib as mpl, matplotlib.pyplot as plt
from cycler import cycler

_COL = ["#000000","#E69F00","#56B4E9","#009E73","#D55E00","#CC79A7","#0072B2","#F0E442"]
_DASH= ["-","--","-.",":","-","--","-.",":"]
_MARK= ["o","s","^","D","v","P","X","*"]
mpl.rcParams.update({
    "figure.dpi":150, "savefig.dpi":600,
    "savefig.bbox":"tight", "savefig.pad_inches":0.03,
    "pdf.fonttype":42, "ps.fonttype":42,
    "font.size":13, "axes.titlesize":15, "axes.labelsize":14,
    "xtick.labelsize":12, "ytick.labelsize":12, "legend.fontsize":12,
    "axes.linewidth":1.0, "lines.linewidth":2.0, "lines.markersize":6,
    "axes.grid":True, "grid.alpha":0.3, "grid.linewidth":0.6,
    "axes.prop_cycle":cycler(color=_COL),
    "figure.constrained_layout.use":True,
})

def save_fig(fig, name):
    import os
    base="/content/drive/MyDrive/fgsan_ckpt/figs"; os.makedirs(base, exist_ok=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(f"{base}/{name}.{ext}")
    print(f"saved {base}/{name}.(pdf|svg|png)")

def style_lines(ax):
    for i,ln in enumerate(ax.get_lines()):
        ln.set_linestyle(_DASH[i % len(_DASH)]); ln.set_marker(_MARK[i % len(_MARK)])
        ln.set_markevery(max(1, len(ln.get_xdata())//12))
    if ax.get_legend_handles_labels()[1]: ax.legend(frameon=False)

print("publication style active.")
