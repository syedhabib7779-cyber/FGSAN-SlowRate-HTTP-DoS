import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
OUT = './'
C = dict(blue=('#3B5A7D', '#EEF2F7'), orange=('#B7802A', '#FCF3E3'), red=('#B5524A', '#FBEBEA'), green=('#4A7D5C', '#EAF3EC'))
TXT, SUB = '#111111', '#3A4654'

def panel(ax, x, y, w, h, key, title):
    ec, fc = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.07', fc=fc, ec=ec, lw=1.1))
    ax.text(x + 0.12, y + h - 0.07, title, color=ec, fontsize=8, fontweight='bold', va='top', ha='left')

def box(ax, x, y, w, h, key, title, sub=None, bold=True, fs=7, sfs=6.2, fc='white'):
    ec = C[key][0]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.045', fc=fc, ec=ec, lw=0.9))
    cx = x + w / 2
    if sub:
        two = '\n' in title
        ax.text(cx, y + h * (0.71 if two else 0.66), title, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=TXT, linespacing=1.0)
        ax.text(cx, y + h * (0.23 if two else 0.28), sub, ha='center', va='center', fontsize=sfs, color=SUB, linespacing=1.1)
    else:
        ax.text(cx, y + h / 2, title, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=TXT, linespacing=1.1)

def arrow(ax, p, q, key, lw=1.0, ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=C[key][0], lw=lw, shrinkA=0, shrinkB=0))

def line(ax, xs, ys, key, lw=1.0):
    ax.plot(xs, ys, color=C[key][0], lw=lw, solid_capstyle='butt')

def save(fig, name):
    for ext in ('pdf', 'svg'): fig.savefig(OUT + name + '.' + ext, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT + name + '.png', dpi=600, bbox_inches='tight', pad_inches=0.02)

# ---------------------------------------------------------------- Fig. 1
W, H = 7.16, 5.05
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
# Stage 1
panel(ax, 0.05, 4.13, W - 0.10, 0.87, 'blue', 'Stage 1   Data preparation')
bw, bh, by = 1.24, 0.50, 4.24
xs = [0.20 + i * ((W - 0.40 - bw) / 4) for i in range(5)]
S1 = [('CICIDS2017', 'Wednesday capture\n450,978 flows (cleaned)'),
      ('Label filtering', 'BENIGN / Slowloris /\nSlowHTTPTest'),
      ('Class balancing', '5:1 benign:attack\n56,475 + 11,295'),
      ('Stratified split', '70 / 15 / 15\n47,439 / 10,165 / 10,166'),
      ('Standardization', 'z-score, training-split\nstatistics only')]
for x, (t, s) in zip(xs, S1): box(ax, x, by, bw, bh, 'blue', t, s)
for a, b in zip(xs[:-1], xs[1:]): arrow(ax, (a + bw, by + bh / 2), (b, by + bh / 2), 'blue')
# Stage 2
panel(ax, 0.05, 3.04, 3.42, 0.97, 'blue', 'Stage 2   Semantic feature grouping')
box(ax, 0.20, 3.47, 3.12, 0.27, 'blue', '27 CICFlowMeter aggregate flow features', bold=False, fs=7)
gw = 0.96; gx = [0.20, 0.20 + (3.12 - gw) / 2, 0.20 + 3.12 - gw]
for x, t in zip(gx, ['Volume (8)', 'Timing (14)', 'Behavior (5)']):
    box(ax, x, 3.12, gw, 0.24, 'blue', t, bold=False, fs=7)
    arrow(ax, (x + gw / 2, 3.47), (x + gw / 2, 3.36), 'blue', ms=6)
arrow(ax, (xs[0] + bw / 2, 4.13), (xs[0] + bw / 2, 4.01), 'blue')
# Stage 3
panel(ax, 3.60, 3.04, W - 3.65, 0.97, 'orange', 'Stage 3   FGSAN forward pass')
s3w = 1.03; s3x = [3.72, 3.72 + s3w + 0.12, 3.72 + 2 * (s3w + 0.12)]
S3 = [('Group\nencoders', '3 × FFN, d = 32'), ('Multi-head\nself-attention', 'H = 4, across groups'), ('Residual + LN,\nmean pooling', 'classification head')]
for x, (t, s) in zip(s3x, S3): box(ax, x, 3.36, s3w, 0.40, 'orange', t, s, fs=6.6, sfs=6)
for a, b in zip(s3x[:-1], s3x[1:]): arrow(ax, (a + s3w, 3.56), (b, 3.56), 'orange', ms=6)
box(ax, 3.72, 3.11, s3x[2] + s3w - 3.72, 0.20, 'orange', 'Focal loss objective  (α = 0.25, γ = 2.0)', bold=False, fs=6.8)
arrow(ax, (3.32, 3.56), (3.72, 3.56), 'blue')
# Stage 4
panel(ax, 0.05, 1.86, W - 0.10, 1.04, 'red', 'Stage 4   Adversarial training loop (Algorithm 1)')
S4 = [(1.15, 'Mini-batch', 'B = 128, 30 epochs\nAdam, lr = 10⁻³'),
      (1.40, 'PGD attack generation', 'ε = 0.05, α = 0.01, T = 7'),
      (1.55, 'Functionality-preserving\nconstraints', 'increase-only packet-length\nand byte features'),
      (1.15, 'Batch mixing', '50% adversarial\n50% clean'),
      (0.70, 'Update θ', None)]
gap = (W - 0.40 - sum(w for w, _, _ in S4)) / 4; x = 0.20; pos4 = []
for w, t, s in S4:
    box(ax, x, 2.17, w, 0.49, 'red', t, s, fs=6.8, sfs=6.0); pos4.append((x, w)); x += w + gap
for (a, wa), (b, _) in zip(pos4[:-1], pos4[1:]): arrow(ax, (a + wa, 2.415), (b, 2.415), 'red')
xl = pos4[0][0] + pos4[0][1] / 2; xr = pos4[-1][0] + pos4[-1][1] / 2
line(ax, [xr, xr], [2.17, 2.00], 'red'); line(ax, [xr, xl], [2.00, 2.00], 'red'); arrow(ax, (xl, 2.00), (xl, 2.17), 'red')
ax.text((pos4[3][0] + xr) / 2 - 0.2, 2.025, 'repeat for E = 30 epochs', fontsize=6.2, color=C['red'][0], style='italic', ha='center', va='bottom')
xm3 = (s3x[1] + s3w / 2)
arrow(ax, (xm3, 3.04), (xm3, 2.90), 'red'); ax.text(xm3 + 0.06, 2.935, 'model under attack', fontsize=5.8, color=C['red'][0], va='center')
# Stage 5
panel(ax, 0.05, 0.05, W - 0.10, 1.66, 'green', 'Stage 5   Evaluation protocol')
xt = pos4[2][0] + pos4[2][1] / 2
arrow(ax, (2.0, 1.86), (2.0, 1.71), 'green'); ax.text(2.06, 1.785, 'trained model', fontsize=5.8, color=C['green'][0], va='center')
S5 = [('Clean-data performance', 'vs. 5 baselines: RF, XGBoost,\nLightGBM, SVM, MLP'),
      ('Four-way ablation', 'attention × adversarial\ntraining'),
      ('Benign-slow stress test', 'timing / volume stretching,\nMcNemar exact test'),
      ('Evasion transferability', 'white-box and black-box\ntransfer to baselines')]
w5 = 1.56; x5 = [0.20 + i * ((W - 0.40 - w5) / 3) for i in range(4)]
for x, (t, s) in zip(x5, S5): box(ax, x, 1.00, w5, 0.48, 'green', t, s, fs=6.8, sfs=6.0)
yb = 0.84
line(ax, [x5[0] + w5 / 2, x5[-1] + w5 / 2], [yb, yb], 'green')
for x in x5: line(ax, [x + w5 / 2, x + w5 / 2], [1.00, yb], 'green')
bx = [(0.95, 2.35, 'Explainability', 'raw attention weights (negative result)\nand occlusion-based importance'),
      (3.85, 2.35, 'Reported outcomes', 'F1, AUC, FPR, FNR, evasion success rate,\ngroup importance by detection outcome')]
for x, w, t, s in bx:
    box(ax, x, 0.20, w, 0.48, 'green', t, s, fs=6.8, sfs=6.0)
    arrow(ax, (x + w / 2, yb), (x + w / 2, 0.68), 'green')
arrow(ax, (0.95 + 2.35, 0.44), (3.85, 0.44), 'green')
save(fig, 'Figure_1_methodology'); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2
P = dict(blue=('#2F3B4A', '#5B7BB2'), green=('#2F3B4A', '#6FA287'), red=('#2F3B4A', '#B9474A'), orange=('#2F3B4A', '#DD8452'),
         purple=('#2F3B4A', '#8172B3'), grey=('#2F3B4A', '#6B6B6B'), dark=('#2F3B4A', '#444444'))
def fbox(ax, cx, cy, w, h, key, text, fs=6.6):
    ec, fc = P[key]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.03', fc=fc, ec=ec, lw=0.7))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color='white', linespacing=1.15)
def farr(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=6.5, color='#2F3B4A', lw=0.8, shrinkA=0, shrinkB=0))
W, H = 7.16, 3.75
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
ax.text(0.05, H - 0.08, '(a)', ha='left', va='top', fontsize=9, fontweight='bold')
ax.text(3.95, H - 0.08, '(b)', ha='left', va='top', fontsize=9, fontweight='bold')
# (a)
cx = 1.85; bh = 0.25
fbox(ax, cx, 3.25, 1.9, bh, 'blue', 'Flow feature vector  x ∈ ℝ²⁷')
gxs = [0.62, 1.85, 3.08]
for g, (n, k) in zip(gxs, [('Volume', 8), ('Timing', 14), ('Behavior', 5)]):
    fbox(ax, g, 2.80, 1.05, bh, 'green', f'{n}\n({k} features)')
    fbox(ax, g, 2.37, 1.05, bh, 'green', 'Group encoder\n(2-layer MLP, d = 32)', fs=6.0)
    farr(ax, (cx, 3.25 - bh / 2), (g, 2.80 + bh / 2)); farr(ax, (g, 2.80 - bh / 2), (g, 2.37 + bh / 2))
    farr(ax, (g, 2.37 - bh / 2), (g, 1.93 + 0.15))
fbox(ax, cx, 1.93, 2.9, 0.30, 'red', 'Multi-head self-attention across groups\n(H = 4 heads, d$_h$ = 8)')
steps = [(1.50, 'orange', 'Residual add + LayerNorm'), (1.12, 'orange', 'Mean pool across groups'),
         (0.74, 'purple', 'Classification head\n(MLP → softmax)'), (0.32, 'dark', 'ŷ: Benign / Attack')]
prev = 1.93 - 0.15
for y, k, t in steps:
    fbox(ax, cx, y, 1.9, bh, k, t); farr(ax, (cx, prev), (cx, y + bh / 2)); prev = y - bh / 2
# (b)
cx = 5.75; w = 1.85
rows = [(3.25, 'blue', 'Clean mini-batch\n(X, Y), size B'), (2.80, 'grey', 'Select first k = ρ·B samples\nfor perturbation'),
        (2.35, 'red', 'PGD attack (T = 7 steps, ε = 0.05)\nfunctionality-preserving mask S'),
        (1.90, 'grey', "X′ = concat(adversarial samples,\nremaining clean samples)"),
        (1.45, 'green', 'FGSAN forward pass\n(panel a)'), (1.00, 'orange', 'Focal loss (α = 0.25, γ = 2.0)'),
        (0.55, 'purple', r'Adam update:  $\theta \leftarrow \theta - \eta\,\nabla_{\theta} L$')]
for i, (y, k, t) in enumerate(rows):
    fbox(ax, cx, y, w, 0.28, k, t, fs=6.2)
    if i: farr(ax, (cx, rows[i - 1][0] - 0.14), (cx, y + 0.14))
lx = cx - w / 2 - 0.35
ax.plot([cx - w / 2, lx, lx], [0.55, 0.55, 2.35], color='#2F3B4A', lw=0.8)
farr(ax, (lx, 2.35), (cx - w / 2, 2.35))
ax.text(lx - 0.07, 1.45, 'θ updated every step\n(attack uses the current model)', rotation=90, ha='right', va='center', fontsize=5.8, style='italic', color='#2F3B4A')
save(fig, 'Figure_2_architecture'); plt.close(fig)
print('ok')
