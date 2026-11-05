import os
"""Fig. 1: COHERE-Resp architecture (grayscale).

Boxes sit on a 5-column x 3-lane grid; arrows are routed orthogonally through the corridors
between lanes, and every arrow is declared explicitly as a polyline so its direction can be
audited.  The dashed arrow is the reverse-mode gradient of the training objective, returned
to the learnable patient module along the left margin."""
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif"})
B, ARROWS = {}, []
FS = 5.7
COLX = [0.5, 3.22, 5.94, 8.66, 11.38]
CW = 2.42


def box(ax, name, col, y, h, text, fill="white", span=1, bold=False):
    x = COLX[col]; w = CW + (COLX[1] - COLX[0]) * (span - 1)
    B[name] = (x, y, w, h)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.10", fc=fill, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, linespacing=1.25,
            weight="bold" if bold else "normal")


def P(name, side, frac=0.5):
    x, y, w, h = B[name]
    return {"r": (x + w, y + h * frac), "l": (x, y + h * frac), "t": (x + w * frac, y + h), "b": (x + w * frac, y)}[side]


def path(ax, pts, ls="-", lw=0.85, label=None, lpos=None, rot=0, tag=""):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color="black", lw=lw, ls=ls, solid_capstyle="butt")
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.85 * (x1 - x0), y0 + 0.85 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=7, lw=lw, color="black", shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=FS - 0.3, ha="center", va="center", style="italic", rotation=rot,
                bbox=dict(fc="white", ec="none", pad=0.5))
    ARROWS.append((tag, pts[0], pts[-1], ls))


def overlaps():
    bad = []
    for (n1, r1), (n2, r2) in itertools.combinations(B.items(), 2):
        x1, y1, w1, h1 = r1; x2, y2, w2, h2 = r2
        g = 0.15
        if x1 < x2 + w2 + g and x2 < x1 + w1 + g and y1 < y2 + h2 + g and y2 < y1 + h1 + g:
            bad.append((n1, n2))
    return bad


def draw(path_out="figs/fig1_architecture.png"):
    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    B.clear(); ARROWS.clear()
    fig, ax = plt.subplots(figsize=(7.1, 4.6), dpi=300)
    ax.set_xlim(0, 14.1); ax.set_ylim(0, 9.1); ax.axis("off")
    G1, G2, G3 = "#F0F0F0", "#DCDCDC", "#C4C4C4"
    LA, LB, LC = 6.55, 3.85, 1.25          # lane bottoms
    H = 1.25
    # lane titles
    ax.text(0.55, LA + H + 0.42, "(a) Privacy-preserving soft linkage", fontsize=7, weight="bold", va="center")
    ax.text(2.6, LB + H + 0.42, "(b) Patient module, parameters $\\theta$", fontsize=7, weight="bold", va="center")
    ax.text(0.55, LC + H + 0.30, "(c) Patient-to-facility coherence", fontsize=7, weight="bold", va="center")
    ax.add_patch(Rectangle((0.38, LB - 0.18), 13.54, H + 0.36, fc="none", ec="#808080", lw=0.6, ls=(0, (1, 1.5))))
    # lane (a)
    box(ax, "src", 0, LA, H, "Fragmented sources:\nward, laboratories,\nimaging, test, clinic", G1)
    box(ax, "bloom", 1, LA, H, "Field-wise Bloom\nfilters of bigrams\n(keyed hashing)")
    box(ax, "odds", 2, LA, H, "Calibrated pair\nodds $o_{ij}$ from\nDice similarities")
    box(ax, "pi", 3, LA, H, "Membership $\\pi_{ij}$\nnormalised with\norphan option $\\kappa$", G2)
    box(ax, "ww", 4, LA, H, "Wastewater pressure:\nSARS-CoV-2, flu, RSV,\nslopes, variant mix", G1)
    # lane (b)
    box(ax, "tok", 0, LB, H, "Evidence tokens\n$(\\mathbf{x},\\mathbf{m},t,s,\\log\\pi)$")
    box(ax, "enc", 1, LB, H, "Membership attention\nwith decay $\\gamma_h$\nand bias $\\beta_h\\log\\pi_j$", G2)
    box(ax, "pool", 2, LB, H, "$\\pi$-weighted\nattention pooling\n$\\to\\ \\mathbf{z}_i$")
    box(ax, "pca", 3, LB, H, "Gated pressure\ncross-attention\n$\\mathbf{z}_i+\\mathbf{g}\\odot\\mathbf{c}_i$", G2)
    box(ax, "haz", 4, LB, H, "Daily hazards\n$h_{i1},\\ldots,h_{i7}$;\n72-h risk and set")
    # lane (c)
    box(ax, "base", 0, LC, H, "Base forecasts per site and horizon:  $\\hat{C}$ = summed inpatient\nprobabilities,  $\\hat{N}$ = forecast admissions $\\times$ onset profile,\n$\\hat{Y}$ = direct facility regressor", G1, span=3)
    box(ax, "mint", 3, LC, H, "MinT\nreconciliation\n$\\tilde{\\mathbf{b}}=\\mathbf{S}\\mathbf{G}_h\\hat{\\mathbf{b}}$", G2)
    box(ax, "out", 4, LC, H, "Coherent O$_2$ demand\n$\\tilde{Y}=\\tilde{C}+\\tilde{N}$;\nscaled risks, intervals")
    # objective strip
    box(ax, "obj", 1, 0.12, 0.62, "Objective $\\mathcal{J}$ = survival NLL $+\\,\\lambda\\,\\cdot$ Poisson coherence (facility counts); FedAvg, secure aggregation", G3, span=4)

    # lane (a) arrows
    for a, b in [("src", "bloom"), ("bloom", "odds"), ("odds", "pi")]:
        path(ax, [P(a, "r"), P(b, "l")], tag=f"{a}->{b}")
    # pi -> tokens through the a/b corridor
    yc1 = LA - 0.55
    x_pi = P("pi", "b", 0.3)[0]; x_tok = P("tok", "t", 0.5)[0]
    path(ax, [(x_pi, LA), (x_pi, yc1), (x_tok, yc1), (x_tok, LB + H)], label="weights $\\pi_{ij}$",
         lpos=((COLX[1] + COLX[2] + CW) / 2, yc1), tag="pi->tok")
    # wastewater -> pressure cross-attention (enters pca at right quarter)
    yc2 = LA - 0.95
    x_ww = P("ww", "b", 0.5)[0]; x_pca = P("pca", "t", 0.78)[0]
    path(ax, [(x_ww, LA), (x_ww, yc2), (x_pca, yc2), (x_pca, LB + H)], label="pressure tokens",
         lpos=(x_ww - 0.05, (LA + yc2) / 2 + 0.05), tag="ww->pca")
    # lane (b) arrows
    for a, b in [("tok", "enc"), ("enc", "pool"), ("pool", "pca"), ("pca", "haz")]:
        path(ax, [P(a, "r"), P(b, "l")], tag=f"{a}->{b}")
    # hazards -> base forecasts through the b/c corridor
    yc3 = LB - 0.55
    x_h = P("haz", "b", 0.5)[0]; x_b1 = 5.6
    path(ax, [(x_h, LB), (x_h, yc3), (x_b1, yc3), (x_b1, LC + H)], label="$p_{ih}$ of current inpatients",
         lpos=(9.2, yc3), tag="haz->base")
    # wastewater -> base forecasts along the right margin, entering base at column 2
    xm = 13.97; yc4 = LB - 0.95
    x_b2 = 7.45
    path(ax, [P("ww", "r"), (xm, P("ww", "r")[1]), (xm, yc4), (x_b2, yc4), (x_b2, LC + H)],
         label="for $\\hat{N}$, $\\hat{Y}$", lpos=(xm - 0.75, yc4 + 0.0), tag="ww->base")
    # lane (c)
    path(ax, [P("base", "r"), P("mint", "l")], tag="base->mint")
    path(ax, [P("mint", "r"), P("out", "l")], tag="mint->out")
    # reverse-mode gradient: objective -> patient module (dashed), along the left margin
    xl = 0.22
    path(ax, [P("obj", "l"), (xl, P("obj", "l")[1]), (xl, LB + H / 2), (0.38, LB + H / 2)], ls=(0, (3.2, 1.8)), lw=0.95,
         label="$\\nabla_{\\theta}\\mathcal{J}$", lpos=(xl, (LC + LB) / 2 + 0.2), rot=90, tag="obj->patient(grad)")
    bad = overlaps()
    fig.savefig(path_out, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    return bad


if __name__ == "__main__":
    print("overlapping boxes:", draw())
    for a in ARROWS:
        print(a)
