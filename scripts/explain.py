"""
Figures that explain what a trained CNN classifier uses, written to figures/xai/ (PNG, plus the numbers of
every figure as CSV next to it in figures/xai/data/ and the headline numbers in summary.json).

    python scripts/explain.py                                   # the 1000 bp model on family split 1
    python scripts/explain.py --model experiments/baselines/ce_rc_cos50_fam_s1/model.pt \\
        --data-dir data/datasets/taxa8fam500_s1 --fragment-size 500 --out figures/xai_500

  01  attribution maps of four fragments (Integrated Gradients, gradient x input, Grad-CAM) and local GC
  02  do the methods find what the prediction relies on? deletion curves against random removal
  03  do the maps depend on what the model learned? correlation with a randomly initialised copy
  04  how much is composition? accuracy on fragments shuffled while keeping base / dinucleotide counts
  05  what the model predicts as a function of GC content
  06  motifs of the first convolution filters that separate one class from the rest
  07  the test genomes the model gets most wrong, and what it calls them instead

Colours: one validated categorical order (blue, orange, aqua) and one blue sequential ramp; text stays in
ink colours. Figure text is in English.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from metapathpredict import explain  # noqa: E402
from metapathpredict.baselines import load_supervised_checkpoint  # noqa: E402
from metapathpredict.genome_eval import TEST_FASTA, read_fragment_accessions  # noqa: E402

# ------------------------------------------------------------------------------------------- style
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = {"ig": "#2a78d6", "gxi": "#eb6834", "cam": "#1baf7a"}  # categorical slots 1-3 (validated all-pairs)
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue 100 -> 700
SEQ = LinearSegmentedColormap.from_list("blue_seq", RAMP)
NAMES = {"ig": "Integrated Gradients", "gxi": "Gradient x input", "cam": "Grad-CAM"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False, "lines.solid_capstyle": "round",
})


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=0, pad=4)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    return ax


def dot_legend(ax_or_fig, entries, **kwargs):
    """Legend with a coloured mark and text in ink (text never wears the series colour)."""
    handles = [plt.Line2D([], [], color=c, lw=2.2, marker="o", markersize=0) for _, c in entries]
    return ax_or_fig.legend(handles, [n for n, _ in entries], labelcolor=INK_2, handlelength=1.6, **kwargs)


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def smooth(a, width):
    return np.convolve(a, np.ones(width) / width, mode="same") if width > 1 else a


def gc_window(idx, width=50):
    return smooth(((idx == 1) | (idx == 2)).astype(float), width)


# ------------------------------------------------------------------------------------------- data
def load(args):
    model = load_supervised_checkpoint(args.model, "cpu")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    class_names = ckpt["config"]["class_names"]
    d = Path(args.data_dir)
    with h5py.File(d / f"encoded_test_{args.fragment_size}.hdf5") as f:
        x, y = f["sequences"][:], f["labels"][:]
    genomes = read_fragment_accessions(d / TEST_FASTA)
    return model, class_names, x, y, genomes, d


def probabilities(model, x, batch=512):
    out = []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            out.append(torch.softmax(model(torch.from_numpy(x[lo:lo + batch])), dim=1))
    return torch.cat(out).numpy()


def pick_confident(probs, y, per_class, rng):
    """Correctly classified fragments, `per_class` per class, most confident first within a random pool."""
    chosen = []
    for c in range(probs.shape[1]):
        ok = np.flatnonzero((probs.argmax(1) == c) & (y == c))
        if len(ok):
            chosen += list(rng.choice(ok, size=min(per_class, len(ok)), replace=False))
    return np.array(sorted(chosen))


# ----------------------------------------------------------------------------------------- figures
def fig_examples(model, x, y, probs, class_names, out, rng, smooth_width=15):
    classes = [n for n in ("bacteria", "fungi", "plant", "virus") if n in class_names]
    picks = []
    for name in classes:
        c = class_names.index(name)
        ok = np.flatnonzero((probs.argmax(1) == c) & (y == c))
        picks.append(int(ok[np.argsort(-probs[ok, c])[: max(len(ok) // 5, 1)]][rng.integers(min(20, max(len(ok) // 5, 1)))]))
    xt = torch.from_numpy(x[picks])
    target = torch.tensor([class_names.index(n) for n in classes])
    ig = explain.position_scores(explain.integrated_gradients(model, xt, target=target, steps=48)).numpy()
    gxi = explain.position_scores(explain.gradient_x_input(model, xt, target=target)).numpy()
    cam = explain.grad_cam(model, xt, target=target).numpy()
    idx = explain.to_indices(x[picks])
    length = x.shape[-1]
    pos = np.arange(length)

    fig = plt.figure(figsize=(13, 8.6))
    outer = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.16, left=0.055, right=0.985, top=0.885, bottom=0.07)
    rows = []
    for k, name in enumerate(classes):
        inner = outer[k // 2, k % 2].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        a, b = fig.add_subplot(inner[0]), fig.add_subplot(inner[1], sharex=None)
        style(a, "y"), style(b, "y")
        norm_ig = smooth(ig[k], smooth_width) / np.abs(smooth(ig[k], smooth_width)).max()
        norm_gxi = smooth(gxi[k], smooth_width) / np.abs(smooth(gxi[k], smooth_width)).max()
        a.axhline(0, color=GRID, lw=1.0)
        a.fill_between(pos, 0, cam[k], color=SERIES["cam"], alpha=0.22, lw=0)
        a.plot(pos, cam[k], color=SERIES["cam"], lw=1.6)
        a.plot(pos, norm_gxi, color=SERIES["gxi"], lw=1.6)
        a.plot(pos, norm_ig, color=SERIES["ig"], lw=1.8)
        a.set_ylim(-1.05, 1.05)
        a.set_xlim(0, length)
        a.set_xticklabels([])
        a.set_ylabel("relative importance")
        p = probs[picks[k], class_names.index(name)]
        a.set_title(f"{name}  (predicted {name}, p = {p:.2f})")
        gc = gc_window(idx[k])
        b.plot(pos, 100 * gc, color=INK_2, lw=1.4)
        b.axhline(100 * gc.mean(), color=GRID, lw=1.0)
        b.set_xlim(0, length)
        b.set_ylim(0, 100)
        b.set_yticks([0, 50, 100])
        b.set_ylabel("GC, %")
        b.set_xlabel("position in the fragment, bp")
        for j in range(length):
            rows.append([name, j, ig[k][j], gxi[k][j], cam[k][j], 100 * gc[j]])
    dot_legend(fig, [(NAMES[k], SERIES[k]) for k in ("ig", "gxi", "cam")], loc="upper left", ncol=3,
               bbox_to_anchor=(0.055, 0.985), fontsize=10)
    fig.text(0.985, 0.985, f"Signed scores smoothed over {smooth_width} bp and scaled to the largest value per fragment;\n"
             "Grad-CAM is non-negative and coarse (one cell = 8 bp). Bottom: GC in a 50 bp window.",
             ha="right", va="top", color=INK_2, fontsize=9)
    fig.savefig(out / "01_attribution_examples.png", dpi=170)
    plt.close(fig)
    write_csv(out / "data" / "01_attribution_examples.csv", ["class", "position", "integrated_gradients", "gradient_x_input",
                                                            "grad_cam", "gc_percent_window50"], rows)


def end_labels(ax, series, x_end, min_gap):
    """Direct labels at the right end of lines, moved apart so they never overlap. Text stays in ink."""
    items = sorted(series, key=lambda s: s[1])
    placed = []
    for label, y_end, colour in items:
        y = y_end if not placed else max(y_end, placed[-1] + min_gap)
        placed.append(y)
        ax.plot([x_end], [y_end], marker="o", ms=5, color=colour, mec=SURFACE, mew=1.2, zorder=5, clip_on=False)
        ax.text(x_end + 0.012, y, label, va="center", ha="left", color=INK_2, fontsize=9.5)


def fig_deletion(model, x, y, probs, class_names, out, rng, n_per_class, summary):
    pick = pick_confident(probs, y, n_per_class, rng)
    xt = torch.from_numpy(x[pick])
    target = torch.from_numpy(y[pick])
    fractions = np.linspace(0, 0.5, 11)
    scores = {
        "ig": explain.position_scores(explain.integrated_gradients(model, xt, target=target, steps=24)),
        "gxi": explain.position_scores(explain.gradient_x_input(model, xt, target=target)),
        "cam": torch.from_numpy(explain.grad_cam(model, xt, target=target).numpy()) + 1e-6 * torch.rand(len(pick), x.shape[-1]),
    }
    curves = {k: explain.deletion_curve(model, xt, s, fractions, target=target) for k, s in scores.items()}
    rand = np.stack([explain.deletion_curve(model, xt, torch.rand(len(pick), x.shape[-1]), fractions, target=target) for _ in range(10)])
    rand_mean, rand_lo, rand_hi = rand.mean(0), rand.min(0), rand.max(0)
    auc = {k: explain.area_under(c, fractions) for k, c in curves.items()}
    auc_random = explain.area_under(rand_mean, fractions)
    summary["deletion_auc"] = {**auc, "random": auc_random, "fragments": int(len(pick))}

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    style(ax, "y")
    ax.fill_between(fractions, rand_lo, rand_hi, color=MUTED, alpha=0.25, lw=0)
    ax.plot(fractions, rand_mean, color=MUTED, lw=2.0)
    for k in ("cam", "gxi", "ig"):
        ax.plot(fractions, curves[k], color=SERIES[k], lw=2.2)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, max(1.0, rand_hi.max()) * 1.02)
    ax.set_xlabel("fraction of positions removed (set to N), most important first")
    ax.set_ylabel("mean probability of the predicted class")
    ax.set_title("Removing the positions a method marks hurts the prediction faster than removing random ones")
    end_labels(ax, [(f"{NAMES[k]}  (area {auc[k]:.3f})", curves[k][-1], SERIES[k]) for k in curves]
               + [(f"random order  (area {auc_random:.3f})", rand_mean[-1], MUTED)], 0.5, 0.045)
    ax.set_xlim(0, 0.5)
    fig.subplots_adjust(left=0.09, right=0.68, top=0.9, bottom=0.12)
    fig.text(0.09, 0.02, f"{len(pick)} correctly classified test fragments, {n_per_class} per class. Grey band: range of 10 random orders.  "
             "Smaller area = better.", color=INK_2, fontsize=8.5)
    fig.savefig(out / "02_deletion_curves.png", dpi=170)
    plt.close(fig)
    write_csv(out / "data" / "02_deletion_curves.csv", ["fraction_removed", "integrated_gradients", "gradient_x_input", "grad_cam",
                                                        "random_mean", "random_min", "random_max"],
              [[f, curves["ig"][i], curves["gxi"][i], curves["cam"][i], rand_mean[i], rand_lo[i], rand_hi[i]] for i, f in enumerate(fractions)])


def fig_randomization(model, x, y, probs, out, rng, n_per_class, summary):
    pick = pick_confident(probs, y, n_per_class, rng)
    xt = torch.from_numpy(x[pick])
    target = torch.from_numpy(y[pick])
    methods = {
        "Integrated Gradients": lambda m, b: explain.position_scores(explain.integrated_gradients(m, b, target=target, steps=16)),
        "Gradient x input": lambda m, b: explain.position_scores(explain.gradient_x_input(m, b, target=target)),
        "Grad-CAM": lambda m, b: torch.from_numpy(explain.grad_cam(m, b, target=target).numpy()),
    }
    corr = {name: explain.randomization_check(model, xt, fn) for name, fn in methods.items()}
    summary["randomization_spearman"] = corr

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    style(ax, "y")
    names = list(corr)
    bars = ax.bar(names, [corr[n] for n in names], width=0.5, color=SERIES["ig"], edgecolor=SURFACE, linewidth=1.5)
    for bar, n in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width() / 2, max(corr[n], 0) + 0.03, f"{corr[n]:.2f}", ha="center", color=INK, fontsize=10)
    ax.set_ylim(-0.1, 1.05)
    ax.text(0.5, 0.9, "A map that came from the input alone, not from the weights,\nwould score close to 1.", transform=ax.transAxes,
            ha="center", va="center", color=INK_2, fontsize=10)
    ax.set_ylabel("rank correlation with a random-weights copy")
    ax.set_title("Maps of the trained model vs an untrained copy\n(low = the maps depend on what was learned)", pad=10)
    fig.tight_layout()
    fig.savefig(out / "03_randomization_check.png", dpi=170)
    plt.close(fig)
    write_csv(out / "data" / "03_randomization_check.csv", ["method", "mean_spearman"], [[n, corr[n]] for n in names])


def fig_shuffle(model, x, y, class_names, out, rng, n_per_class, summary):
    keep = np.concatenate([rng.choice(np.flatnonzero(y == c), size=min(n_per_class, int((y == c).sum())), replace=False)
                           for c in range(len(class_names))])
    xs, ys = x[keep], y[keep]
    modes = {"original order": "none", "same base counts, order shuffled": "mono", "same dinucleotide counts, order shuffled": "dinuc"}
    results = {label: explain.shuffle_accuracy(model, xs, ys, mode, seed=0) for label, mode in modes.items()}
    summary["shuffle_accuracy"] = {label: r["accuracy"] for label, r in results.items()}

    groups = ["all classes"] + list(class_names)
    values = {label: [r["accuracy"]] + [r["per_class"][c] for c in range(len(class_names))] for label, r in results.items()}
    colours = [SERIES["ig"], SERIES["gxi"], SERIES["cam"]]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    style(ax, "y")
    width, base = 0.26, np.arange(len(groups))
    for i, (label, colour) in enumerate(zip(modes, colours)):
        ax.bar(base + (i - 1) * width, values[label], width=width, color=colour, edgecolor=SURFACE, linewidth=1.5, label=label)
        ax.text(base[0] + (i - 1) * width, values[label][0] + 0.015, f"{values[label][0]:.2f}", ha="center", color=INK, fontsize=9)
    ax.axhline(1 / len(class_names), color=MUTED, lw=1.0)
    for i, label in enumerate(modes):  # a bar of height 0 is invisible: say so
        for g, v in enumerate(values[label]):
            if v < 0.005:
                ax.text(base[g] + (i - 1) * width, 0.012, "0", ha="center", color=INK_2, fontsize=9)
    ax.set_xticks(base)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of fragments classified correctly")
    fig.text(0.06, 0.955, "How much of the prediction is composition? Accuracy after shuffling the fragments", fontsize=11, color=INK, va="top")
    dot_legend(ax, [(k, c) for k, c in zip(modes, colours)] + [(f"chance (1/{len(class_names)})", MUTED)], loc="lower left",
               bbox_to_anchor=(0, 1.02), ncol=4, borderaxespad=0, fontsize=9.5)
    fig.text(0.06, 0.01, f"{len(keep)} test fragments, {n_per_class} per class. Dinucleotide shuffle: Altschul-Erikson (exact dinucleotide counts).",
             color=INK_2, fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.12)
    fig.savefig(out / "04_shuffle_test.png", dpi=170)
    plt.close(fig)
    write_csv(out / "data" / "04_shuffle_test.csv", ["group"] + list(modes), [[g] + [values[m][i] for m in modes] for i, g in enumerate(groups)])


def fig_gc(x, y, probs, class_names, out, summary):
    idx = explain.to_indices(x)
    gc = ((idx == 1) | (idx == 2)).mean(axis=1)
    edges = np.arange(0.25, 0.751, 0.05)
    bins = np.clip(np.digitize(gc, edges) - 1, 0, len(edges) - 2)
    n_bins, n_cls = len(edges) - 1, len(class_names)
    pred = probs.argmax(1)
    share = np.full((n_cls, n_bins), np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = bins == b
        count[b] = sel.sum()
        if count[b] >= 50:
            share[:, b] = np.bincount(pred[sel], minlength=n_cls) / count[b]
    summary["gc_bins_with_data"] = int((count >= 50).sum())

    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4.2], width_ratios=[40, 1.3], hspace=0.08, wspace=0.05,
                          left=0.14, right=0.93, top=0.9, bottom=0.11)
    top, ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    centers = np.arange(n_bins)
    style(top, "y")
    top.bar(centers, count, width=0.86, color=SERIES["ig"], edgecolor=SURFACE, linewidth=1.5)
    top.set_xlim(-0.5, n_bins - 0.5)
    top.set_xticks([])
    top.set_ylabel("fragments")
    top.set_title("Predicted class by GC content of the fragment (test set)")
    image = ax.imshow(np.ma.masked_invalid(share), aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(class_names)
    labels = [f"{100 * edges[i]:.0f}-{100 * edges[i + 1]:.0f}" for i in range(n_bins)]
    labels[0], labels[-1] = f"<{100 * edges[1]:.0f}", f">{100 * edges[-2]:.0f}"
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("GC content of the fragment, % (bins with fewer than 50 fragments left blank)")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_bins, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for c in range(n_cls):
        for b in range(n_bins):
            if share[c, b] >= 0.3:
                ax.text(b, c, f"{100 * share[c, b]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if share[c, b] > 0.5 else INK)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("share of the bin's fragments predicted as this class", color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, colors=INK_2)
    fig.savefig(out / "05_gc_dependence.png", dpi=170)
    plt.close(fig)
    write_csv(out / "data" / "05_gc_dependence.csv", ["gc_bin", "fragments"] + [f"share_{n}" for n in class_names],
              [[labels[b], int(count[b])] + [("" if np.isnan(share[c, b]) else share[c, b]) for c in range(n_cls)] for b in range(n_bins)])


def fig_motifs(model, x, y, class_names, out, rng, n_per_class, summary, per_class_cap=3, n_filters=12):
    keep = np.concatenate([rng.choice(np.flatnonzero(y == c), size=min(n_per_class, int((y == c).sum())), replace=False)
                           for c in range(len(class_names))])
    result = explain.first_layer_motifs(model, x[keep], y[keep], num_classes=len(class_names), top_windows=300)
    order, taken, chosen = np.argsort(-result["specificity"]), {}, []
    for f in order:
        c = int(result["best_class"][f])
        if taken.get(c, 0) < per_class_cap and result["information"][f] > 3:
            chosen.append(int(f))
            taken[c] = taken.get(c, 0) + 1
        if len(chosen) == n_filters:
            break
    summary["motif_filters"] = {"total": int(result["pwm"].shape[0]), "shown": len(chosen)}

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 2.35 * rows + 0.9))
    axes = np.atleast_1d(axes).ravel()
    csv_rows = []
    for ax, f in zip(axes, chosen):
        pwm = result["pwm"][f]
        consensus = "".join(explain.BASES[i] for i in pwm.argmax(axis=0))
        ax.imshow(pwm, cmap=SEQ, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(4))
        ax.set_yticklabels(list(explain.BASES), fontsize=9)
        ax.set_xticks(range(pwm.shape[1]))
        ax.set_xticklabels(list(consensus), fontsize=9)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"filter {f}: {class_names[int(result['best_class'][f])]}\n{result['information'][f]:.1f} bits", fontsize=9.5)
        csv_rows.append([f, class_names[int(result["best_class"][f])], consensus, result["information"][f], result["specificity"][f]]
                        + [pwm[b, j] for b in range(4) for j in range(pwm.shape[1])])
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("First-layer filters that separate one class from the rest: base frequencies of the best-matching windows",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Letters under a matrix: the most frequent base at each position. Darker = more frequent. "
             "Filters chosen by how specifically one class excites them (at most 3 per class, >3 bits).", color=INK_2, fontsize=8.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.87 if rows > 2 else 0.82, bottom=0.06, hspace=0.75, wspace=0.35)
    fig.savefig(out / "06_first_layer_motifs.png", dpi=170)
    plt.close(fig)
    width = result["pwm"].shape[2]
    write_csv(out / "data" / "06_first_layer_motifs.csv",
              ["filter", "best_class", "consensus", "information_bits", "specificity"] + [f"{'ACGT'[b]}{j + 1}" for b in range(4) for j in range(width)],
              csv_rows)


def fig_genomes(probs, y, genomes, class_names, data_dir, out, summary, per_class=3):
    organism = {}
    with open(data_dir / "split_assignments.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            organism[row["accession"]] = row["organism"]
    n_cls = len(class_names)
    rows = []
    for acc in np.unique(genomes):
        sel = genomes == acc
        c = int(y[sel][0])
        rows.append((c, float((probs[sel].argmax(1) == c).mean()), acc, probs[sel].mean(0), int(sel.sum())))
    picked = []
    for c in range(n_cls):
        worst = sorted([r for r in rows if r[0] == c], key=lambda r: r[1])[:per_class]
        picked += worst
    fig, ax = plt.subplots(figsize=(9.8, 0.3 * len(picked) + 2.4))
    matrix = np.stack([r[3] for r in picked])
    image = ax.imshow(matrix, aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(picked)))
    ax.set_yticklabels([f"{organism.get(r[2], r[2])[:34]}  [{class_names[r[0]]}, {100 * r[1]:.0f}% right]" for r in picked], fontsize=8.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cls, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(picked), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i, r in enumerate(picked):
        ax.add_patch(Rectangle((r[0] - 0.44, i - 0.44), 0.88, 0.88, fill=False, ec=INK, lw=1.6, zorder=6))
        for c in range(n_cls):
            if matrix[i, c] >= 0.25:
                ax.text(c, i, f"{100 * matrix[i, c]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if matrix[i, c] > 0.5 else INK)
    fig.suptitle(f"The {per_class} test genomes per class the model gets most wrong: mean predicted probability of each class",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=11, color=INK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mean probability over the genome's fragments", color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, colors=INK_2)
    fig.text(0.01, 0.005, "Outlined cell = the true class. Numbers are percentages, shown where at least 25%.", color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(out / "07_hardest_genomes.png", dpi=170)
    plt.close(fig)
    summary["hardest_genomes"] = [{"accession": r[2], "organism": organism.get(r[2], ""), "class": class_names[r[0]], "accuracy": r[1]} for r in picked]
    write_csv(out / "data" / "07_hardest_genomes.csv", ["accession", "organism", "true_class", "fragments", "accuracy"] + [f"mean_p_{n}" for n in class_names],
              [[r[2], organism.get(r[2], ""), class_names[r[0]], r[4], r[1]] + list(r[3]) for r in picked])


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="experiments/baselines/ce_rc_fam1000_s1/model.pt")
    ap.add_argument("--data-dir", default="data/datasets/taxa8fam_s1")
    ap.add_argument("--fragment-size", type=int, default=1000)
    ap.add_argument("--out", default="figures/xai")
    ap.add_argument("--per-class", type=int, default=30, help="fragments per class for attribution-based figures")
    ap.add_argument("--only", nargs="+", type=int, help="figure numbers to draw, e.g. --only 2 4")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, class_names, x, y, genomes, data_dir = load(args)
    probs = probabilities(model, x)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"model": args.model, "data_dir": args.data_dir, "fragments": int(len(y)),
                    "accuracy": float((probs.argmax(1) == y).mean()), "classes": class_names})
    want = set(args.only) if args.only else set(range(1, 8))
    if 1 in want:
        fig_examples(model, x, y, probs, class_names, out, rng)
    if 2 in want:
        fig_deletion(model, x, y, probs, class_names, out, rng, args.per_class, summary)
    if 3 in want:
        fig_randomization(model, x, y, probs, out, rng, max(args.per_class // 3, 5), summary)
    if 4 in want:
        fig_shuffle(model, x, y, class_names, out, rng, 150, summary)
    if 5 in want:
        fig_gc(x, y, probs, class_names, out, summary)
    if 6 in want:
        fig_motifs(model, x, y, class_names, out, rng, 400, summary)
    if 7 in want:
        fig_genomes(probs, y, genomes, class_names, data_dir, out, summary)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("hardest_genomes", "classes")}, indent=2))


if __name__ == "__main__":
    main()
