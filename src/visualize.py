from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
          "#9C27B0", "#00BCD4", "#795548"]

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Vẽ Confusion Matrix chuẩn hóa. Trả về Figure (chưa lưu file)."""
    n       = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)),
                           facecolor="white")

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, where=row_sum != 0,
                        out=np.zeros_like(cm, dtype=float))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names,
                                                 rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    thresh = cm_norm.max() / 2
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i,j] > thresh else "black")
    plt.tight_layout()
    return fig

def save_epoch_plot(history, save_dir, label=""):
    if len(history) < 2:
        return

    ep    = [r["epoch"]      for r in history]
    loss  = [r["loss"]       for r in history]
    map50 = [r["map_50"]     for r in history]
    map_  = [r["map"]        for r in history]
    mar   = [r["mar_100"]    for r in history]
    lr    = [r["lr"]         for r in history]
    best  = [r["best_map50"] for r in history]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.suptitle(f"Training Results  {label}  (epoch {ep[-1]})",
                 fontsize=13, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    panels = [
        ("Train Loss",      loss,  "#1f77b4", "Loss"),
        ("mAP@0.5",         map50, "#2ca02c", "mAP"),
        ("mAP@0.5:0.95",    map_,  "#ff7f0e", "mAP"),
        ("mAR@100",         mar,   "#d62728", "mAR"),
        ("Learning Rate",   lr,    "#9467bd", "LR"),
        ("mAP@0.5 vs Best", None,  "#2ca02c", "mAP"),
    ]
    for i, (title, y, color, ylabel) in enumerate(panels):
        ax = axs[i]
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel,  fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if i < 5:
            ax.plot(ep, y, color=color, linewidth=1.8,
                    marker="o", markersize=2.5)
        else:
            ax.plot(ep, map50, color=color, linewidth=1.8,
                    marker="o", markersize=2.5, label="mAP@0.5")
            ax.plot(ep, best,  color=color, linewidth=1.5,
                    linestyle="--", alpha=0.5, label="Best")
            ax.legend(fontsize=8)

    if map50:
        bi = map50.index(max(map50))
        axs[1].axvline(ep[bi], color="gray", linestyle=":", alpha=0.5)
        axs[1].annotate(f"  best={max(map50):.3f}",
                        xy=(ep[bi], max(map50)), fontsize=8, color="gray",
                        xytext=(4, -14), textcoords="offset points")

    fig.text(0.5, 0.005,
             f"Epoch {ep[-1]}  |  Loss {loss[-1]:.4f}  |  "
             f"mAP@0.5 {map50[-1]:.4f}  |  Best {max(best):.4f}",
             ha="center", fontsize=9, color="#555", style="italic")

    plt.savefig(Path(save_dir) / "results.png",
                dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_kfold_plot(all_histories, save_dir, k):
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(f"{k}-Fold Cross Validation Results",
                 fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    best_maps      = []
    all_map_curves = []

    for i, history in enumerate(all_histories):
        c     = COLORS[i % len(COLORS)]
        ep    = [r["epoch"]   for r in history]
        map50 = [r["map_50"]  for r in history]
        mar   = [r["mar_100"] for r in history]
        best  = max(map50) if map50 else 0.0
        best_maps.append(best)
        all_map_curves.append(map50)

        ax1.plot(ep, map50, color=c, linewidth=1.8,
                 marker="o", markersize=3,
                 label=f"Fold {i+1} (best={best:.3f})")
        ax1.scatter([ep[map50.index(best)]], [best],
                    color=c, s=70, zorder=5)
        ax3.plot(ep, mar, color=c, linewidth=1.5,
                 label=f"Fold {i+1}")

    max_len   = max(len(c) for c in all_map_curves)
    padded    = [c + [c[-1]] * (max_len - len(c)) for c in all_map_curves]
    avg_curve = [sum(col) / len(col) for col in zip(*padded)]
    avg       = sum(best_maps) / len(best_maps)
    ax1.plot(range(1, max_len + 1), avg_curve, color="black",
             linewidth=2.5, linestyle="--", label=f"TB ({avg:.3f})")

    for ax, title, ylabel in [
        (ax1, "mAP@0.5 theo Epoch", "mAP@0.5"),
        (ax3, "mAR@100 theo Epoch", "mAR@100"),
    ]:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.3)

    fold_labels = [f"Fold {i+1}" for i in range(k)]
    bars = ax2.bar(fold_labels, best_maps, color=COLORS[:k],
                   alpha=0.85, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, best_maps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    best_i = best_maps.index(max(best_maps))
    bars[best_i].set_edgecolor("gold"); bars[best_i].set_linewidth(2.5)
    ax2.axhline(avg, color="red", linestyle="--",
                linewidth=1.5, label=f"TB = {avg:.3f}")
    ax2.set_title("Best mAP@0.5 mỗi Fold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("mAP@0.5")
    ax2.set_ylim(0, min(1.0, max(best_maps) * 1.15))
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")

    TAIL     = 10
    box_data = [[r["map_50"] for r in h][-TAIL:] for h in all_histories]
    bp       = ax4.boxplot(box_data, labels=fold_labels,
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], COLORS[:k]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax4.set_title(f"Phân phối mAP@0.5 ({TAIL} epoch cuối)\n"
                  "Box hẹp = ổn định cao",
                  fontsize=10, fontweight="bold")
    ax4.set_ylabel("mAP@0.5"); ax4.grid(True, alpha=0.3, axis="y")

    std = (sum((f - avg) ** 2 for f in best_maps) / len(best_maps)) ** 0.5
    fig.text(0.5, 0.005,
             f"K={k}  |  Avg={avg:.4f}  |  Std={std:.4f}  |  "
             f"Best=Fold {best_i+1} ({max(best_maps):.4f})",
             ha="center", fontsize=9, color="#444", style="italic")

    plt.savefig(Path(save_dir) / "kfold_results.png",
                dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  [OK] kfold_results.png")