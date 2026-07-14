"""Draw the feed-forward network diagram for the report (report/nn_architecture.pdf).

Network: NN_5(4, 30, 30, 30, 30, 88) with ELU between hidden layers and a linear (id) output.
The figure mirrors the previous hand-drawn diagram's style but with the correct FOUR hidden
layers (the old PDF showed only three). Run with the project's python:

    python make_nn_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

BLUE = "#5b9bd5"          # connection / arrow colour
NODE_LW = 1.4
CONN_LW = 0.5

# Column x-positions: input, 4 hidden, output.
XS = [0.0, 1.8, 3.6, 5.4, 7.2, 9.4]
R = 0.20                  # node radius

# Visible-node y-positions per column (representative nodes; vertical dots fill the gaps).
input_y  = [3.4, 2.6, 1.8, 1.0]                       # 4 nodes  theta_1..theta_4
hidden_y = [4.0, 2.1, 0.2]                            # "1", (middle), "30"
output_y = [3.7, 2.55, 1.4, 0.25]                    # "Output 1" .. "Output 88"

input_labels  = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"]
hidden_labels = ["1", "", "30"]
headers = ["Input\nlayer", "1st Hidden\nlayer", "2nd Hidden\nlayer",
           "3rd Hidden\nlayer", "4th Hidden\nlayer", "Output\nlayer"]

fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(-1.3, 11.4)
ax.set_ylim(-0.6, 5.6)
ax.axis("off")

# columns of visible nodes, in draw order
cols = [input_y, hidden_y, hidden_y, hidden_y, hidden_y, output_y]


def draw_node(x, y, label=""):
    ax.add_patch(Circle((x, y), R, facecolor="white", edgecolor="black",
                         lw=NODE_LW, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=10, zorder=4)


# 1) connection lines (drawn first, behind nodes)
for c in range(len(XS) - 1):
    for y0 in cols[c]:
        for y1 in cols[c + 1]:
            ax.plot([XS[c] + R, XS[c + 1] - R], [y0, y1],
                    color=BLUE, lw=CONN_LW, alpha=0.7, zorder=1)

# 2) nodes + vertical dots
# input
for y, lab in zip(input_y, input_labels):
    draw_node(XS[0], y, lab)
    ax.annotate("", xy=(XS[0] - R, y), xytext=(XS[0] - 0.85, y),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
    ax.text(XS[0] - 0.95, y, f"Input {input_labels.index(lab)+1}",
            ha="right", va="center", fontsize=9)

# hidden layers
for c in range(1, 5):
    x = XS[c]
    for y, lab in zip(hidden_y, hidden_labels):
        draw_node(x, y, lab)
    # vertical dots between the three representative nodes
    for ya, yb in [(hidden_y[0], hidden_y[1]), (hidden_y[1], hidden_y[2])]:
        ym = 0.5 * (ya + yb)
        for dy in (-0.18, 0.0, 0.18):
            ax.plot(x, ym + dy, marker="o", ms=1.8, color="black", zorder=3)

# output
for k, y in enumerate(output_y):
    draw_node(XS[5], y)
    ax.annotate("", xy=(XS[5] + 0.85, y), xytext=(XS[5] + R, y),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
# output dots between the middle nodes
for ya, yb in [(output_y[1], output_y[2])]:
    ym = 0.5 * (ya + yb)
    for dy in (-0.18, 0.0, 0.18):
        ax.plot(XS[5], ym + dy, marker="o", ms=1.8, color="black", zorder=3)
ax.text(XS[5] + 0.95, output_y[0], "Output 1", ha="left", va="center", fontsize=9)
ax.text(XS[5] + 0.95, output_y[-1], "Output 88", ha="left", va="center", fontsize=9)

# 3) column headers
for x, h in zip(XS, headers):
    ax.text(x, 5.35, h, ha="center", va="center", fontsize=11, fontweight="bold")

# 4) activation arrows between columns (ELU x4, id x1), drawn just below the headers
act_labels = ["ELU", "ELU", "ELU", "ELU", "id"]
for c in range(len(XS) - 1):
    xm = 0.5 * (XS[c] + XS[c + 1])
    ax.annotate("", xy=(xm + 0.45, 4.85), xytext=(xm - 0.45, 4.85),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8))
    ax.text(xm, 4.55, act_labels[c], ha="center", va="center", fontsize=9, style="italic")

plt.tight_layout()
fig.savefig("../report/nn_architecture.pdf", bbox_inches="tight")
print("wrote ../report/nn_architecture.pdf")
