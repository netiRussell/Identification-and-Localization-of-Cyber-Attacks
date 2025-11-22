# Boxplot graph generator

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

data_ieee14_SW = {
    "MLP":      {"gt95": 20,    "lt5": 5},
    "CNN":      {"gt95": 20,    "lt5": 5},
    "LSTM":     {"gt95": 20,    "lt5": 5},
    "ChebConv": {"gt95": 20,    "lt5": 5},
    "ARMAConv": {"gt95": 20,    "lt5": 5},
    "ACEOT":    {"gt95": 20,    "lt5": 5},
}

data_ieee14_NW = {
    "MLP":      {"gt95": 20,    "lt5": 5},
    "CNN":      {"gt95": 20,    "lt5": 5},
    "LSTM":     {"gt95": 20,    "lt5": 5},
    "ChebConv": {"gt95": 20,    "lt5": 5},
    "ARMAConv": {"gt95": 20,    "lt5": 5},
    "ACEOT":    {"gt95": 20,    "lt5": 5},
}

data_ieee300_SW = {
    "MLP":      {"gt95": 54.95, "lt5": 37.71},
    "CNN":      {"gt95": 50,    "lt5": 5},
    "LSTM":     {"gt95": 62.85,    "lt5": 13.07},
    "ChebConv": {"gt95": 66.13,    "lt5": 9.38},
    "ARMAConv": {"gt95": 70.47, "lt5": 3.72},
    "ACEOT":    {"gt95": 81.98, "lt5": 6.23},
}

data_ieee300_NW = {
    "MLP":      {"gt95": 42.67, "lt5": 28.33},
    "CNN":      {"gt95": 20,    "lt5": 5},
    "LSTM":     {"gt95": 45.67,    "lt5": 5.33},
    "ChebConv": {"gt95": 49,    "lt5": 4.33},
    "ARMAConv": {"gt95": 47.61, "lt5": 3.00},
    "ACEOT":    {"gt95": 51.00, "lt5": 0.71},
}

networks = ["IEEE14", "IEEE300"]
data_by_net = {
    "IEEE14": data_ieee14_NW,
    "IEEE300": data_ieee300_NW,
}


# Text parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# Prepare grid
models = list(data_by_net["IEEE14"].keys())
n_models = len(models)
metrics = ["lt5", "gt95"]

label_map = {
    "gt95": r"$F_1 \geq 95$",
    "lt5":  r"$F_1 \leq 5$",
}
colors = {
    "lt5":  "#f797a5",
    "gt95": "#97f798",
}

fig = plt.figure(figsize=(5.5, 4.5))
ax = fig.add_subplot(111, projection='3d')

# Bring the 3D box closer / better framed
ax.set_box_aspect((1.4, 0.7, 0.8))

# Bar geometry
dx = 0.35        # bar width in x
dy = 0.25        # bar depth in y
offset = 0.15    # x-offset between lt5 and gt95

xpos_list = []
ypos_list = []
zpos_list = []
dx_list = []
dy_list = []
dz_list = []
color_list = []

for i_model, model in enumerate(models):
    for i_net, net in enumerate(networks):
        model_stats = data_by_net[net][model]

        # lt5 bar (shift left)
        x_lt5 = i_model - offset
        y_lt5 = i_net
        z0 = 0
        val_lt5 = model_stats["lt5"]

        xpos_list.append(x_lt5)
        ypos_list.append(y_lt5)
        zpos_list.append(z0)
        dx_list.append(dx)
        dy_list.append(dy)
        dz_list.append(val_lt5)
        color_list.append(colors["lt5"])

        # gt95 bar (shift right)
        x_gt95 = i_model + offset
        y_gt95 = i_net
        val_gt95 = model_stats["gt95"]

        xpos_list.append(x_gt95)
        ypos_list.append(y_gt95)
        zpos_list.append(z0)
        dx_list.append(dx)
        dy_list.append(dy)
        dz_list.append(val_gt95)
        color_list.append(colors["gt95"])

xpos = np.array(xpos_list)
ypos = np.array(ypos_list)
zpos = np.array(zpos_list)
dx = np.array(dx_list)
dy = np.array(dy_list)
dz = np.array(dz_list)

# Build colors with different alpha for IEEE14 (row y == 0)
facecolors = []
for y, base_c in zip(ypos, color_list):
    alpha = 0.6 if y == 0 else 1.0   # IEEE14 semi-transparent, IEEE300 opaque
    facecolors.append(mcolors.to_rgba(base_c, alpha=alpha))

# Box plot
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=facecolors, edgecolor="black", linewidth=0.6)

# Axes labels and ticks
ax.set_zlabel("Percentage (%)", labelpad=0)
ax.zaxis.set_label_coords(0.9, 0.5)

ax.set_xticks(range(n_models))
ax.set_xticklabels(models, rotation=15, ha="right", va="top")

ax.set_yticks(range(len(networks)))
ax.set_yticklabels([])

ax.tick_params(axis='x', pad=-2)  # move x tick labels toward the axis
ax.tick_params(axis='y', pad=0)  # move y tick labels toward the axis
ax.tick_params(axis='z', pad=0, labelsize=7)

ax.set_zticks(np.arange(0, 55, 5))  # 86 SW, 55 NW


# put Networks next to their rows
z_mid = 1  # height where the network names will sit
x_label_pos = 7  # slightly in front/left of the first bars

for i_net, net in enumerate(networks):
    ax.text(
        x_label_pos, i_net, z_mid,
        net,
        ha="center", va="center",
        rotation=90,  # vertical labels
    )


# Legend (fake handles with patches)
'''
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=colors["lt5"],  edgecolor="black", label=label_map["lt5"]),
    Patch(facecolor=colors["gt95"], edgecolor="black", label=label_map["gt95"]),
]
ax.legend(handles=legend_handles, loc="upper left", ncol=2)
'''

# Viewing angle so the two "rows" are visible
ax.view_init(elev=25, azim=-60)

# Save
plt.savefig("./plots/boxplot.png")

plt.show()
print("Figure size (inches):", fig.get_size_inches())


