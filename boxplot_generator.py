# Boxplot graph generator

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np


data_configNW = {
        "MLP": {
                "gt95": 42.67,
                "lt5": 28.33,
            },
        
        "CNN": {
                "gt95": 20,
                "lt5": 5,
            },
        
        "LSTM": {
                "gt95": 20,
                "lt5": 5,
            },
        
        "ChebConv": {
                "gt95": 20,
                "lt5": 5,
            },
        
        "ARMAConv": {
                "gt95": 47.61,
                "lt5": 3.00,
            },
        
        "ACEOT": {
                "gt95": 51.00,
                "lt5": 0.71,
            },
        }

data_configSW = {
        "MLP": {
                "gt95": 54.95,
                "lt5": 37.71,
            },
        
        "CNN": {
                "gt95": 50,
                "lt5": 5,
            },
        
        "LSTM": {
                "gt95": 50,
                "lt5": 5,
            },
        
        "ChebConv": {
                "gt95": 50,
                "lt5": 5,
            },
        
        "ARMAConv": {
                "gt95": 70.47,
                "lt5": 3.72,
            },
        
        "ACEOT": {
                "gt95": 81.98,
                "lt5": 6.23,
            },
        }

# Inialize the graph
figure, axes = plt.subplots(layout='constrained', figsize=(6, 3.5))

# Text parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Plot parameters
width = 0.25
gap = 0.10 

# Turn data_config into plottable data
y_ax_model_names = [key for key in data_configSW.keys()]

stats = {
    "lt5": np.zeros(len(y_ax_model_names)),
    "gt95": np.zeros(len(y_ax_model_names)),
    }
for i, (_, model_stats) in enumerate(data_configSW.items()):
    stats["gt95"][i] = model_stats["gt95"]
    stats["lt5"][i] = model_stats["lt5"]


# Generate a horizontal boxplot
counter = 1
x = np.arange(len(y_ax_model_names)) 

label_map = {
    "gt95": r"$F_1 \geq 95$",
    "lt5":  r"$F_1 \leq 5$",
}

colors = ["#f797a5", "#97f798"]


# Left bar
bar1 = axes.bar(
    x - (width/2 + gap/2), stats["lt5"], width,
    label=label_map["lt5"], color=colors[0],
    edgecolor="black", linewidth=1
)

# Right bar
bar2 = axes.bar(
    x + (width/2 + gap/2), stats["gt95"], width,
    label=label_map["gt95"], color=colors[1],
    edgecolor="black", linewidth=1
)

# Add labels above bars
axes.bar_label(bar1, padding=3, fmt="%.2f")
axes.bar_label(bar2, padding=3, fmt="%.2f")


# Add labels and title for clarity
axes.set_ylabel("Percentage (%)")
axes.set_ylim(0, 95)
axes.set_xticks(x, y_ax_model_names)
axes.legend(loc='upper left', ncols=2)
axes.set_title('SW evaluation of localization performance')


# Display the plot
plt.show()

