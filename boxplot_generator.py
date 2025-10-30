# Boxplot graph generator

import matplotlib.pyplot as plt
import numpy as np

data_config = {
        "MLP": {
                "gt95": 50,
                "lt5": 10,
            },
        
        "CNN": {
                "gt95": 62,
                "lt5": 7,
            },
        
        "LSTM": {
                "gt95": 65,
                "lt5": 6,
            },
        
        "ChebConv": {
                "gt95": 72,
                "lt5": 4,
            },
        
        "ARMAConv": {
                "gt95": 76,
                "lt5": 1,
            },
        
        "ACEOT": {
                "gt95": 82,
                "lt5": 0.5,
            },
        }


# Turn data_config into plottable data
y_ax_model_names = [key for key in data_config.keys()]

stats = {
    "lt5": np.zeros(len(y_ax_model_names)),
    "gt95": np.zeros(len(y_ax_model_names)),
    }
for i, (_, model_stats) in enumerate(data_config.items()):
    stats["gt95"][i] = model_stats["gt95"]
    stats["lt5"][i] = model_stats["lt5"]


# Inialize the graph
figure, axes = plt.subplots(layout='constrained')


# Turn into a horizontal boxplot
width = 0.25
counter = 1
x = np.arange(len(y_ax_model_names)) 

label_map = {
    "gt95": r"$F_1 \geq 95$",
    "lt5":  r"$F_1 \leq 5$",
}

colors = ["#f797a5", "#97f798"]
for i, (stat_name, stat_data) in enumerate(stats.items()):
    offset = x + (width*counter)
    bar = axes.bar( offset, stat_data, width, label=label_map[stat_name], color=colors[i] )
    axes.bar_label(bar, padding = 3)
    counter += 1


# Add labels and title for clarity
axes.set_ylabel("Percentage (%)")
axes.set_ylim(0, 90)
axes.set_xticks(x + 1.5*width, y_ax_model_names)
axes.legend(loc='upper left', ncols=2)
axes.set_title('Box Chart Example')


# Display the plot
plt.show()

