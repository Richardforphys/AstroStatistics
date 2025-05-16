# my_matplotlib_style.py
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 5),
    'grid.color': 'gray',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.grid': True,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'axes.titlepad': 20,
    'figure.autolayout': True,
})
