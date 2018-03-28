import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import sttomog


rho = np.array([
    [0, 0, 0, 0],
    [0, 1, 1j, 0],
    [0, -1j, 1, 0],
    [0, 0, 0, 0]
])

sttomog.plot_rho(rho)
plt.show()
plt.clf()

sttomog.plot_rho(rho, real_imag="imag", box_size=.4, color_style="bw")
plt.show()
plt.clf()

cm_colors = [
    '#ff7764',
    '#ffc1ff',
    '#85aaff',
    '#ffc1ff',
    '#ff7764'
]
cm_values = range(len(cm_colors))
color_list = []
for v, c in zip(cm_values, cm_colors):
    color_list.append((v / np.max(cm_values), c))
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', color_list)

sttomog.plot_rho(rho, colormap=custom_cmap)
plt.show()
plt.clf()
