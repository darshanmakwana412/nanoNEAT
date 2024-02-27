"""
utilities for plotting the genome
"""

import matplotlib.patches as patches

def plot_square(ax, center, node, size=0.4, is_hidden=False):
    color = "green" if is_hidden else "grey"
    ax.add_patch(patches.Rectangle((center[0]-size/2, center[1]-size/2), size, size, fill=True, color=color, linewidth=0, linestyle='-', zorder=5))
    ax.text(center[0], center[1], str(node), ha='center', va='center', weight='bold', color="black", fontsize=15, zorder=6)

def plot_edge(ax, start, end, weight, enabled):
    if enabled:
        if weight > 0:
            color = "blue"
        else:
            color = "red"
        linestyle = '-'
    else:
        color = "grey"
        linestyle = ':'
    linewidth = abs(weight * 10) + 1
    ax.annotate("", xy=end, xycoords='data', xytext=start, textcoords='data', arrowprops=dict(arrowstyle="->", color=color, linestyle=linestyle, linewidth=linewidth, zorder=3))