from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle

import utils.math_ops as math_ops


def set_axis_info(axis, title_x='x', title_y='y', x_range=None, y_range=None):
    if x_range:
        axis.set_xlim(x_range)
    if y_range:
        axis.set_ylim(y_range)
    axis.set_xlabel(title_x)
    axis.set_ylabel(title_y)


def get_dummy_element():
    return Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


def add_point(ax, pos, x_range, y_range, name, color):
    x, y = pos
    if math_ops.is_in_range(x, x_range) and math_ops.is_in_range(y, y_range):
        ax.scatter(x, y, marker='.', facecolors='none', edgecolor=color, zorder=1)
        ax.text(math_ops.scale_in_range(x, x_range, 0.01), math_ops.scale_in_range(y, y_range, 0.01),
                name, color=color, size=8, zorder=2)
    else:
        act_x, act_y = (x, x), (y, y)
        ha, va = 'center', 'center'
        if not math_ops.is_in_range(y, y_range):
            if y < y_range[0]:
                act_y = (math_ops.scale_in_range(y_range[0], y_range, 0.05),
                         math_ops.scale_in_range(y_range[0], y_range, 0.01))
                va = 'bottom'
            else:
                act_y = (math_ops.scale_in_range(y_range[1], y_range, -0.05),
                         math_ops.scale_in_range(y_range[1], y_range, -0.01))
                va = 'top'
        if not math_ops.is_in_range(x, x_range):
            if x < x_range[0]:
                act_x = (math_ops.scale_in_range(x_range[0], x_range, 0.05),
                         math_ops.scale_in_range(x_range[0], x_range, 0.01))
                ha = 'left'
            else:
                act_x = (math_ops.scale_in_range(x_range[1], x_range, -0.05),
                         math_ops.scale_in_range(x_range[1], x_range, -0.01))
                ha = 'right'
        arrow = FancyArrowPatch((act_x[0], act_y[0]), (act_x[1], act_y[1]),
                                mutation_scale=5.,
                                arrowstyle=ArrowStyle('-|>'),
                                fc=color, ec=color)
        ax.add_patch(arrow)
        ax.text(act_x[0], act_y[0], name, ha=ha, va=va, color=color, size=8, zorder=2)
