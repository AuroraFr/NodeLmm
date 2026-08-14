"""
Custom legend handlers that draw mini trajectory schematics
for each profile in the PDP legend.

Usage:
    from profile_legend import get_profile_legend_handles
    handles, labels = get_profile_legend_handles(PROFILE_ORDER, PROFILE_COLOURS, PROFILE_LABELS)
    ax.legend(handles=handles, labels=labels, handler_map=..., ...)

Or simply:
    from profile_legend import add_profile_legend
    add_profile_legend(ax, profiles_in_plot)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
import numpy as np


# ── Profile shapes (normalised to [0,1] × [0,1]) ────────────────────

def _shape_stable_low():
    """Flat line at bottom."""
    x = [0, 1]
    y = [0.2, 0.2]
    return x, y

def _shape_stable_high():
    """Flat line at top."""
    x = [0, 1]
    y = [0.8, 0.8]
    return x, y

def _shape_late_spike():
    """Q25 → Q75: low then step up at midpoint."""
    x = [0, 0.48, 0.52, 1]
    y = [0.2, 0.2, 0.8, 0.8]
    return x, y

def _shape_late_decline():
    """Q75 → Q25: high then step down at midpoint."""
    x = [0, 0.48, 0.52, 1]
    y = [0.8, 0.8, 0.2, 0.2]
    return x, y

def _shape_gradual_rise():
    """Linear rise from low to high."""
    x = [0, 1]
    y = [0.2, 0.8]
    return x, y

def _shape_gradual_decline():
    """Linear decline from high to low."""
    x = [0, 1]
    y = [0.8, 0.2]
    return x, y


PROFILE_SHAPES = {
    "stable_low":      _shape_stable_low,
    "stable_high":     _shape_stable_high,
    "late_spike":      _shape_late_spike,
    "late_decline":    _shape_late_decline,
    "early_burden":    _shape_late_decline,   # alias
    "gradual_rise":    _shape_gradual_rise,
    "gradual_decline": _shape_gradual_decline,
}


# ── Custom legend handler ────────────────────────────────────────────

class ProfileSchemaHandler(HandlerBase):
    """
    Draws a mini trajectory schematic in the legend.
    """
    def __init__(self, color, shape_func, **kwargs):
        self.color = color
        self.shape_func = shape_func
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Get shape in [0,1] normalised coords
        sx, sy = self.shape_func()

        # Scale to legend box
        pad = 2
        xs = [pad + xi * (width - 2 * pad) for xi in sx]
        ys = [ydescent + yi * height for yi in sy]

        line = plt.Line2D(xs, ys,
                          color=self.color, linewidth=2.0,
                          solid_capstyle='round',
                          transform=trans)
        return [line]


# ── Convenience function ─────────────────────────────────────────────

def add_profile_legend(ax, profiles_in_plot, profile_colours, profile_labels,
                       **legend_kwargs):
    """
    Add a legend with mini trajectory schematics to an axes.

    Args:
        ax:                matplotlib Axes
        profiles_in_plot:  list of profile names present in the plot
        profile_colours:   dict {profile_name: color}
        profile_labels:    dict {profile_name: display label}
        **legend_kwargs:   passed to ax.legend()
    """
    handles = []
    labels = []
    handler_map = {}

    for pname in profiles_in_plot:
        if pname not in PROFILE_SHAPES:
            continue
        color = profile_colours.get(pname, "grey")
        label = profile_labels.get(pname, pname)

        # Dummy handle (the handler_map overrides rendering)
        handle = mpatches.Patch(color=color, label=label)
        handles.append(handle)
        labels.append(label)

        handler_map[handle] = ProfileSchemaHandler(
            color=color,
            shape_func=PROFILE_SHAPES[pname],
        )

    defaults = dict(
        loc='best', fontsize=9, framealpha=0.9,
        handlelength=3.0, handleheight=1.5,
        borderpad=0.8, labelspacing=0.6,
    )
    defaults.update(legend_kwargs)

    ax.legend(handles=handles, labels=labels,
              handler_map=handler_map, **defaults)