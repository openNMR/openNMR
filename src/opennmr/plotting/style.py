"""
Plotting style helpers shared by ``oned``/``twod``.

``apply_tick_style``/``apply_spine_style``/``get_stack_colormap``/
``get_contour_cmap``/``downsample_2d``/``save_figure`` are ``ax``-
scoped (they only ever touch the ``Axes``/data passed to them), so
they're safe to call from GUI code embedding a canvas -- ported from
``nmrtools2``'s ``plotting.py``. ``downsample_2d`` has since diverged
from that original (still evenly-spaced decimation there) -- see its
own docstring for why.

``setup_plotting`` is the one exception: it calls
``plt.rcParams.update(...)``, a *global* matplotlib style change. That
is harmless for a one-off standalone figure (a script or notebook cell
calling one of ``plotting.oned``/``plotting.twod``'s convenience
wrappers), which is the only place this package still calls it -- but
it must never be called from GUI code, since restyling one embedded
canvas would silently restyle every other open canvas/tab sharing the
same matplotlib process.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

# Registers the colour-blind-friendly qualitative palettes (Okabe-Ito,
# Tol) with matplotlib, so every colormap argument in the package --
# notebook API included, which imports this module -- accepts them.
from .accessibility import is_qualitative, register_palettes

register_palettes()

VALID_SAVE_FORMATS = ("png", "pdf", "svg")

#: Relative luminance above which a colour is treated as **too light to
#: use for a line on a white background**, and dropped from every
#: colormap this module hands out.
#:
#: Derived rather than picked: WCAG 1.4.11 sets 3:1 as the minimum
#: contrast for a graphical object, and contrast against white is
#: ``1.05 / (L + 0.05)``, so 3:1 means ``L <= 0.30``.
#:
#: The colours this removes are not marginal. Measured across the
#: colormaps this package offers, their lightest entries sit at
#: **1.0-1.4:1** against white -- pale yellow, near-white blue, the
#: white middle of every diverging map. On paper or a projector they
#: are simply not there. Even ``tab10``'s lightest manages only 2.0:1.
#:
#: Set to ``1.0`` to disable the filtering entirely; every function
#: below also takes a ``max_luminance`` argument for a one-off
#: override.
MAX_LINE_LUMINANCE = 0.30

#: How finely a colormap is sampled when working out which parts of it
#: are usable. 256 matches matplotlib's own default resolution.
_LUMINANCE_SAMPLES = 256

CONTOUR_CMAP_NAMES = [
    'magma', 'coolwarm', 'cividis', 'viridis', 'plasma', 'inferno',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn',
    'Spectral', 'bwr', 'seismic', 'Greys', 'Purples', 'Blues', 'Greens',
    'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
    'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
]


#: Never shrink a label below this (points) trying to make it fit --
#: past here it is unreadable, so a warning is more use than a smaller
#: font. See :func:`fit_labels_within_figure`.
LABEL_FIT_MIN_FONTSIZE = 5.0

#: Multiplier per shrink attempt, and how many attempts before giving
#: up. 0.85^6 ~ 0.38, i.e. a 16 pt label can reach ~6 pt.
_LABEL_FIT_FACTOR = 0.85
_LABEL_FIT_MAX_STEPS = 6


def _layout_quietly(fig) -> bool:
    """``fig.tight_layout()``, without letting matplotlib's own "Tight
    layout not applied" ``UserWarning`` reach the user.

    Returns whether the layout pass ran cleanly. That warning is not
    noise -- it is matplotlib saying the decorations do not fit, which
    is exactly the condition :func:`fit_labels_within_figure` then
    acts on -- but it names no spectrum and offers no remedy, so it is
    captured and answered rather than printed."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 -- a layout pass must never take a redraw down
            return False
    return not any("ight layout" in str(w.message) for w in caught)


#: Slack (pixels) before a label counts as clipped. A tick label at
#: the very end of an axis legitimately overhangs the figure edge by a
#: pixel or two, and matplotlib does not treat that as a problem.
_LABEL_FIT_TOLERANCE_PX = 2.0


def _axis_label_artists(fig):
    """The persistent text artists: axis labels and titles. These are
    single objects that live as long as the Axes, so setting a font
    size on one sticks."""
    for ax in fig.axes:
        for artist in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if artist.get_text():
                yield artist


def _tick_labelsize(axis) -> float:
    """The font size ``axis``'s tick labels are currently drawn at."""
    for label in axis.get_ticklabels():
        if label.get_text():
            return float(label.get_fontsize())
    return float(plt.rcParams.get("font.size", 10.0))


def _shrink_text_once(fig, factor: float, min_fontsize: float) -> bool:
    """Shrink every label, title and tick on ``fig`` by ``factor``,
    down to ``min_fontsize``. Returns whether anything actually moved.

    **Tick labels are shrunk through ``set_tick_params``, not by
    setting a size on the individual tick artists.** A Tick's label is
    not a persistent object the way an axis label is: matplotlib keeps
    a pool of them and creates more from the axis's *stored* tick
    parameters whenever a view needs more ticks than the pool holds.
    Writing a size onto the artists therefore shrank the ticks that
    happened to exist and left the stored size untouched, so the next
    pan, zoom or level scroll brought some ticks back at full size
    while the axis labels stayed small -- reported as the DOSY still
    having "tiny labels" while "the tickmarks now seem to be better
    matched". Going through the axis makes the change durable and
    keeps labels and ticks in step.
    """
    changed = False
    for artist in _axis_label_artists(fig):
        current = artist.get_fontsize()
        new = max(min_fontsize, current * factor)
        if new < current:
            artist.set_fontsize(new)
            changed = True
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            current = _tick_labelsize(axis)
            new = max(min_fontsize, current * factor)
            if new < current:
                axis.set_tick_params(which="both", labelsize=new)
                changed = True
    return changed


def _settle_label_positions(fig) -> None:
    """Put every axis label and title where the *current* layout
    actually puts it, before anything measures one.

    Matplotlib positions these during ``Axis.draw``, not when the Axes
    move. Between a ``tight_layout`` that repositioned the Axes and the
    next real draw, ``label.get_window_extent()`` therefore reports
    where the label was under the *previous* layout -- which for a
    freshly rebuilt figure is a position no layout ever chose.

    Measuring that reports a clip that is not there, and
    :func:`fit_labels_within_figure` then shrinks the whole figure's
    text trying to fix it. Measured on a DOSY map, whose Axes are
    rebuilt from a new gridspec on every redraw: at 8 x 6 inches, with
    both axis labels comfortably inside the figure, every label and
    tick came out at 6 pt against a requested 16. Reported as "ILT axes
    have tiny text". A 2D contour rebuilt by a level scroll had the
    same problem at a large label size.

    ``draw_without_rendering`` rather than poking
    ``Axis._update_label_position`` directly: that updates the label
    from tick boxes that are themselves stale, and measured *still*
    disagreed with a real draw by ~190 px on a shared-axis figure. A
    full artist walk is the only thing that agrees with what will be
    drawn, and it is cheap -- the expensive part of a contour is
    building its paths, which has already happened by the time this
    runs. Measured at 16 ms for a 512 x 4096 contour, against 0.3 s for
    the ``canvas.draw()`` this used to be assumed to cost.
    """
    try:
        fig.draw_without_rendering()
    except Exception:  # noqa: BLE001 -- a layout check must never take a redraw down
        pass


def clipped_labels(fig) -> list:
    """Axis labels and titles falling outside ``fig``'s own bounds --
    i.e. the ones a viewer would see cut off at an edge.

    **Deliberately not tick labels.** The label at either end of an
    axis routinely overhangs the figure edge slightly -- that is
    ordinary matplotlib behaviour, not a defect -- so testing them
    would report a perfectly good figure as clipped and shrink it for
    no reason (measured: a plain 6 x 4 inch plot, driven to the 5 pt
    floor by two end-of-axis tick labels). They are still *shrunk* as
    part of the remedy; they just do not *trigger* it.

    Measured against a renderer directly rather than by drawing the
    figure: a full draw of a large 2D contour costs 0.3-1.3 s, far too
    much to spend on a layout check, while measuring a text artist is
    well under a millisecond."""
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        try:
            renderer = fig._get_renderer()
        except Exception:  # noqa: BLE001 -- no renderer, nothing measurable
            return []
    _settle_label_positions(fig)
    bounds = fig.bbox
    tol = _LABEL_FIT_TOLERANCE_PX
    clipped = []
    for ax in fig.axes:
        for artist in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if not artist.get_text():
                continue
            try:
                box = artist.get_window_extent(renderer)
            except Exception:  # noqa: BLE001 -- an unmeasurable artist is not a clip
                continue
            if (box.x0 < bounds.x0 - tol or box.y0 < bounds.y0 - tol
                    or box.x1 > bounds.x1 + tol or box.y1 > bounds.y1 + tol):
                clipped.append(artist)
    return clipped


def fit_labels_within_figure(fig, *, min_fontsize: float = LABEL_FIT_MIN_FONTSIZE) -> dict:
    """Lay ``fig`` out so its axis labels are not cut off, shrinking
    the text as far as necessary (and no further) to achieve it.

    ``tight_layout`` alone is not a safeguard: it reserves room for
    decorations *if the figure is big enough*, and simply gives up
    (with a warning nobody sees) when it is not. A GUI canvas is sized
    by the window, so "not big enough" is an ordinary state, not an
    edge case -- measured, a 1D canvas below about 1.4 x 0.9 inches
    and a 12-row stacked Compare below about 2.5 inches tall both put
    the x label off the bottom of the figure entirely.

    So: lay out, measure, and if anything is clipped, shrink every
    label/tick/title by :data:`_LABEL_FIT_FACTOR` and lay out again,
    down to ``min_fontsize``. Returns
    ``{"fitted": bool, "steps": int, "clipped": [str, ...]}`` --
    ``fitted`` False means the text is at the floor and *still* does
    not fit, which is a real "make this panel bigger" situation the
    caller should surface rather than silently accept.

    Font sizes are changed on the *artists*, never on the caller's own
    style settings, so this is a display-time accommodation: the next
    rebuild re-applies the user's chosen sizes from scratch and a
    canvas that has been made larger again comes back at full size.
    """
    _layout_quietly(fig)
    clipped = clipped_labels(fig)
    steps = 0
    while clipped and steps < _LABEL_FIT_MAX_STEPS:
        if not _shrink_text_once(fig, _LABEL_FIT_FACTOR, min_fontsize):
            break          # everything is already at the floor
        _layout_quietly(fig)
        clipped = clipped_labels(fig)
        steps += 1
    return {
        "fitted": not clipped,
        "steps": steps,
        "clipped": sorted({a.get_text() for a in clipped}),
    }


def apply_tick_style(ax, direction='in', length=4.0, width=1.0, pad=4.0,
                      num_ticks=None, axis='both', minor=True, labelsize=None):
    """Apply consistent tick styling to an ``ax``: direction, size,
    and (optionally) an explicit target tick count.

    ``length``/``width`` : major tick length/width. Minor ticks (when
    ``minor=True``) are drawn at 3/4 the width and half the length.
    ``num_ticks`` : if given, an approximate target major-tick count
    via ``MaxNLocator`` on the requested axis/axes.
    ``labelsize`` : if given, sets the tick label font size directly
    (equivalent to ``setup_plotting``'s ``xtick.labelsize``/
    ``ytick.labelsize`` rcParams, but ax-scoped -- needed anywhere
    ``setup_plotting`` itself can't be called, e.g. a GUI canvas).
    """
    tick_kwargs = {} if labelsize is None else {'labelsize': labelsize}
    ax.tick_params(axis=axis, which='major', direction=direction,
                    length=length, width=width, pad=pad, **tick_kwargs)

    if minor:
        ax.minorticks_on()
        ax.tick_params(axis=axis, which='minor', direction=direction,
                        length=length / 2, width=0.75 * width)
    else:
        ax.minorticks_off()

    if num_ticks is not None:
        if axis in ('x', 'both'):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
        if axis in ('y', 'both'):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))

    return ax


def apply_spine_style(ax, top=False, right=False, bottom=True, left=True, linewidth=None):
    """Show/hide individual axis spines. Defaults match the package's
    usual "open" plot style (top and right hidden).

    ``linewidth`` : if given, also sets every spine's line width
    (equivalent to ``setup_plotting``'s ``axes.linewidth`` rcParam,
    but ax-scoped) -- needed anywhere ``setup_plotting`` itself can't
    be called, e.g. a GUI canvas, since a spine's width otherwise just
    falls back to matplotlib's own default regardless of what
    ``apply_tick_style`` was given."""
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)
    if linewidth is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
    return ax


def set_colorbar_gap(fig, gap_pt) -> list:
    """Set the gap between each vertical colour bar in ``fig`` and the
    plot it belongs to, to ``gap_pt`` points (1/72 inch at print size).
    User request (2026-10-01): "a way to control the spacing between the
    colourbar and the heatmap ... or the surface plots of the tensors".

    * A 2D plot (a Czjzek map) has its right edge moved to ``gap_pt``
      from the bar. The bar stays where it is, so nothing to its right
      (a spectrum panel) is disturbed and nothing is pushed off the
      figure.
    * A 3D picture (a tensor surface) keeps its place and the bar moves,
      measured from the drawn box's right-hand corner at the current
      view. It stops at the figure's edge, labels included.

    ``gap_pt`` None or negative leaves the figure as drawn. Call after
    any ``tight_layout``, which re-spaces the plots and undoes it.
    Returns the colour-bar Axes it moved a gap for.
    """
    if gap_pt is None or float(gap_pt) < 0:
        return []
    gap = float(gap_pt) / 72.0 / fig.get_figwidth()     # figure fraction
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = None
    done = []
    for cax in fig.axes:
        bar = getattr(cax, "_colorbar", None)
        if bar is None or getattr(bar, "orientation", "vertical") != "vertical":
            continue
        cpos = cax.get_position()
        owners = [ax for ax in fig.axes
                  if ax is not cax and getattr(ax, "_colorbar", None) is None
                  and ax.get_position().x0 < cpos.x0
                  and min(ax.get_position().y1, cpos.y1) > max(ax.get_position().y0, cpos.y0)]
        if not owners:
            continue
        owner = max(owners, key=lambda ax: ax.get_position().x1 if ax.get_position().x1 <= cpos.x0 + 1e-6
                    else -1.0)
        if getattr(owner, "name", "") == "3d":
            from mpl_toolkits.mplot3d import proj3d
            M = owner.get_proj()
            xs = []
            for x in owner.get_xlim3d():
                for y in owner.get_ylim3d():
                    for z in owner.get_zlim3d():
                        px, py, _ = proj3d.proj_transform(x, y, z, M)
                        xs.append(owner.transData.transform((px, py))[0])
            right = max(xs) / fig.bbox.width
            x0 = right + gap
            if renderer is not None:
                tight = cax.get_tightbbox(renderer)
                overhang = (tight.x1 - cax.get_window_extent(renderer).x1) / fig.bbox.width
                x0 = min(x0, 1.0 - cpos.width - overhang - 2.0 / fig.bbox.width)
            cax.set_position([x0, cpos.y0, cpos.width, cpos.height])
        else:
            pos = owner.get_position(original=True)
            x1 = cpos.x0 - gap
            if x1 - pos.x0 < 0.25 * pos.width:
                continue           # would squash the plot: leave it
            owner.set_position([pos.x0, pos.y0, x1 - pos.x0, pos.height], which="both")
        done.append(cax)
    return done


def apply_default_font(font='Arial'):
    """Set the process-wide matplotlib default typeface -- both for
    ordinary text (``font.family``/``font.sans-serif``, falling back
    through a chain of very-likely-installed sans faces so a machine
    without ``font`` installed still gets *a* clean sans-serif rather
    than a warning and a silent substitution to matplotlib's own
    default) and for mathtext (``mathtext.rm``/``it``/``bf``, which
    govern the actual glyphs behind every ``\mathregular{}`` in a
    nuclide axis label -- see :func:`opennmr.core.registry.NUC_label`
    -- since those are never reachable through a per-``Text``-artist
    ``set_fontfamily()`` call).

    This is deliberately **not** ``setup_plotting``: that also pins
    tick/spine sizing and figure dimensions, which the GUI canvas has
    to keep per-instance (see this module's own docstring for why the
    GUI never calls ``setup_plotting`` itself). Typeface is different
    -- there is exactly one matplotlib font manager and one mathtext
    engine per process, so "the default face" can only ever be one
    global choice, however many canvases are open, and setting it
    once is not the kind of per-canvas restyling that docstring warns
    against.

    Called once at GUI startup (see
    :class:`opennmr.gui.main_window.MainWindow`) so the very first
    plot already has a pleasant default instead of matplotlib's own
    DejaVu Sans -- explicit user request: "add Arial as the default
    font, not the unpleasant matplotlib default". ``setup_plotting``
    also calls this, so a standalone script/notebook figure gets the
    same default typeface as the GUI, not a second one.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Bitstream Vera Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = font
    plt.rcParams['mathtext.it'] = f'{font}:italic'
    plt.rcParams['mathtext.bf'] = f'{font}:bold'
    plt.rcParams['mathtext.default'] = 'regular'


def setup_plotting(font='Arial', fontsize=12, pl_width=3.375, pl_height=3.375,
                    output_format=''):
    """Apply the shared matplotlib style and resolve figure dimensions
    for a standalone (non-GUI) figure -- see the module docstring for
    why this must not be called from GUI code.

    ``output_format`` : ``''``, ``'manuscript'`` (3.375x3.375in,
    fontsize 12), or ``'presentation'`` (10x10in, fontsize 24).

    Returns a dict with the resolved ``fontsize``, ``pl_width``,
    ``pl_height``, ``lw`` (line width, scaled to figure width), and
    ``ticklen``.
    """
    if output_format == 'manuscript':
        fontsize = 12
        pl_height = 3.375
        pl_width = 3.375
    elif output_format == 'presentation':
        fontsize = 24
        pl_height = 10
        pl_width = 10

    lw = pl_width / 3.375
    ticklen = lw * 5

    # Typeface (family + mathtext) is factored out into its own
    # function -- see apply_default_font's docstring for why the GUI
    # also calls it directly, on its own, without the rest of what
    # this function sets.
    apply_default_font(font)

    params = {
        'legend.fontsize': 'large',
        # 'font.family'/'font.sans-serif' -- set by apply_default_font above.
        'font.weight': 'regular',
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'xtick.major.width': lw,
        'xtick.major.size': ticklen,
        'xtick.minor.visible': True,
        'xtick.minor.width': 3 / 4 * lw,
        'xtick.minor.size': ticklen / 2,
        'ytick.labelsize': fontsize,
        'ytick.major.width': lw,
        'ytick.major.size': ticklen,
        'ytick.minor.visible': True,
        'ytick.minor.width': 3 / 4 * lw,
        'ytick.minor.size': ticklen / 2,
        'axes.labelweight': 'regular',
        'axes.linewidth': lw,
        'axes.titlepad': 25,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    }

    plt.rcParams.update(params)

    return {
        'fontsize': fontsize,
        'pl_width': pl_width,
        'pl_height': pl_height,
        'lw': lw,
        'ticklen': ticklen,
    }


def relative_luminance(color) -> float:
    """WCAG relative luminance of ``color``, 0 (black) to 1 (white).

    The sRGB channels are linearised before weighting, which is what
    makes this a *perceptual* measure rather than a naive channel
    average: pure yellow and pure blue have the same mean channel
    value and wildly different apparent brightness (0.93 against 0.07),
    and only the weighted-linear form gets that right. Averaging
    channels instead would keep yellow and discard blue -- exactly
    backwards.
    """
    red, green, blue = cl.to_rgb(color)
    channels = np.array([red, green, blue], dtype=float)
    linear = np.where(channels <= 0.03928, channels / 12.92,
                      ((channels + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def contrast_against_white(color) -> float:
    """Contrast ratio of ``color`` against a white background -- the
    number :data:`MAX_LINE_LUMINANCE` is derived from, exposed so a
    caller can check a colour it chose itself."""
    return 1.05 / (relative_luminance(color) + 0.05)


def is_too_light(color, max_luminance: float = None) -> bool:
    """Whether ``color`` is too light to read as a line on white."""
    cap = MAX_LINE_LUMINANCE if max_luminance is None else max_luminance
    return relative_luminance(color) > cap


def usable_cmap_positions(cmap, max_luminance: float = None,
                          samples: int = _LUMINANCE_SAMPLES) -> np.ndarray:
    """Positions along ``cmap`` whose colour is dark enough to use.

    Returns positions rather than colours so callers can resample the
    map at whatever resolution they need.

    **Position-based, not range-based**, and that matters: where the
    too-light part of a map sits depends entirely on the map. It is at
    the top of ``viridis`` (yellow), the bottom of ``Greys``, and the
    *middle* of every diverging map (``coolwarm``, ``RdYlBu``). A
    fixed "trim the last 20%" rule would fix one of those three and
    silently do nothing for the others.

    Empty is never returned: a map with nothing dark enough in it
    yields its single darkest position, so callers always have
    something to work with and :func:`_darkened` can take it from
    there.
    """
    positions = np.linspace(0.0, 1.0, int(samples))
    # Vectorised form of relative_luminance over every sample at once
    # (same linearisation and weights) -- the per-sample loop was the
    # bulk of the time the palette accessibility checks took.
    channels = np.asarray(cmap(positions), dtype=float)[:, :3]
    linear = np.where(channels <= 0.03928, channels / 12.92,
                      ((channels + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    cap = MAX_LINE_LUMINANCE if max_luminance is None else max_luminance
    usable = positions[luminance <= cap]
    if usable.size:
        return usable
    return positions[[int(np.argmin(luminance))]]


def _darkened(color, max_luminance: float = None):
    """``color`` scaled toward black until it clears the cap, keeping
    its alpha and as much of its hue as scaling allows.

    The last-resort path, for a colormap with nothing dark enough
    anywhere in it. Scaling RGB uniformly preserves hue and saturation
    ratios while dropping brightness -- the alternative, clipping
    individual channels, shifts the hue instead, which is worse: the
    colour then no longer identifies the same trace it did before.
    """
    cap = MAX_LINE_LUMINANCE if max_luminance is None else max_luminance
    rgba = list(cl.to_rgba(color))
    for _attempt in range(24):
        if relative_luminance(rgba[:3]) <= cap:
            break
        rgba[:3] = [channel * 0.85 for channel in rgba[:3]]
    return tuple(rgba)


def get_stack_colormap(color='black'):
    """Colour map used to distinguish a stack of traces (multiple 1D
    spectra, or extracted 2D row/column slices).

    ``color`` : ``'black'`` (every trace black), ``'diverging'`` (a
    four-stop brown/teal diverging map), or any matplotlib colormap
    name (falls back to black if not recognised).
    """
    if color == 'diverging':
        return cl.LinearSegmentedColormap.from_list(
            "", ["#a6611a", "#dfc27d", "#80cdc1", "#018571"]
        )

    if color == 'black':
        return cl.LinearSegmentedColormap.from_list(
            "", ["#000000", "#000000", "#000000", "#000000"]
        )

    try:
        return plt.get_cmap(color)
    except (ValueError, KeyError):
        print(f"Colour scheme '{color}' not recognised; defaulting to black.")
        return cl.LinearSegmentedColormap.from_list(
            "", ["#000000", "#000000", "#000000", "#000000"]
        )


def stack_colors(n, color='black', max_luminance: float = None):
    """``n`` RGBA colours spread along :func:`get_stack_colormap`, with
    the too-light part of the map skipped -- the colour-resolving logic
    ``plot_1d_stack`` uses per trace, factored out so other
    stacked-trace views (a GUI canvas plotting each trace on its own
    subplot, the Compare stack, resolved SCORE components) match it
    without duplicating the setup.

    Colours are taken from :func:`usable_cmap_positions` rather than
    from an even sweep of the whole map, so a stack never contains a
    trace nobody can see. That does mean the spacing is even *in the
    usable part of the map*, not in the map as a whole -- which is the
    point: the discarded stretch was contributing invisible traces, not
    distinguishable ones.

    Every returned colour is guaranteed under the cap: a map with
    nothing dark enough in it (an all-white one) has its darkest colour
    scaled toward black rather than being handed back unusable.

    A *qualitative* palette (a short listed map -- ``okabe_ito``,
    ``tol_muted``, ``Dark2``, ``tab10`` ...) is not sampled at all:
    series take its entries in turn, see
    :func:`opennmr.plotting.accessibility.ordered_qualitative_colours`.
    """
    if n <= 0:
        return []
    if color not in ('black', 'diverging') and is_qualitative(color):
        # A set of categories, not a ramp: series take its entries in
        # turn (cycling past the end), in the order that keeps the
        # first n as far apart as possible -- sampling it like a ramp
        # skipped entries and, via readable_cmap, blended neighbours
        # into colours belonging to neither. See
        # accessibility.ordered_qualitative_colours.
        from .accessibility import ordered_qualitative_colours
        palette = ordered_qualitative_colours(color, max_luminance)
        return [palette[i % len(palette)] for i in range(n)]
    cmap = get_stack_colormap(color)
    usable = usable_cmap_positions(cmap, max_luminance)
    if n == 1:
        # One trace has nothing to be distinguished *from*, so it takes
        # the darkest usable colour rather than the map's own start,
        # which for a light-ended map is the worst available choice.
        picks = [usable[0] if relative_luminance(cmap(usable[0])[:3])
                 <= relative_luminance(cmap(usable[-1])[:3]) else usable[-1]]
    else:
        picks = np.interp(np.linspace(0.0, 1.0, n),
                          np.linspace(0.0, 1.0, usable.size), usable)
    return [_darkened(cmap(p), max_luminance) for p in picks]


def get_contour_cmap(name, max_luminance: float = None):
    """A named colormap for 2D contour plots, restricted to
    :data:`CONTOUR_CMAP_NAMES`, **rebuilt from only its usable
    colours**.

    A contour plot here draws *lines*, so the same visibility problem
    applies as to a trace stack: a near-white contour on a white
    background is a contour nobody can see. The map is resampled over
    :func:`usable_cmap_positions` and handed back as a new colormap of
    the same length, so it is a drop-in for the original everywhere --
    ``levels``/``norm`` handling is untouched, only the colours change.

    For a sequential map this trims the light end. **For a diverging
    map it removes the pale middle**, and the two dark halves meet:
    that is deliberate. The middle of a diverging map is where its
    colours are lightest, contour levels are never drawn at zero
    anyway (they start above the noise threshold), and the two sides
    stay distinguishable by hue.
    """
    if name not in CONTOUR_CMAP_NAMES:
        raise ValueError(
            f"Unknown contour colour scheme '{name}'. "
            f"Available options: {sorted(CONTOUR_CMAP_NAMES)}"
        )
    try:
        base = plt.get_cmap(name)
    except AttributeError:  # pragma: no cover - very old matplotlib
        base = cm.get_cmap(name)

    return readable_cmap(base, max_luminance)


def readable_cmap(cmap, max_luminance: float = None):
    """``cmap`` resampled over only its usable colours -- the reusable
    core of :func:`get_contour_cmap`, for any colormap rather than only
    the named contour ones.

    Accepts a colormap object or a matplotlib name, and returns a new
    colormap of the standard length, so it substitutes for the original
    anywhere without touching ``levels``/``norm`` handling.
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    usable = usable_cmap_positions(cmap, max_luminance)
    resampled = np.interp(np.linspace(0.0, 1.0, _LUMINANCE_SAMPLES),
                          np.linspace(0.0, 1.0, usable.size), usable)
    colors = [_darkened(cmap(p), max_luminance) for p in resampled]
    return cl.LinearSegmentedColormap.from_list(f"{cmap.name}_readable", colors)


def qualitative_colors(n, name: str | None = None, max_luminance: float = None):
    """``n`` visually distinct colours for things that are *categories*
    rather than points on a scale -- the components of a SCORE
    resolution, the individual peaks under a fitted lineshape.

    A qualitative palette, deliberately, not a sample of a continuous
    map: a diverging or sequential map washes out at one end or in the
    middle, which made a peak of middling rank nearly invisible once
    shaded. ``tab10`` has no such midpoint -- but two of its ten
    entries (the olive and the cyan) still only reach 2.0:1 against
    white, so each colour is put through the same luminance cap as
    everything else here and darkened if it misses.

    Cycles once past the palette's own length, as its callers already
    did.

    ``name`` None (the default since 2026-10-01; it was ``tab10``): a
    palette that is colour-blind friendly for ``n``, measured --
    :func:`opennmr.plotting.accessibility.default_palette`.
    """
    if name is None:
        from .accessibility import default_series_colours
        return [tuple(float(v) for v in c) for c in default_series_colours(int(n), False,
                                                                            max_luminance)]
    palette = plt.get_cmap(name)
    count = getattr(palette, "N", 10)
    return [_darkened(palette(i % count), max_luminance) for i in range(int(n))]


def save_figure(fig, fig_name='nmr_plot', save_format='png', dpi=300):
    """Save a figure using the package's shared saving convention.

    ``fig_name`` : output path/filename, with or without an
    extension. If it already ends in ``.png``/``.pdf``/``.svg``, that
    extension is used as-is and ``save_format`` is ignored; otherwise
    ``.{save_format}`` is appended.

    Returns the path the figure was actually saved to.
    """
    if save_format not in VALID_SAVE_FORMATS:
        raise ValueError(
            f"save_format must be one of {VALID_SAVE_FORMATS}, got '{save_format}'."
        )

    root, ext = os.path.splitext(fig_name)
    path = fig_name if ext.lower().lstrip('.') in VALID_SAVE_FORMATS else f"{fig_name}.{save_format}"

    fig.savefig(path, dpi=dpi)
    return path


def downsample_2d(data, x, y, max_points_x=2000, max_points_y=500):
    """Downsample a 2D array (and its axes) for fast contour
    rendering, keeping each output cell's own largest-magnitude
    (signed) value rather than a plain evenly-spaced index pick.

    ``nmrtools2``'s original (and this function's own, until this
    fix) picked evenly-spaced *indices* (``np.linspace``) and threw
    everything else away -- fine for a smooth, broad feature, but a
    thin/sparse one (an EXSY/NOESY diagonal ridge, often only 1-2
    points wide at real digital resolution) can fall entirely between
    the sampled indices and vanish, or survive at scattered points
    that no longer look like a continuous ridge -- even though the
    underlying data never changed. Real user report, confirmed by
    direct reproduction against a real 2H EXSY dataset: the Processing
    tab's own preview (this function, ~16x fewer total points at the
    default 0.25 contour fraction) and the Spectrum tab's own full-
    resolution view of the *exact same spectrum object* looked like
    "two different 2Ds" -- the diagonal visible in one, not the other.
    Max-magnitude-per-bin instead of index-skipping means a bin
    containing any part of a real feature always keeps *a* point from
    it, not nothing.

    Reduces one axis at a time (X then Y) via :func:`_max_abs_bin` --
    a separable approximation of a true joint-2D block-max (which
    would need reducing both axes of each block simultaneously), but
    a large improvement over index-skipping, and correct for the
    common case (contour fraction applied independently per axis,
    matching every existing caller's own ``contour_fraction_x``/
    ``contour_fraction_y`` convention).

    **Not free, unlike the old index-skip -- has to touch every input
    element at least once to find each bin's own max, where index-
    skipping only ever touched the output-sized subset.** Measured
    directly on a real, large conventional-2D array (1024x65536, a
    real EXSY dataset): ~0.35s here vs. a few ms for the old method.
    Fine for a one-off settled redraw (spectrum selection, a drag
    settling), but a real, known cost during a *live 2D phase-drag*
    on data this large specifically, since that redraws on every
    mouse-move tick with no in-place update path for a contour (see
    ``MplCanvas.set_full_resolution``'s own docstring on why). Not
    optimized further here -- the actual fix for 2D drag
    responsiveness at any data size is phasing only the relevant row/
    column during the drag itself, not re-rendering (let alone
    downsampling) the whole contour on every tick; see this project's
    own TODO.md "2D roadmap" for that separate, queued work.

    ``data`` : shape ``(len(y), len(x))``. Returns ``data_ds, x_ds,
    y_ds`` unchanged (well, each axis fed back through unchanged) if
    already within the point-count limits.
    """
    n_y, n_x = data.shape

    max_points_x = max(2, min(max_points_x, n_x))
    max_points_y = max(2, min(max_points_y, n_y))

    # X (columns, the last/contiguous axis) first -- reducing along
    # the last axis keeps every argmax/take_along_axis's own memory
    # access contiguous; reducing Y (a middle axis once reshaped)
    # first was measured directly to be *slower* on a real-sized
    # array despite operating on fewer total elements, since each of
    # its per-bin comparisons strides across memory instead of
    # scanning it in order.
    data_ds, x_ds = _max_abs_bin(data, x, max_points_x, axis=1)
    data_ds, y_ds = _max_abs_bin(data_ds, y, max_points_y, axis=0)

    return data_ds, x_ds, y_ds


def _max_abs_bin(data, coords, n_bins, axis):
    """Reduce ``data``/``coords`` along ``axis`` to ``n_bins`` bins --
    each bin keeps the *signed* value with the largest magnitude
    across its own span (not just an evenly-spaced index pick, see
    :func:`downsample_2d`'s own docstring for why), with the bin's own
    middle-index coordinate standing in for the whole span. A no-op
    (returns ``data``/``coords`` unchanged) if ``n_bins`` already
    covers every index along ``axis``.

    Fully vectorized (reshape + one ``argmax``/``take_along_axis`` --
    no Python-level loop over bins) -- an earlier version used
    ``numpy.array_split`` plus a per-bin loop instead, simpler but
    ~0.7s on a real 1024x65536 contour, unusably slow for a live 2D
    phase-drag tick (a fresh downsample on every mouse-move, since a
    ``QuadContourSet`` has no in-place data swap the way a ``Line2D``
    does). Achieves the uniform bin size a reshape needs by trimming
    at most ``n_bins - 1`` trailing indices (never more than one bin's
    own width) rather than requiring ``n_bins`` to evenly divide the
    axis length -- a negligible edge effect for any real acquisition
    size, and the same "drop a few, don't fail" spirit as this
    project's own raw-``ser``-shape floor-division fallback."""
    n = data.shape[axis]
    if n_bins >= n:
        return data, coords
    bin_size = n // n_bins
    trimmed_n = bin_size * n_bins
    if trimmed_n != n:
        data = np.take(data, np.arange(trimmed_n), axis=axis)
    new_shape = data.shape[:axis] + (n_bins, bin_size) + data.shape[axis + 1:]
    reshaped = data.reshape(new_shape)
    peak_index = np.argmax(np.abs(reshaped), axis=axis + 1, keepdims=True)
    reduced = np.take_along_axis(reshaped, peak_index, axis=axis + 1)
    reduced = np.squeeze(reduced, axis=axis + 1)
    reduced_coords = coords[:trimmed_n].reshape(n_bins, bin_size)[:, bin_size // 2]
    return reduced, reduced_coords


#: Default cap for :func:`downsample_1d` (and the older, no-longer-
#: used-internally :func:`downsample_indices`) -- rendering (not
#: fitting/processing) more points than this per 1D line rarely adds
#: visible detail at any normal screen resolution, but does cost real
#: time redrawing, especially for a *dynamic* view (phase-dragging
#: redraws on every mouse-move tick) or *many* lines at once (a
#: Compare-tab stack of dozens of traces) -- both explicitly reported
#: as slow for high-point-count data (diffusion/imaging acquisitions,
#: or just a high digital resolution, routinely land in the
#: 10^5-10^6 point range).
DOWNSAMPLE_1D_MAX_POINTS = 8000


def downsample_indices(n, max_points=DOWNSAMPLE_1D_MAX_POINTS):
    """Evenly-spaced indices for downsampling a length-``n`` 1D array
    to at most ``max_points``, for fast *display* rendering only --
    the same idea as :func:`downsample_2d`, for a 1D line instead of a
    2D contour. Returns ``None`` (not an index array) when ``n`` is
    already within ``max_points``, so a caller can cheaply skip
    indexing altogether in the (usual) case no downsampling is
    actually needed.

    Deterministic in ``n``/``max_points`` alone (no data-dependent
    peak-preservation logic) -- naive even-striding, kept here for any
    external caller that wants exactly that. **Not used by this
    project's own GUI rendering any more** (see :func:`downsample_1d`):
    reported directly as a real problem, not just a cosmetic one --
    a narrow, well-digitized peak can fall entirely between two
    evenly-spaced samples, showing "a well-defined peak" as a mostly
    flat line broken by a couple of stray points, since which sample
    each stride happens to land on has nothing to do with where the
    signal actually is.
    """
    if n <= max_points:
        return None
    return np.linspace(0, n - 1, max_points, dtype=int)


def downsample_1d(x, y, max_points=DOWNSAMPLE_1D_MAX_POINTS):
    """Peak-preserving downsample of a 1D line for *display* only --
    :func:`downsample_2d`'s own ``_max_abs_bin`` binning, applied to a
    line instead of a contour's rows/columns. Splits ``y`` into
    ``max_points`` contiguous bins and keeps the largest-magnitude
    (signed) sample in each bin, rather than an arbitrary evenly-
    spaced pick (:func:`downsample_indices`'s naive striding) that can
    land entirely off a narrow peak and miss it altogether -- reported
    directly as "a peak that's well-defined elsewhere is only made of
    three or four points" in a downsampled view.

    Returns ``(x, y)`` unchanged (same objects, not copies) when
    ``len(y) <= max_points`` -- no downsampling needed at all, the
    same no-op contract :func:`downsample_2d` and
    :func:`downsample_indices` both already have.

    ``x`` may be ``None`` when only the downsampled *y*-values are
    needed and the corresponding x is already known to be unchanged --
    e.g. refreshing an existing line's y-data in place after a phase
    change, where the x already on screen doesn't need re-deriving.
    The bin *boundaries* only ever depend on ``len(y)``/``max_points``
    (never on ``x``'s own values, or ``y``'s), so re-calling this with
    new ``y`` values always reproduces bins aligned to whatever x was
    already plotted -- only *which* sample wins each bin can change,
    because that part *is* data-dependent (a phase change can move
    where the largest-magnitude sample in a bin actually is). Returns
    ``None`` in place of the x half whenever the input ``x`` was
    ``None``.
    """
    y = np.asarray(y)
    n = len(y)
    if n <= max_points:
        return x, y
    coords = np.arange(n) if x is None else np.asarray(x)
    y_ds, x_ds = _max_abs_bin(y, coords, max_points, axis=0)
    return (None if x is None else x_ds), y_ds
