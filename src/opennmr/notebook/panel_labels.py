"""
Panel labels -- **(a)**, **(b)**, ... on the panels of a multi-panel figure.

Works on any figure whose panels are separate Axes: the notebook's
:func:`opennmr.notebook.compare_panels.plot_compare_panels` (1D, 2D and
pulse-sequence panels in a row, column or grid) and the GUI's Compare tile
call it, and so can you for a figure of your own::

    from opennmr.notebook import panel_labels as pl

    labels = pl.add_panel_labels(fig, axes)                    # (a) (b) (c) ...
    labels = pl.add_panel_labels(fig, axes, style="A)", start="b", order="column")

    pl.label_text(3, "(A)")         # '(C)'
    pl.label_text(4, "I.")          # 'IV.'

**Style** -- one of :data:`LABEL_STYLES`: ``(a)  a)  a.  a  A  A.  A)  (A)
I  I.  I)  (I)`` (``I`` is a roman numeral) and the lower-case roman ``(i)
i)  i.  i``.

**Order** -- which panel is "first": ``"row"`` (left to right, then the next
row down), ``"column"`` (top to bottom, then the next column) or a list with
one entry per panel (``[2, 1, 3, None]``: the position each panel takes in
the sequence; ``None`` leaves that panel unlabelled). ``start`` moves the
whole sequence: ``start="b"`` (or ``2``) makes the first panel ``(b)``.

**Placement**

``"auto"`` (default)
    Above the top-left corner of each panel, with the bottom of the text
    ``pad`` points above the panel (above its projections too). The labels of
    one row of the grid share a baseline and those of one column share a left
    edge, so the set reads as a set. A label is raised, as a whole row, until
    it touches no tick label, axis label or title.
``"inside"``
    The same, but inside the top-left corner (clear of legends).
``"manual"``
    Where you say: ``positions={"all": (dx, dy), 2: (dx, dy)}`` -- the
    label's lower-left corner, in **points right of / above the panel's
    top-left corner** (``"all"`` for every panel, a 0-based panel number for
    one; a panel's own entry wins). Panels with neither sit at the automatic
    spot.

The labels are Text artists attached to their axes, so ``tight_layout`` and
saving at another size keep them with their panels; :meth:`PanelLabels.settle`
(run for you by ``plot_compare_panels``) re-checks the placement after a layout.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["LABEL_STYLES", "ORDERS", "PLACEMENTS", "PanelLabels", "add_panel_labels",
           "label_text", "parse_start", "label_numbers", "describe_style"]

#: Every style, in the order a menu should list them.
LABEL_STYLES = ("(a)", "a)", "a.", "a", "A", "A.", "A)", "(A)",
                "I", "I.", "I)", "(I)", "(i)", "i)", "i.", "i")
ORDERS = ("row", "column")
PLACEMENTS = ("auto", "inside", "manual")

# style -> (kind, opening, closing)
_PARTS = {
    "(a)": ("alpha", False, "(", ")"), "a)": ("alpha", False, "", ")"),
    "a.": ("alpha", False, "", "."), "a": ("alpha", False, "", ""),
    "A": ("alpha", True, "", ""), "A.": ("alpha", True, "", "."),
    "A)": ("alpha", True, "", ")"), "(A)": ("alpha", True, "(", ")"),
    "I": ("roman", True, "", ""), "I.": ("roman", True, "", "."),
    "I)": ("roman", True, "", ")"), "(I)": ("roman", True, "(", ")"),
    "(i)": ("roman", False, "(", ")"), "i)": ("roman", False, "", ")"),
    "i.": ("roman", False, "", "."), "i": ("roman", False, "", ""),
}

_ROMAN = (("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100), ("XC", 90),
          ("L", 50), ("XL", 40), ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1))


def describe_style(style: str) -> str:
    """The first three labels of ``style``, for a menu: ``'(a)  (b)  (c)'``."""
    return "  ".join(label_text(k, style) for k in (1, 2, 3))


def _check_style(style: str) -> tuple:
    try:
        return _PARTS[style]
    except KeyError:
        raise ValueError(f"style must be one of {LABEL_STYLES}, not {style!r}.") from None


def _alpha(n: int) -> str:
    """1 -> a, 26 -> z, 27 -> aa, 28 -> ab ..."""
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(97 + r) + out
    return out


def _roman(n: int) -> str:
    if not 1 <= n <= 3999:
        raise ValueError(f"A roman numeral label needs a number from 1 to 3999, not {n}.")
    out = ""
    for sym, val in _ROMAN:
        while n >= val:
            out += sym
            n -= val
    return out


def label_text(number: int, style: str = "(a)") -> str:
    """The label for the ``number``-th panel (1-based) in ``style``."""
    kind, upper, opening, closing = _check_style(style)
    number = int(number)
    if number < 1:
        raise ValueError(f"A panel label needs a number of 1 or more, not {number}.")
    core = _alpha(number) if kind == "alpha" else _roman(number).lower()
    return f"{opening}{core.upper() if upper else core}{closing}"


def parse_start(start, style: str = "(a)") -> int:
    """The number ``start`` stands for in ``style``: an int is itself;
    a string is read in the style's own alphabet -- ``"b"`` or ``"(B)"`` is 2
    for a letter style, ``"iv"`` is 4 for a roman one -- and ``"3"`` is 3."""
    kind = _check_style(style)[0]
    if isinstance(start, bool):
        raise ValueError("start must be a number or a label such as 'b'.")
    if isinstance(start, (int, float)):
        if int(start) != start or start < 1:
            raise ValueError(f"start must be a whole number of 1 or more, not {start!r}.")
        return int(start)
    text = str(start).strip().strip("().").strip()
    if not text:
        raise ValueError("start is empty.")
    if text.isdigit():
        value = int(text)
        if value < 1:
            raise ValueError("start must be 1 or more.")
        return value
    if kind == "alpha":
        if not text.isalpha():
            raise ValueError(f"Cannot read {start!r} as a letter label.")
        value = 0
        for ch in text.lower():
            value = value * 26 + (ord(ch) - 96)
        return value
    # roman
    up = text.upper()
    if not set(up) <= set("IVXLCDM"):
        raise ValueError(f"Cannot read {start!r} as a roman numeral.")
    total, prev = 0, 0
    for ch in reversed(up):
        val = dict((s, v) for s, v in _ROMAN if len(s) == 1)[ch]
        total += val if val >= prev else -val
        prev = max(prev, val)
    if total < 1 or _roman(total) != up:
        raise ValueError(f"{start!r} is not a valid roman numeral.")
    return total


def label_numbers(cells, order="row", start=1, style: str = "(a)") -> list:
    """The number of the label each panel gets, as a list in the order of
    ``cells`` (``[(row, col), ...]``, one per panel); ``None`` for a panel
    left without a label. ``order``: ``"row"``, ``"column"`` or a list of the
    sequence positions (see the module notes)."""
    n = len(cells)
    first = parse_start(start, style)
    if isinstance(order, str):
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS} or a list, not {order!r}.")
        key = (lambda i: (cells[i][0], cells[i][1])) if order == "row" \
            else (lambda i: (cells[i][1], cells[i][0]))
        ranked = sorted(range(n), key=key)
        position = {i: rank + 1 for rank, i in enumerate(ranked)}
        return [first + position[i] - 1 for i in range(n)]
    order = list(order)
    if len(order) != n:
        raise ValueError(f"order lists {len(order)} positions for {n} panels.")
    out = []
    for value in order:
        if value is None:
            out.append(None)
            continue
        if int(value) != value or value < 1:
            raise ValueError(f"Each entry of order is a position of 1 or more (or None), not {value!r}.")
        out.append(first + int(value) - 1)
    return out


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        return fig._get_renderer()


def _cell(ax) -> tuple:
    """``(row, column)`` of ``ax`` in its grid (0, 0 when it has none)."""
    try:
        spec = ax.get_subplotspec()
        return int(spec.rowspan.start), int(spec.colspan.start)
    except Exception:  # noqa: BLE001 -- an axes placed by hand
        return 0, 0


class PanelLabels:
    """The labels added by :func:`add_panel_labels`.

    ``texts`` -- one Text per panel (``None`` where a panel has none);
    ``axes`` -- the panels; ``numbers`` -- each panel's label number.
    Change ``placement`` / ``positions`` / the style attributes and call
    :meth:`place` (or :meth:`settle`) to move them.
    """

    def __init__(self, fig, axes, texts, numbers, *, style, placement, positions, pad,
                 fontsize) -> None:
        self.fig = fig
        self.axes = list(axes)
        self.texts = list(texts)
        self.numbers = list(numbers)
        self.style = style
        self.placement = placement
        self.positions = dict(positions or {})
        self.pad = float(pad)
        self.fontsize = fontsize
        #: Points right of / above each panel's top-left corner, as last placed.
        self.offsets: list = [(0.0, self.pad)] * len(self.axes)

    # -- public --------------------------------------------------------

    def texts_of(self) -> list:
        """The label strings, ``None`` for an unlabelled panel."""
        return [t.get_text() if t is not None else None for t in self.texts]

    def remove(self) -> None:
        for t in self.texts:
            if t is not None:
                try:
                    t.remove()
                except (ValueError, NotImplementedError):
                    pass
        self.texts = [None] * len(self.axes)
        if getattr(self.fig, "_opennmr_panel_labels", None) is self:
            self.fig._opennmr_panel_labels = None

    def settle(self, layout=None, passes: int = 3) -> bool:
        """Lay out (``layout()``, e.g. ``fig.tight_layout``), place the labels,
        and repeat until they stop moving -- the layout reserves room for the
        labels, and where they end up depends on the layout. Returns True when
        the placement is stable."""
        stable = False
        for _ in range(max(1, passes)):
            if layout is not None:
                layout()
            if not self.place():
                stable = True
                break
        if layout is not None:
            layout()
        return stable

    def place(self) -> bool:
        """Put every label where :attr:`placement` says. Returns True when any
        label moved by more than half a point."""
        live = [i for i, t in enumerate(self.texts) if t is not None]
        if not live:
            return False
        offsets = list(self.offsets)
        if self.placement in ("auto", "inside"):
            auto = self._automatic(live)
        else:
            auto = {i: (0.0, self.pad) for i in live}
        for i in live:
            if self.placement == "manual" or i in self.positions or "all" in self.positions:
                chosen = self.positions.get(i, self.positions.get("all"))
                offsets[i] = tuple(float(v) for v in chosen) if chosen is not None else auto[i]
            else:
                offsets[i] = auto[i]
        moved = False
        for i in live:
            old = self.offsets[i]
            if abs(old[0] - offsets[i][0]) > 0.5 or abs(old[1] - offsets[i][1]) > 0.5:
                moved = True
            self._apply(i, offsets[i])
        self.offsets = offsets
        return moved

    # -- internals -----------------------------------------------------

    def _apply(self, i: int, offset) -> None:
        from matplotlib.transforms import ScaledTranslation
        ax, text = self.axes[i], self.texts[i]
        text.set_transform(ax.transAxes + ScaledTranslation(offset[0] / 72.0, offset[1] / 72.0,
                                                            self.fig.dpi_scale_trans))

    def _members(self, ax, renderer) -> list:
        """``ax`` and the axes that belong to its panel (projection strips,
        insets): those whose centre lies in ``ax``'s cell of the grid."""
        try:
            cell = ax.get_subplotspec().get_position(self.fig).transformed(self.fig.transFigure)
        except Exception:  # noqa: BLE001
            return [ax]
        out = [ax]
        for other in self.fig.axes:
            if other is ax or other in self.axes or not other.get_visible():
                continue
            box = other.get_window_extent(renderer)
            cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
            if cell.x0 <= cx <= cell.x1 and cell.y0 <= cy <= cell.y1:
                out.append(other)
        return out

    def _obstacles(self, renderer, inside: bool) -> list:
        """Boxes a label must not cover: tick labels and axis labels, titles
        (and, for an inside label, legends), in pixels."""
        boxes = []
        for a in self.fig.axes:
            if not a.get_visible():
                continue
            if not inside:
                for axis in (a.xaxis, a.yaxis):
                    try:
                        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                            for lab in (tick.label1, tick.label2):
                                if lab.get_visible() and lab.get_text():
                                    boxes.append(lab.get_window_extent(renderer))
                    except Exception:  # noqa: BLE001
                        pass
                    lab = axis.label
                    if lab.get_visible() and lab.get_text():
                        boxes.append(lab.get_window_extent(renderer))
                for title in (a.title, getattr(a, "_left_title", None), getattr(a, "_right_title", None)):
                    if title is not None and title.get_visible() and title.get_text():
                        boxes.append(title.get_window_extent(renderer))
            legend = a.get_legend()
            if legend is not None and legend.get_visible() and inside:
                boxes.append(legend.get_window_extent(renderer))
        return boxes

    def _automatic(self, live: list) -> dict:
        """``{panel: (dx, dy)}`` in points -- see the module notes."""
        fig = self.fig
        r = _renderer(fig)
        inside = self.placement == "inside"
        scale = fig.dpi / 72.0
        pad = self.pad * scale
        try:                                     # tick labels, titles and locators where the layout puts them
            fig.draw_without_rendering()
        except Exception:  # noqa: BLE001 -- measuring must never take a redraw down
            pass
        for a in fig.axes:
            try:
                a.get_tightbbox(r)
            except Exception:  # noqa: BLE001
                pass
        info = {}
        for i in live:
            ax = self.axes[i]
            box = ax.get_window_extent(r)
            members = self._members(ax, r)
            top = max(m.get_window_extent(r).y1 for m in members)
            size = self.texts[i].get_window_extent(r)
            info[i] = {"left": box.x0, "main_top": box.y1, "top": top,
                       "w": size.width, "h": size.height, "cell": _cell(ax)}
        rows: dict = {}
        cols: dict = {}
        for i in live:
            rows.setdefault(info[i]["cell"][0], []).append(i)
            cols.setdefault(info[i]["cell"][1], []).append(i)
        # x: a column shares its left edge; y: a row shares its baseline
        x = {}
        for members in cols.values():
            edge = (max if inside else min)(info[i]["left"] for i in members)
            for i in members:
                x[i] = edge + (pad if inside else 0.0)
        y = {}
        for members in rows.values():
            if inside:
                base = min(info[i]["top"] for i in members) - pad - max(info[i]["h"] for i in members)
            else:
                base = max(info[i]["top"] for i in members) + pad
            for i in members:
                y[i] = base
        # clear every obstacle: move the whole row (up; down for an inside label)
        obstacles = self._obstacles(r, inside)
        step = (-1.0 if inside else 1.0) * scale
        for members in rows.values():
            for _ in range(int(40 / 1.0)):
                clash = False
                for i in members:
                    mine = (x[i], y[members[0]], x[i] + info[i]["w"], y[members[0]] + info[i]["h"])
                    for ob in obstacles:
                        if ob.x0 < mine[2] - 0.5 and ob.x1 > mine[0] + 0.5 \
                                and ob.y0 < mine[3] - 0.5 and ob.y1 > mine[1] + 0.5:
                            clash = True
                            break
                    if clash:
                        break
                if not clash:
                    break
                for i in members:
                    y[i] += step
        return {i: ((x[i] - info[i]["left"]) / scale, (y[i] - info[i]["main_top"]) / scale)
                for i in live}


def add_panel_labels(fig, axes, *, style: str = "(a)", start=1, order="row",
                     placement: str = "auto", positions: dict | None = None, pad: float = 4.0,
                     fontsize: float | None = None, weight: str = "bold", color="black",
                     family: str | None = None, layout=None) -> PanelLabels:
    """Label ``axes`` (one per panel) ``(a)``, ``(b)`` ... and return the
    :class:`PanelLabels`. See the module notes for ``style``, ``start``,
    ``order``, ``placement`` and ``positions``.

    ``pad`` is the clear space (points) between a label and its panel;
    ``fontsize`` defaults to 14 pt; ``weight``, ``color`` and ``family``
    style the text. ``layout`` is a function that lays the figure out
    (``fig.tight_layout``): given, the labels are settled with it straight
    away. A figure holds one set of labels: calling this again replaces it.
    """
    from matplotlib.transforms import ScaledTranslation  # noqa: F401 -- fail early if unusable
    _check_style(style)
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, not {placement!r}.")
    axes = [a for a in (axes if isinstance(axes, Iterable) else [axes]) if a is not None]
    if not axes:
        raise ValueError("Nothing to label: give at least one axes.")
    previous = getattr(fig, "_opennmr_panel_labels", None)
    if previous is not None:
        previous.remove()
    cells = [_cell(a) for a in axes]
    numbers = label_numbers(cells, order=order, start=start, style=style)
    size = 14.0 if fontsize is None else float(fontsize)
    texts = []
    for ax, number in zip(axes, numbers):
        if number is None:
            texts.append(None)
            continue
        kw = {"family": family} if family else {}
        text = ax.text(0.0, 1.0, label_text(number, style), transform=ax.transAxes,
                       ha="left", va="bottom", fontsize=size, fontweight=weight, color=color,
                       clip_on=False, zorder=20, **kw)
        text._opennmr_panel_label = True
        texts.append(text)
    cleaned = {}
    for key, value in (positions or {}).items():
        key = key if key == "all" else int(key)
        if len(value) != 2:
            raise ValueError("Each position is (dx, dy) in points.")
        cleaned[key] = (float(value[0]), float(value[1]))
    labels = PanelLabels(fig, axes, texts, numbers, style=style, placement=placement,
                         positions=cleaned, pad=pad, fontsize=size)
    fig._opennmr_panel_labels = labels
    for i in range(len(axes)):
        labels._apply(i, labels.offsets[i])
    if layout is not None:
        labels.settle(layout)
    return labels
