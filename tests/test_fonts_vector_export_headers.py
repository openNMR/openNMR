"""2026-10-17 (round 2): every installed typeface for the plots
(notebook.fonts and Plot Style ▸ Font family, maths included), vector
exports that open whole in Illustrator (plotting.vector_export: no path
over ~20k points, TrueType fonts), section headers one height whatever
their characters, and a width cap so a capped-amplitude fit converges."""
import io
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from opennmr.notebook import fonts  # noqa: E402
from opennmr.notebook.fit_display import draw_fit_components  # noqa: E402
from opennmr.plotting import vector_export  # noqa: E402

from test_gui import qapp  # noqa: F401,E402


# ---- fonts --------------------------------------------------------------------

def test_available_fonts_are_sorted_unique_text_faces():
    names = fonts.available_fonts()
    assert names, "matplotlib always ships DejaVu"
    assert "DejaVu Sans" in names
    assert names == sorted(set(names), key=str.casefold)
    assert not any("emoji" in n.lower() or "wingdings" in n.lower() for n in names)
    assert fonts.font_available("DejaVu Serif") and fonts.font_available("serif")
    assert not fonts.font_available("No Such Face 1234")


def test_font_rc_sets_text_maths_and_truetype_embedding():
    rc = fonts.font_rc("DejaVu Serif")
    assert rc["font.family"] == "DejaVu Serif"
    assert rc["mathtext.fontset"] == "custom" and rc["mathtext.rm"] == "DejaVu Serif"
    assert rc["pdf.fonttype"] == 42
    assert fonts.font_rc(None) == {}
    assert "mathtext.rm" not in fonts.font_rc("serif")
    with fonts.use_font("DejaVu Serif"):
        assert matplotlib.rcParams["font.family"] == ["DejaVu Serif"]
    fig, _ax = fonts.plot_font_samples(["DejaVu Sans", "DejaVu Serif"])
    plt.close(fig)
    samples = fonts.font_samples(["DejaVu Serif"])      # IPython's HTML where installed
    assert "DejaVu Serif" in getattr(samples, "data", samples)


def test_plot_style_offers_installed_faces_drawn_in_themselves(qapp):
    from PySide6.QtCore import Qt
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    combo = window.output_font_family_combo
    data = [combo.itemData(i) for i in range(combo.count())]
    for generic in fonts.GENERIC_FAMILIES:
        assert generic in data
    assert "DejaVu Serif" in data
    index = combo.findData("DejaVu Serif")
    assert combo.itemData(index, Qt.ItemDataRole.FontRole).family() == "DejaVu Serif"


def test_a_named_face_reaches_the_maths_in_labels(qapp):
    from matplotlib.text import Text
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    canvas.ax.set_xlabel(r"$^{13}$C shift (ppm)")
    canvas.set_style(font_family="DejaVu Serif")
    canvas.canvas.draw()
    label = canvas.ax.xaxis.label
    assert label.get_fontfamily() == ["DejaVu Serif"]
    assert label.get_math_fontfamily() == "custom"
    canvas.set_style(font_family="")
    canvas.canvas.draw()
    assert label.get_math_fontfamily() == matplotlib.rcParams["mathtext.fontset"]
    assert all(isinstance(t, Text) for t in canvas.figure.findobj(Text))


def test_a_face_missing_here_is_shown_not_swapped(qapp):
    from opennmr.gui.main_window import MainWindow
    window = MainWindow()
    window.canvas.set_style(font_family="No Such Face 1234")
    window._load_style_controls_from(window.canvas)
    combo = window.output_font_family_combo
    assert combo.currentData() == "No Such Face 1234"
    assert "not installed here" in combo.currentText()


# ---- vector export --------------------------------------------------------------

def _fit_figure(n=40000):
    x = np.linspace(200, -200, n)
    c1 = np.exp(-(x / 2) ** 2)
    c2 = 0.5 * np.exp(-((x - 10) / 3) ** 2)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    draw_fit_components(ax, x, c1 + c2, c1 + c2, components=[c1, c2, 0 * x], labels=["A", "B"])
    ax.legend()
    return fig, ax


def _longest_svg_path(fig, safe):
    buffer = io.BytesIO()
    if safe:
        vector_export.save_vector_safe(fig, buffer, format="svg")
    else:
        fig.savefig(buffer, format="svg")
    text = buffer.getvalue().decode()
    return max(len(re.findall(r"[LM] ", d)) for d in re.findall(r' d="([^"]*)"', text))


def test_component_fills_stay_under_illustrators_path_limit():
    fig, ax = _fit_figure()
    assert _longest_svg_path(fig, safe=False) > 32767      # what Illustrator dropped
    assert _longest_svg_path(fig, safe=True) <= vector_export.MAX_PATH_POINTS + 10
    # ... and the figure itself is untouched afterwards
    assert len(ax.collections[0].get_paths()[0].vertices) > 2 * 40000
    plt.close(fig)


def test_thinned_figure_looks_the_same():
    fig, _ax = _fit_figure()
    plain = io.BytesIO()
    fig.savefig(plain, format="png")
    with vector_export.vector_safe(fig):
        thinned = io.BytesIO()
        fig.savefig(thinned, format="png")
    a = plt.imread(io.BytesIO(plain.getvalue()))
    b = plt.imread(io.BytesIO(thinned.getvalue()))
    assert a.shape == b.shape
    assert np.mean(np.abs(a - b)) < 2e-3
    plt.close(fig)


def test_thin_indices_keep_a_narrow_spike():
    y = np.zeros(100000)
    y[54321] = 1.0
    keep = vector_export.thin_indices(y, 1000)
    assert keep.size <= 1002 and 54321 in keep and keep[0] == 0 and keep[-1] == y.size - 1


def test_pdf_embeds_truetype_fonts(tmp_path):
    fig, ax = plt.subplots()
    ax.set_xlabel("shift (ppm)")
    path = tmp_path / "f.pdf"
    vector_export.save_vector_safe(fig, str(path))
    assert b"FontFile2" in path.read_bytes()          # TrueType, editable as text
    plt.close(fig)


def test_gui_save_of_a_fit_is_illustrator_safe(qapp, tmp_path):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()
    x = np.linspace(200, -200, 40000)
    c1, c2 = np.exp(-(x / 2) ** 2), 0.5 * np.exp(-((x - 10) / 3) ** 2)
    canvas.show_slice_fit(x, c1 + c2, c1 + c2, components=[c1, c2, 0 * x],
                          peak_labels=["A", "B"])
    path = tmp_path / "fit.svg"
    canvas.figure.savefig(str(path))                    # through print_figure, as the dialog
    text = path.read_text()
    longest = max(len(re.findall(r"[LM] ", d)) for d in re.findall(r' d="([^"]*)"', text))
    assert longest <= vector_export.MAX_PATH_POINTS + 10


# ---- section headers ------------------------------------------------------------

def test_section_header_height_does_not_depend_on_its_characters(qapp):
    from opennmr.gui.widgets.collapsible_section import CollapsibleSection
    plain = CollapsibleSection("Regularisation")
    odd = CollapsibleSection("Regularisation (T₁/T₂/D distributions, 2D maps) ☃ ₃ ✓")
    assert plain._header.sizeHint().height() == odd._header.sizeHint().height()


# ---- a capped amplitude converges -------------------------------------------------

def test_a_width_without_its_own_limit_is_capped_at_twice_the_span():
    from opennmr.core.spectrum import NMRAxis, NMRMetadata, NMRSpectrum
    from opennmr.processing import fitting
    x = np.linspace(3.0, 7.0, 400)
    y = 2.0 * np.exp(-4 * np.log(2) * ((x - 5.0) / 0.2) ** 2)
    spectrum = NMRSpectrum(data=y, axes=[NMRAxis(kind="frequency", ppm=x, hz=x * 400.0)],
                           metadata=NMRMetadata(vendor="bruker", nucleus="1H", sfo1=400.0))
    peaks = [{"type": "gaussian", "shift": 5.0, "fwhm": 0.2,
              "bounds": {"amplitude": (0.0, 0.8)}}]
    # Normalised, as the GUI fits (the bound is in those units).
    peaks[0]["bounds"]["amplitude"] = (0.0, 0.4)
    popt, *_ = fitting.fit_spectrum(spectrum, peaks, fit_regions=[(4.95, 5.05)])
    assert popt[2] == pytest.approx(0.4, rel=1e-6)       # held at the cap
    assert popt[1] <= 2 * 4.0 + 1e-9                     # width stopped at the cap


# ---- quantity symbols italic (2026-10-17) -----------------------------------------

@pytest.mark.parametrize("plain, italic", [
    ("T (K)", "$T$ (K)"),
    ("t (min)", "$t$ (min)"),
    ("T₁ρ (ms)", r"$T_{1\rho}$ (ms)"),
    ("δ (ppm)", r"$\delta$ (ppm)"),
    ("1000/T (K)", "1000/$T$ (K)"),
    ("ln(T₁ (s))", "ln($T_{1}$ (s))"),
    ("x$_{EFG}$", r"$x_{\mathrm{EFG}}$"),
    (r"D ($\mathregular{m^{2}\,s^{-1}}$)", r"$D$ ($\mathregular{m^{2}\,s^{-1}}$)"),
    # words, units and existing mathtext are left alone
    ("Delay (s)", "Delay (s)"), ("Contact time (µs)", "Contact time (µs)"),
    ("10/s rate", "10/s rate"), ("1000/$T$ (K$^{-1}$)", "1000/$T$ (K$^{-1}$)"),
    ("Linewidth (FWHM) (Hz)", "Linewidth (FWHM) (Hz)"),
])
def test_quantity_symbols_are_italic(plain, italic):
    from opennmr.plotting.labels import italicise_symbols, label_problem
    assert italicise_symbols(plain) == italic
    assert italicise_symbols(italic) == italic          # idempotent
    assert label_problem(italic) is None                  # valid mathtext


def test_mathtext_letters_are_italic_by_default():
    from opennmr.plotting import style
    try:
        style.apply_default_font("DejaVu Sans")
        assert matplotlib.rcParams["mathtext.default"] == "it"
    finally:
        matplotlib.rcdefaults()


def test_the_canvas_italicises_plain_axis_labels(qapp):
    from opennmr.gui.widgets.canvas import MplCanvas
    canvas = MplCanvas()

    def draw(fig):
        ax = fig.add_subplot(111)
        ax.plot([1, 2], [1, 2])
        ax.set_xlabel("T (K)")
        ax.set_ylabel("T₁ (s)")
    canvas.show_external_plot(draw, kind="curve")
    ax = canvas.figure.axes[0]
    assert ax.get_xlabel() == "$T$ (K)" and ax.get_ylabel() == "$T_{1}$ (s)"


def test_novelty_typefaces_are_not_offered_but_still_work():
    """No Jokerman and other silly typefaces among the choices (2026-11-02)."""
    for silly in ("Jokerman", "Comic Sans MS", "Papyrus", "Curlz MT", "Brush Script MT",
                  "Segoe Script", "Lucida Handwriting", "Chiller", "Vivaldi", "Zapfino"):
        assert fonts.is_novelty_font(silly), silly
    for serious in ("Arial", "Times New Roman", "Helvetica", "Calibri", "Cambria", "DejaVu Sans",
                    "Liberation Serif", "Latin Modern Roman", "Palatino Linotype", "Georgia",
                    "Courier New", "Segoe UI", "Verdana", "Garamond", "Century Schoolbook"):
        assert not fonts.is_novelty_font(serious), serious
    offered = set(fonts.available_fonts())
    everything = set(fonts.available_fonts(publication=False))
    assert offered <= everything
    assert not any(fonts.is_novelty_font(n) for n in offered)
    assert all(fonts.is_novelty_font(n) for n in everything - offered)
    # A figure that names one is not broken by the list: it is still "available".
    assert fonts.font_available("serif") and fonts.font_available("DejaVu Sans")


def test_the_font_pickers_leave_novelty_faces_out_and_keep_a_saved_one(qapp, monkeypatch):
    from opennmr.gui.main_window import MainWindow
    from opennmr.notebook import fonts as nb_fonts
    window = MainWindow()
    for combo in (window.output_font_family_combo, window.integral_labels_font_combo):
        names = [combo.itemText(i) for i in range(combo.count())]
        assert not any(nb_fonts.is_novelty_font(n) for n in names), names
    # A figure saved in one of them, and installed here, still shows its own plain name.
    monkeypatch.setattr(nb_fonts, "font_available", lambda name: True)
    window.canvas.set_style(font_family="Jokerman")
    window._load_style_controls_from(window.canvas)
    combo = window.output_font_family_combo
    assert combo.currentData() == "Jokerman" and combo.currentText() == "Jokerman"
