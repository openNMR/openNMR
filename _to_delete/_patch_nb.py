import pathlib
p = pathlib.Path("src/opennmr/notebook/simpson.py"); s = p.read_text()
def sub(old, new):
    global s
    assert s.count(old) == 1, old[:50]
    s = s.replace(old, new)

# shape: draw into a given figure
sub("""               font='Arial', fontsize=12, save_fig=False,
               fig_name='pulse_shape', save_format='png'):
    \"\"\"
    Plot amplitude""", """               font='Arial', fontsize=12, save_fig=False,
               fig_name='pulse_shape', save_format='png', fig=None):
    \"\"\"
    Plot amplitude""")
sub("""    duration_us : float, optional
        Total pulse duration in microseconds. If given, the x-axis is
        time (us); if omitted, it's point index.
""", """    duration_us : float, optional
        Total pulse duration in microseconds. If given, the x-axis is
        time (us); if omitted, it's point index.
    fig : matplotlib Figure, optional
        Draw into this figure (the GUI does) instead of making one.
""")
sub("""    style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                            pl_height=pl_height)
    pl_width, pl_height = style['pl_width'], style['pl_height']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, shape['amplitude']""", """    if fig is None:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                                pl_height=pl_height)
        pl_width, pl_height = style['pl_width'], style['pl_height']
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.set_figheight(pl_height)
        fig.set_figwidth(pl_width)
    else:
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(x, shape['amplitude']""")
sub("""    fig.set_figheight(pl_height)
    fig.set_figwidth(pl_width)
    fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, (ax1, ax2)""", """    if not getattr(fig, '_opennmr_managed', False):
        fig.tight_layout()

    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)

    return fig, (ax1, ax2)""")

s += '''

# ============================================================
# Parameter-scan tables and input scripts
# ============================================================

def SIMPSONread_table(path):
    """
    Read a headerless three-column SIMPSON scan table (x, y, z) -- e.g. the
    efficiency of a pulse over a grid of rf amplitudes and offsets.

    A file whose first column is one evenly spaced axis is x, real, imaginary
    (a spectrum or FID): read that with :func:`SIMPSONload`.

    Returns
    -------
    dict : ``x``, ``y``, ``z`` (the columns); and, when x and y form a complete
    grid, ``grid=True`` with ``xs``, ``ys`` and ``Z`` (``Z[j, i]`` at
    ``ys[j]``, ``xs[i]``).
    """
    from ..io import simpson as _simpson_io
    return _simpson_io.read_table(path)


def SIMPSONread_script(path):
    """
    Read the nucleus and field out of a SIMPSON input script (.tcl), to fill
    in ``nuclide=`` and ``field=`` for :func:`SIMPSONload`. The script is
    never run; only numbers and arithmetic on them are evaluated.

    Returns
    -------
    dict : ``nucleus``, ``channels``, ``proton_mhz`` (the ``proton_frequency``
    in MHz -- pass ``field=str(proton_mhz)``), ``spin_rate``, ``sw``, ``np``;
    ``None`` where the script does not give a plain number.
    """
    from ..io import simpson as _simpson_io
    return _simpson_io.read_tcl_settings(path)


def SIMPSONplot_table(table, mode='auto', xlabel='x', ylabel='y', zlabel='z',
                      cmap='viridis', ax=None, pl_width=3.375, pl_height=2.5,
                      font='Arial', fontsize=12, save_fig=False,
                      fig_name='simpson_table', save_format='png'):
    """
    Plot a scan table from :func:`SIMPSONread_table` (or its path).

    mode : {"auto", "map", "lines"}
        ``"map"``: z as colour over the x-y grid (the default when the table is
        a complete grid); ``"lines"``: z against x, one line per y (or a
        single line when the table is not a grid).
    xlabel, ylabel, zlabel : str
        Axis and colour-bar text (quantity symbols are italicised).

    Returns ``(fig, ax)``.
    """
    from ..plotting.accessibility import default_series_colours
    from ..plotting.labels import italicise_symbols
    if isinstance(table, str):
        table = SIMPSONread_table(table)
    if mode not in ('auto', 'map', 'lines'):
        raise ValueError("mode must be 'auto', 'map' or 'lines'.")
    if mode == 'auto':
        mode = 'map' if table.get('grid') else 'lines'
    if mode == 'map' and not table.get('grid'):
        raise ValueError('mode="map" needs x and y to form a complete grid.')
    if ax is None:
        style = setup_plotting(font=font, fontsize=fontsize, pl_width=pl_width,
                               pl_height=pl_height)
        fig, ax = plt.subplots()
        fig.set_figwidth(style['pl_width'])
        fig.set_figheight(style['pl_height'])
    else:
        fig = ax.figure
    if mode == 'map':
        xs, ys, grid = table['xs'], table['ys'], table['Z']
        mesh = ax.pcolormesh(xs, ys, grid, cmap=cmap, shading='nearest')
        colour_bar = fig.colorbar(mesh, ax=ax)
        colour_bar.set_label(italicise_symbols(zlabel))
        ax.set_ylabel(italicise_symbols(ylabel))
    else:
        if table.get('grid'):
            ys = table['ys']
            colours = default_series_colours(len(ys))
            for j, y_value in enumerate(ys):
                ax.plot(table['xs'], table['Z'][j], color=colours[j % len(colours)],
                        linewidth=1.2, label=f'{y_value:g}')
            if len(ys) <= 8:
                ax.legend(frameon=False, title=italicise_symbols(ylabel))
        else:
            ax.plot(table['x'], table['z'], color='black', linewidth=1.2)
        ax.set_ylabel(italicise_symbols(zlabel))
    ax.set_xlabel(italicise_symbols(xlabel))
    for side in ('top', 'right'):
        if mode == 'lines':
            ax.spines[side].set_visible(False)
    if not getattr(fig, '_opennmr_managed', False):
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 -- layout is cosmetic
            pass
    if save_fig:
        save_figure(fig, fig_name=fig_name, save_format=save_format)
    return fig, ax
'''
sub("def SIMPSONload(path, nuclide=None, field=None, SFO1=None, reference_ppm=0.0, kind=None, **processing):",
    "def SIMPSONload(path, nuclide=None, field=None, SFO1=None, reference_ppm=0.0, kind=None,\n                element=0, **processing):")
sub("""    kind : {"fid", "spe"}, optional
        Needed only for a headerless ``fsave -xreim`` .dat file.
    \"\"\"""", """    kind : {"fid", "spe"}, optional
        For a headerless ``fsave -xreim`` .dat file (x, real, imaginary
        columns): left out, a file whose x column runs through zero is read
        as a spectrum (Hz about the carrier) and one that starts at or after
        zero as a FID (give ``sw=`` or a time unit with ``time_unit=``).
    element : int
        Which of the FIDs/spectra of a file with ``NELEM`` > 1 (0-based); see
        :func:`opennmr.io.simpson.n_elements`.

    A FID or spectrum may have real and imaginary parts (``re im`` per line in
    SIMP text, three columns in a .dat); both are kept. A pulse shape, scan
    table or input script is not read here: use :func:`SIMPSONread_shape`,
    :func:`SIMPSONread_table` and :func:`SIMPSONread_script`.
    \"\"\"""")
sub("""reference_ppm=reference_ppm, kind=kind, **processing)""", """reference_ppm=reference_ppm, kind=kind, element=element, **processing)""")
p.write_text(s)

for path, old, new in (
    ("src/opennmr/notebook/__init__.py", "    SIMPSONplot_fid_series, SIMPSONload,", "    SIMPSONplot_fid_series, SIMPSONload, SIMPSONread_table, SIMPSONread_script, SIMPSONplot_table,"),
    ("src/opennmr/notebook/__init__.py", '    "SIMPSONplot_fid_series", "SIMPSONload",', '    "SIMPSONplot_fid_series", "SIMPSONload", "SIMPSONread_table", "SIMPSONread_script",\n    "SIMPSONplot_table",')):
    q = pathlib.Path(path); t = q.read_text(); assert t.count(old) == 1; q.write_text(t.replace(old, new))
print("ok")
