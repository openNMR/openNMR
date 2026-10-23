import pathlib
p = pathlib.Path("src/opennmr/gui/simpson_panel.py"); s = p.read_text()
def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)

sub('FILE_FILTER = "SIMPSON files (*.spe *.fid *.dat *.txt *.xreim *.out);;All files (*)"',
    'FILE_FILTER = ("SIMPSON files (*.spe *.fid *.csdf *.dat *.shape *.tcl *.txt *.xreim *.out);;"\n               "All files (*)")')

# add_paths: scripts fill in the boxes instead of joining the list
sub('''    def add_paths(self, paths) -> None:
        have = set(self.paths())
        for path in sorted(str(p) for p in paths):
            if path in have:
                continue''', '''    def add_paths(self, paths) -> None:
        have = set(self.paths())
        for path in sorted(str(p) for p in paths):
            if path in have:
                continue
            if path.lower().endswith(".tcl"):
                self.apply_script(path)
                continue''')

sub('''    def _on_add_files(self) -> None:''', '''    def apply_script(self, path: str) -> dict | None:
        """Fill in the nuclide and field from a SIMPSON input script (.tcl).
        The script is only read, never run."""
        try:
            settings = simpson_io.read_tcl_settings(path)
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            self._say(f"Could not read {os.path.basename(path)}: {exc}")
            return None
        done = []
        if settings.get("nucleus"):
            self.nuclide_combo.setCurrentText(settings["nucleus"])
            done.append(f"nuclide {settings['nucleus']}")
        if settings.get("proton_mhz"):
            self._proton_spin.setValue(float(settings["proton_mhz"]))
            done.append(f"{settings['proton_mhz']:g} MHz ¹H")
        self._say(f"{os.path.basename(path)}: set " + (", ".join(done) if done else "nothing")
                  + ". Check the nuclide and field; the carrier is not in the script.")
        return settings

    def _on_add_files(self) -> None:''')

# element spin in spectra box
sub('''        row = QHBoxLayout()
        self.plot_button = QPushButton("Plot")''', '''        row = QHBoxLayout()
        row.addWidget(QLabel("Element to plot:"))
        self.element_spin = TypedSpinBox()
        self.element_spin.setRange(1, 9999)
        self.element_spin.setValue(1)
        self.element_spin.setToolTip("A SIMPSON file with NELEM > 1 holds several FIDs or spectra "
                                     "(e.g. one per detect operator). Plot shows this one; Add to "
                                     "Open experiments adds all of them, as name [k/n].")
        row.addWidget(self.element_spin)
        row.addStretch(1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.plot_button = QPushButton("Plot")''')

sub('''    def read_spectrum(self, path: str):
        return simpson_io.read_1d(path, **self.read_kwargs())
''', '''    def read_spectrum(self, path: str, element: int | None = None):
        kwargs = self.read_kwargs()
        kwargs["element"] = (self.element_spin.value() - 1) if element is None else element
        return simpson_io.read_1d(path, **kwargs)

    def _plot_table(self, path: str) -> None:
        from ..notebook import simpson as simpson_nb
        table = simpson_nb.SIMPSONread_table(path)

        def draw(fig):
            fig.clear()
            ax = fig.add_subplot(111)
            simpson_nb.SIMPSONplot_table(table, ax=ax)
            return ax

        self._draw_plot(draw)

    def _plot_shape(self, path: str) -> None:
        from ..notebook import simpson as simpson_nb
        simpson_nb.SIMPSONread_shape(path)  # raises with a clear message if malformed

        def draw(fig):
            _fig, (ax_amplitude, _ax_phase) = simpson_nb.SIMPSONplot_shape(path, fig=fig)
            return ax_amplitude

        self._draw_plot(draw)
''')

sub('''        try:
            self.spectrum = self.read_spectrum(paths[0])
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "SIMPSON", f"Could not read {os.path.basename(paths[0])}:\\n{exc}")
            return
        self._plot_spectrum(self.spectrum)
        self._say(f"Plotted {os.path.basename(paths[0])}.")''', '''        name = os.path.basename(paths[0])
        role = simpson_io.describe(paths[0])["role"]
        try:
            if role == "table":
                self._plot_table(paths[0])
                self._say(f"Plotted {name} as a table of x, y, z (a parameter scan). It is not "
                          "a spectrum, so it cannot be added to Open experiments.")
                return
            if role == "shape":
                self._plot_shape(paths[0])
                self._say(f"Plotted the pulse shape {name} (amplitude and phase).")
                return
            n_elements = simpson_io.describe(paths[0]).get("elements", 1)
            self.element_spin.setMaximum(max(n_elements, 1))
            self.spectrum = self.read_spectrum(paths[0])
        except Exception as exc:  # noqa: BLE001 -- shown to the user
            QMessageBox.warning(self, "SIMPSON", f"Could not read {name}:\\n{exc}")
            return
        self._plot_spectrum(self.spectrum)
        self._say(f"Plotted {name}"
                  + (f" (element {self.element_spin.value()} of {n_elements})"
                     if n_elements > 1 else "") + ".")''')

sub('''        self._open_files(paths, self.read_kwargs())
        self._say(f"Added {len(paths)} file(s) to Open experiments.")''', '''        added, skipped, count = [], [], 0
        for path in paths:
            info = simpson_io.describe(path)
            if info["role"] != "data":
                skipped.append(f"{os.path.basename(path)} ({info['detail']})")
                continue
            elements = info.get("elements", 1)
            for k in range(elements):
                kwargs = self.read_kwargs()
                if elements > 1:
                    kwargs["element"] = k
                self._open_files([path], kwargs)
                count += 1
            added.append(path)
        text = f"Added {count} spectrum/FID(s) from {len(added)} file(s) to Open experiments."
        if skipped:
            text += " Not spectra or FIDs, so not added: " + "; ".join(skipped) + "."
        self._say(text)''')
p.write_text(s); print("ok")
