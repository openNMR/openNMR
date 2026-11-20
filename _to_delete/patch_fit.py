import sys
root = sys.argv[1]
p = root + "/src/opennmr/gui/main_window.py"
s = open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
s = s.replace("\r\n", "\n")
def rep(a, b):
    global s
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)
rep("from .motion_panel import AnisotropicMotionPanel\n",
    "from .motion_fit_panel import MotionFitPanel\nfrom .motion_panel import AnisotropicMotionPanel\n")
rep('''("_motion_section",)),''', '''("_motion_section", "_motion_fit_section")),''')
rep('''"simpson": "SIMPSON import", "bms": "BMS shifts",
                          "motion": "Anisotropic motion"}''', '''"simpson": "SIMPSON import", "bms": "BMS shifts",
                          "motion": "Anisotropic motion",
                          "motion_fit": "Anisotropic motion fit"}''')
rep('''            # Reading files is not a simulation run: nothing to re-run.
            "simpson": _Section(''', '''            # A fit can take minutes: restoring puts the inputs back and leaves it one click away.
            "motion_fit": _Section(get_state=self.motion_fit_panel.get_state,
                                   set_state=self.motion_fit_panel.set_state,
                                   rerun=None, summary=panel_summary(self.motion_fit_panel),
                                   anchor=lambda: self.motion_fit_panel),
            # Reading files is not a simulation run: nothing to re-run.
            "simpson": _Section(''')
rep('''        if key == "motion":
            self._motion_section.setVisible(True)      # an optional section may be hidden
        self._show_side_tab("Dynamics" if key == "motion" else "Simulate")''',
    '''        if key in ("motion", "motion_fit"):
            for name in ("_motion_section", "_motion_fit_section"):
                getattr(self, name).setVisible(True)   # an optional section may be hidden
        self._show_side_tab("Dynamics" if key in ("motion", "motion_fit") else "Simulate")''')
rep('''        layout.addWidget(self._motion_section)
''', '''        layout.addWidget(self._motion_section)
        # Fitting it (rate per spectrum + Arrhenius line, or a global fit); notebook.motion_fit.
        self.motion_fit_panel = MotionFitPanel(
            self.motion_panel, canvas=self.fit_canvas, show_view=self._raise_fit_canvas,
            on_run=lambda: self._journal_simulation("motion_fit"))
        self._motion_fit_section = CollapsibleSection(
            "Fit Anisotropic Motion to Spectra (Rates and Barriers)", content=self.motion_fit_panel)
        layout.addWidget(self._motion_fit_section)
''')
open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", nl))
print("ok")
