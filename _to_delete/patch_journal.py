import sys, io
root = sys.argv[1]
p = root + "/src/opennmr/gui/main_window.py"
s = open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
s = s.replace("\r\n", "\n")
def rep(a, b):
    global s
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)
rep('''"simpson": "SIMPSON import", "bms": "BMS shifts"}''',
    '''"simpson": "SIMPSON import", "bms": "BMS shifts",
                          "motion": "Anisotropic motion"}''')
rep('''            # Reading files is not a simulation run: nothing to re-run.
            "simpson": _Section(''', '''            # Dynamics ▸ Anisotropic motion: re-runs what was run last (seconds).
            "motion": _Section(get_state=self.motion_panel.get_state,
                               set_state=self.motion_panel.set_state,
                               rerun=self.motion_panel.rerun,
                               summary=panel_summary(self.motion_panel),
                               anchor=lambda: self.motion_panel),
            # Reading files is not a simulation run: nothing to re-run.
            "simpson": _Section(''')
rep('''        self._show_side_tab("Simulate")
        anchor = section.anchor()''', '''        if key == "motion":
            self._motion_section.setVisible(True)      # an optional section may be hidden
        self._show_side_tab("Dynamics" if key == "motion" else "Simulate")
        anchor = section.anchor()''')
rep('''get_simulate_settings=lambda: self.combined_sim_panel.settings()''',
    '''get_simulate_settings=lambda: self.combined_sim_panel.settings(),
            on_run=lambda: self._journal_simulation("motion")''')
open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", nl))
print("ok")
