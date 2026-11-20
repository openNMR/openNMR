import sys
root=sys.argv[1]
def patch(path, edits):
    t=open(path,encoding='utf-8').read()
    for a,b in edits:
        assert t.count(a)==1,(path,a[:60],t.count(a))
        t=t.replace(a,b)
    open(path,'w',encoding='utf-8').write(t)

patch(root+"/src/opennmr/gui/main_window.py",[
("from .dynamics_panels import ExchangeLineshapePanel, VTRelaxationPanel\n",
 "from .dynamics_panels import ExchangeLineshapePanel, VTRelaxationPanel\nfrom .motion_panel import AnisotropicMotionPanel\n"),
('''        "exchange": ("Exchange lineshape (Dynamics)", ("_exchange_section",)),
''','''        "exchange": ("Exchange lineshape (Dynamics)", ("_exchange_section",)),
        "motion": ("Anisotropic motion: rotating tensors (Dynamics)", ("_motion_section",)),
'''),
('''        layout.addWidget(self._exchange_section)
        # Tikhonov regularisation, its own section''','''        layout.addWidget(self._exchange_section)
        # Rotations of CSA / quadrupolar / dipolar tensors as well as hops, with Simulate as
        # the source of the interactions (explicit user request, 2026-11-18); maths
        # notebook.motion, drawn on the Fit view.
        self.motion_panel = AnisotropicMotionPanel(
            canvas=self.fit_canvas, show_view=self._raise_fit_canvas,
            get_spectra=self._exchange_fit_spectra,
            get_simulate_sites=self._motion_simulate_sites,
            get_simulate_settings=lambda: self.combined_sim_panel.settings())
        self._motion_section = CollapsibleSection(
            "Anisotropic Motion (Rotations and Hops)", content=self.motion_panel)
        layout.addWidget(self._motion_section)
        # Tikhonov regularisation, its own section'''),
('''    def _exchange_fit_spectra(self) -> list[dict]:''','''    def _motion_simulate_sites(self) -> list:
        """Simulate ▸ Solid-state NMR's sites of the observed nucleus, for Dynamics ▸
        Anisotropic motion."""
        panel = self.combined_sim_panel
        observed = panel.observed()
        sites = [s for s in panel.build_sites() if s.nucleus == observed]
        if not sites:
            raise ValueError(f"Simulate has no {observed} sites.")
        return sites

    def _exchange_fit_spectra(self) -> list[dict]:'''),
])
patch(root+"/src/opennmr/gui/welcome_wizard.py",[
('"batch_pseudo2d", "bpp", "exchange", "advanced")),','"batch_pseudo2d", "bpp", "exchange", "motion", "advanced")),')])
