"""Settings ▸ Updates: is there a newer openNMR, and how do I get it?

The checking itself is :mod:`opennmr.update` (it never downloads or
installs). This panel shows which version is running and how it was
installed -- the app or a pip install, which update differently -- runs
the check off the GUI thread, and then offers the right next step:

* the app: **Install** downloads this platform's installer (checksum
  checked first), closes openNMR, installs and starts it again --
  :mod:`opennmr.update_install` does the work; "Download in browser" is
  the fallback when a release cannot be verified or installed that way;
* a pip install from PyPI: **Update to X and restart** (one question,
  then openNMR closes, a hidden helper upgrades it with pip and
  reopens it -- :func:`opennmr.update_install.plan_pip_upgrade`; nothing
  is shown, and the old version comes back if the upgrade fails). It is
  offered only where pip may change this environment; otherwise, and
  always beside it, the command is shown with a Copy button;
* a git install: **Update with pip** runs pip in this environment
  (restart afterwards), and the command is shown with a Copy button;
* a source folder: the ``git pull`` command only -- it is never run for
  the user, since it could overwrite their own changes.

Remembered in ``~/.opennmr/ui_settings.json`` (the dict MainWindow
passes in):

* ``update_check_at_start`` -- check when openNMR starts, at most once
  every :data:`STARTUP_INTERVAL_S`. Off until the user turns it on (a
  network call nobody asked for); the environment variable
  ``OPENNMR_UPDATE_CHECK=0`` turns it off regardless.
* ``update_last_checked`` -- epoch seconds of the last check.
* ``update_source`` -- ``pypi``, ``owner/repo`` or a manifest URL
  (default :func:`opennmr.update.default_source`: PyPI for a pip install,
  :data:`opennmr.update.DEFAULT_SOURCE` otherwise).
* ``update_token`` -- a GitHub token, only needed while the repository
  is private. Stored as plain text in that file, which the tooltip says.
"""

from __future__ import annotations

import sys
import threading
import time

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import update, update_install
from . import loader

#: A start-up check is skipped if the last check was more recent than this.
STARTUP_INTERVAL_S = 20 * 3600

#: Release notes longer than this are cut in the panel (the full notes
#: are on the release page).
NOTES_SHOWN = 600


class _CheckRelay(QObject):
    """Carries a finished check from the worker thread to the GUI thread
    (a queued signal); parented to the panel so it dies with it."""

    finished = Signal(object, bool)


class _InstallRelay(QObject):
    """Carries an install's progress and outcome from the worker thread
    to the GUI thread. ``progress(done, total)`` for a download, or
    ``progress(line, None)`` for a line of pip's output."""

    progress = Signal(object, object)
    done = Signal(object)


class UpdatesPanel(QWidget):
    """The Updates section of the Settings panel.

    ``settings`` is MainWindow's ``_ui_settings`` dict (edited in place)
    and ``save`` writes it; ``notify`` shows a short status-bar message.
    """

    #: Emitted (on the GUI thread) with the :class:`update.UpdateCheck`
    #: of every finished check.
    checkFinished = Signal(object)

    def __init__(self, settings: dict, save=None, notify=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._save = save or (lambda: None)
        self._notify = notify or (lambda _text: None)
        self._kind = update.install_kind()
        self._default_source = update.default_source(self._kind)
        self._checking = False
        self.last_result: update.UpdateCheck | None = None
        #: Background thread or not; tests (conftest pins
        #: OPENNMR_BACKGROUND_LOAD=0) get the same code, called directly.
        self.background = loader.background_load_enabled()
        self._relay = _CheckRelay(self)
        self._relay.finished.connect(self._on_check_finished)
        self._installing = False
        self._cancel_event = threading.Event()
        self._install_relay = _InstallRelay(self)
        self._install_relay.progress.connect(self._on_install_progress)
        self._install_relay.done.connect(self._on_install_done)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.version_label = QLabel(
            f"This is openNMR {update.__version__}, running as "
            f"{update.INSTALL_KINDS.get(self._kind, self._kind)}.")
        self.version_label.setWordWrap(True)
        layout.addWidget(self.version_label)

        check_row = QHBoxLayout()
        self.check_button = QPushButton("Check for Updates")
        self.check_button.setToolTip("Ask the update source whether a newer openNMR exists. "
                                     "Nothing is downloaded or installed.")
        self.check_button.clicked.connect(lambda: self.check_now())
        check_row.addWidget(self.check_button)
        check_row.addStretch(1)
        layout.addLayout(check_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        # The next step, shown once a newer version is found.
        self.install_button = QPushButton("Install")
        self.install_button.setToolTip(
            "Download the update, check it, and install it. For the app, openNMR closes and "
            "starts again; for a pip install, pip updates this environment.")
        self.install_button.clicked.connect(lambda: self._on_install_clicked())
        self.install_button.hide()
        self.download_button = QPushButton("Download in browser")
        self.download_button.setToolTip("Open the download in your web browser and install it "
                                        "yourself.")
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.hide()
        self.release_page_button = QPushButton("What's New…")
        self.release_page_button.setToolTip("Open the release page (its notes say what changed).")
        self.release_page_button.clicked.connect(self._on_release_page_clicked)
        self.release_page_button.hide()
        action_row = QHBoxLayout()
        action_row.addWidget(self.install_button)
        action_row.addWidget(self.download_button)
        action_row.addWidget(self.release_page_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(lambda: self._cancel_event.set())
        self.cancel_button.hide()
        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.cancel_button)
        layout.addLayout(progress_row)

        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setToolTip("Run this in a terminal (or a notebook cell, prefixed with %).")
        self.copy_command_button = QPushButton("Copy")
        self.copy_command_button.clicked.connect(self._on_copy_command)
        command_row = QHBoxLayout()
        command_row.addWidget(self.command_edit, 1)
        command_row.addWidget(self.copy_command_button)
        self.command_row = QWidget()
        self.command_row.setLayout(command_row)
        command_row.setContentsMargins(0, 0, 0, 0)
        self.command_row.hide()
        layout.addWidget(self.command_row)

        self.startup_checkbox = QCheckBox("Check when openNMR starts (at most once a day)")
        self.startup_checkbox.setToolTip(
            "In the background, a few seconds after start-up. If a newer version exists the "
            "status bar says so; nothing is downloaded.")
        self.startup_checkbox.setChecked(bool(settings.get("update_check_at_start", False)))
        self.startup_checkbox.toggled.connect(self._on_startup_toggled)
        layout.addWidget(self.startup_checkbox)

        form = QFormLayout()
        self.source_edit = QLineEdit(str(settings.get("update_source") or self._default_source))
        self.source_edit.setToolTip(
            "Where to look: 'pypi', a GitHub repository as owner/name (its latest Release), or "
            "the address of a version manifest (JSON).")
        self.source_edit.editingFinished.connect(self._on_source_edited)
        source_reset = QPushButton("Reset")
        source_reset.setToolTip(f"Back to {self._default_source}.")
        source_reset.clicked.connect(self._on_source_reset)
        self.source_reset_button = source_reset
        source_row = QHBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(source_reset)
        source_widget = QWidget()
        source_widget.setLayout(source_row)
        form.addRow("Releases from:", source_widget)

        self.token_edit = QLineEdit(str(settings.get("update_token") or ""))
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("Only while the repository is private")
        self.token_edit.setToolTip(
            "A GitHub token with read access to the repository (a fine-grained token with "
            "'Contents: read-only' is enough). Needed only while the repository is private.\n\n"
            "It is stored as plain text in ~/.opennmr/ui_settings.json. Alternatively leave this "
            f"empty and set the environment variable {update.TOKEN_ENV}.")
        self.token_edit.editingFinished.connect(self._on_token_edited)
        form.addRow("Access token:", self.token_edit)
        # A PyPI install asks PyPI: no repository or token to choose.
        self.advanced_box = QWidget()
        self.advanced_box.setLayout(form)
        form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.advanced_box)
        if self._kind == "pip":
            self.advanced_box.hide()

        notebook = QLabel(
            "The app and the notebook package are separate copies: updating one does not "
            "change the other. To use openNMR in Jupyter or Colab, pip install it (see README).")
        notebook.setWordWrap(True)
        notebook.setStyleSheet("color: gray;")
        layout.addWidget(notebook)

        self._show_last_upgrade_result()

    def _show_last_upgrade_result(self) -> None:
        """What an in-app upgrade did, said once and calmly at the first
        start after it (a line here and in the status bar; never a dialog)."""
        result = update_install.take_last_result()
        if not result:
            return
        text = update_install.describe_result(result, update.__version__)
        if text:
            self.status_label.setText(text)
            self._notify(text)

    # -- settings ---------------------------------------------------------

    def source(self) -> str:
        return self.source_edit.text().strip() or self._default_source

    def token(self) -> str | None:
        return self.token_edit.text().strip() or None

    def _on_startup_toggled(self, on: bool) -> None:
        self._settings["update_check_at_start"] = bool(on)
        self._save()

    def _on_source_edited(self) -> None:
        text = self.source_edit.text().strip()
        if text == self._default_source or not text:
            self._settings.pop("update_source", None)
        else:
            self._settings["update_source"] = text
        self._save()

    def _on_source_reset(self) -> None:
        self.source_edit.setText(self._default_source)
        self._on_source_edited()

    def _on_token_edited(self) -> None:
        text = self.token_edit.text().strip()
        if text:
            self._settings["update_token"] = text
        else:
            self._settings.pop("update_token", None)
        self._save()

    # -- checking ---------------------------------------------------------

    def startup_check_due(self, now: float | None = None) -> bool:
        """Whether a start-up check should run now: enabled here and by the
        environment, and none in the last :data:`STARTUP_INTERVAL_S`."""
        if not self.startup_checkbox.isChecked() or not update.startup_check_enabled():
            return False
        now = time.time() if now is None else now
        try:
            last = float(self._settings.get("update_last_checked") or 0)
        except (TypeError, ValueError):
            last = 0.0
        return now - last >= STARTUP_INTERVAL_S

    def maybe_check_at_startup(self) -> bool:
        """Run a quiet check if one is due; returns whether one started."""
        if not self.startup_check_due():
            return False
        self.check_now(quiet=True)
        return True

    def check_now(self, quiet: bool = False) -> None:
        """Check the source. ``quiet`` (start-up): only a newer version is
        announced in the status bar; errors stay in this panel."""
        if self._checking:
            return
        self._checking = True
        self.check_button.setEnabled(False)
        self.status_label.setText("Checking…")
        source, token = self.source(), self.token()

        def work():
            result = update.check(source, token=token)
            try:
                self._relay.finished.emit(result, quiet)
            except RuntimeError:  # the window closed while checking
                pass

        if self.background:
            threading.Thread(target=work, name="opennmr-update-check", daemon=True).start()
        else:
            work()

    def _on_check_finished(self, result, quiet: bool) -> None:
        self._checking = False
        self.check_button.setEnabled(True)
        self.last_result = result
        self._settings["update_last_checked"] = time.time()
        self._save()
        self._show_result(result)
        if result.status == "newer":
            self._notify(f"openNMR {result.info.version} is available – Settings ▸ Updates.")
        elif not quiet and result.status == "current":
            self._notify("openNMR is up to date.")
        self.checkFinished.emit(result)

    def _show_result(self, result) -> None:
        self.install_button.hide()
        self.download_button.hide()
        self.release_page_button.hide()
        self.command_row.hide()
        checked = time.strftime("%H:%M")
        if result.status == "error":
            self.status_label.setText(f"Could not check ({checked}): {result.message}")
            return
        if result.status == "current":
            latest = f" (latest release: {result.latest_version})" if result.latest_version else ""
            self.status_label.setText(f"Up to date{latest}. Checked at {checked}.")
            return
        info = result.info
        notes = info.notes
        if len(notes) > NOTES_SHOWN:
            notes = notes[:NOTES_SHOWN].rstrip() + " …"
        upgrade_blocker = (update_install.upgrade_blocker(self._kind)
                           if self._kind == "pip" else "")
        text = f"openNMR {info.version} is available (this is {result.current_version}).\n"
        if self._kind == "pip" and not upgrade_blocker:
            text += "Updating closes openNMR and opens it again."
        else:
            text += update.update_instructions(self._kind, info.version, self.source())
            if upgrade_blocker:
                text += f"\n({upgrade_blocker})"
        no_installer = self._kind == "app" and not info.asset_name
        if no_installer:
            # The release exists but has nothing for this system attached (a source-code
            # archive is never offered as an installer): say so and leave the release
            # page as the way on.
            text += ("\n\nThis release has no installer for your system attached yet, so openNMR "
                     "cannot install it automatically. The release page lists what is available.")
        if notes:
            text += f"\n\nWhat's new:\n{notes}"
        self.status_label.setText(text)
        if no_installer:
            pass
        elif self._kind == "app":
            self.install_button.setText(f"Install openNMR {info.version}")
            self.install_button.show()
            self.download_button.show()
        elif self._kind == "pip":
            if not upgrade_blocker:
                self.install_button.setText(f"Update to {info.version} and restart")
                self.install_button.show()
        elif update_install.pip_argv(self._kind, self.source()) is not None:
            self.install_button.setText(f"Update with pip to {info.version}")
            self.install_button.show()
        self.install_button.setEnabled(not self._installing)
        command = update.pip_command(self._kind, self.source())
        if command:
            self.command_edit.setText(command)
            self.command_row.show()
        if result.release_page:
            self.release_page_button.show()

    # -- actions ----------------------------------------------------------

    def _open_url(self, url: str) -> None:
        """Seam for tests: opening a browser under test would be rude."""
        QDesktopServices.openUrl(QUrl(url))

    def _on_download_clicked(self) -> None:
        if self.last_result is not None and self.last_result.info is not None:
            self._open_url(self.last_result.info.url)

    # -- installing -------------------------------------------------------
    # Seams for tests: none of these should really run under test.

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _launch(self, plan) -> None:
        update_install.launch(plan)

    def _open_path(self, path: str) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _plan_pip_upgrade(self, version: str):
        return update_install.plan_pip_upgrade(version)

    def _launch_upgrade(self, plan) -> None:
        update_install.launch_upgrade(plan)

    def _exit_app(self) -> None:
        window = self.window()
        if window is not None:
            window.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_install_clicked(self) -> None:
        result = self.last_result
        if self._installing or result is None or result.info is None:
            return
        info = result.info
        if self._kind == "app":
            size = f" ({info.size / 1e6:.0f} MB)" if info.size else ""
            question = (f"Download openNMR {info.version}{size}, then close openNMR, install it "
                        "and start it again?\n\nSave any work you want to keep first. Your "
                        "settings are kept.")
        elif self._kind == "pip":
            question = (f"Update openNMR to {info.version}?\n\nopenNMR will close, update itself "
                        "and open again, which takes a minute or so. Your settings are kept.")
            if self._confirm("Update openNMR", question):
                self._start_pip_upgrade(info)
            return
        else:
            question = (f"Update openNMR to {info.version} with pip, in this Python environment "
                        f"({sys.executable})?\n\nRestart openNMR afterwards to use it.")
        if self._confirm("Update openNMR", question):
            self._start_install(info)

    def _start_pip_upgrade(self, info) -> None:
        """Hand the upgrade to the detached helper and quit. If the helper
        cannot even be started nothing has happened: openNMR stays open
        and the pip command is shown, in plain words."""
        try:
            plan = self._plan_pip_upgrade(info.version)
            self._launch_upgrade(plan)
        except (update_install.InstallError, OSError) as exc:
            self.status_label.setText(
                "The update could not be started, so openNMR was left as it is. "
                "To update by hand, run the command below.")
            command = update.pip_command(self._kind)
            if command:
                self.command_edit.setText(command)
                self.command_row.show()
            self._notify(f"The update could not be started ({exc}).")
            return
        self.status_label.setText("Updating – openNMR will close and open again.")
        self._exit_app()

    def _start_install(self, info) -> None:
        self._installing = True
        self._cancel_event.clear()
        self.install_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("")
        self.progress_bar.show()
        self.cancel_button.show()
        self.status_label.setText("Preparing the update…")
        token, kind, source = self.token(), self._kind, self.source()
        relay, cancelled = self._install_relay, self._cancel_event.is_set

        def work():
            try:
                if kind == "app":
                    path = update_install.fetch_installer(
                        info, token=token, cancelled=cancelled,
                        progress=lambda done, total: relay.progress.emit(done, total))
                    payload = {"ok": True, "plan": update_install.plan_install(path)}
                else:
                    ok, output = update_install.run_pip(
                        update_install.pip_argv(kind, source), cancelled=cancelled,
                        on_line=lambda line: relay.progress.emit(line, None))
                    tail = "\n".join(output.strip().splitlines()[-6:])
                    payload = {"ok": ok, "pip": True, "version": info.version,
                               "message": "" if ok else f"pip could not update openNMR:\n{tail}"}
            except update_install.InstallError as exc:
                payload = {"ok": False, "message": str(exc)}
            except Exception as exc:  # a worker must never die without telling the panel
                payload = {"ok": False, "message": f"Unexpected error: {exc}"}
            try:
                relay.done.emit(payload)
            except RuntimeError:  # the window closed meanwhile
                pass

        if self.background:
            threading.Thread(target=work, name="opennmr-update-install", daemon=True).start()
        else:
            work()

    def _on_install_progress(self, done, total) -> None:
        if isinstance(done, str):  # a line of pip's output
            self.status_label.setText(done[:200] or "Updating…")
            return
        if total:
            self.progress_bar.setRange(0, int(total))
            self.progress_bar.setValue(int(done))
            self.progress_bar.setFormat(f"{done / 1e6:.0f} / {total / 1e6:.0f} MB")
            self.status_label.setText("Downloading the update…")
        else:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText(f"Downloading the update… {done / 1e6:.0f} MB")

    def _on_install_done(self, payload) -> None:
        self._installing = False
        self.progress_bar.hide()
        self.cancel_button.hide()
        self.check_button.setEnabled(True)
        self.install_button.setEnabled(True)
        if not payload.get("ok"):
            self.status_label.setText(f"The update did not install: {payload.get('message', '')}")
            self._notify("The update did not install – see Settings ▸ Updates.")
            return
        if payload.get("pip"):
            text = f"openNMR was updated to {payload.get('version', 'the newest version')}. " \
                   "Restart openNMR to use it."
            self.status_label.setText(text)
            self._notify(text)
            return
        plan = payload["plan"]
        if plan.manual:
            self.status_label.setText(plan.message)
            self._open_path(plan.open_path)
            return
        try:
            self._launch(plan)
        except update_install.InstallError as exc:
            self.status_label.setText(f"The update did not install: {exc}")
            return
        self.status_label.setText("Installing – openNMR will close and start again.")
        self._exit_app()

    def _on_release_page_clicked(self) -> None:
        if self.last_result is not None and self.last_result.release_page:
            self._open_url(self.last_result.release_page)

    def _on_copy_command(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.command_edit.text())
        self._notify("Copied.")


def schedule_startup_check(panel: UpdatesPanel, delay_ms: int = 4000) -> None:
    """After start-up has settled (a check must never slow the window
    appearing), run the quiet check if one is due."""
    QTimer.singleShot(delay_ms, panel, panel.maybe_check_at_startup)
