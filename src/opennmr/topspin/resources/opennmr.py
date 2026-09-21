# -*- coding: utf-8 -*-
"""
OpenNMR toolbar launcher for TopSpin.

    opennmr         add the OpenNMR button to TopSpin's main icon toolbar
    opennmr -kill   remove it again

This file is a template. ``opennmr-topspin-install`` copies it into
``<TopSpin>/exp/stan/nmr/py/user/`` and fills in the settings block below
with the exact Python that has OpenNMR installed, so nothing is looked up
on PATH at run time. Re-run the installer after moving or recreating that
Python environment.

The button sits at the end of the icon row, where TopSpin shows the
user-defined icon group. Nothing is written into TopSpin's own files: the
button is added to the running GUI and lasts until -kill or until TopSpin
exits (the installer can add ``opennmr`` to autostart.mac).

TopSpin keeps one cached toolbar per dataset dimension (1D, 2D, 3D, ...)
and swaps them in and out of a fixed host panel. The launcher listens to
that host panel and moves its button onto whichever toolbar is shown.

Runs in TopSpin's own Jython (verified on TopSpin 3.5, 4.5 and 5.0).
"""

import os
import sys
import threading

from java.awt import BorderLayout, Container, RenderingHints
from java.awt.event import ActionListener, ContainerAdapter
from java.awt.image import BufferedImage
from java.io import File
from java.lang import ProcessBuilder, String, System
from java.util import ArrayList
from javax.imageio import ImageIO
from javax.swing import (AbstractButton, ImageIcon, JButton, JOptionPane,
                         JToolBar, SwingUtilities)

# The handlers below are bare on purpose. TopSpin runs every script under
# its TopCmds prelude, whose "from java.lang import *" rebinds Exception in
# this module's globals to java.lang.Exception -- and Jython keeps the Java
# and Python exception families apart, so "except Exception" would then let
# every Python error through. A bare except catches both.


# BEGIN INSTALLER SETTINGS
COMMAND = []       # interpreter and arguments that start the OpenNMR GUI
PATH_PREPEND = []  # folders put in front of PATH for the GUI
ICON = u""         # the logo inside the installed opennmr package
# END INSTALLER SETTINGS

IS_WINDOWS = System.getProperty("os.name").lower().startswith("windows")

REINSTALL_HINT = (
    "Run opennmr-topspin-install again from the Python environment "
    "that has OpenNMR installed.")


class GuiCommand(object):
    """The OpenNMR GUI, started in the background with a fixed command line.

    The program starts without a shell, with its output going to a log file
    so a chatty program can never stall on a full pipe. If it exits with an
    error, ``on_error(exit_code, log_path)`` is called from a worker thread.
    """

    def __init__(self, command, path_prepend, on_error=None):
        self.command = list(command)
        self.path_prepend = list(path_prepend)
        self.on_error = on_error
        self.log_path = os.path.join(
            System.getProperty("java.io.tmpdir"), "opennmr", "opennmr-gui.log")

    def start(self):
        """Start the program. Raises EnvironmentError if it cannot start."""
        if not self.command:
            raise EnvironmentError("OpenNMR is not set up.\n\n" + REINSTALL_HINT)
        if not os.path.isfile(self.command[0]):
            raise EnvironmentError(
                "OpenNMR's Python was not found:\n%s\n\n%s"
                % (self.command[0], REINSTALL_HINT))

        log_file = File(self.log_path)
        log_file.getParentFile().mkdirs()

        command = ArrayList()
        for part in self.command:
            command.add(String(part))
        builder = ProcessBuilder(command)
        builder.directory(File(System.getProperty("user.home")))
        if self.path_prepend:
            # On Windows this map ignores case, so "PATH" also matches "Path".
            environment = builder.environment()
            current = environment.get("PATH")
            environment.put("PATH", os.pathsep.join(
                self.path_prepend + ([current] if current else [])))
        builder.redirectInput(File("NUL" if IS_WINDOWS else "/dev/null"))
        builder.redirectErrorStream(True)
        builder.redirectOutput(log_file)
        try:
            process = builder.start()
        except:  # java.io.IOException, or anything else the JVM raises
            raise EnvironmentError(
                "Could not start OpenNMR:\n%s" % sys.exc_info()[1])

        watcher = threading.Thread(target=self._wait, args=(process,))
        watcher.setDaemon(True)
        watcher.start()

    def _wait(self, process):
        exit_code = process.waitFor()
        if exit_code != 0 and self.on_error is not None:
            self.on_error(exit_code, self.log_path)


class _FollowToolbar(ContainerAdapter):
    """Moves the launcher onto each toolbar TopSpin swaps into the host."""

    def __init__(self, launcher):
        ContainerAdapter.__init__(self)
        self.launcher = launcher

    def componentAdded(self, event):
        if isinstance(event.getChild(), JToolBar):
            self.launcher.dock(event.getChild())


class _Click(ActionListener):
    def __init__(self, callback):
        self.callback = callback

    def actionPerformed(self, event):
        self.callback()


class ToolbarLauncher(object):
    """A button at the end of TopSpin's main icon toolbar.

    The instance is stored as a client property on the toolbar's host panel,
    so a later run of the script (``-kill``, or a second ``opennmr``) finds
    the one already installed.
    """

    def __init__(self, key, tooltip, icon_path, command):
        self.key = key
        self.command = command
        self.icon_path = icon_path
        self.host = None
        self.listener = _FollowToolbar(self)
        self.button = JButton()
        self.button.setName(key)
        self.button.setToolTipText(tooltip)
        self.button.setFocusable(False)
        # FlatLaf (TopSpin 5) styles it like the neighbouring toolbar icons.
        self.button.putClientProperty("JButton.buttonType", "toolBarButton")
        self.button.addActionListener(_Click(self.launch))

    # -- finding TopSpin's toolbar ------------------------------------------

    @staticmethod
    def current_toolbar():
        """Return the toolbar TopSpin is showing now, or None."""
        try:
            from de.bruker.nmr.mfw.base import UtilToolbar
            return UtilToolbar.getUI2Toolbar()
        except:
            return None

    @classmethod
    def installed(cls, key):
        """Return the launcher installed under ``key``, or None."""
        toolbar = cls.current_toolbar()
        host = toolbar.getParent() if toolbar is not None else None
        return host.getClientProperty(key) if host is not None else None

    # -- install / remove (on the Swing thread) -----------------------------

    def install(self):
        toolbar = self.current_toolbar()
        if toolbar is None or toolbar.getParent() is None:
            raise EnvironmentError("TopSpin's main toolbar was not found.")
        self.host = toolbar.getParent()
        self._set_icon(toolbar)
        self.dock(toolbar)
        self.host.addContainerListener(self.listener)
        self.host.putClientProperty(self.key, self)

    def remove(self):
        if self.host is not None:
            self.host.removeContainerListener(self.listener)
            self.host.putClientProperty(self.key, None)
        parent = self.button.getParent()
        if parent is not None:
            parent.remove(self.button)
            parent.revalidate()
            parent.repaint()

    def dock(self, toolbar):
        """Append the button to the icon row of ``toolbar``.

        The toolbar's centre panel holds the icon groups in a FlowLayout,
        with the user-defined group last. Adding to a new parent takes the
        button off the toolbar it was on before.
        """
        row = None
        layout = toolbar.getLayout()
        if isinstance(layout, BorderLayout):
            row = layout.getLayoutComponent(BorderLayout.CENTER)
        if not isinstance(row, Container):
            row = toolbar
        if self.button.getParent() is not row:
            row.add(self.button)
            row.revalidate()
            row.repaint()

    def _set_icon(self, toolbar):
        """Load the icon at the size of TopSpin's own toolbar icons."""
        try:
            image = ImageIO.read(File(self.icon_path))
        except:
            image = None
        if image is None:
            self.button.setText("OpenNMR")
            return
        size = _sibling_icon_size(toolbar) or 24
        scaled = [_scaled(image, size * f) for f in (1, 2)]
        try:
            # Sharp on HiDPI screens (Java 9+).
            from java.awt.image import BaseMultiResolutionImage
            self.button.setIcon(ImageIcon(BaseMultiResolutionImage(scaled)))
        except:  # Java 8 (TopSpin 3.x): one resolution only
            self.button.setIcon(ImageIcon(scaled[0]))

    # -- clicking ------------------------------------------------------------

    def launch(self):
        try:
            self.command.start()
        except:
            _show_error(self.button, "OpenNMR", str(sys.exc_info()[1]))


def _scaled(image, height):
    """Return ``image`` scaled to ``height``, halving in steps for quality."""
    while image.getHeight() > 2 * height:
        image = _draw(image, image.getWidth() // 2, image.getHeight() // 2)
    width = max(1, image.getWidth() * height // image.getHeight())
    return _draw(image, width, height)


def _draw(image, width, height):
    result = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    graphics = result.createGraphics()
    graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                              RenderingHints.VALUE_INTERPOLATION_BICUBIC)
    graphics.drawImage(image, 0, 0, width, height, None)
    graphics.dispose()
    return result


def _sibling_icon_size(component):
    """Return the icon height of the first toolbar button found, or None."""
    if isinstance(component, AbstractButton) and component.getIcon() is not None:
        return component.getIcon().getIconHeight()
    if isinstance(component, Container):
        for child in component.getComponents():
            size = _sibling_icon_size(child)
            if size:
                return size
    return None


def _show_error(anchor, title, message):
    parent = SwingUtilities.getWindowAncestor(anchor) if anchor else None
    JOptionPane.showMessageDialog(parent, message, title, JOptionPane.ERROR_MESSAGE)


def _log_tail(path, lines=15):
    try:
        handle = open(path, "rb")
        try:
            text = handle.read().decode("utf-8", "replace")
        finally:
            handle.close()
        return "\n".join(text.splitlines()[-lines:])
    except (IOError, OSError):
        return ""


def _report_exit(exit_code, log_path):
    message = "OpenNMR exited with code %d.\n\nLog: %s\n\n%s" % (
        exit_code, log_path, _log_tail(log_path))
    SwingUtilities.invokeLater(lambda: _show_error(None, "OpenNMR", message))


def _on_swing_thread(action):
    """Run ``action`` on the Swing thread; re-raise anything it raised."""
    errors = []

    def run():
        try:
            action()
        except:
            errors.append(sys.exc_info()[1])

    if SwingUtilities.isEventDispatchThread():
        run()
    else:
        SwingUtilities.invokeAndWait(run)
    if errors:
        raise errors[0]


KEY = "opennmr.toolbarLauncher"


def main():
    kill = "-kill" in [str(arg).lower() for arg in sys.argv[1:]]

    def apply():
        existing = ToolbarLauncher.installed(KEY)
        if kill:
            if existing is not None:
                existing.remove()
            return
        if existing is None:
            ToolbarLauncher(
                key=KEY,
                tooltip="Launch OpenNMR",
                icon_path=ICON,
                command=GuiCommand(COMMAND, PATH_PREPEND, on_error=_report_exit),
            ).install()

    try:
        _on_swing_thread(apply)
    except:
        _show_error(None, "OpenNMR", str(sys.exc_info()[1]))


main()
