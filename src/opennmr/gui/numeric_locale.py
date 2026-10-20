"""Numbers are always written with a full stop as the decimal point.

Qt formats and parses every spin box and number validator with the
system locale, so on a French, German, Spanish, ... computer one box
showed ``1,5`` while the plots, tables and logs (plain Python and
matplotlib, which never look at the locale) showed ``1.5`` -- explicit
user report, 2026-10-20: "sometimes commas appear as the decimal,
sometimes points; it should ALWAYS be a full stop".

:func:`use_decimal_point` makes English (United States) Qt's default
locale. Call it before any widget is built (``app.main`` does, and
``MainWindow`` does again for embedding hosts and tests); calling it more
than once is harmless.
"""

from __future__ import annotations

from PySide6.QtCore import QLocale

#: What every number box, validator and Qt-formatted number uses.
LOCALE = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)


def use_decimal_point() -> QLocale:
    """Make ``.`` the decimal point in every Qt number box. Returns the
    locale now in force."""
    QLocale.setDefault(LOCALE)
    return LOCALE
