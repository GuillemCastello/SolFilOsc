"""Shared helpers for the nested on-disk date layout.

Both preprocessed data and analysis results are stored under a nested date
hierarchy following the usual GONG naming convention::

    <root>/<year>/<month>/<day>/...

where ``year = YYYY`` (e.g. ``2014``), ``month = YYYYMM`` (e.g. ``201401``) and
``day = YYYYMMDD`` (e.g. ``20140102``). These helpers are the single source of
truth for that mapping.
"""

import os


def day_relpath(day: str) -> str:
    """Map a ``YYYYMMDD`` day string to its ``YYYY/YYYYMM/YYYYMMDD`` subpath."""
    day = str(day)
    return os.path.join(day[:4], day[:6], day)


def day_dir(root: str, day: str) -> str:
    """Directory holding one day's files: ``<root>/YYYY/YYYYMM/YYYYMMDD``."""
    return os.path.join(root, day_relpath(day))


def discover_days(root: str) -> list[str]:
    """Return sorted ``YYYYMMDD`` day strings found under ``<root>/YYYY/YYYYMM/YYYYMMDD``.

    Folder names are filtered to 4/6/8-digit numeric components, so siblings such
    as ``data/raw`` are ignored.
    """
    days = []
    if not os.path.isdir(root):
        return days
    for y in os.listdir(root):
        yp = os.path.join(root, y)
        if not (len(y) == 4 and y.isdigit() and os.path.isdir(yp)):
            continue
        for m in os.listdir(yp):
            mp = os.path.join(yp, m)
            if not (len(m) == 6 and m.isdigit() and os.path.isdir(mp)):
                continue
            for d in os.listdir(mp):
                if len(d) == 8 and d.isdigit() and os.path.isdir(os.path.join(mp, d)):
                    days.append(d)
    return sorted(days)
