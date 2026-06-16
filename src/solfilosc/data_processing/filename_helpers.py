"""Helpers for parsing SolFilOsc preprocessing filenames.

The preprocessing pipeline receives metadata from GONG-style filenames.  Keep
that knowledge in this module so pipeline code does not rely on positional
string slices such as ``file[-16:-10]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable


DEFAULT_OBSERVATORIES = frozenset({"L", "M", "U", "C", "T", "B"})
FITS_FZ_SUFFIX = ".fits.fz"
FITS_SUFFIX = ".fits"
UPDATED_SUFFIX = "_updated"

_OBSERVATION_RE = re.compile(
    r"(?P<date>\d{8})(?P<time>\d{6})(?P<observatory>[A-Za-z])(?P<product>[A-Za-z0-9]*)"
)


@dataclass(frozen=True)
class FilenameInfo:
    """Parsed metadata from a raw or updated preprocessing FITS filename."""

    path: str
    name: str
    stem: str
    canonical_stem: str
    date: str
    time: str
    observatory: str
    product: str
    is_updated: bool
    observation_datetime: datetime

    @property
    def year(self) -> str:
        return self.date[:4]

    @property
    def month(self) -> str:
        return self.date[4:6]

    @property
    def day(self) -> str:
        return self.date[6:8]

    @property
    def seconds_since_midnight(self) -> int:
        return (
            int(self.time[:2]) * 3600
            + int(self.time[2:4]) * 60
            + int(self.time[4:6])
        )

    @property
    def hhmmss(self) -> str:
        return self.time

    @property
    def yyyymmdd(self) -> str:
        return self.date

    @property
    def updated_name(self) -> str:
        return f"{self.canonical_stem}{UPDATED_SUFFIX}{FITS_SUFFIX}"


class FilenameParseError(ValueError):
    """Raised when a preprocessing filename cannot be parsed."""


def parse_filename(
    path: str | Path,
    *,
    allowed_observatories: Iterable[str] | None = DEFAULT_OBSERVATORIES,
) -> FilenameInfo:
    """Parse a raw ``*.fits.fz`` or generated ``*_updated.fits`` filename.

    The parser searches the filename stem for ``YYYYMMDDHHMMSS<site><product>``
    rather than relying on the metadata living at a fixed absolute index.  If a
    future naming convention changes, this function should be the only place
    that needs to learn the new pattern.
    """

    path_obj = Path(path)
    name = path_obj.name
    stem, is_updated = _normalise_stem(name)

    match = _OBSERVATION_RE.search(stem)
    if match is None:
        raise FilenameParseError(
            f"Could not parse observation metadata from filename {name!r}. "
            "Expected a stem containing YYYYMMDDHHMMSS followed by an "
            "observatory letter."
        )

    date = match.group("date")
    time = match.group("time")
    observatory = match.group("observatory").upper()
    product = match.group("product")

    allowed = _normalise_allowed_observatories(allowed_observatories)
    if allowed is not None and observatory not in allowed:
        raise FilenameParseError(
            f"Unknown observatory {observatory!r} in filename {name!r}. "
            f"Expected one of {sorted(allowed)!r}."
        )

    try:
        observation_datetime = datetime.strptime(f"{date}{time}", "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise FilenameParseError(
            f"Invalid date/time in filename {name!r}: {date} {time}."
        ) from exc

    canonical_stem = match.group(0)
    return FilenameInfo(
        path=str(path_obj),
        name=name,
        stem=stem,
        canonical_stem=canonical_stem,
        date=date,
        time=time,
        observatory=observatory,
        product=product,
        is_updated=is_updated,
        observation_datetime=observation_datetime,
    )


def parse_filenames(paths: Iterable[str | Path]) -> list[FilenameInfo]:
    """Parse a collection of filenames and return the metadata objects."""

    return [parse_filename(path) for path in paths]


def observation_date(path: str | Path) -> str:
    """Return the observation date as ``YYYYMMDD``."""

    return parse_filename(path).date


def observation_time(path: str | Path) -> str:
    """Return the observation time as ``HHMMSS``."""

    return parse_filename(path).time


def observation_seconds(path: str | Path) -> int:
    """Return seconds since midnight for the observation timestamp."""

    return parse_filename(path).seconds_since_midnight


def observation_datetime(path: str | Path) -> datetime:
    """Return the observation timestamp as a ``datetime``."""

    return parse_filename(path).observation_datetime


def observatory(path: str | Path) -> str:
    """Return the single-letter observatory/site code."""

    return parse_filename(path).observatory


def canonical_stem(path: str | Path) -> str:
    """Return the canonical observation stem used by derived products."""

    return parse_filename(path).canonical_stem


def updated_filename(path: str | Path) -> str:
    """Return the generated ``*_updated.fits`` basename for a source file."""

    return parse_filename(path).updated_name


def updated_filepath(path: str | Path, output_dir: str | Path) -> str:
    """Return the generated ``*_updated.fits`` path under ``output_dir``."""

    return str(Path(output_dir) / updated_filename(path))


def same_observatory(first: str | Path, second: str | Path) -> bool:
    """Return whether two files belong to the same observatory/site."""

    return observatory(first) == observatory(second)


def observatory_change_indices(paths: Iterable[str | Path]) -> list[int]:
    """Return indices where consecutive files switch observatory/site."""

    infos = parse_filenames(paths)
    return [
        index
        for index, (previous, current) in enumerate(zip(infos, infos[1:]), start=1)
        if previous.observatory != current.observatory
    ]


def time_offsets_seconds(paths: Iterable[str | Path]) -> list[float]:
    """Return observation-time offsets from the first file in seconds."""

    infos = parse_filenames(paths)
    if not infos:
        return []

    reference = infos[0].observation_datetime
    return [
        (info.observation_datetime - reference).total_seconds()
        for info in infos
    ]


def sort_by_observation_time(paths: Iterable[str | Path]) -> list[str | Path]:
    """Sort paths by parsed observation timestamp, then observatory."""

    parsed_paths = [(parse_filename(path), path) for path in paths]
    parsed_paths.sort(
        key=lambda item: (
            item[0].observation_datetime,
            item[0].observatory,
            Path(item[1]).name,
        )
    )
    return [path for _, path in parsed_paths]


def formatted_observation_date(path: str | Path) -> str:
    """Return the observation date formatted as ``YYYY-MM-DD``."""

    info = parse_filename(path)
    return f"{info.year}-{info.month}-{info.day}"


def group_by_observatory(
    paths: Iterable[str | Path],
    observatory_order: Iterable[str] = ("L", "M", "U", "C", "T", "B"),
) -> dict[str, list[str | Path]]:
    """Group paths by observatory, preserving the requested observatory order."""

    grouped = {site.upper(): [] for site in observatory_order}
    for path in paths:
        site = observatory(path)
        grouped.setdefault(site, []).append(path)
    return grouped


def _normalise_stem(name: str) -> tuple[str, bool]:
    lower_name = name.lower()
    is_updated = False

    if lower_name.endswith(FITS_FZ_SUFFIX):
        stem = name[: -len(FITS_FZ_SUFFIX)]
    elif lower_name.endswith(FITS_SUFFIX):
        stem = name[: -len(FITS_SUFFIX)]
    else:
        stem = Path(name).stem

    if stem.lower().endswith(UPDATED_SUFFIX):
        stem = stem[: -len(UPDATED_SUFFIX)]
        is_updated = True

    return stem, is_updated


def _normalise_allowed_observatories(
    allowed_observatories: Iterable[str] | None,
) -> frozenset[str] | None:
    if allowed_observatories is None:
        return None
    return frozenset(site.upper() for site in allowed_observatories)


def filter_science_files(files):
    """Return only the science (GONG) files, skipping non-science ones in-memory.

    Unlike the previous ``move_non_science_files`` helper, this never touches the
    source directory on disk, so it is safe to run against read-only or
    remote-mounted raw-data folders.
    """

    science_files = []
    skipped = 0

    for file in files:
        info = parse_filename(file, allowed_observatories=None)
        if info.observatory in DEFAULT_OBSERVATORIES:
            science_files.append(file)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} non-science files")

    return science_files