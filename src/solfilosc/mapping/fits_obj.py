"""Placeholder for the original FITS object reference.

The previous file contained an incomplete IDL-style FITS object sketch, not
valid executable Python. It is not used by the preprocessing or analysis
pipeline. The module is kept so imports fail explicitly with a useful message
instead of a syntax error.
"""


class FitsObjectNotImplementedError(NotImplementedError):
    """Raised when the legacy FITS object placeholder is used."""


class fits:
    """Non-functional placeholder for the legacy IDL FITS object."""

    def __init__(self, *args, **kwargs):
        raise FitsObjectNotImplementedError(
            "solfilosc.mapping.fits_obj.fits is a legacy placeholder and is "
            "not implemented. Use solfilosc.mapping.fits2map.fits2map instead."
        )
