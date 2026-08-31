"""Layout geometry for this variant, which builds one architecture only.

The builder registry that used to sit here mapped a style name onto one of two
builders. This variant has a single architecture, so the registry would be a
one entry dictionary whose only purpose is to let a caller ask for a style that
does not exist; the builder is exported directly instead.
"""

from .finfet import build_finfet_layout

__all__ = ["build_finfet_layout"]
