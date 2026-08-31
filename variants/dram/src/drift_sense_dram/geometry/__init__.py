"""Layout builders.

This variant builds DRAM and nothing else, so there is one builder and no
style dispatch. A registry keyed by architecture would advertise a choice the
tool cannot make.
"""

from .dram import build_dram_layout

__all__ = ["build_dram_layout"]
