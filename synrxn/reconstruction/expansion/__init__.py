"""Typed, source-preserving companion to the frozen SynRXN reaction tables.

Expansion observations are not interchangeable with independent reactions.
The catalog records the counting unit, evidence, parents and license partition
for each artifact. Optional native dependencies are imported only on use.
"""

from .catalog import ExpansionCatalog, group_split

__all__ = ["ExpansionCatalog", "group_split"]
