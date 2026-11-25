"""SynRXN's public loading API, imported only when requested.

Reconstruction planning and manifest inspection use the standard library and
should not import the optional dataframe/chemistry stack merely to show help.
"""

from importlib import import_module

from ._compat import MODULE_ALIASES, install as _install_module_aliases

_install_module_aliases()

__version__ = "1.5.0"

__all__ = ["DataLoader", "DatasetCatalog", "DatasetMetadata", "Task", "__version__"]


def __getattr__(name):
    alias = MODULE_ALIASES.get(__name__ + '.' + name)
    if alias is not None:
        value = import_module(alias)
        globals()[name] = value
        return value
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".data", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
