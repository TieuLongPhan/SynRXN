"""Weak controls for qualified task evaluations; model training is separate."""

from .dummy import DummyPredictor, DummyProtocol, get_dummy_protocol, list_dummy_protocols

__all__ = ["DummyPredictor", "DummyProtocol", "get_dummy_protocol", "list_dummy_protocols"]
