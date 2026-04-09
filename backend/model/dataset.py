"""Compatibility layer for older imports.

Use model.data_loader for new code.
"""

from .data_loader import get_dataloaders

__all__ = ["get_dataloaders"]
