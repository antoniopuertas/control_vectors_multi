"""
Compatibility patches for repeng library.

Patches ControlModule to properly forward attributes to wrapped layers,
which is required for models like Qwen2 that access layer attributes
(e.g., attention_type) during the forward pass.
"""

import torch.nn as nn
from repeng.control import ControlModule

# Store the original nn.Module.__getattr__ to call as fallback
_original_module_getattr = nn.Module.__getattr__


def _control_module_getattr(self, name: str):
    """Forward unknown attribute lookups to the wrapped block."""
    # First, try the original nn.Module.__getattr__ for _modules, _parameters, etc.
    try:
        return _original_module_getattr(self, name)
    except AttributeError:
        pass

    # If not found, forward to the wrapped block
    modules = self.__dict__.get("_modules", {})
    block = modules.get("block")
    if block is not None and hasattr(block, name):
        return getattr(block, name)
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def patch_repeng_for_qwen():
    """
    Apply compatibility patches to repeng for Qwen2 models.

    This patches ControlModule.__getattr__ to forward unknown attributes
    to the underlying block, which fixes the 'attention_type' error.
    """
    if not hasattr(ControlModule, "_patched_for_qwen"):
        ControlModule.__getattr__ = _control_module_getattr
        ControlModule._patched_for_qwen = True


# Auto-apply patch on import
patch_repeng_for_qwen()
