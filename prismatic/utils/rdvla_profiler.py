# prismatic/utils/rdvla_profiler.py

from contextlib import nullcontext

try:
    import torch
    from torch.profiler import record_function
except Exception:
    torch = None
    record_function = None


class RDVLAProfiler:
    """
    Lightweight wrapper around torch.profiler.record_function.

    Usage:
        with rdvla_range("RDVLA/vlm/language_model_forward"):
            ...
    """

    enabled = False

    @classmethod
    def set_enabled(cls, enabled: bool):
        cls.enabled = bool(enabled)


def rdvla_range(name: str):
    if RDVLAProfiler.enabled and record_function is not None:
        return record_function(name)
    return nullcontext()
