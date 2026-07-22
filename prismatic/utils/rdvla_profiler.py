# prismatic/utils/rdvla_profiler.py

import time
from collections import defaultdict
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
    timing_enabled = False
    timing_cuda_sync = False
    _current_timing_record = None

    @classmethod
    def set_enabled(cls, enabled: bool):
        cls.enabled = bool(enabled)

    @classmethod
    def set_timing_enabled(cls, enabled: bool):
        cls.timing_enabled = bool(enabled)

    @classmethod
    def set_timing_cuda_sync(cls, enabled: bool):
        cls.timing_cuda_sync = bool(enabled)

    @classmethod
    def start_timing_record(cls, metadata=None):
        cls._current_timing_record = {
            "metadata": dict(metadata or {}),
            "timings_ms": defaultdict(float),
            "counts": defaultdict(int),
            "cuda_events": [],
        }
        return cls._current_timing_record

    @classmethod
    def finish_timing_record(cls):
        record = cls._current_timing_record
        cls._current_timing_record = None
        if record is None:
            return None

        cuda_events = record.pop("cuda_events", [])
        if cuda_events:
            torch.cuda.synchronize()
            for name, start_event, end_event in cuda_events:
                record["timings_ms"][name] += float(start_event.elapsed_time(end_event))
                record["counts"][name] += 1

        return {
            "metadata": record["metadata"],
            "timings_ms": dict(record["timings_ms"]),
            "counts": dict(record["counts"]),
        }

    @classmethod
    def _timing_active(cls):
        return cls.timing_enabled and cls._current_timing_record is not None


class _RDVLATimingRange:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.start_event = None
        self.end_event = None
        self.use_cuda_events = (
            torch is not None
            and torch.cuda.is_available()
            and not RDVLAProfiler.timing_cuda_sync
        )

    def __enter__(self):
        if not RDVLAProfiler._timing_active():
            return self

        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            if torch is not None and torch.cuda.is_available() and RDVLAProfiler.timing_cuda_sync:
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not RDVLAProfiler._timing_active():
            return False

        record = RDVLAProfiler._current_timing_record
        if self.use_cuda_events:
            self.end_event.record()
            record["cuda_events"].append((self.name, self.start_event, self.end_event))
        else:
            if torch is not None and torch.cuda.is_available() and RDVLAProfiler.timing_cuda_sync:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            record["timings_ms"][self.name] += elapsed_ms
            record["counts"][self.name] += 1
        return False


class _CombinedContext:
    def __init__(self, *contexts):
        self.contexts = contexts

    def __enter__(self):
        for context in self.contexts:
            context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        suppress = False
        for context in reversed(self.contexts):
            suppress = bool(context.__exit__(exc_type, exc, tb)) or suppress
        return suppress


def rdvla_range(name: str):
    contexts = []
    if RDVLAProfiler.enabled and record_function is not None:
        contexts.append(record_function(name))
    if RDVLAProfiler._timing_active():
        contexts.append(_RDVLATimingRange(name))

    if not contexts:
        return nullcontext()
    if len(contexts) == 1:
        return contexts[0]
    return _CombinedContext(*contexts)


def rdvla_timing_range(name: str):
    if RDVLAProfiler._timing_active():
        return _RDVLATimingRange(name)
    return nullcontext()
