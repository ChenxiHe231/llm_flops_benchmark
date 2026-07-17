"""Locate a task's candidate implementation.

Convention: tasks/{operator}/{phase}/impl.py exposing `def run(inputs) -> out`.
If absent, fall back to the reference backend call (so latency/mfu measure the
real-backend BASELINE out of the box, and verify trivially passes at cosine≈1).
"""
import importlib.util
import os
from functools import partial

from . import specs

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HARNESS_DIR)
TASKS_DIR = os.path.join(_ROOT, "tasks")


def candidate_path(op: str, phase: str) -> str:
    return os.path.join(TASKS_DIR, op, phase, "impl.py")


def load_candidate(op: str, phase: str):
    """Return (run_callable, source_str). source_str is 'reference' or the path."""
    path = candidate_path(op, phase)
    if os.path.isfile(path):
        spec = importlib.util.spec_from_file_location(f"impl_{op}_{phase}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "run"):
            raise AttributeError(f"{path} has no run(inputs) function")
        return mod.run, path
    return partial(specs.reference, op, phase), "reference"
