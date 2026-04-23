"""Small import surface for the `annotation` utilities.

Most of the real logic lives in `annotation.libero_demo_replay`. This package
file exists so callers can import a few shared dataclasses and helpers without
paying the import cost for LIBERO / TensorFlow until they are actually needed.

Typical usage:
    from annotation import load_rlds_episode, resolve_demo_replay_spec, render_demo
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DemoReplaySpec",
    "NextObjectHighlightPlan",
    "NextObjectHighlightRenderResult",
    "RenderResult",
    "ResolutionTrace",
    "RldsEpisode",
    "build_next_object_highlight_plan",
    "inspect_demo_replay_resolution",
    "load_rlds_episode",
    "match_demo_key",
    "render_demo",
    "render_next_object_highlighted_demo",
    "resolve_demo_replay_spec",
    "resolve_demo_replay_spec_from_episode",
    "trace_bddl_file_resolution",
    "trace_demo_hdf5_resolution",
    "write_frame_sequence",
    "write_rlds_episode_visualization",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Keep imports lazy so simple callers do not pull in heavier replay
    # dependencies until they actually ask for one of these symbols.
    module = import_module("annotation.libero_demo_replay")
    return getattr(module, name)
