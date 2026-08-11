"""Inspect-backed agent adapters."""

from inspect_similarity.agents.inspect_agent import (
    InspectCoTAgent,
    InspectIOAgent,
    create_inspect_agents,
)

__all__ = ["InspectCoTAgent", "InspectIOAgent", "create_inspect_agents"]
