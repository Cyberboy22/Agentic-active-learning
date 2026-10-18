"""Agentic active learning for expensive experiments.

A compact, domain-agnostic framework in which an LLM agent orchestrates
surrogate models, uncertainty estimates and acquisition functions to decide
which experiments to run next, with transparent, evidence-based rationales.

The included example oracle is a benign materials-engineering task
(concrete compressive strength). Swap `oracle.Oracle` for any expensive
evaluator (a simulation, an assay, a docking run) without touching the agent.
"""

__version__ = "0.1.0"
