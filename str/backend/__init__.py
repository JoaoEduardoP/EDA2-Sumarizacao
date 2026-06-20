"""backend package initializer for EDA2-Sumarizacao.

This file makes the `backend` directory a Python package so
imports like `backend.graph_summarizer` are resolvable by tools.
"""

__all__ = [
    "wiki_fetcher",
    "graph_summarizer",
    "preprocessor",
    "exporters",
    "main",
    "visualization",
]
