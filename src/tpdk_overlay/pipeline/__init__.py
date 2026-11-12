"""Orchestration, event bus, session state, and persistence.

``orchestrator.py`` owns thread lifetimes and wires ports; ``events.py`` is an
async publish/subscribe bus feeding the WebSocket manager; ``session.py`` keeps
the transcript store, checkpoint windows, and on-disk artifacts.
"""
