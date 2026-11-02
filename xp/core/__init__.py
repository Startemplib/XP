"""Shared domain: the document model, geometry, and image comparison.

Nothing here knows about a worker, a window or a card. If a module
in this package imports from `xp.workers`, the dependency is the
wrong way round.
"""
