"""The card: backend selection, kernels, and what runs on them.

`backend` decides whether there is a card at all and hands back cupy
or numpy accordingly, so everything above can be written once.
Every module here falls back to the processor rather than failing.
"""
