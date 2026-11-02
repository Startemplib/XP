"""The three things the program does, each with its own package.

  converter  a PDF becomes an editable notebook
  merger     two documents become one, by hand, with a brush
  arranger   pages are reordered across notebooks

A worker owns its own pipeline, its own GUI and its own vocabulary.
Shared capability belongs in core, render or gpu -- not here.
"""
