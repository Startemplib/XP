"""The .xopp document model and its serialisation.

`xopp_io` is the one implementation of reading and writing a
notebook; Converter and Merger both use it rather than each
keeping their own. `bigxopp` is the streaming reader for the
files too large to hold as one string.
"""
