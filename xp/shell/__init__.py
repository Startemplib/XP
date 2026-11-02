"""The Explorer-side half of XP: a menu built when it is opened.

Nothing in here is imported by the program itself. The shell loads
`contextmenu.py` **inside Explorer's own process** -- the CLSID registered by
`xp.runtime.windows` names `pythoncom312.dll` as an in-proc server, and that
DLL imports the class named by `PythonCOM`. That is why an out-of-proc server
is not here: the shell does not load one for a context-menu handler.
"""
