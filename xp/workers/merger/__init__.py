"""Two documents, a brush, and one document out.

`sides` holds what a source is -- and a PDF side keeps its ink as
ink: `Side.strokes` returns None for a PDF on purpose, because
turning it into strokes is postponed until the user has said which
parts the user wants. That conversion happens at export, through
`xp.workers.converter`, which is this package's one dependency
on another worker.
"""
