"""Command-line entry point.

    xp                                             launch the dashboard (GUI + embedded console)
    xp gui                                          same as above, explicitly
    xp convert <pdf> -o out.xopp [--profile hand|elec|path.json] [--pages 1-5]
        [--interactive [--save-profile p.json]]   page by page: preview, then accept/adjust/quit
    xp probe <pdf> [--pages 1-5]           what the page actually measures as
    xp repair <file.xopp> [-o out.xopp]     make a file Xournal++ refuses openable
    xp tune <pdf> [--page N] [--profile ...] -o tuned_profile.json
    xp calibrate <samples_dir> -o profile.json
    xp diff <a.xopp|.pdf> <b.xopp|.pdf> [--profile ...]
    xp merge <a.xopp|.pdf> <b.xopp|.pdf> [-o out.xopp]
        [--mode overlay|append|image] [--profile ...] [--pages 1-5]

`-o` is optional for `--mode overlay`, which opens a window: opening two
documents to look at them asks nothing, and the window asks where to save when
the user commits. `--mode append` and `--mode image` open no window, so they still
need it up front -- there is nothing to look at first, and so nothing to ask
about afterwards.

Everything is scriptable without a window except `overlay`-mode merge/diff,
which opens the tkinter brush-review GUI since that step inherently needs
drawing.

The dashboard (app.py) runs this exact same parser against whatever you
type into its embedded console, so CLI and GUI aren't two different tools --
the console just is the CLI, inline.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path
from typing import List, Optional

from xp.workers.converter.convert import convert_pdf
from xp.workers.merger.merge import MergeSession, append_merge
from xp.workers.converter.profiles import Profile, calibrate_from_samples
from xp.core.document.xopp_io import XoppDocument


def parse_pages(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    indices: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            indices.extend(range(int(a) - 1, int(b)))
        else:
            indices.append(int(part) - 1)
    return indices


def load_side(path: str, profile: Profile, pages: Optional[List[int]] = None) -> XoppDocument:
    """A side as a document of strokes, converting a PDF if it is one.

    Still here, and still what `append` and the page-map editor use: those
    work on strokes and have nothing to show you while they wait.

    The merge and diff windows do **not** use it any more. They take the
    file itself (see `open_side`), because a PDF can be looked at and
    brushed over without being converted, and converting it first is about
    twenty minutes spent before anything is on screen.
    """
    if str(path).lower().endswith(".pdf"):
        return convert_pdf(path, profile, pages=pages)
    return XoppDocument.load(path)


def open_side(path: str):
    """A side for the merge window: read and renderable, never converted."""
    from xp.workers.merger.sides import Side

    return Side(path)


def _profile_arg(name_or_path: str) -> Profile:
    return Profile.named(name_or_path)


def _auto_tune(pdf_path, profile, page_number):
    import pymupdf

    from xp.workers.converter.optimization.autotune import tune_page
    from xp.workers.converter.convert import (page_geometry, render_pdf_page,
                                               usable_dpi)
    from xp.workers.converter.probe import probe_page

    with pymupdf.open(pdf_path) as pdf:
        if not 1 <= page_number <= pdf.page_count:
            raise SystemExit(f"--auto-tune: choose a page from 1 to {pdf.page_count}")
        # Clamp before the geometry, which is given the same dpi: past the
        # pixel cap MuPDF returns a white page without raising.
        dpi, note = usable_dpi(pdf[page_number - 1], profile.dpi)
        if note:
            print(note)
        if dpi != profile.dpi:
            profile = replace(profile, dpi=dpi)
        raw = render_pdf_page(pdf[page_number - 1], dpi)
        geometry = page_geometry(pdf[page_number - 1], profile, raw)
    print(f"Auto-tuning on page {page_number}...")
    result = tune_page(raw, profile, probe_page(raw), geometry=geometry)
    tuned = result.profile
    print(f"Page {page_number}: {result.summary()}")
    print(f"Settings: threshold window {tuned.adaptive_block_size}, offset {tuned.adaptive_c:g}, "
          f"specks below {tuned.min_area:g}")
    print(f"Cleaned first: denoise {tuned.denoise:g}, sharpen {tuned.sharpen:g}, "
          f"open {tuned.morph_open:g}, close {tuned.morph_close:g}")
    if result.bridge is not None:
        print("Drawn with: " + ", ".join(f"{name} {value:g}"
                                        for name, value in result.bridge.settings.items()))
    return tuned


def cmd_convert(args) -> None:
    profile = _profile_arg(args.profile)
    pages = parse_pages(args.pages)

    if args.auto_tune is not None:
        profile = _auto_tune(args.pdf, profile, args.auto_tune)
        if args.save_profile and not args.interactive:
            profile.save(args.save_profile)
            print(f"Saved tuned profile to {args.save_profile}")

    if args.interactive:
        from xp.workers.converter.optimization.tune import interactive_convert

        tuned = interactive_convert(args.pdf, args.output, profile, pages=pages,
                                     auto=not args.no_auto)
        if args.save_profile:
            tuned.save(args.save_profile)
            print(f"Saved tuned profile to {args.save_profile}")
        return

    # **The background is decided before the notebook is written**, because the page element has
    # to carry the reference when it is saved. `place_the_background` keeps the Q base template in
    # the folder the user chose for it and answers that path. It is absolute because Xournal++ 1.2.5
    # refuses `domain="relative"` outright, and it is a fixed folder rather than wherever the
    # output happens to be, because a conversion is staged in the system temp directory before it
    # is named -- which is how the user's notebooks came to point at `%TEMP%`.
    from xp.workers.converter.convert import place_the_background

    background = None if getattr(args, 'no_background', False) else place_the_background(args.output)
    doc = convert_pdf(args.pdf, profile, pages=pages, auto=not args.no_auto,
                      background=background)
    doc.save(args.output)
    print(f"Wrote {args.output}")


def cmd_probe(args) -> None:
    import pymupdf

    from xp.workers.converter.convert import render_pdf_page, usable_dpi
    from xp.workers.converter.probe import probe_page

    profile = _profile_arg(args.profile)
    doc = pymupdf.open(args.pdf)
    indices = parse_pages(args.pages) or range(doc.page_count)
    for i in indices:
        dpi, note = usable_dpi(doc[i], profile.dpi)
        if note:
            print(note)
        img = render_pdf_page(doc[i], dpi)
        print(f"=== page {i + 1}/{doc.page_count} ===")
        print(probe_page(img).summary())
        print()


def cmd_calibrate(args) -> None:
    ranges = calibrate_from_samples(args.samples_dir)
    print(json.dumps(ranges, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(ranges, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


def cmd_tune(args) -> None:
    from xp.workers.converter.optimization.tune import interactive_tune

    base = _profile_arg(args.profile)
    tuned = interactive_tune(args.pdf, page_index=args.page - 1, base_profile=base)
    if args.output:
        tuned.save(args.output)
        print(f"Saved tuned profile to {args.output}")
    else:
        print("(no -o given, tuned profile not saved -- pass -o path.json to keep it)")


def _has_strokes(side) -> bool:
    from xp.workers.merger.sides import PDF, Side

    return not (isinstance(side, Side) and side.kind == PDF)


def build_alignment(left, right, mode: str, master=None):
    """Work out the left-page -> right-page mapping for a session.

    `pagealign` matches pages by their ink, which it reads from strokes. A
    PDF side has none until something converts it, and converting it to
    decide a page order is the twenty minutes this is avoiding. So a PDF
    side is paired by position, and says so rather than quietly doing
    something else.
    """
    if mode == "index":
        return None
    if not (_has_strokes(left) and _has_strokes(right)):
        print("Pairing pages by position: page matching reads strokes, and a "
              "PDF side has none until you ask for it to be converted.")
        return None

    from xp.workers.merger.matching.pagealign import align_pages

    left_doc = getattr(left, 'document', left)
    right_doc = getattr(right, 'document', right)
    alignment = align_pages(left_doc, right_doc)
    print(alignment.summary())

    if mode == "manual":
        from xp.workers.merger.gui.mapgui import edit_mapping

        alignment = edit_mapping(left_doc, right_doc, alignment, master=master)
        print(alignment.summary())

    return alignment


def cmd_diff(args) -> None:
    from xp.workers.merger.gui.window import run_review

    left = open_side(args.left)
    right = open_side(args.right)
    alignment = build_alignment(left, right, args.align, getattr(args, "master", None))
    session = MergeSession(left, right, alignment=alignment)
    run_review(session, read_only=True, master=getattr(args, "master", None))


def cmd_merge(args) -> None:
    if args.mode in ("image", "append") and not args.output:
        # Neither of these opens a window. There is nothing to look at
        # first, so there is nothing to decide at the end either: the name
        # has to come up front, and saying so is better than writing
        # somewhere it guessed.
        raise SystemExit(
            f"xp merge --mode {args.mode} needs -o/--output: it shows you "
            f"nothing first, so it cannot ask afterwards.")

    if args.mode == "image":
        cmd_merge_image(args)
        return

    if args.mode == "append":
        # Appending is not a merge window: it joins two documents of strokes,
        # so a PDF side has to be converted first and there is nothing to
        # look at while it is.
        profile = _profile_arg(args.profile)
        pages = parse_pages(args.pages)
        result = append_merge(load_side(args.left, profile, pages),
                              load_side(args.right, profile, pages))
        result.save(args.output)
        print(f"Wrote {args.output}")
        return

    from xp.workers.merger.gui.window import run_review

    left = open_side(args.left)
    right = open_side(args.right)
    alignment = build_alignment(left, right, args.align, getattr(args, "master", None))
    classify_dpi = getattr(args, 'classify_dpi', None)
    session = MergeSession(left, right, alignment=alignment,
                           **({} if classify_dpi is None
                              else {'classify_dpi': classify_dpi}))

    # The user named an output, or the user did not, and that is the only difference
    # between the two ways in. Named: the window has nothing to ask, so the
    # closure that writes is the whole of the save. Not named: the window is
    # a window -- the user may only be looking at the two documents -- so it opens
    # asking nothing and asks where to save when the user commits.
    if args.output:
        def on_commit():
            # The window does the writing, on a worker and with the panel up.
            # This only says so.
            print(f"Wrote {args.output}")
    else:
        on_commit = None

    run_review(session, read_only=False, on_commit=on_commit,
               output=args.output or None,
               master=getattr(args, "master", None))


def cmd_merge_image(args) -> None:
    """Compare first, convert nothing.

    The other modes go through `load_side`, which sends a `.pdf` through
    `convert_pdf` before anything is shown -- about twenty minutes for the user's
    notebook, spent before anybody has seen a single difference. This one
    never calls it: it rasterizes both sides, compares the pixels, and only
    then does anything. Rasterizing a PDF and converting a PDF are not the
    same operation, and the whole saving is in the difference.
    """
    from xp.workers.converter import incremental
    dpi = args.dpi or incremental.COMPARE_DPI
    with incremental.Sources(args.left, args.right, dpi=dpi) as sources:
        pages = len(sources.base)

        back = chr(13)   # written as a character: a literal carriage
                         # return in the source is a line break to the
                         # next tool that reads this file

        def say(number, total):
            print(back + f"Comparing page {number} of {total}...",
                  end="", flush=True)

        made = incremental.look(sources, progress=say)
        print(back + " " * 44 + back, end="")
        print(made.summary())
        for page in made.pages:
            if page.touched or page.conflicts:
                print(f"  page {page.number}: {page.note}")

        if args.dry_run:
            print("--dry-run: nothing written")
            return
        if not made.patches:
            print("Nothing to merge.")
            return

        wanted = parse_pages(args.pages)
        chosen = None if wanted is None else [n + 1 for n in wanted]
        done = incremental.fast_visual(sources, made, pages=chosen)
        sources.document.save(args.output)

    print(f"Merged {done['patches']} patch(es) into {done['pages']} page(s), "
          f"{done['bytes'] / 1024:.1f} KB of PNG")
    if done['conflicts']:
        # Never applied, and said out loud rather than buried: an image
        # cannot delete a stroke, and a merge that pretended otherwise would
        # leave the old ink live underneath a picture of its absence.
        print(f"{done['conflicts']} region(s) need a decision and were left "
              f"alone: a patch can add ink, it cannot remove or rewrite it.")
    print(f"Wrote {args.output}")


def cmd_gui(_args) -> None:
    from xp.runtime.app import App

    App().run()


def cmd_convert_gui(args) -> None:
    from xp.workers.converter.gui.convertgui import ConversionWindow
    ConversionWindow(args.pdf, mode=args.workflow, profile=Profile.named(args.profile)).run()


def cmd_open(args) -> None:
    from xp.runtime.app import App
    app = App()
    app.root.after(50, lambda: app.open_sources(args.files))
    app.run()


def cmd_arrange(args) -> None:
    """Hand the files to Arranger and return; XP opens no window for this.

    The page arranger is its own program in this tree. `workers/arranger/__init__.py` says why
    it is a subprocess and not an import.
    """
    from xp.workers.arranger.bridge import arranger_dir, launch_arranger
    launch_arranger(args.files)
    print('Arranger started from %s with %d file(s).'
          % (arranger_dir(), len(args.files)))


def cmd_repair(args) -> None:
    """Make a .xopp Xournal++ will open again.

    Xournal++ refuses a whole document over one stroke with fewer than two
    points, and files written by XP before that was fixed have them: a full
    stop was traced as a single point. This rewrites those as the dots they
    are and leaves everything else exactly as it was.
    """
    source = Path(args.file)
    document = XoppDocument.load(source)
    fixed = document.repair()
    target = Path(args.output) if args.output else source
    if not fixed and not args.output:
        print(f"{source.name}: nothing to repair")
        return
    document.save(target)
    print(f"Repaired {fixed:,} strokes; wrote {target}")


def cmd_pagemap(args) -> None:
    """The page map on its own: pair the pages, print the result, and keep it
    if asked. Merging is a separate decision."""
    from xp.workers.merger.gui.mapgui import edit_mapping

    profile = _profile_arg(args.profile)
    left = load_side(args.left, profile, None)
    right = load_side(args.right, profile, None)
    alignment = edit_mapping(left, right)
    print(alignment.summary())
    if args.output:
        Path(args.output).write_text(
            json.dumps({str(l): r for l, r in alignment.mapping.items()}, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


def cmd_testbench(args) -> None:
    from xp.runtime.testbench import run, write_doc

    if args.write_doc:
        print(f'Wrote {write_doc(samples=args.samples)}')
        return
    run(args.samples)


def cmd_shell(args) -> None:
    from xp.runtime.windows import install, uninstall
    print(install() if args.command == 'install-shell' else uninstall())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xp")
    sub = p.add_subparsers(dest="command")  # not required: no subcommand launches the dashboard

    po = sub.add_parser('open', help='Open selected PDF/xopp files in the conversion or merge window')
    po.add_argument('files', nargs='+')
    po.set_defaults(func=cmd_open)
    pa = sub.add_parser('arrange',
                        help='Reorder pages across notebooks in Arranger (a separate program)')
    pa.add_argument('files', nargs='+')
    pa.set_defaults(func=cmd_arrange)
    pgc = sub.add_parser('convert-gui', help='Editable automatic settings with live stage previews')
    pgc.add_argument('pdf')
    pgc.add_argument('--workflow', choices=['auto', 'live'], default='live')
    pgc.add_argument('--profile', default='hand')
    pgc.set_defaults(func=cmd_convert_gui)
    rp = sub.add_parser('repair', help='Fix a .xopp Xournal++ refuses to open')
    rp.add_argument('file')
    rp.add_argument('-o', '--output', help='write here instead of over the file')
    rp.set_defaults(func=cmd_repair)

    pm = sub.add_parser('pagemap', help='Pair two documents\' pages, without merging them')
    pm.add_argument('left')
    pm.add_argument('right')
    pm.add_argument('--profile', default='hand')
    pm.add_argument('-o', '--output', help='save the pairing as JSON')
    pm.set_defaults(func=cmd_pagemap)

    pb = sub.add_parser('testbench', help='Try every function by hand on real documents')
    pb.add_argument('--samples', help="folder holding the real documents (default: the owner's, found by xp.where)")
    pb.add_argument('--write-doc', action='store_true', help='write TESTING.md and exit')
    pb.set_defaults(func=cmd_testbench)
    for verb in ('install-shell', 'uninstall-shell'):
        ps = sub.add_parser(verb, help=('Install' if verb == 'install-shell' else 'Remove') + ' Windows Explorer integration for this user')
        ps.set_defaults(func=cmd_shell)

    pc = sub.add_parser("convert", help="Convert a PDF to a fresh .xopp")
    pc.add_argument("pdf")
    pc.add_argument("-o", "--output", required=True)
    pc.add_argument("--profile", default="hand", help="'hand', 'elec', or a path to a profile JSON")
    pc.add_argument("--pages", help="e.g. '1-5,8'")
    pc.add_argument("--interactive", action="store_true",
                    help="go page by page: preview each, then accept / adjust / quit (needs arpy's vp)")
    pc.add_argument("--auto-tune", nargs="?", type=int, const=1, metavar="PAGE",
                    help="first search the settings that trace PAGE (default 1) closest to the PDF "
                         "with the fewest strokes, and convert every page with them")
    pc.add_argument("--save-profile", help="with --interactive or --auto-tune, save the settings you end up with")
    pc.add_argument("--no-auto", action="store_true",
                    help="skip probing the page; assume a white background and use the profile's own thresholds")
    pc.add_argument("--no-background", action="store_true",
                    help="do not point the notebook's pages at the Q base template PDF; the blue "
                         "border then comes from nowhere, because it is not traced either")
    pc.set_defaults(func=cmd_convert)

    pp = sub.add_parser("probe", help="Report a page's measured paper color, ink contrast and stroke thickness")
    pp.add_argument("pdf")
    pp.add_argument("--pages", help="e.g. '1-5,8' (default: every page)")
    pp.add_argument("--profile", default="hand")
    pp.set_defaults(func=cmd_probe)

    pt = sub.add_parser("tune", help="Interactively tune convert's CV parameters on one page (needs arpy's vp)")
    pt.add_argument("pdf")
    pt.add_argument("--page", type=int, default=1, help="1-based page number to tune on (default: 1)")
    pt.add_argument("--profile", default="hand", help="starting point: 'hand', 'elec', or a path to a profile JSON")
    pt.add_argument("-o", "--output", help="save the tuned profile as JSON here")
    pt.set_defaults(func=cmd_tune)

    pcal = sub.add_parser("calibrate", help="Derive HSV ranges from sample ink-color crops (ColorProb.py, generalized)")
    pcal.add_argument("samples_dir", help="Directory with one subfolder of .png crops per ink color")
    pcal.add_argument("-o", "--output", help="Write the resulting ranges to this JSON path")
    pcal.set_defaults(func=cmd_calibrate)

    pd = sub.add_parser("diff", help="Open a read-only visual diff of two sources")
    pd.add_argument("left")
    pd.add_argument("right")
    pd.add_argument("--profile", default="hand")
    pd.add_argument("--pages")
    pd.add_argument("--align", choices=["auto", "index", "manual"], default="index",
                    help="'index' (the default) pairs page 1 with page 1 and so on, so opening is cheap; 'auto' matches pages by content (handles inserted, removed and reordered pages, and costs a signature pass over both documents); 'manual' opens the two-column mapping editor")
    pd.set_defaults(func=cmd_diff)

    pm = sub.add_parser("merge", help="Merge two sources (.xopp and/or .pdf)")
    pm.add_argument("left")
    pm.add_argument("right")
    pm.add_argument("-o", "--output",
                    help="where the merge goes. Not needed for the window "
                         "modes: opening two documents to look at them asks "
                         "nothing, and the window asks when the user is done. The "
                         "modes that open no window still need it up front.")
    pm.add_argument("--mode", choices=["overlay", "append", "image"], default="overlay",
                    help="'image' compares the two as pictures and patches in "
                         "only what is new -- it never converts the PDF to "
                         "strokes, so it costs what changed and not what the "
                         "document already contains")
    pm.add_argument("--dpi", type=int, default=None,
                    help="with --mode image: pixels per inch for the "
                         "comparison and the patches (default 150)")
    pm.add_argument("--classify-dpi", type=int, default=None,
                    help="resolution the merge window identifies the page at, "
                         "in pixels per inch (default 288 = 4 pixels per page "
                         "unit). The zoom only says how big the page looks; "
                         "this says how finely both sides' content is compared")
    pm.add_argument("--dry-run", action="store_true",
                    help="with --mode image: say what would change and write "
                         "nothing")
    pm.add_argument("--profile", default="hand")
    pm.add_argument("--pages")
    pm.add_argument("--align", choices=["auto", "index", "manual"], default="index",
                    help="'index' (the default) pairs page 1 with page 1 and so on, so opening is cheap; 'auto' matches pages by content (handles inserted, removed and reordered pages, and costs a signature pass over both documents); 'manual' opens the two-column mapping editor")
    pm.set_defaults(func=cmd_merge)

    pg = sub.add_parser("gui", help="Launch the dashboard (a window with an embedded console plus buttons for every command)")
    pg.set_defaults(func=cmd_gui)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        cmd_gui(args)
        return
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
