"""The dashboard: one window with an embedded console (type CLI commands,
see their output inline) plus toolbar buttons that open file pickers for
the same commands. Both paths run through the exact same argparse parser as
`python -m xp <command>` (see cli.py) -- the console typed commands
and the CLI are not two different code paths, just two ways of feeding the
same one.

No GPU/CUDA/exe packaging dependency here: this is plain Python + tkinter
(stdlib), launched with `python -m xp`. GPU use, if any, stays inside
convert.py/gpu.py and falls back to CPU automatically -- the dashboard
itself never touches it directly.
"""

from __future__ import annotations

from pathlib import Path
import queue
import shlex
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from xp.runtime.cli import build_alignment, build_parser, open_side, parse_pages
from xp.workers.merger.merge import MergeSession
from xp.workers.converter.profiles import Profile


class _LogWriter:
    """File-like object that appends to the log file.

    There is no console any more -- the user's order: "delete that stupid terminal ... if it has an error
    then just write in the log". `print()` from the converter, the calibration and the CLI lands in
    `%LOCALAPPDATA%\\XP\\xp.log`, which is outside the repository and survives a cleanup.
    """

    def write(self, s: str) -> None:
        if s:
            from xp.runtime import log

            log.write(s)

    def flush(self) -> None:
        pass


class App:
    # Defaults on the class, not only in `__init__`, because the lifecycle
    # methods below are reached from an `after` callback that a test (or a
    # future caller) may have set up on an instance built with `__new__` to
    # avoid a window. A counter that only exists after `__init__` turns that
    # into an AttributeError deep inside an event handler, where the symptom is
    # "the window never opened" rather than "this object was not initialised".
    _tasks = 0
    _failed = False
    _closing = False

    def __init__(self):
        self.root = tk.Tk()
        from xp.runtime.shutdown import arm
        arm(self.root)
        # **Which code is this?** The question after a click that showed the old
        # behaviour: "it didn't update". The title now carries the tree's version, so
        # that question is answerable at a glance instead of by restarting hopefully.
        from xp.runtime.version import title as versioned_title

        self.root.title(versioned_title("XP"))
        # **Maximised, like every other XP window.** The user's order (2026-10-27):
        # "by default the window of XP GUI should mostly be full screen, unless
        # those loading bars". The dashboard was 900x600 and then widened to fit
        # its buttons -- a window that sizes itself to its own console, which is
        # the opposite of what the user asked for. Minimised restore still works: this
        # asks for the `zoomed` state, not for a borderless fullscreen.
        from xp.runtime.shutdown import fill_the_screen

        fill_the_screen(self.root)
        from xp.render.appearance import decorate
        decorate(self.root)

        self._out_queue = None          # no console: output goes to the log file
        self._busy = False
        # How many windows this launcher has opened that are still doing
        # something, and whether any of them ended badly. The launcher closes
        # itself when the last one is done and nothing failed, and stays with
        # a note in the log when something did -- see `_task_finished`.
        self._tasks = 0
        self._failed = False
        self._closing = False

        self._build_toolbar()
        from xp.runtime import log

        log.stamp('dashboard opened')

    # ---------------- layout ----------------

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(bar, text="Convert PDF -> xopp", command=self._dialog_convert).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Probe", command=self._dialog_probe).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Diff", command=self._dialog_diff).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Merge", command=self._dialog_merge).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Merge (map pages)", command=self._dialog_merge_manual).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Page map", command=self._dialog_page_map).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Test bench", command=self._open_testbench).pack(side=tk.LEFT, padx=2)
        self.status = ttk.Label(bar, text="idle")
        self.status.pack(side=tk.RIGHT, padx=6)
        # **The row of seven is wider than the old 900 px window.** A screenshot
        # taken at the default size showed the sixth button cut off and the
        # seventh missing: a control nobody can press is a control that is not
        # there. The window is widened to what the buttons ask for, once, after
        # they are laid out.
        self.root.update_idletasks()
        needed = bar.winfo_reqwidth() + 24
        current = 900
        try:
            current = int(str(self.root.geometry()).split('+')[0].split('x')[0])
        except (ValueError, IndexError):
            pass
        if needed > current:
            self.root.geometry('%dx620' % needed)

    # ---------------- output ----------------

    def _print(self, text: str) -> None:
        """Everything that used to be written into the console goes to the log file."""
        from xp.runtime import log

        log.write(text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.status.configure(text="working..." if busy else "idle")

    # ---------------- the launcher's own lifecycle ----------------

    def _task_opened(self) -> None:
        """A window that is going to do something is now open.

        Opening is not finishing. A launcher that vanished when the merge
        window appeared would take the merge with it, because the merge window
        is a Toplevel **under this root** and every Toplevel dies with its
        root. So the count goes up here, and only the window itself says when
        it is done.
        """
        self._tasks += 1

    def the_tool_is_the_whole_window(self, tool_root) -> None:
        """Tie the hidden launcher's life to the tool window's.

        The click opened a converter, not a dashboard, so when the converter closes there is nothing
        left to keep the process alive for -- and a hidden Tk root that outlives its tool is a
        process the next click cannot reuse.
        """
        def gone(event=None):
            # **Only when the window itself goes.** A `<Destroy>` binding on a
            # Toplevel also fires for every *descendant*, because a widget's
            # bindtags include its nearest toplevel -- measured: destroying one
            # Label inside the tool window fired this handler, with
            # `event.widget` set to the Label. And this handler destroys the
            # launcher's **root**, which takes the whole Tk interpreter and
            # every window in the process with it. The window stays painted on
            # the desktop with nothing behind it, so every key and click after
            # that does nothing -- "the window is there and the hotkeys are
            # dead", with no traceback, because there is nobody left to raise.
            if event is not None:
                try:
                    if str(getattr(event, 'widget', '')) != str(tool_root):
                        return
                except Exception:
                    return
            try:
                self.root.destroy()
            except Exception:
                pass

        try:
            tool_root.bind('<Destroy>', gone, add='+')
        except Exception:
            pass

    def _task_finished(self, clean: bool = True) -> None:
        """One of those windows is done -- or has failed.

        Clean, and nothing left: the launcher closes and takes its windows with
        it, because a window sitting on the desktop with nothing to do is the
        thing the user complained about.

        Not clean: it stays, with its console, because the console is where the
        reason is. A *recoverable* warning is not this -- a page the tuner
        could not improve says so in the window and is not a failure, and a
        conversion that was cancelled on purpose is not one either.
        """
        self._tasks = max(0, self._tasks - 1)
        if not clean:
            self._failed = True
            self._print("A task ended badly. The launcher stays so the reason "
                        "above can be read and the command run again.\n")
            return
        if self._tasks or self._failed or self._busy:
            return
        self._close_launcher()

    def _close_launcher(self) -> None:
        """Close the launcher, and with it whatever it still has open."""
        if self._closing:
            return
        self._closing = True
        self._print("Everything finished.\n")
        # `after`, not now: this runs inside an event handler that may be
        # inside the window being closed, and a toplevel destroyed while Tk is
        # still dispatching an event inside it is a crashed interpreter rather
        # than a closed window.
        self.root.after(250, self.root.destroy)

    # ---------------- command dispatch ----------------

    def run_command_line(self, line: str) -> None:
        if line in ("help", "?", "-h", "--help"):
            self._print(build_parser().format_help() + "\n")
            return
        if self._busy:
            self._print("A command is already running -- wait for it to finish.\n")
            return

        try:
            argv = shlex.split(line)
        except ValueError as e:
            self._print(f"Could not parse that: {e}\n")
            return
        self.run_arguments(argv)

    def run_arguments(self, argv) -> None:
        """File pickers and Explorer pass argv directly, preserving Windows paths."""
        if self._busy:
            self._print("A command is already running -- wait for it to finish.\n")
            return
        try:
            args = build_parser().parse_args(argv)
        except SystemExit:
            return

        if args.command is None:
            return  # blank/whitespace-only after parsing, nothing to do
        if args.command == "gui":
            self._print("Already in the dashboard.\n")
            return
        if args.command == 'open':
            self.open_sources(args.files)
            return
        if args.command == 'arrange':
            self.open_sources(args.files, action='arrange')
            return
        if args.command == 'convert-gui' or (args.command == 'convert' and args.interactive):
            self.open_conversion(args.pdf, mode=getattr(args, 'workflow', 'live'), profile=args.profile)
            return

        # Only `overlay` opens a window. `diff` is read-only review and opens
        # the same one; `append` and `image` open nothing, so they are
        # ordinary headless commands and have to come through the other door.
        #
        # They did not. Every mode that was not `append` came through here,
        # so `--mode image` -- which never looks at a stroke and never opens a
        # window -- was handed to the brush-review window instead, and
        # quietly did a stroke merge. The chooser's "Image level" entry goes
        # through `run_arguments`, so that was the entry the user clicks, and the
        # image merge was unreachable from the dashboard by any route.
        opens_a_window = (
            args.command == "diff"
            or (args.command == "merge"
                and getattr(args, "mode", "overlay") == "overlay"))
        if opens_a_window:
            self._run_gui_command(args)
        else:
            self._run_headless_command(args)

    def _run_headless_command(self, args) -> None:
        self._set_busy(True)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _LogWriter()
            try:
                args.func(args)
            except SystemExit as exc:
                # A command that refuses its own arguments -- `merge --mode
                # image` with nowhere to write, say. That is an answer, not a
                # fault, so it is printed as one instead of arriving as a
                # traceback from a dead worker thread.
                if exc.code:
                    from xp.runtime import log

                    log.write('%s\n' % exc.code)
            except Exception as e:
                from xp.runtime import log

                log.write('ERROR: %r\n' % (e,))
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _run_gui_command(self, args) -> None:
        """diff / merge --mode overlay: opening happens off the main thread;
        the brush-review Toplevel is then built ON the main thread once it is
        open, since Tk widgets must be created on the thread running
        mainloop().

        **Opening, not converting.** This used to call `load_side`, which
        sends a `.pdf` through the whole stroke pipeline -- about twenty
        minutes for the user's notebook, spent before the window existed. The
        dashboard has its own copy of this path, so fixing `cli.cmd_merge`
        did not fix the thing the user actually clicks; that is the whole of why
        it still converted after the merge was supposed to stop doing it.
        """
        from xp.workers.merger.gui.window import MergeCue, run_review

        self._set_busy(True)
        # **`index` unless the user asked for matching.** Opening a merge used to run
        # the content match on the way in; on the sample pair that is 6.6 s of
        # signatures before the window can appear, and it is the one thing the user
        # asked not to happen by default. `--align auto` still does it, and the
        # page map's Auto Match button does it when the user wants it.
        align_mode = getattr(args, "align", "index")
        cue = MergeCue(self.root, 'Opening the merge')

        # **The worker never touches Tk.** It used to call `root.after` from its
        # own thread for every stage, and that can wedge: the main thread is
        # inside `update()` and the worker blocks on the interpreter's Tk
        # marshalling, so the merge never opens at all. It posts progress to a
        # queue instead and the main thread drains it.
        messages = queue.Queue()

        def stage(done, detail):
            messages.put(('step', (done, detail)))

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _LogWriter()
            try:
                stage(1, 'opening %s' % Path(args.left).name)
                left = open_side(args.left)
                stage(2, 'opening %s' % Path(args.right).name)
                right = open_side(args.right)
                # Matching pages by content is plain computation, so it
                # belongs here off the main thread; only the manual editor
                # (a window) has to wait for open_gui below.
                auto_alignment = None
                if align_mode == "auto":
                    stage(3, 'matching the pages')
                    auto_alignment = build_alignment(left, right, align_mode)
            except Exception as e:
                self._out_queue.put(f"ERROR: {e!r}\n")
                sys.stdout, sys.stderr = old_out, old_err
                messages.put(('error', e))
                return
            sys.stdout, sys.stderr = old_out, old_err
            messages.put(('open', (left, right, auto_alignment)))

        def open_gui(left, right, auto_alignment):
            if True:
                self._set_busy(False)
                stage(3, 'building the merge')
                alignment = auto_alignment
                if align_mode == "manual":
                    from xp.workers.merger.gui.mapgui import edit_mapping

                    alignment = edit_mapping(getattr(left, 'document', left),
                                             getattr(right, 'document', right),
                                             auto_alignment, master=self.root)
                    self._print(alignment.summary() + "\n")

                session = MergeSession(left, right, alignment=alignment)
                self._task_opened()
                if args.command == "diff":
                    run_review(session, read_only=True, master=self.root,
                               on_finish=self._task_finished, cue=cue)
                else:
                    # The same rule as `cli.cmd_merge`, and it has to be
                    # repeated here because this is the copy the user actually
                    # clicks -- the Explorer menu and the dashboard both come
                    # through this door, not through the command line. The user
                    # named an output or the user did not: named, and the window
                    # has nothing to ask; not named, and the window asks when
                    # the user commits rather than before it opens.
                    if args.output:
                        def on_commit():
                            # The merge window writes it, on a worker
                            # with its progress panel up; this is the report.
                            self._print(f"Wrote {args.output}\n")
                    else:
                        on_commit = None

                    # Swapping is a launcher job, not a window job: the session
                    # holds a parsed left and a parsed right and the merged
                    # page's size is the left's, so "the other way round" is a
                    # new session. This is what the old Choose-sides window's
                    # Swap did, moved to where the work is.
                    def on_swap():
                        self._print("Swapping sides...\n")
                        self.run_arguments(['merge', args.right, args.left,
                                            '--align', align_mode])

                    run_review(session, read_only=False, on_commit=on_commit,
                               output=getattr(args, 'output', None) or None,
                               on_swap=on_swap, on_finish=self._task_finished,
                               master=self.root, cue=cue)

        done = {'yes': False}

        def pump():
            if done['yes']:
                return
            try:
                while True:
                    kind, payload = messages.get_nowait()
                    if kind == 'step':
                        cue.step(payload[0], 3, payload[1])
                        continue
                    done['yes'] = True
                    if kind == 'error':
                        error = payload
                        cue.failed('The merge did not open',
                                   '%s: %s' % (type(error).__name__, error))
                        self._set_busy(False)
                        return
                    left, right, auto_alignment = payload
                    open_gui(left, right, auto_alignment)
                    return
            except queue.Empty:
                pass
            self.root.after(100, pump)

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(0, pump)

    # ---------------- toolbar dialogs (build a command line, then run it) ----------------

    def _dialog_convert(self) -> None:
        pdf = filedialog.askopenfilename(title="Choose a PDF", filetypes=[("PDF", "*.pdf")])
        if not pdf:
            return
        self.open_conversion(pdf)

    def open_conversion(self, pdf, mode='live', profile='hand'):
        from xp.workers.converter.gui.convertgui import ConversionWindow
        try:
            window = ConversionWindow(pdf, master=self.root, mode=mode,
                                      profile=Profile.named(profile),
                                      on_finish=self._task_finished)
        except Exception as exc:
            messagebox.showerror('Could not open PDF', str(exc), parent=self.root)
            return None
        # Counted only once it exists: a PDF that would not open is not a task
        # that is running, and counting it would leave the launcher waiting for
        # a window that was never made.
        self._task_opened()
        if not self.root.winfo_ismapped():
            # The click opened a converter, not a dashboard: this window is the whole application.
            self.the_tool_is_the_whole_window(window.root)
        return window

    def open_sources(self, paths, action='open'):
        """Open the tool the menu asked for, and **only that window**.

        When the launcher has already put its dashboard away (`_came_from_a_click`), a tool that opens
        here is the whole application: the hidden launcher goes when the tool does.
        """
        from xp.workers.merger.gui.opengui import MergeFilesWindow, normalized_sources
        # **`live` means the same as `auto` here.** The launcher's own argument list accepts `live`
        # (it is a workflow name), while the menu's table offers `auto` for a PDF -- so a `live` that
        # arrived from anywhere else was refused by `check_action` with "That selection cannot be
        # converted", which is a trap rather than a rule: both open the Converter.
        if action == 'live':
            action = 'auto'
        try:
            if action in ('auto', 'live', 'merge', 'arrange'):
                # One rule for the menu and for the door behind it.
                # `check_action` is the same table that draws the menu -- a pdf
                # cannot be arranged, a notebook cannot be converted, and a
                # selection of both can only be merged -- so a verb arriving
                # another way is refused with the rule rather than with a
                # second opinion. Raised inside this `try`, because the
                # Explorer path turns it into one box rather than a traceback.
                from xp.runtime.windows import check_action
                check_action(action, paths)
            if action == 'arrange':
                # **Before `normalized_sources`, and that is the point.** That
                # function refuses more than two paths, because one or two is
                # what converting and merging mean. Arranging is pages out of
                # *any number* of notebooks, which is the whole reason it is a
                # separate program, so it gets its own, looser check: existing
                # .xopp files, or folders, which Arranger reads for the
                # notebooks directly inside them.
                #
                # A separate program, started and let go of: it has its own
                # window and its own Tk loop, and XP neither waits for it nor
                # learns what it wrote. See `xp/arrange.py`.
                from xp.workers.arranger.bridge import launch_arranger
                wanted = []
                for value in paths:
                    path = Path(value).expanduser().resolve()
                    if path.is_dir() or (path.is_file()
                                         and path.suffix.lower() == '.xopp'):
                        if path not in wanted:
                            wanted.append(path)
                    else:
                        raise ValueError(
                            'Arrange takes .xopp notebooks or folders of '
                            'them:\n%s' % path)
                if not wanted:
                    return
                launch_arranger(wanted)
                self._print('Arranger opened with %d file(s).\n' % len(wanted))
                return
            sources = normalized_sources(paths)
            if not sources:
                return
            if action in ('auto', 'live'):
                if len(sources) != 1 or sources[0].suffix.lower() != '.pdf':
                    raise ValueError('Choose one PDF for conversion.')
                # **The menu's "Converter" does not start a search.** Its verb is
                # named `auto` for history, but the user's rules are that the converter
                # opens on `Original`, in live stages, and that Auto-tune never
                # starts by itself. Opening from Explorer used to run the whole
                # parameter search on the way in: the combo said `Auto-tune`, the
                # stage was whatever it had reached, and the picture under it was
                # a guess. `--workflow auto` is still there for anyone who asks
                # for the search in words.
                return self.open_conversion(sources[0], mode='live')
            if len(sources) == 1 and sources[0].suffix.lower() == '.pdf' and action not in ('merge', 'diff'):
                return self.open_conversion(sources[0])
            if len(sources) >= 2:
                # **The door the user actually comes through.** The Explorer verb and
                # `xp open` both arrive here, not at any toolbar button, and
                # this line used to open `MergeFilesWindow` whenever it had two
                # files -- a form asking for the two files it had just been
                # handed. That is the "Choose sides" window the user kept seeing
                # after it had been removed from the dashboard, because the
                # dashboard's button was a different function and only that one
                # had been fixed.
                #
                # Two sources is an answer, not a question: run the merge (or
                # the diff) with them. The chooser is now only for the case it
                # is actually needed -- one file, and no idea what to do with
                # it -- and the settings that used to live in it are in the
                # merge window: Swap is a button there, the page map is `m`, and
                # the merge mode is `--mode` on the console for the modes that
                # open no window.
                verb = 'diff' if action == 'diff' else 'merge'
                self.run_arguments([verb, str(sources[0]), str(sources[1])])
                return
            return MergeFilesWindow(self, sources, mode='diff' if action == 'diff' else 'merge')
        except Exception as exc:
            # **The log, not a box.** The user's order: "if it has an error then just write in the log. I
            # don't need that." A dialog here also blocks the process: it is parented to a window the user
            # never sees when the click came from Explorer, so the click looked like "nothing
            # happened" and the dialog sat behind everything.
            from xp.runtime import log

            log.stamp('could not open %r: %r' % (list(paths), exc))
            self.status.configure(text='could not open the selection -- see the log')
            if not self.root.winfo_ismapped():
                # The click opened a tool that then refused the file: nothing is showing, so this
                # process must not sit here invisible (the user's click would appear to have done nothing,
                # and a second click would find a launcher it cannot use).
                self.root.after(200, self.root.destroy)

    def _dialog_page_map(self) -> None:
        """Pair up two documents' pages without merging them."""
        left = filedialog.askopenfilename(title="Left document", filetypes=[("xopp/pdf", "*.xopp *.pdf")])
        if not left:
            return
        right = filedialog.askopenfilename(title="Right document", filetypes=[("xopp/pdf", "*.xopp *.pdf")])
        if not right:
            return
        self.run_arguments(['pagemap', left, right])

    def _open_testbench(self) -> None:
        """Every function of XP, on real documents, with somewhere to record
        what you thought of each one (see testbench.py)."""
        from xp.runtime.testbench import TestBench
        TestBench(self)

    def _dialog_probe(self) -> None:
        pdf = filedialog.askopenfilename(title="Choose a PDF to measure", filetypes=[("PDF", "*.pdf")])
        if not pdf:
            return
        self.run_command_line(f'probe "{pdf}" --pages 1')

    def _dialog_diff(self) -> None:
        left = filedialog.askopenfilename(title="Left side: choose a .xopp or .pdf",
                                           filetypes=[("xopp/pdf", "*.xopp *.pdf")])
        if not left:
            return
        right = filedialog.askopenfilename(title="Right side: choose a .xopp or .pdf",
                                            filetypes=[("xopp/pdf", "*.xopp *.pdf")])
        if not right:
            return
        self.run_command_line(f'diff "{left}" "{right}"')

    def _dialog_merge(self, align: str = "index") -> None:
        """Two sides, then straight into the merge. No window in between.

        This used to open `MergeFilesWindow` with **both fields empty**, so the
        first thing a Merge was, was a form asking which two files it was for.
        The two files are what the button's two dialogs are for, and the
        window that does the work holds the rest of the settings -- so there is
        one window for choosing and working rather than two.

        What the old form had, and where it is now:

            the two paths      these two dialogs
            Browse             the dialogs are the browse
            Swap sides         a button in the merge window itself (`on_swap`)
            page pairing       `m`, and the *Page map...* button, as before
            the merge mode     `--mode append|image` on the console, because
                               neither opens this window; nothing is
                               pre-selected on the way in
            the alignment      `--align` here, and `manual` still stops at the
                               page map before the window opens
        """
        left = filedialog.askopenfilename(title="Left side: choose a .xopp or .pdf",
                                          filetypes=[("xopp/pdf", "*.xopp *.pdf")])
        if not left:
            return
        right = filedialog.askopenfilename(title="Right side: choose a .xopp or .pdf",
                                           filetypes=[("xopp/pdf", "*.xopp *.pdf")])
        if not right:
            return
        self.run_command_line(f'merge "{left}" "{right}" --align {align}')

    def _dialog_merge_manual(self) -> None:
        """Same as Merge, but stop to let you fix the page mapping first."""
        self._dialog_merge(align="manual")

    def _ask_profile(self) -> str:
        # Toolbar flow defaults to 'hand'; type a full command in the
        # console instead (e.g. `convert x.pdf -o y.xopp --profile elec`)
        # for a typed/printed-source PDF.
        return "hand"

    # **No window, no process.** The user's rule, 2026-11-01: *"the design of this
    # system, this XP, should be no back-end shit. When I close it, when I
    # terminate it, it is dead. That is all. When the GUI is done, it dies!
    # If you really want it to exist, at least add a tray."*
    #
    # The launcher breaks that by construction. It withdraws its own dashboard
    # so a click does not produce two windows (`xp_launcher.py`, and the
    # reason is a report of the user's), and then sits in `mainloop` -- a process
    # with no window at all. Measured on the user's machine: `pythonw` 7880, started
    # 07:14, no main window, still alive an hour and a half later, with a
    # second `pythonw` under it. Nothing was watching for the condition the
    # comment claimed -- "the launcher ends with the tool" -- so nothing made
    # it true.
    #
    # This watches for it. Every few seconds it asks whether any window of
    # this process is actually on the screen; when none has been for two
    # checks running, the process ends. Two checks and not one because a tool
    # window is withdrawn for a moment while it is built, and a launcher that
    # quit in that gap would take the tool with it.
    IDLE_CHECK_MS = 4000

    def _any_window_on_screen(self) -> bool:
        """Whether this process has a window a person could point at."""
        import tkinter as tk

        for window in (self.root, *self.root.winfo_children()):
            try:
                if not isinstance(window, (tk.Tk, tk.Toplevel)):
                    continue
                if window.winfo_exists() and window.winfo_ismapped():
                    return True
            except Exception:
                continue
        return False

    def _die_when_nothing_is_shown(self) -> None:
        if self._any_window_on_screen():
            self._unseen = 0
        else:
            self._unseen = getattr(self, '_unseen', 0) + 1
            if self._unseen >= 2:
                # Said, because a process that vanishes without a word is the
                # other half of the same complaint.
                print('[xp] no window left; the launcher is exiting rather '
                      'than waiting in the background', flush=True)
                try:
                    self.root.destroy()
                except Exception:
                    pass
                return
        try:
            self.root.after(self.IDLE_CHECK_MS, self._die_when_nothing_is_shown)
        except Exception:
            pass

    def run(self) -> None:
        self._unseen = 0
        # Not from the first instant: the window this launcher was started to
        # open has not been built yet.
        self.root.after(self.IDLE_CHECK_MS * 2, self._die_when_nothing_is_shown)
        self.root.mainloop()


def main() -> None:
    # The same call as the launcher's: `python -m xp` must lay itself out the
    # same way a shortcut does.
    try:
        from xp.runtime.dpi import be_dpi_aware

        be_dpi_aware()
    except Exception:
        pass
    App().run()
