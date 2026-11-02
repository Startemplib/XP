"""The XP submenu, built from the whole selection every time it is opened.

A static submenu can only speak per file type: it cannot know that a PDF and a
notebook are selected together, because it is written into the registry before
anything is selected. Explorer therefore has to ask something at the moment the
menu opens, and that something is this handler:

    IShellExtInit::Initialize    the selection arrives as a data object --
                                 every selected file, not just the one under
                                 the pointer
    IContextMenu::QueryContextMenu
                                 build the XP submenu from `verbs_for(paths)`,
                                 which is the same table the launcher refuses
                                 with
    IContextMenu::InvokeCommand  start XP **once**, with every path

It runs in-proc -- inside Explorer -- so every method answers instead of
raising: an exception here is the shell's context menu failing, not a traceback
in a console.

The parsing of the selection is a plain function (`paths_from_hdrop`) so the
part that can be wrong can be tested without a shell.
"""
from __future__ import annotations

import subprocess
import struct
import sys
from pathlib import Path

import pythoncom
import win32con
import win32gui
from win32com.shell import shell as _shell

# Fixed, and written down in one place: the installer, the uninstaller and a
# test all have to name the same component. Never change it without removing
# the old registration first -- an orphaned CLSID is a menu entry that cannot
# be removed by name.
CLSID = '{6F2A8B7C-3D4E-4F5A-9B6C-7D8E9F0A1B2C}'
POPUP_LABEL = 'XP'


def set_menu_bitmap(hmenu, position: int, bitmap) -> bool:
    """Attach `bitmap` to the item at `position`, and say whether it stuck.

    **`SetMenuItemBitmaps` returned success and nothing was attached**: reading
    the item back (`GetMenuItemInfo`, MIIM_BITMAP) showed a null bitmap on the
    "XP" entry, which is why the user's menu kept the generic document icon however
    often the handler was restarted. `SetMenuItemInfoW` with `MIIM_BITMAP` is the
    call that both attaches it and can be verified; this asks the menu afterwards
    instead of trusting a return code that lies.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class MENUITEMINFO(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.UINT), ('fMask', wintypes.UINT),
                ('fType', wintypes.UINT), ('fState', wintypes.UINT),
                ('wID', wintypes.UINT), ('hSubMenu', wintypes.HANDLE),
                ('hbmpChecked', wintypes.HANDLE),
                ('hbmpUnchecked', wintypes.HANDLE),
                ('dwItemData', wintypes.LPVOID),
                ('dwTypeData', wintypes.LPWSTR), ('cch', wintypes.UINT),
                ('hbmpItem', wintypes.HANDLE),
            ]

        MIIM_BITMAP = 0x80
        info = MENUITEMINFO()
        info.cbSize = ctypes.sizeof(MENUITEMINFO)
        info.fMask = MIIM_BITMAP
        info.hbmpItem = int(bitmap)
        if not ctypes.windll.user32.SetMenuItemInfoW(int(hmenu), int(position),
                                                     True, ctypes.byref(info)):
            return False
        check = MENUITEMINFO()
        check.cbSize = ctypes.sizeof(MENUITEMINFO)
        check.fMask = MIIM_BITMAP
        if not ctypes.windll.user32.GetMenuItemInfoW(int(hmenu), int(position),
                                                     True, ctypes.byref(check)):
            return False
        return bool(check.hbmpItem)
    except Exception as exc:
        _note('set_menu_bitmap failed: %r' % (exc,))
        return False


def popup_bitmap(keepalive=None):
    """A **bitmap** handle for XP's mark, or None.

    **The first version of this returned an `HICON`, and Explorer drew a sheet
    of paper.** The user sent the picture: "Squarize" and "Open with Code" carry their
    own marks, and XP carried the generic document icon. `SetMenuItemBitmaps`
    takes an HBITMAP, not an HICON, so the call was refused (it is logged) and
    the shell fell back to the default. The icon is loaded, then
    `GetIconInfo` is asked for the colour bitmap inside it, which is what the
    menu wants.

    `keepalive` (a list, if given) receives the HICON and the HBITMAP: Windows
    does not copy them, and a handle that gets garbage-collected mid-menu is a
    menu that draws nothing.

    Never fatal: a menu without its picture is a menu.
    """
    try:
        from xp.render.appearance import icon_path

        ico = str(icon_path())
        icon = win32gui.LoadImage(0, ico, win32con.IMAGE_ICON, 16, 16,
                                  win32con.LR_LOADFROMFILE)
        if not icon:
            return None
        info = win32gui.GetIconInfo(icon)
        # (fIcon, xHotspot, yHotspot, hbmMask, hbmColor)
        bitmap = info[4] or info[3]
        if keepalive is not None:
            keepalive.extend([icon, bitmap])
        return bitmap or None
    except Exception as exc:                # a menu is more important than a mark
        _note('popup_bitmap failed: %r' % (exc,))
        return None
CF_HDROP = 15

# Said once, when the module is imported -- i.e. when the shell loads this code
# into its own process. Every later line in the log is a click; this one says
# *which* file they are running against, because an in-proc module stays in the
# host process's `sys.modules` until Explorer restarts: a fixed file on disk is
# not a fixed handler in memory, and that is exactly what made "it still does
# nothing" look like the fix had failed.
try:
    import os as _os
    _mt = _os.path.getmtime(__file__)
    import time as _time
    _when = _time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(_mt))
    with open(_os.path.join(_os.environ.get('LOCALAPPDATA', ''), 'XP', 'menu-server.log'), 'a', encoding='utf-8') as _h:
        _h.write('--- module loaded: %s (written %s)\n' % (__file__, _when))
except Exception:
    pass


def _note(text) -> None:
    """One line in the server's log. Never raises: a diagnostic that can break
    the thing it is diagnosing is worse than no diagnostic."""
    try:
        import os
        log = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'XP' / 'menu-server.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as handle:
            handle.write(text + '\n')
    except Exception:
        pass


def paths_from_hdrop(raw) -> list[str]:
    """The file names inside a CF_HDROP blob.

    The structure is `DROPFILES`: a DWORD offset to the file list, two points,
    then `fNC` and `fWide`. The names follow, NUL-separated, and are UTF-16
    when `fWide` is set -- which it is for every current Windows. A parser that
    assumed wide would read half the paths wrong on an old shell, and one that
    assumed narrow would read every path as garbage here.
    """
    if not raw or len(raw) < 20:
        return []
    offset = struct.unpack_from('<I', raw, 0)[0]
    wide = struct.unpack_from('<i', raw, 16)[0]
    if offset <= 0 or offset >= len(raw):
        return []
    body = bytes(raw[offset:])
    if wide:
        text = body.decode('utf-16-le', 'replace')
    else:
        text = body.decode('mbcs', 'replace')
    return [part for part in text.split('\x00') if part.strip()]


def _paths_of_array(array) -> list[str]:
    """The filesystem paths of an `IShellItemArray`, or [] if it is not one."""
    from win32com.shell import shellcon

    try:
        count = array.GetCount()
    except Exception:
        return []
    found = []
    for index in range(count):
        try:
            found.append(array.GetItemAt(index).GetDisplayName(
                shellcon.SIGDN_FILESYSPATH))
        except Exception:
            continue
    return found


def paths_from_data_object(data_object) -> list[str]:
    """Everything the selection holds, from the object Explorer hands over.

    **Three shapes, and the array is not always wrapped in a data object.** The
    shell calls `Initialize` with either an `IShellItemArray` or a `DataObject`;
    this used to assume the second and convert it, so when it was handed an array
    (as the shell does for a single selection in Explorer 11, and as the probe
    below does) `SHCreateShellItemArrayFromDataObject` raised `TypeError: The
    Python instance can not be converted to a COM object`, the CF_HDROP fallback
    found nothing, and the handler offered no menu at all. Asked directly first,
    then converted, then the drop list, which is what callers that really send one
    mean.
    """
    if isinstance(data_object, (list, tuple)):
        return [str(one) for one in data_object]
    if isinstance(data_object, str):
        return [data_object]
    direct = _paths_of_array(data_object)
    if direct:
        return direct
    try:
        from win32com.shell import shell, shellcon
        import pythoncom

        raw = shell.SHCreateShellItemArrayFromDataObject(
            data_object, pythoncom.IID_IUnknown)
        converted = _paths_of_array(raw.QueryInterface(shell.IID_IShellItemArray))
        if converted:
            return converted
    except Exception as exc:
        _note('shell item array failed: %r' % (exc,))
    try:
        raw = data_object.GetData(CF_HDROP)
    except Exception:
        return []
    return paths_from_hdrop(raw)


def _a_windowed_python() -> Path:
    """The `pythonw.exe` to launch, or a path that does not exist.

    **`sys.prefix` is not reliable here.** In-proc this module runs inside Explorer, where
    `sys.prefix` is whichever interpreter the COM server belongs to -- and the old code fell back to
    `python.exe` beside it when `pythonw` was absent, which is a **console** build. That fallback is
    exactly the "stupid terminal" that appeared from the right-click menu: the menu worked, and it
    opened a black window for as long as the child lived.

    The order is: this interpreter, its base prefix, `%LOCALAPPDATA%/Programs/Python/*`, and the
    interpreter the installer recorded beside the COM registration.
    """
    candidates = [Path(sys.prefix) / 'pythonw.exe',
                  Path(getattr(sys, 'base_prefix', sys.prefix)) / 'pythonw.exe']
    try:
        local = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python'
        if local.is_dir():
            for folder in sorted(local.glob('Python3*'), reverse=True):
                candidates.append(folder / 'pythonw.exe')
    except Exception:
        pass
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Classes\CLSID\{6F2A8B7C-3D4E-4F5A-9B6C-7D8E9F0A1B2C}\PythonCOMPath') as key:
            recorded = Path(winreg.QueryValueEx(key, '')[0])
            candidates.append(recorded / 'pythonw.exe')
    except Exception:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def launch(action: str, paths) -> None:
    """Start XP once, for the whole selection, and let it go.

    One process, because the selection is one question: Explorer's per-file
    invocation model is what the queue exists to undo, and here there is
    nothing to undo.
    """
    from xp.runtime.paths import ROOT
    launcher = Path(ROOT) / 'xp_launcher.py'
    # **Not `sys.executable`.** In-proc, this module runs inside Explorer's
    # process and `sys.executable` there is not python.exe -- the click then
    # starts a path that does not exist and nothing happens, with no console to
    # say so. `sys.prefix` is the interpreter's own directory even when
    # embedded, and the launcher's `pythonw` is looked up there first; the
    # registry value the installer writes is the fallback, and `sys.executable`
    # only if both are missing.
    pythonw = _a_windowed_python()
    # **Never `python.exe`.** It is the console build: the menu item then flashes a black window for
    # as long as the child runs, which is the "stupid terminal" the user kept seeing from the right-click
    # menu. If `pythonw` cannot be found the click does nothing and says why in the log, because a
    # console window is not an acceptable fallback.
    if not pythonw.is_file():
        from xp.runtime import log

        log.stamp('menu: pythonw.exe was not found next to %s; nothing launched' % sys.executable)
        return
    command = [str(pythonw), str(launcher), '--action', action,
               *[str(path) for path in paths]]
    _note('launch %s with %d path(s): %s' % (action, len(paths), command[0]))
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0                      # SW_HIDE
    subprocess.Popen(command, close_fds=True,
                     creationflags=subprocess.CREATE_NO_WINDOW,
                     startupinfo=startup)


class XPMenu:
    """One menu, one selection. A new instance per right-click."""

    _com_interfaces_ = [_shell.IID_IShellExtInit, _shell.IID_IContextMenu]
    # Named so pywin32 can build the vtable for the two interfaces without a
    # typelib: the order does not matter, the names have to match COM's.
    _public_methods_ = ['Initialize', 'QueryContextMenu', 'InvokeCommand',
                        'GetCommandString']
    _reg_clsid_ = CLSID
    _reg_desc_ = 'XP: Converter, Merger, Arranger'
    _reg_clsctx_ = pythoncom.CLSCTX_LOCAL_SERVER
    _reg_threading_ = 'Apartment'

    def __init__(self):
        _note('XPMenu()')
        self.paths: list[str] = []
        self.verbs: list[str] = []
        self.first_id = 1
        # The popup has to stay alive while the menu is on screen: destroying
        # the handle is how a submenu becomes empty the moment it is opened.
        self._popup = None

    # ---- IShellExtInit -------------------------------------------------

    def Initialize(self, pidlFolder, dataObject, hkeyProgID):
        # **This runs inside the shell's process.** An exception here is not a
        # traceback in a console, it is Explorer's context menu failing, so
        # every entry point answers "nothing" instead of raising.
        try:
            self.paths = paths_from_data_object(dataObject)
            _note('Initialize: %d path(s) %r' % (len(self.paths), self.paths[:3]))
        except BaseException as exc:
            _note('Initialize failed: %r' % (exc,))
            self.paths = []
        return 0

    # ---- IContextMenu --------------------------------------------------

    def QueryContextMenu(self, hMenu, indexMenu, idCmdFirst, idCmdLast, flags):
        try:
            return self._build(hMenu, indexMenu, idCmdFirst, idCmdLast)
        except BaseException as exc:
            _note('QueryContextMenu failed: %r' % (exc,))
            return 0

    def _build(self, hMenu, indexMenu, idCmdFirst, idCmdLast):
        from xp.runtime.windows import verbs_for, label_of

        verbs = verbs_for(self.paths)
        _note('QueryContextMenu: %d verb(s) %r, ids from %d'
              % (len(verbs), verbs, idCmdFirst))
        if not verbs:
            return 0                       # no XP entry at all for this selection
        popup = win32gui.CreatePopupMenu()
        # **The popup itself takes one id.** The shell counts every id the
        # handler consumed, and the submenu *is* one of them -- so the verbs
        # have to start one past `idCmdFirst`, and the count returned has to
        # include the popup. Without this the shell's dispatch was shifted by
        # one and a click on Arranger arrived as offset 1, which this code then
        # read as "the second verb": the menu appeared, the click landed, and
        # the wrong tool opened.
        self._by_id, self._by_offset = {}, {}
        for offset, action in enumerate(verbs):
            menu_id = idCmdFirst + 1 + offset
            win32gui.AppendMenu(popup, win32con.MF_STRING, menu_id,
                                label_of(action))
            self._by_id[menu_id] = action
            self._by_offset[1 + offset] = action
        win32gui.InsertMenu(hMenu, indexMenu,
                            win32con.MF_BYPOSITION | win32con.MF_POPUP,
                            popup, POPUP_LABEL)
        # **The mark, on the "XP" entry itself -- which lives in the parent
        # menu.** It was being set on `popup`'s item 0, i.e. on "Arranger": the
        # picture went one level too deep and Explorer drew the XP line with the
        # generic document icon the user sent the screenshot of. And it is a *bitmap*
        # the call wants (an icon handle is refused); see `popup_bitmap`.
        self._bitmaps = []
        bitmap = popup_bitmap(self._bitmaps)
        if bitmap is not None:
            if set_menu_bitmap(hMenu, indexMenu, bitmap):
                _note('bitmap on the XP entry: %r' % (bitmap,))
            else:
                _note('bitmap refused for the XP entry: %r' % (bitmap,))
        else:
            _note('no popup bitmap')
        self._popup, self.verbs, self.first_id = popup, verbs, idCmdFirst + 1
        return len(verbs) + 1

    def _label_of_id(self, value: int) -> str:
        """What the popup's item with that id says, or '' when it has none.

        Through `user32.GetMenuStringW` and not `win32gui`: this pywin32 build
        has no `GetMenuString`, and the whole point of asking is that it is the
        only answer that cannot be argued with.
        """
        popup = getattr(self, '_popup', None)
        if popup is None:
            return ''
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(128)
            written = ctypes.windll.user32.GetMenuStringW(
                int(popup), int(value), buffer, len(buffer),
                win32con.MF_BYCOMMAND)
            return buffer.value if written else ''
        except Exception:
            return ''

    def _chosen(self, ci) -> str | None:
        """Which action was picked, from whichever shape the caller uses.

        **`lpVerb` is an id *offset*, not an absolute id.** COM passes the
        command's offset from the `idCmdFirst` the handler was given, and the
        shell rewrites it before dispatching: the click that reached the log as
        `InvokeCommand: (0, 0, 1, ...)` meant "the second item", while this
        code was subtracting `first_id` from it and landing on a negative
        index -- so the menu appeared, the click arrived, and nothing happened.
        Both shapes are accepted because a caller may pass either.
        """
        try:
            raw = ci['lpVerb'] if isinstance(ci, dict) else ci[2]
        except Exception:
            return None
        if isinstance(raw, (bytes, bytearray, str)):
            text = (raw if isinstance(raw, str)
                    else bytes(raw).decode('mbcs', 'replace')).strip('\x00')
            if text in self.verbs:
                return text
            return None
        try:
            value = int(raw)
        except Exception:
            return None
        # **The shell numbers our ids, then dispatches the offset.** It was
        # measured, not read: with the popup at `idCmdFirst` and the verbs at
        # `idCmdFirst + 1` and `+ 2`, a click on the *first* verb arrived as
        # 2 and a click on the second as 3 -- one more than the id we used,
        # because the shell counts its own numbering from 1. So the offset is
        # `verb_index + 2`.
        # **Ask the menu what that number is.** Explorer has dispatched both an
        # id and a 1-based offset for these items -- the log has an
        # `InvokeCommand: (..., 1, ...)` that meant the first item, and clicks
        # that arrived as 2 and 3 for the same two items. Arithmetic cannot tell
        # those apart, and the guess it made was the bug the user reported: clicking
        # "Merger" ran "Arranger". `GetMenuString` with the item's id answers
        # what the clicked item actually says, and an offset that matches no id
        # answers nothing -- which is itself the answer.
        label = self._label_of_id(value)
        if label:
            from xp.runtime.windows import label_of as _label

            for action in self.verbs:
                if _label(action) == label:
                    _note('InvokeCommand: id %r is %r' % (value, label))
                    return action
        # **Then the numbering, in the two shapes the logs contain.** A number
        # that came in without a label is an offset, and the offsets seen here
        # were `index + 2` (in a menu whose ids started at 500) and `index + 1`
        # (in one whose ids started at 1); the label above already answered the
        # second kind, so what is left is `index + 2`.
        if value in getattr(self, '_by_id', {}):
            return self._by_id[value]
        # **The mapping the menu made when it built itself.** `_build` records
        # `_by_offset[1 + index]` for every verb -- the popup consumes
        # `idCmdFirst`, so the verbs start one past it -- and then this method
        # never read it. It guessed instead, and the guess it tried first was
        # `value - 2`, which turns offset 2 into index 0. Offset 2 **is** the
        # second verb under the handler's own numbering, and the second verb
        # is Merger on both file types, so clicking Merger ran whatever came
        # first: Arranger on a notebook, Converter on a PDF. The user's words at
        # 03:36 on 2026-10-27: "HWY I CHOCK MERGE RIOT APPEA RTO BE
        # ARREANGER". The answer was in the object the whole time.
        by_offset = getattr(self, '_by_offset', {})
        if value in by_offset:
            return by_offset[value]
        # Then the shapes that are not this handler's own numbering: an id
        # relative to `first_id`, and a plain 0-based index. `value - 2` is
        # gone; it could only ever shadow `_by_offset`.
        index = value - self.first_id
        if 0 <= index < len(self.verbs):
            return self.verbs[index]
        if 0 <= value < len(self.verbs):
            return self.verbs[value]
        # **Last, the `+2` shape an older log recorded.** Kept so a dispatch
        # that really does arrive as `index + 2` still lands -- it is only
        # ever reached for a number `_by_offset` does not know, so it can no
        # longer shadow the handler's own numbering. The one value the two
        # readings disagree about is 2, and that one goes to `_by_offset`:
        # the API says the offset is `id - idCmdFirst`, which makes 2 the
        # second verb, and the user's report settles it, because under the other
        # reading the user's click on Merger would have opened Merger.
        index = value - 2
        if 0 <= index < len(self.verbs):
            return self.verbs[index]
        return None

    def InvokeCommand(self, ci):
        try:
            action = self._chosen(ci)
            _note('InvokeCommand: %r -> %r' % (ci, action))
            if action is None:
                return
            launch(action, self.paths)
        except BaseException as exc:
            _note('InvokeCommand failed: %r' % (exc,))

    def GetCommandString(self, idCmd, uFlags):
        """The status-bar help, and the string Explorer shows for a key."""
        try:
            index = int(idCmd) - self.first_id
            label = self.verbs[index]
        except Exception:
            return ''
        from xp.runtime.windows import label_of
        return label_of(label)
