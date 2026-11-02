"""Current-user Explorer verbs, Send To, and shortcuts; no default-app change."""
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from xp.runtime.paths import ROOT          # the checkout, found not counted
BACKSLASH = chr(92)
KEY_PREFIX = r'Software\Classes\SystemFileAssociations'


# What the XP submenu offers, per file type: (action, label). The action is
# what the launcher is told to do with the selection.
# **Exactly three entries, these three names, on both file types.** The user's
# specification of 2026-10-21, and it is a specification rather than a
# preference: three tools, three doors, one list to learn.
#
#     Converter   the PDF -> .xopp converter        (action 'auto')
#     Merger      the brush-based merger            (action 'merge')
#     Arranger    the page arranger in this tree       (action 'arrange')
#
# An earlier version of this table offered each tool only on the type it can
# act on -- two entries per type. The user overruled that: the menu is the same
# everywhere. The two combinations that cannot work (a PDF handed to the
# arranger, a notebook handed to the converter) are refused in one plain
# sentence by `app.open_sources`, which is better than a menu that changes
# shape depending on what you right-clicked.
#
# Nothing was removed from the program, only from this menu. Where the three
# that came out still live:
#   live stages   `python -m xp convert-gui <pdf> --workflow live`, or the
#                 dashboard's Convert button
#   compare       `python -m xp diff a b`, or the dashboard
#   Open in XP    the Desktop / Start shortcuts, and Send To > XP, which takes
#                 a whole selection
THREE_TOOLS = [('auto', 'Converter'),
               ('merge', 'Merger'),
               ('arrange', 'Arranger')]

# What the *file type* offers when the selection is one type only. The
# selection rule below decides for a mixed one; this table is what a static
# submenu can express, and `registration_plan` writes it.
SUBMENU = {
    '.pdf': ['auto', 'merge'],
    # Arranger first, Merger second: **Merger is the second item on both file
    # types**, which is what the user asked for on 2026-10-23 -- the second line of
    # the submenu is where the user's hand goes, and it should not mean a different
    # tool depending on what the user right-clicked.
    '.xopp': ['arrange', 'merge'],
}

# The menu is built by the shell when it opens, not written into the registry
# ahead of time: a static submenu cannot know that a PDF and a notebook were
# selected together, and that is exactly the case the user's rule names. The handler
# is `xp/shell/contextmenu.py`; this is its component id, and it must match the
# one in that file (a test holds them together).
CONTEXT_MENU_CLSID = '{6F2A8B7C-3D4E-4F5A-9B6C-7D8E9F0A1B2C}'
# **On.** The user's rule of 2026-10-26 is about the whole selection -- a PDF and a
# notebook selected together must offer *Merger only* -- and a static per-type
# submenu cannot express that, because Explorer shows it for the file type the user
# right-clicked. The handler has been written and tested for a while
# (`xp/shell/contextmenu.py`, with the dispatch offsets the log taught us); it
# was only switched off. `pythoncom` is an optional dependency, so the install
# falls back to the static menu when it is not there -- see `_the_menu_plan`.
DYNAMIC_MENU = True


def _the_menu_plan(pythonw):
    """The dynamic plan when pywin32 is there, the static one otherwise.

    `pythoncom` is in `requirements-optional.txt`: without it the shell has no
    in-proc COM server to load, and an install that raised here would leave a
    user with no menu at all instead of a simpler one.
    """
    if DYNAMIC_MENU:
        try:
            import pythoncom  # noqa: F401  (the probe, not the use)
        except Exception:
            pass
        else:
            return menu_plan(pythonw)
    return registration_plan(pythonw, ROOT / 'xp_launcher.py',
                             str(icon_of_the_day()))


def menu_plan(pythonw=None):
    """The keys that make Explorer build the menu instead of reading one.

    **In-proc, because that is what the shell loads.** A classic
    `IContextMenu` handler lives inside Explorer's own process: COM reads
    `InprocServer32`, loads that DLL, and calls into it on the shell's thread.
    An out-of-proc (`LocalServer32`) server is not the mechanism -- which is
    what the earlier attempt got wrong: the server started, published its class
    factory and the caller failed before ever reaching our class.

    No C compiler is needed for that, because pywin32 is itself an in-proc COM
    server: `InprocServer32` names `pythoncom312.dll`, `PythonCOM` names the
    Python class to build, and `PythonCOMPath` puts the project on `sys.path`
    inside the shell's process. Everything is written under
    `HKEY_CURRENT_USER\\Software\\Classes`, so it is this user's menu, needs no
    administrator, and removing the five keys is the whole of the undo.
    """
    import pythoncom

    dll = str(Path(pythoncom.__file__))
    clsid = r'Software\Classes\CLSID' + BACKSLASH + CONTEXT_MENU_CLSID
    plan = [(clsid, '', 'XP menu'),
            (clsid + r'\InprocServer32', '', dll),
            (clsid + r'\InprocServer32', 'ThreadingModel', 'Apartment'),
            (clsid + r'\PythonCOM', '', 'xp.shell.contextmenu.XPMenu'),
            (clsid + r'\PythonCOMPath', '', str(ROOT))]
    # **The menu entry's picture.** A handler with no `DefaultIcon` is drawn with
    # a blank square or the system's generic one -- the "wrong image" the user pointed
    # at. The installer writes it, and it lives inside the component's own key,
    # so `menu_roots()` already takes it away on uninstall.
    icon = icon_of_the_day()
    if icon is not None:
        plan.append((clsid + r'\DefaultIcon', '', str(icon)))
    for extension in SUBMENU:
        plan.append((KEY_PREFIX + BACKSLASH + extension
                     + r'\shellex\ContextMenuHandlers\XP', '',
                     CONTEXT_MENU_CLSID))
    return plan


def icon_of_the_day():
    """The `.ico` the shortcuts use, without making one (that is `install`)."""
    from xp.render.appearance import icon_path

    try:
        icon = Path(icon_path())
        if icon.exists():
            return icon
    except Exception:
        pass
    return ROOT / 'xp' / 'render' / 'assets' / 'xp.ico'


def _the_dynamic_plan(pythonw):
    """The dynamic plan, or None when pywin32 is not installed."""
    try:
        import pythoncom  # noqa: F401
    except Exception:
        return None
    try:
        return menu_plan(pythonw)
    except Exception:
        return None


def describe_install():
    """What an install of this menu actually writes, as JSON.

    The verify step used to count verbs per file type, which is a fact about
    the *static* menu and is simply wrong for the selection-aware one: there
    are no per-type verbs to count, there is a COM registration. So the
    installer asks for the plan itself and reads the registry back entry by
    entry -- same source, no second copy to go stale.
    """
    import json

    from xp.render.appearance import icon_path

    pythonw = Path(sys.executable).with_name('pythonw.exe')
    # **A description must not write.** `icon_path()` regenerates the .ico when
    # the mark is newer, so `-CheckOnly` -- documented as changing nothing --
    # rewrote a file. The path is what the plan needs; the file is made by
    # `install()`, which is the call that is allowed to write.
    icon = icon_path()
    try:
        icon = str(Path(icon))
        if not Path(icon).exists():
            icon = str(ROOT / 'xp' / 'render' / 'assets' / 'xp.ico')
    except Exception:
        icon = str(ROOT / 'xp' / 'render' / 'assets' / 'xp.ico')
    plan = _the_menu_plan(pythonw)
    dynamic = plan is not None and plan == _the_dynamic_plan(pythonw)
    return json.dumps({
        'mode': 'selection-aware' if dynamic else 'static',
        'clsid': CONTEXT_MENU_CLSID if DYNAMIC_MENU else '',
        'entries': [{'key': path, 'name': name, 'value': value,
                     'exists': True} for path, name, value in plan],
    })


def menu_roots():
    """The keys a menu install owns, for clearing and for pruning."""
    clsid = r'Software\Classes\CLSID' + BACKSLASH + CONTEXT_MENU_CLSID
    roots = [clsid]
    for extension in SUBMENU:
        roots.append(KEY_PREFIX + BACKSLASH + extension
                     + r'\shellex\ContextMenuHandlers\XP')
    return roots

# What a *selection* can be given, as opposed to what a file type offers.
#
# The user's rule of 2026-10-23, and it is about the whole selection rather than the
# file the user happened to right-click:
#
#     pdf only            Converter, Merger      -- a converter takes one pdf,
#                                                  a merger takes two things
#     xopp only           Arranger, Merger       -- a notebook is already
#                                                 converted, so it is arranged
#                                                 or merged; Merger is second
#                                                 on both types on purpose
#     both                Merger                 -- the only tool that reads both
#
# A notebook and a pdf cannot be arranged together, and a pdf cannot be
# converted *with* a notebook in the selection: the tools that would have to
# refuse are not offered. This function is the one place that rule lives; the
# menu asks it, the launcher asks it, and the tests ask it.
PDF_SUFFIXES = ('.pdf',)
XOPP_SUFFIXES = ('.xopp',)


def kinds_of(paths):
    """How many pdfs and how many notebooks a selection holds."""
    pdfs = notebooks = 0
    for path in paths:
        suffix = os.path.splitext(str(path))[1].lower()
        if suffix in PDF_SUFFIXES:
            pdfs += 1
        elif suffix in XOPP_SUFFIXES:
            notebooks += 1
    return pdfs, notebooks


def verbs_for(paths):
    """The actions this selection can be given, in menu order.

    Empty for anything that is neither a pdf nor a notebook, so a right-click
    on some other file shows no XP entry at all rather than a menu that cannot
    act.
    """
    pdfs, notebooks = kinds_of(paths)
    if pdfs and notebooks:
        return ['merge']
    if pdfs:
        return ['auto', 'merge']
    if notebooks:
        return ['arrange', 'merge']
    return []


def label_of(action):
    for name, label in THREE_TOOLS:
        if name == action:
            return label
    return action


def check_action(action, paths):
    """The selection rule, as a refusal. Nothing to return when it is allowed.

    The menu asks `verbs_for` what to show; the launcher asks this, so a verb
    that arrives by another road (a stale menu, a console line, a test) is
    refused by the same table instead of by a second opinion.
    """
    allowed = verbs_for([str(path) for path in paths])
    if allowed and action not in allowed:
        spoken = {'auto': 'converted', 'live': 'converted',
                  'merge': 'merged', 'arrange': 'arranged'}.get(action, action)
        raise ValueError('That selection cannot be %s: it takes %s.'
                         % (spoken, ', '.join(label_of(a) for a in allowed)))


def registration_plan(pythonw, launcher, icon):
    """One XP entry per file type that opens a submenu of the actions.

    A cascading verb (MUIVerb + an empty SubCommands, with the real verbs in
    a nested `shell` key) keeps the context menu to a single XP line while
    still offering everything: converting a PDF either way, merging, or
    comparing read-only. Flat verbs would add one line each to every PDF's
    menu, and a single verb would make you open the window before you could
    say what you wanted.

    MultiSelectModel is Document: Explorer invokes the command once per
    selected file, and shellqueue collects those back into one selection, so
    picking two files and choosing Merge does the obvious thing.
    """
    plan = []
    for extension, actions in SUBMENU.items():
        root_key = KEY_PREFIX + '\\' + extension + r'\shell\XP'
        plan.extend([(root_key, 'MUIVerb', 'XP'), (root_key, 'Icon', str(icon)),
                     (root_key, 'SubCommands', '')])
        for position, action in enumerate(actions):
            # Explorer lists a cascading menu's verbs in key order, which is
            # alphabetical -- hence the numeric prefix; the name isn't shown.
            key = root_key + '\\shell\\%02d_%s' % (position, action)
            command = subprocess.list2cmdline([str(pythonw), str(launcher), '--shell', '--action', action])
            plan.extend([(key, 'MUIVerb', label_of(action)), (key, 'Icon', str(icon)),
                         (key, 'MultiSelectModel', 'Document'),
                         (key + r'\command', '', command + ' "%1"')])
    return plan


def submenu_roots():
    """The one key per file type that holds the whole XP submenu."""
    return [KEY_PREFIX + '%s%s%sshell%sXP' % (BACKSLASH, extension, BACKSLASH, BACKSLASH)
            for extension in SUBMENU]


def _delete_tree(winreg, path):
    """Remove one of our verb keys and everything under it.

    The actions are numbered by their position in the menu, so changing the
    menu renames them: adding one entry turned `04_open` into `05_open` and
    left the old key behind, which Explorer duly showed as a second "Open in
    XP". Writing the new plan over the old values cannot fix that -- the
    stale key has to go -- so an install now clears its own subtree first.
    """
    if not _is_ours(path):
        raise ValueError('Refusing to delete a key that is not ours: ' + path)
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            children = [winreg.EnumKey(key, i) for i in range(winreg.QueryInfoKey(key)[0])]
    except FileNotFoundError:
        return
    for child in children:
        _delete_tree(winreg, path + BACKSLASH + child)
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)


def _is_ours(path):
    """Only the keys this module writes, so a tampered-with record cannot be
    used to delete anything else: XP's verb subtrees under the file-type
    associations, the two `shellex` keys that point at our handler, and our
    own CLSID -- nothing else in `Software\\Classes` is ours."""
    lowered = path.lower()
    if lowered.startswith(KEY_PREFIX.lower()):
        return r'\shell\xp' in lowered or (
            r'\shellex\contextmenuhandlers\xp' in lowered)
    clsid = (r'software\classes\clsid' + BACKSLASH
             + CONTEXT_MENU_CLSID).lower()
    return lowered == clsid or lowered.startswith(clsid + BACKSLASH)


def _manifest_path():
    return Path(os.environ['LOCALAPPDATA']) / 'XP/shell-install.json'


def _read_value(winreg, path, name):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            value, kind = winreg.QueryValueEx(key, name)
        return {'exists': True, 'value': value, 'kind': kind}
    except FileNotFoundError:
        return {'exists': False}


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _shortcut(path, pythonw, launcher, icon, sendto=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    arguments = subprocess.list2cmdline([str(launcher), '--action', 'open'])
    command = "$s=New-Object -ComObject WScript.Shell\n$l=$s.CreateShortcut(" + _ps_quote(path) + ")\n"
    for property_name, value in [('TargetPath', pythonw), ('Arguments', arguments), ('WorkingDirectory', ROOT), ('IconLocation', icon), ('Description', 'Convert PDFs and merge Xournal++ notebooks')]:
        command += '$l.' + property_name + '=' + _ps_quote(value) + '\n'
    command += '$l.Save()'
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-EncodedCommand', encoded],
                   check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def _uninstall_shortcut(path, icon):
    """A Start Menu entry that runs `uninstall.cmd`.

    The installer had no visible counterpart: removing XP meant knowing that
    `install.cmd -Uninstall` exists. A tool a person cannot find is a tool that
    is not there, so the installer leaves this beside its own Start Menu entry
    and the uninstaller takes it away again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    script = ROOT / 'uninstall.cmd'
    arguments = subprocess.list2cmdline(['/c', str(script)])
    command = "$s=New-Object -ComObject WScript.Shell\n$l=$s.CreateShortcut(" + _ps_quote(path) + ")\n"
    for property_name, value in [('TargetPath', os.environ.get('COMSPEC', 'cmd.exe')),
                                 ('Arguments', arguments),
                                 ('WorkingDirectory', str(ROOT)),
                                 ('IconLocation', icon),
                                 ('Description', 'Remove the XP Explorer actions and shortcuts')]:
        command += '$l.' + property_name + '=' + _ps_quote(value) + '\n'
    command += '$l.Save()'
    encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-EncodedCommand', encoded],
                   check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def install():
    if os.name != 'nt':
        raise RuntimeError('Explorer integration is only available on Windows.')
    import winreg
    from xp.render.appearance import icon_path
    pythonw = Path(sys.executable).with_name('pythonw.exe')
    if not pythonw.is_file():
        raise RuntimeError(f'Windowed Python was not found: {pythonw}')
    launcher = ROOT / 'xp_launcher.py'
    icon = icon_path()
    manifest_path = _manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'registry': [], 'shortcuts': []}
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Clear our own submenu first: entries are keyed by their position, so a
    # menu that has changed since the last install leaves keys behind that
    # Explorer still shows. Values recorded for keys that no longer exist are
    # dropped with them, since restoring them would put the orphans back.
    for root_key in submenu_roots() + menu_roots():
        _delete_tree(winreg, root_key)
    manifest['registry'] = [entry for entry in manifest['registry']
                            if not any(entry['path'].lower().startswith(root.lower())
                                       for root in submenu_roots() + menu_roots())]
    plan = _the_menu_plan(pythonw)
    for path, name, value in plan:
        existing = next((v for v in manifest['registry'] if v['path'] == path and v['name'] == name), None)
        if existing is None:
            existing = {'path': path, 'name': name, 'before': _read_value(winreg, path, name)}
            manifest['registry'].append(existing)
        existing['installed'] = value
        save()
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    # Resolve redirected Desktop/Documents paths instead of assuming USERPROFILE.
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
        desktop = Path(os.path.expandvars(winreg.QueryValueEx(key, 'Desktop')[0]))
        documents = Path(os.path.expandvars(winreg.QueryValueEx(key, 'Personal')[0]))
    appdata = Path(os.environ['APPDATA'])
    shortcuts = [desktop / 'XP.lnk', appdata / 'Microsoft/Windows/Start Menu/Programs/XP.lnk',
                 appdata / 'Microsoft/Windows/SendTo/XP.lnk']
    if (documents / 'XP').is_dir():
        shortcuts.append(documents / 'XP/XP.lnk')
    for path in shortcuts:
        entry = next((v for v in manifest['shortcuts'] if v['path'] == str(path)), None)
        if entry is None:
            entry = {'path': str(path), 'before': base64.b64encode(path.read_bytes()).decode('ascii') if path.exists() else None}
            manifest['shortcuts'].append(entry)
        save()
        _shortcut(path, pythonw, launcher, icon)
        entry['installed_hash'] = hashlib.sha256(path.read_bytes()).hexdigest()
        save()
    # The way back out, next to the way in.
    uninstall_link = appdata / 'Microsoft/Windows/Start Menu/Programs/Uninstall XP.lnk'
    entry = next((v for v in manifest['shortcuts'] if v['path'] == str(uninstall_link)), None)
    if entry is None:
        entry = {'path': str(uninstall_link),
                 'before': base64.b64encode(uninstall_link.read_bytes()).decode('ascii')
                 if uninstall_link.exists() else None}
        manifest['shortcuts'].append(entry)
    save()
    _uninstall_shortcut(uninstall_link, icon)
    entry['installed_hash'] = hashlib.sha256(uninstall_link.read_bytes()).hexdigest()
    save()
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)
    return 'Installed Explorer actions, Send To, and desktop/Start shortcuts for this user. On Windows 11, use Show more options.'


def _prune_container(winreg, path):
    r"""Delete a container XP created by writing through it, if it is empty.

    `_prune_empty` refuses anything that is not literally XP's verb key, which
    is the right guard -- but the containers `...\shell` and
    `...\shellex\ContextMenuHandlers` are XP's furniture too: they did not
    exist before the first verb was written, and after `...\shell\XP` goes they
    are empty. This deletes one **only when it is completely empty**, so a key
    another tool has written into is left alone, and so is anything under it.
    """
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            children, values, _ = winreg.QueryInfoKey(key)
    except FileNotFoundError:
        return True
    if children or values:
        return False
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
        return True
    except OSError:
        return False


def _prune_empty(winreg, path):
    """Delete one of our keys if nothing is left in it, children first.

    Only ever our own verb keys -- `_is_ours` is the same guard `_delete_tree`
    uses -- and only ever *empty* ones: a key somebody else has written a value
    into is left exactly where it is, and its parents with it. So this cannot
    take anything away that XP did not put there.

    Returns True if the key is gone (or was never there).
    """
    if not _is_ours(path):
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            children = [winreg.EnumKey(key, i)
                        for i in range(winreg.QueryInfoKey(key)[0])]
    except FileNotFoundError:
        return True
    for child in children:
        _prune_empty(winreg, path + BACKSLASH + child)
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            children, values, _ = winreg.QueryInfoKey(key)
        if children == values == 0:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
            return True
    except FileNotFoundError:
        return True
    return False


def uninstall():
    if os.name != 'nt':
        raise RuntimeError('Explorer integration is only available on Windows.')
    import winreg
    path = _manifest_path()
    # **Before the record is consulted at all.** An XP submenu container that
    # holds nothing is ours by construction and has nothing in it to lose, so
    # it goes whether or not there is a manifest to justify it. Without this,
    # an uninstall that half-finished could never be finished: it removes the
    # record on its way out, so the next run returned "No XP Explorer
    # integration is installed" while two empty submenus sat in the registry
    # and the verify step called them a failure. Found by running it twice,
    # 2026-10-21.
    stubborn = [root for root in submenu_roots()
                if not _prune_empty(winreg, root)]
    if not path.exists():
        if stubborn:
            return ('No installation record, and these still hold values that '
                    'XP did not write: ' + ', '.join(stubborn))
        return 'No XP Explorer integration is installed.'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    # Every key this ever wrote, not just the ones the current menu would
    # write: a manifest from an older layout still names keys the plan no
    # longer has, and it still has to be removable. The check below is what
    # keeps that honest -- only our own verb keys, never an arbitrary path.
    allowed = ({key for key, _, _ in registration_plan('', '', '')}
               | {key for key, _, _ in menu_plan('')}
               | {entry['path'] for entry in manifest['registry']})
    skipped = []
    for entry in reversed(manifest['registry']):
        if entry['path'] not in allowed or not _is_ours(entry['path']):
            raise ValueError('Unexpected registry path in shell installation record.')
        current = _read_value(winreg, entry['path'], entry['name'])
        if not current['exists']:
            continue
        if current['value'] != entry['installed']:
            skipped.append(entry['path'])
            continue
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, entry['path'], 0, winreg.KEY_SET_VALUE) as key:
            before = entry['before']
            if before['exists']:
                winreg.SetValueEx(key, entry['name'], 0, before['kind'], before['value'])
            else:
                winreg.DeleteValue(key, entry['name'])
    # Prune from the submenu root downwards, not key by key.
    #
    # **This is a bug the uninstaller had until 2026-10-21, and only running it
    # showed it.** The verbs live as values inside a tree --
    # `...\.pdf\shell\XP\shell\02_arrange\command` -- and taking every value
    # away leaves the containers behind. The loop that stood here deleted the
    # empty ones it could *name*, and the plan never names the intermediate
    # `...\shell\XP\shell`, because that key is created implicitly by writing
    # through it. So `\shell` survived, its parent `XP` therefore still had a
    # child, and uninstalling left an empty XP submenu on both file types --
    # with the script reporting "Explorer integration removed" and its own
    # verify step then calling it a failure, which is at least honest.
    marker = (BACKSLASH + 'shell' + BACKSLASH + 'XP').lower()
    roots = set()
    for key_path in allowed:
        at = key_path.lower().find(marker)
        if at != -1:
            roots.add(key_path[:at + len(marker)])
    for root in sorted(roots | set(menu_roots())):
        _prune_empty(winreg, root)
    # **And the containers it merely wrote through.** Removing `...\shell\XP`
    # leaves `...\shell` -- which XP created the day it wrote the first verb --
    # and an empty `...\shellex\ContextMenuHandlers` from the selection-aware
    # menu. A person looking at the registry after "removed" should not find
    # XP's empty furniture in it.
    for extension in SUBMENU:
        for tail in (r'\shell', r'\shellex\ContextMenuHandlers'):
            _prune_container(winreg, KEY_PREFIX + '\\' + extension + tail)
    for entry in manifest['shortcuts']:
        shortcut = Path(entry['path'])
        if not shortcut.exists():
            continue
        if hashlib.sha256(shortcut.read_bytes()).hexdigest() != entry.get('installed_hash'):
            skipped.append(str(shortcut))
            continue
        if entry['before'] is None:
            shortcut.unlink()
        else:
            shortcut.write_bytes(base64.b64decode(entry['before']))
    if not skipped:
        path.unlink()
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)
    return 'Explorer integration removed.' + (f' Kept {len(skipped)} settings modified later; installation record retained.' if skipped else '')
