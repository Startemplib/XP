"""The shared visual identity: the XP mark, and the one palette every window is painted from."""
import sys

from pathlib import Path
from PIL import Image, ImageTk
from tkinter import ttk

# One dark ground, and light written on it. ABYSS is the container; the three
# inks are what it holds. Every window in XP is painted out of this list.
ABYSS = '#101923'
INK = '#8fc9ff'      # Xournal ink, the heavy stroke
PALE = '#eef5fb'     # the hairline that crosses it
PAPER = '#4cddd5'    # PDF cyan
TEXT = '#d8e5ef'
MUTED = '#96adbf'

# One face, the user's choice, everywhere the panel writes words. Tk's own named
# fonts are what ttk's widgets read, so setting those is what makes a theme
# take -- styling each widget would leave the comboboxes, the entries and the
# menus in the system face and the window half one thing and half another.
#
# `MONO` stays a monospace face on purpose and is not the user's to object to yet:
# it is for the places where characters have to line up under each other --
# a path, a number beside another number -- and Times would make a column
# that is not a column. Nothing writes prose in it.
FACE = 'Times New Roman'
MONO = 'Consolas'
BASE = 10                  # Times reads about a point smaller than Segoe at
                           # the same nominal size, so the body is set one up
                           # from the 9 the system face was using.


def font(size=BASE, weight='normal', mono=False):
    """The panel's face at a size. One place, so a size never drifts."""
    return (MONO if mono else FACE, size, weight)

MARK = Path(__file__).parent / 'assets' / 'xp-mark.png'


def icon_image(size=128):
    """The XP mark, as supplied: a neon XP written in one hand on black.

    It is an asset, not something drawn here -- this only scales it, and
    squares it off first if it ever arrives non-square, so that scaling it
    can't stretch the letters.
    """
    image = Image.open(MARK).convert('RGBA')
    if image.width != image.height:
        side = max(image.size)
        square = Image.new('RGBA', (side, side), (0, 0, 0, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = square
    return image.resize((size, size), Image.Resampling.LANCZOS)


def icon_path():
    """The .ico on disk, rewritten whenever the mark it comes from changes.

    Windows keeps the icon it was handed at install time, so an .ico written
    only once means a new mark never actually appears.
    """
    path = Path(__file__).parent / 'assets' / 'xp.ico'
    source = max(Path(__file__).stat().st_mtime, MARK.stat().st_mtime)
    if not path.exists() or path.stat().st_mtime < source:
        path.parent.mkdir(exist_ok=True)
        icon_image(256).save(path, sizes=[(16, 16), (32, 32), (48, 48),
                                          (64, 64), (128, 128), (256, 256)])
    return path


def use_the_face(root):
    """Every named font Tk hands to a widget, in the panel's face.

    ttk widgets do not read a style's font unless the style sets one, and
    almost none of ours do -- they take `TkDefaultFont` and the rest. So the
    face is set *there*, once, and a combobox, an entry, a button and a label
    all change together. Setting it widget by widget is how a window ends up
    half in one face and half in another.

    `TkFixedFont` is left alone: it is what the places that line characters
    up under each other use, and a proportional face there would stop them
    lining up.
    """
    import tkinter.font as tkfont

    for name, size in (('TkDefaultFont', BASE), ('TkTextFont', BASE),
                       ('TkMenuFont', BASE), ('TkHeadingFont', BASE),
                       ('TkCaptionFont', BASE), ('TkSmallCaptionFont', BASE - 1),
                       ('TkIconFont', BASE), ('TkTooltipFont', BASE - 1)):
        try:
            held = tkfont.nametofont(name, root=root)
        except Exception:
            continue           # not every build defines every one of them
        held.configure(family=FACE, size=size)


def decorate(root):
    root.configure(background=ABYSS)
    root._xopp_icon = ImageTk.PhotoImage(icon_image(64), master=root)
    root.iconphoto(True, root._xopp_icon)
    use_the_face(root)
    style = ttk.Style(root)
    for name in ('Lab.TFrame', 'Lab.TLabelframe'):
        style.configure(name, background=ABYSS)
    style.configure('Lab.TLabel', background=ABYSS, foreground=TEXT,
                    font=font())
    for name in ('Lab.TCheckbutton', 'Lab.TRadiobutton'):
        style.configure(name, background=ABYSS, foreground=TEXT, font=font())
    style.configure('Muted.Lab.TLabel', background=ABYSS, foreground=MUTED,
                    font=font())
    style.configure('Title.Lab.TLabel', background=ABYSS, foreground=PAPER,
                    font=font(18, 'bold'))
    # **The plain styles too, not only the `Lab.` ones.** A window that asks
    # for a bare `ttk.Frame` used to get the system theme's light grey, so the
    # merge window was a light toolbar over a dark root while the converter and
    # the arranger were dark throughout -- which is what the user reported on
    # 2026-10-30: the merger's GUI should be "the same as the converter and the
    # arranger: dark background, blue font, Times New Roman". Styling the base
    # names means a window gets the palette without having to name a style on
    # all forty of its widgets, and the `Lab.` ones still exist for the places
    # that want a particular ink.
    style.configure('TFrame', background=ABYSS)
    style.configure('TLabelframe', background=ABYSS)
    style.configure('TLabelframe.Label', background=ABYSS, foreground=MUTED)
    style.configure('TLabel', background=ABYSS, foreground=TEXT)
    style.configure('TCheckbutton', background=ABYSS, foreground=TEXT)
    style.configure('TRadiobutton', background=ABYSS, foreground=TEXT)
    style.configure('TPanedwindow', background=ABYSS)
    # Buttons, entries and comboboxes are deliberately **not** repainted here.
    # On Windows the native theme draws those itself and ignores `background`,
    # so colouring them would take `theme_use('clam')` for the whole
    # interpreter -- which would change the converter and the arranger, the
    # two windows the user is asking the merger to match. They keep the face, which
    # is the part that does take, and the native chrome, which is what the
    # other two have.
    for name in ('TButton', 'TEntry', 'TCombobox', 'TSpinbox', 'TNotebook.Tab',
                 'TLabelframe.Label', 'TMenubutton'):
        try:
            style.configure(name, font=font())
        except Exception:
            pass
    # **The loading bar, at the size of a line of writing.** The default
    # `ttk.Progressbar` is 22 px of light grey with a chunky green block: next to
    # 10 pt text it reads as a different program, and on the user's screen it was the
    # "loading bar too large" the user pointed at. Six pixels, the panel's own trough,
    # the accent colour for what has been done.
    try:
        style.configure('XP.Horizontal.TProgressbar', troughcolor='#1b2a38',
                        background=PAPER, bordercolor=ABYSS,
                        lightcolor=PAPER, darkcolor=PAPER, thickness=6)
    except Exception:
        pass


def be_dpi_aware() -> bool:
    """Kept for callers that already have this module open.

    The real function lives in `xp.runtime.dpi`, which does not import tkinter:
    the launcher must be able to answer a bad verb without importing Tk, and
    this module imports it for the palette.
    """
    from xp.runtime.dpi import be_dpi_aware as ask

    return ask()
