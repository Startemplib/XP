<#
    Install XP for the current user.

    What it does, in order: find a Python that can run XP, make sure the
    libraries it needs are importable, register the Explorer verbs and the
    shortcuts *from this folder*, and then read the registry back to prove
    they point here.

    <~ 为什么这个脚本必须从「自己所在的文件夹」注册,而不是从一个记下来的路径:
       右键菜单是注册表里一串写死的绝对路径。项目搬过一次家,注册表没跟着动,
       于是菜单项还在,点下去 pythonw.exe 去跑一个不存在的 xp_launcher.py ——
       pythonw 没有控制台,失败一声不吭,看起来就是「菜单在但没反应」。
       所以 $PSScriptRoot 是唯一的事实来源:重跑一次这个脚本就是搬家之后的修复动作,
       而 -CheckOnly 能在不改任何东西的前提下回答「菜单现在指对了没有」。 ~>

    No administrator rights, and nothing outside HKEY_CURRENT_USER: the verbs
    live under Software\Classes\SystemFileAssociations and the shortcuts in
    this user's own Desktop / Start Menu / SendTo. Default applications are
    not touched.

    Usage:
      install.cmd                 install or repair (double-click; bypasses the
                                  execution policy, which install.ps1 needs)
      install.ps1                 install or repair
      install.ps1 -CheckOnly      diagnose only, change nothing
      install.ps1 -NoDeps         skip the library check
      install.ps1 -NoPause        do not wait for Enter at the end (for scripts)
      install.ps1 -WithGpu        also install the optional CUDA extras
      install.ps1 -Uninstall      remove what a previous install added

    **Install only from the folder you want to keep.** The registration bakes
    absolute paths; running this from a copy in Temp would point the menu at
    Temp, which gets cleaned.
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$NoDeps,
    [switch]$WithGpu,
    [switch]$Uninstall,
    [switch]$NoPause
)

$ErrorActionPreference = 'Continue'
# Every python call below imports XP, and an import writes `__pycache__`. A check
# documented as "changes nothing" should not leave compiled bytecode behind, so
# the whole script asks Python not to write it.
$env:PYTHONDONTWRITEBYTECODE = '1'
$root = $PSScriptRoot
$launcher = Join-Path $root 'xp_launcher.py'
$problems = @()

function Say([string]$text, [string]$color = 'Gray') { Write-Host $text -ForegroundColor $color }
function Good([string]$text) { Say "  ok    $text" 'Green' }
function Bad([string]$text) { Say "  FAIL  $text" 'Red'; $script:problems += $text }
function Note([string]$text) { Say "  --    $text" 'DarkGray' }

Say ''
Say '=== XP ===' 'Cyan'
Say "  from  $root"
Say ''

if (-not $Uninstall -and -not (Test-Path $launcher)) {
    Bad 'xp_launcher.py is not next to this script; run the copy that lives in the project'
    if (-not $NoPause) { Read-Host 'Press Enter to close' }
    exit 1
}

# ---- 1. Python ------------------------------------------------------------
# XP needs a windowed interpreter too: windows.py registers pythonw.exe so a
# verb does not flash a console. A Python without it cannot be registered.
Say 'Python' 'White'
$python = $null
$candidates = @()
$pyExe = Get-Command 'py.exe' -ErrorAction SilentlyContinue
if ($pyExe) {
    $found = & $pyExe.Source -3 -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -eq 0 -and $found) { $candidates += $found.Trim() }
}
$onPath = Get-Command 'python.exe' -ErrorAction SilentlyContinue
if ($onPath) { $candidates += $onPath.Source }
$candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')

foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate)) { continue }
    $version = & $candidate -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($LASTEXITCODE -ne 0) { continue }
    $parts = $version.Trim().Split('.')
    if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
        Note "$candidate is Python $($version.Trim()) -- XP needs 3.10 or newer"
        continue
    }
    $windowed = Join-Path (Split-Path $candidate -Parent) 'pythonw.exe'
    # **An uninstall needs no windowed interpreter**: it removes registry keys
    # and shortcuts, which is what python.exe does. Refusing to uninstall
    # because pythonw is missing would leave the menu pointing at a program
    # that is not there -- the exact failure this script exists to prevent.
    if (-not $Uninstall -and -not (Test-Path $windowed)) {
        Note "$candidate has no pythonw.exe beside it -- cannot be registered"
        continue
    }
    $python = $candidate
    Good "Python $($version.Trim()) at $python"
    break
}
if (-not $python) {
    Bad 'no usable Python found (needs 3.10+ with pythonw.exe beside it)'
    Say ''
    Say '  Install Python for this user from python.org, then run this again.' 'Yellow'
    if (-not $NoPause) { Read-Host 'Press Enter to close' }
    exit 1
}

# ---- 2. Libraries ---------------------------------------------------------
# Keyed by import name, not package name: a wheel that installed but cannot
# be imported is the failure that matters here.
# **The names the code imports**, and every name a wheel may expose them under.
# `pymupdf` is checked as `pymupdf` **and** `fitz`: 1.24.3 and later expose the
# first, older wheels only the second, and the tree imports the first. Checking
# one name reported a working install as broken (or a broken one as working,
# depending on the build) -- a package that imports is not the same as a module
# the code can use.
$imports = @{
    'opencv-python' = @('cv2')
    'numpy'         = @('numpy')
    'pymupdf'       = @('pymupdf', 'fitz')
    'pillow'        = @('PIL')
}
if (-not $NoDeps -and -not $Uninstall) {
    Say ''
    Say 'Libraries' 'White'
    # requirements.txt is the list; everything from the first comment that
    # says "Optional" onwards is the CUDA group, which XP falls back from.
    $required = @()
    $optional = $false
    foreach ($line in Get-Content (Join-Path $root 'requirements.txt')) {
        $text = $line.Trim()
        if ($text -match '^#') {
            # The **line**, not the word. Prose in this file says "optional"
            # in its first paragraph ("Everything below the REQUIRED block is
            # optional"), and matching the word made every package optional:
            # a fresh machine then had nothing checked and nothing installed,
            # and the installer said "Libraries" and printed no lines under it.
            # Found by running it 2026-10-23.
            if ($text -match '^#\s*=+\s*Optional from here down') { $optional = $true }
            continue
        }
        if (-not $text) { continue }
        if (-not $optional) { $required += $text }
    }
    $missing = @()
    foreach ($package in $required) {
        # The name for pip can carry a version range; the import name cannot.
        $base = ($package -split '[<>=!~]')[0].Trim().ToLower()
        $names = $imports[$base]
        # **The import name, never the requirement line.** The fallback used to
        # be `@($package)`, so a new requirement like `foo>=1.0` was imported as
        # `import foo>=1.0` -- a syntax error reported as "still cannot be
        # imported" after a successful pip install.
        if (-not $names) { $names = @(($base -replace '-', '_')) }
        $found = $null
        foreach ($name in $names) {
            & $python -c "import $name" | Out-Null
            if ($LASTEXITCODE -eq 0) { $found = $name; break }
        }
        if ($found) { Good "$package ($found)" }
        else { Note "$package is missing (none of: $($names -join ', '))"; $missing += $package }
    }
    if ($missing.Count -gt 0) {
        if ($CheckOnly) {
            Bad "$($missing.Count) librar(ies) missing: $($missing -join ', ')"
        } else {
            Say "  installing $($missing -join ', ') ..." 'Yellow'
            & $python -m pip install --disable-pip-version-check @missing
            foreach ($package in $missing) {
                $base = ($package -split '[<>=!~]')[0].Trim().ToLower()
                $names = $imports[$base]
                if (-not $names) { $names = @(($base -replace '-', '_')) }
                $found = $null
                foreach ($name in $names) {
                    & $python -c "import $name" | Out-Null
                    if ($LASTEXITCODE -eq 0) { $found = $name; break }
                }
                if ($found) { Good "$package ($found)" } else { Bad "$package still cannot be imported" }
            }
        }
    }
    # **The tree's own imports, not just the packages'.** A package can import
    # and the program still not start (a module renamed between versions, an
    # extra the tree took for granted). This is the same import the tools do.
    if ($missing.Count -eq 0) {
        # `xp/render/appearance.py` imports tkinter at module scope, so a Python
        # built without Tcl/Tk dies with a traceback rather than a sentence.
        & $python -c "import tkinter" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { Good 'tkinter (the window toolkit)' }
        else { Bad 'this Python has no tkinter -- XP cannot open a window; reinstall Python with Tcl/Tk' }
        & $python -c "import xp.core.document.xopp_io, xp.workers.converter.convert, xp.workers.merger.merge" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { Good 'XP imports its own modules' }
        else { Note 'XP could not import its own modules -- run it once to see the error' }
    }
    if ($WithGpu -and -not $CheckOnly) {
        # Optional by design: gpu.py falls back to numpy on the CPU, so a
        # failure here is reported and not fatal.
        Say '  installing the CUDA extras ...' 'Yellow'
        # **From the file, not from here.** The wheel has to match the driver
        # (`nvidia-smi`), and requirements-optional.txt is where that is written
        # down; a hard-coded cupy-cuda12x here could disagree with the doc.
        $cuda = @()
        $optionalFile = Join-Path $root 'requirements-optional.txt'
        if (Test-Path $optionalFile) {
            foreach ($line in Get-Content $optionalFile) {
                $text = $line.Trim()
                if (-not $text -or $text.StartsWith('#')) { continue }
                if ($text -match '^(cupy|nvidia-)') { $cuda += $text }
            }
        }
        if ($cuda.Count -eq 0) { $cuda = @('cupy-cuda12x') }
        & $python -m pip install --disable-pip-version-check @cuda
        & $python -c "import cupy" | Out-Null
        if ($LASTEXITCODE -eq 0) { Good 'cupy' } else { Note 'cupy did not import -- XP will use the CPU path' }
    }
}

# ---- 3. The Explorer verbs and the shortcuts ------------------------------
Say ''
Say 'Explorer integration' 'White'
if ($Uninstall) {
    if ($CheckOnly) {
        Note '-CheckOnly given, so nothing was removed'
    } else {
        # **`xp.runtime.windows.uninstall`, not `python -m xp`.** The CLI imports
        # the converter and the merger at module scope, so it needs cv2, numpy,
        # PyMuPDF and Pillow -- on the machine where they never installed, the
        # natural state before an uninstall, removal would fail.
        $ask = "import sys; sys.path.insert(0, r'$root'); " +
               "from xp.runtime.windows import uninstall; uninstall(); print('removed')"
        $output = & $python -c $ask
        if ($LASTEXITCODE -eq 0) { Good ($output | Select-Object -Last 1) }
        else { Bad "uninstall failed: $($output | Select-Object -Last 1)" }
    }
} elseif (-not $CheckOnly) {
    # install-shell is idempotent: it clears XP's own subtree first, so a
    # stale registration from an older location is replaced rather than
    # added to. Run from $root so the paths it bakes in are this folder's.
    Push-Location $root
    $output = & $python -m xp install-shell
    Pop-Location
    if ($LASTEXITCODE -eq 0) { Good ($output | Select-Object -Last 1) } else { Bad 'install-shell failed' }
} else {
    Note '-CheckOnly given, so the registry was not written'
}

# ---- 4. Prove it, by reading the registry back ----------------------------
# The whole point of the exercise: a verb whose command names a file that is
# not there looks perfectly healthy in the menu and does nothing when clicked.
Say ''
Say 'Verify' 'White'
# Asked of the code, not written down here. This used to count verbs per file
# type, which is a fact about the *static* menu and is wrong for the
# selection-aware one: that one is a COM registration, with no per-type verbs
# to count. So the installer asks for the plan and reads the registry back
# entry by entry -- same source, no second copy to drift.
$ask = "import sys; sys.path.insert(0, r'$root'); " +
       "from xp.runtime.windows import describe_install; print(describe_install())"
$planJson = & $python -c $ask 2>$null
$plan = $null
if ($LASTEXITCODE -eq 0 -and $planJson) {
    try { $plan = $planJson | ConvertFrom-Json } catch { $plan = $null }
}
if (-not $plan) {
    Note 'could not ask xp.runtime.windows what the menu should be; skipping the check'
} else {
    Note "menu: $($plan.mode)"
    $wrong = 0
    foreach ($entry in $plan.entries) {
        $key = "HKCU:\$($entry.key)"
        if ($Uninstall) {
            # After removing, every entry XP wrote must be gone. The old code
            # skipped this loop and then announced "the COM registration is
            # gone" about a CLSID that static mode never writes -- a check that
            # could not fail.
            if (Test-Path $key) {
                $value = (Get-ItemProperty $key -ErrorAction SilentlyContinue).($entry.name)
                if ($entry.name -eq '') {
                    $value = (Get-ItemProperty $key -ErrorAction SilentlyContinue).'(default)'
                }
                if ($null -ne $value -and "$value" -eq "$($entry.value)") {
                    Bad "$($entry.key)\$($entry.name) is still there"; $wrong++
                }
            }
            continue
        }
        if (-not (Test-Path $key)) { Bad "missing $($entry.key)"; $wrong++; continue }
        $value = (Get-ItemProperty $key -ErrorAction SilentlyContinue).($entry.name)
        if ($entry.name -eq '') { $value = (Get-ItemProperty $key -ErrorAction SilentlyContinue).'(default)' }
        if ($null -eq $value -or "$value" -ne "$($entry.value)") {
            Bad "$($entry.key)\$($entry.name) is '$value', expected '$($entry.value)'"
            $wrong++
        }
    }
    if ($Uninstall) {
        if ($wrong -eq 0) { Good 'every registry entry XP wrote is gone' }
    } elseif ($wrong -eq 0) {
        Good "every registry entry matches the plan ($($plan.entries.Count) entries)"
    }
    if (-not $Uninstall -and $plan.mode -eq 'selection-aware') {
        # The static submenu must be gone: two menus for one tool is worse than
        # one, and a leftover static tree is exactly what a reinstall leaves.
        foreach ($extension in '.pdf', '.xopp') {
            $stale = "HKCU:\Software\Classes\SystemFileAssociations\$extension\shell\XP"
            if (Test-Path $stale) { Bad "$extension still has the old static submenu" }
            else { Good "$extension has no stale static submenu" }
        }
    }
}

# A shortcut that survived a move is the same failure in a different place.
$shell = New-Object -ComObject WScript.Shell
$links = @((Join-Path ([Environment]::GetFolderPath('Desktop')) 'XP.lnk'),
           (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\XP.lnk'),
           (Join-Path $env:APPDATA 'Microsoft\Windows\SendTo\XP.lnk'))
# **The fourth shortcut is conditional.** `windows.py` writes one into
# `Documents\XP` when that folder already exists, and nobody verified it -- so a
# stale one there would survive a move silently.
$studies = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'XP'
if (Test-Path $studies) { $links += (Join-Path $studies 'XP.lnk') }
# The way back out: the installer leaves an Uninstall XP entry in the Start menu.
$links += (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Uninstall XP.lnk')
foreach ($link in $links) {
    $label = Split-Path (Split-Path $link -Parent) -Leaf
    if (-not (Test-Path $link)) {
        if ($Uninstall) { Good "$label shortcut removed" } else { Bad "$label shortcut is missing: $link" }
        continue
    }
    if ($Uninstall) { Bad "$label shortcut is still there: $link"; continue }
    # **All three fields, not just the arguments.** A shortcut whose target or
    # working directory survived a move still starts the wrong interpreter from
    # the wrong folder, and used to print `ok`.
    $shortcut = $shell.CreateShortcut($link)
    $expectedTarget = Join-Path (Split-Path $python -Parent) 'pythonw.exe'
    $isUninstaller = $link -like '*Uninstall XP.lnk'
    $argumentsOk = if ($isUninstaller) {
        $shortcut.Arguments -like '*uninstall.cmd*'
    } else {
        $shortcut.Arguments -like "*$launcher*"
    }
    $targetOk = if ($isUninstaller) {
        (-not $shortcut.TargetPath) -or ($shortcut.TargetPath -like '*cmd.exe')
    } else {
        (-not $shortcut.TargetPath) -or ($shortcut.TargetPath -eq $expectedTarget)
    }
    $folderOk = (-not $shortcut.WorkingDirectory) -or
                ($shortcut.WorkingDirectory.TrimEnd('\') -eq $root.TrimEnd('\'))
    if ($argumentsOk -and $targetOk -and $folderOk) { Good "$label shortcut" }
    else {
        Bad "$label shortcut points somewhere else (target '$($shortcut.TargetPath)', arguments '$($shortcut.Arguments)', folder '$($shortcut.WorkingDirectory)')"
    }
}

# ---- 5. What XP needs but does not install --------------------------------
Say ''
Say 'Alongside' 'White'
$xournal = @("$env:ProgramFiles\Xournal++\bin\xournalpp.exe",
             "${env:ProgramFiles(x86)}\Xournal++\bin\xournalpp.exe") |
           Where-Object { Test-Path $_ } | Select-Object -First 1
if ($xournal) { Good "Xournal++ at $xournal" }
else { Note 'Xournal++ was not found -- XP writes .xopp files either way, but test_xournalpp_opens_our_files.py needs it' }
# The same search `xp/where.py` does, and for the same reason: sample material
# gets moved and a written-down path goes stale without saying so. XP_MATERIAL
# wins; otherwise walk up from this folder through the folder names it may live
# under. Nothing absolute, so the project can be moved whole.
$material = $null
if ($env:XP_MATERIAL -and (Test-Path $env:XP_MATERIAL)) { $material = $env:XP_MATERIAL }
else {
    $tails = @('doing', 'material', 'samples')
    $folder = Get-Item -LiteralPath $root
    while ($folder -and -not $material) {
        foreach ($tail in $tails) {
            $candidate = Join-Path $folder.FullName $tail
            if (Test-Path $candidate) { $material = $candidate; break }
        }
        $folder = $folder.Parent
    }
}
if ($material) { Good "material at $material" }
else { Note 'sample material was not found; set XP_MATERIAL to the folder holding the .xopp notebooks and PDFs' }

# ---- 6. Say what happened -------------------------------------------------
Say ''
if ($problems.Count -eq 0) {
    Say 'Done. Right-click a .pdf or .xopp -- on Windows 11 under "Show more options".' 'Green'
    Say 'Moved the project? Run this script again from the new folder; that is the repair.' 'DarkGray'
} else {
    Say "$($problems.Count) thing(s) are wrong:" 'Red'
    foreach ($problem in $problems) { Say "  - $problem" 'Red' }
}
Say ''
if (-not $NoPause) { Read-Host 'Press Enter to close' }
if ($problems.Count -eq 0) { exit 0 } else { exit 1 }
