# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Calcify.

Build with:   pyinstaller Calcify.spec --noconfirm

Produces:
  - macOS:   dist/Calcify.app   (double-clickable bundle)
  - Windows: dist/Calcify/Calcify.exe  (one-folder app)
"""

import sys
from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs,
)

block_cipher = None

# --- Make sure nothing matplotlib/scipy/pandas/numpy-related gets dropped ---
hiddenimports = []
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('pandas')
# Explicit belt-and-suspenders for the exact functions used on the
# crash paths (peak detection, smoothing, filtering, interpolation).
hiddenimports += [
    'scipy.signal',
    'scipy.signal._savitzky_golay',
    'scipy.signal._peak_finding',
    'scipy.signal._peak_finding_utils',
    'scipy.interpolate',
    'scipy.special',
    'scipy.special._ufuncs',
    'scipy.special._cdflib',
    'scipy.linalg',
    'scipy._lib.array_api_compat.numpy.fft',
    'scipy._cyutility',
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.window.aggregations',
]

# Data files + compiled libs (fonts, mpl-data, scipy/numpy .so/.dylib/.dll).
datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('scipy')

binaries = []
binaries += collect_dynamic_libs('scipy')
binaries += collect_dynamic_libs('numpy')

# Pick the platform icon (ignored gracefully if missing).
icon_file = 'assets/Calcify.icns' if sys.platform == 'darwin' else 'assets/Calcify.ico'

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'PySide2', 'PySide6', 'PyQt6'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Calcify',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # windowed app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,    # lets macOS "open with" / drag-drop pass file args
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Calcify',
)

# macOS .app bundle
app = BUNDLE(
    coll,
    name='Calcify.app',
    icon=icon_file,
    bundle_identifier='com.calcify.app',
    info_plist={
        'CFBundleName': 'Calcify',
        'CFBundleDisplayName': 'Calcify',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
    },
)
