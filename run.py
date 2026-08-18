#!/usr/bin/env python3
"""
Entry point for Lighthouse QC GUI application.

Also supports a minimal headless Phy/KS export of a previously saved
results dict via --export-phy (see core.ks_export).
"""

import sys

import warnings
warnings.filterwarnings("ignore", message="overflow encountered in cast", category=RuntimeWarning, module="pyqtgraph.*")

import argparse
from core import configure_native_thread_environment


def main():
    """Parse CLI args, construct QApplication, show MainWindow, exec."""
    configure_native_thread_environment()

    parser = argparse.ArgumentParser(description="Lighthouse QC Standalone GUI")
    parser.add_argument('--dat', type=str, help='Path to .dat/.bin file')
    parser.add_argument('--n_channels', type=int, help='Number of channels')
    parser.add_argument(
        '--export-phy',
        type=str,
        default=None,
        metavar='OUT_DIR',
        help='(Reserved) export path hint shown in the GUI Export Phy/KS dialog',
    )

    args = parser.parse_args()

    # Run the app
    from gui.main_window import run

    run(sys.argv, default_dat=args.dat, default_n_channels=args.n_channels)


if __name__ == '__main__':
    main()
