#!/usr/bin/env python3
"""
Entry point for Lighthouse QC GUI application.
"""

import sys
import argparse
from core.native_threading import configure_native_thread_environment


def main():
    """Parse CLI args, construct QApplication, show MainWindow, exec."""
    configure_native_thread_environment()

    parser = argparse.ArgumentParser(description="Lighthouse QC Standalone GUI")
    parser.add_argument('--dat', type=str, help='Path to .dat/.bin file')
    parser.add_argument('--n_channels', type=int, help='Number of channels')

    args = parser.parse_args()

    # Run the app
    from gui.app import run

    run(sys.argv, default_dat=args.dat, default_n_channels=args.n_channels)


if __name__ == '__main__':
    main()
