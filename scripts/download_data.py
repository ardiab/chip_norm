#!/usr/bin/env python
"""Script for downloading ENTEx data files."""

import argparse

from chipvi.utils.public_data_processing.download_entex_files import process_downloaded_files


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download BAM files and convert them to numpy arrays",
    )
    parser.add_argument(
        "--url_list_fpath",
        type=str,
        help="Path to a text file containing URLs to download",
    )
    
    return parser.parse_args()


def main() -> None:
    """Download and process ENTEx files."""
    args = parse_args()
    process_downloaded_files(args.url_list_fpath)


if __name__ == "__main__":
    main()