#!/usr/bin/env python
"""Script for processing ENTEx metadata."""

from chipvi.utils.public_data_processing.get_entex_metadata import process_entex_metadata


def main() -> None:
    """Process ENTEx metadata."""
    process_entex_metadata()


if __name__ == "__main__":
    main()