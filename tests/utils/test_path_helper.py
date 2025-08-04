"""Tests for PathHelper class."""

import pytest
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from chipvi.utils.path_helper import PathHelper


def test_path_helper_resolves_paths_from_config():
    """Test that PathHelper correctly resolves paths from configuration."""
    # Create a simple test configuration
    test_config = {
        "paths": {
            "project_root": "/test/project",
            "data_base": "/test/project/data",
            "data_raw": "/test/project/data/raw",
            "data_processed": "/test/project/data/processed",
            "entex_base": "/test/project/data/entex_files",
            "entex_raw": "/test/project/data/entex_files/raw",
            "entex_processed": "/test/project/data/entex_files/proc",
            "outputs": "/test/project/outputs"
        }
    }
    cfg = OmegaConf.create(test_config)
    
    # Instantiate PathHelper with the configuration
    path_helper = PathHelper(cfg)
    
    # Assert that raw_data_dir attribute resolves to absolute path of data/raw
    expected_raw_data_dir = Path("/test/project/data/raw")
    assert path_helper.raw_data_dir == expected_raw_data_dir
    
    # Assert that entex_proc_file_dir resolves to absolute path of data/entex_files/proc
    expected_entex_proc_dir = Path("/test/project/data/entex_files/proc")
    assert path_helper.entex_proc_file_dir == expected_entex_proc_dir