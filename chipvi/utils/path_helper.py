"""Path management for the chipvi package."""

from __future__ import annotations

from pathlib import Path
from omegaconf import DictConfig


class PathHelper:
    """Configuration-driven path helper for the chipvi package."""

    def __init__(self, cfg: DictConfig):
        """Initialize PathHelper with configuration.
        
        Args:
            cfg: Configuration object containing path definitions.
        """
        # Core paths from configuration
        self.project_root = Path(cfg.paths.project_root)
        self.data_base = Path(cfg.paths.data_base)
        self.raw_data_dir = Path(cfg.paths.data_raw)
        self.proc_data_dir = Path(cfg.paths.data_processed)
        
        # ENTEx paths
        self.entex_base = Path(cfg.paths.entex_base)
        self.entex_raw_file_dir = Path(cfg.paths.entex_raw)
        self.entex_proc_file_dir = Path(cfg.paths.entex_processed)
        
        # Output paths
        self.outputs = Path(cfg.paths.outputs)
        
        # ENTEx metadata paths (derived from raw data dir)
        self.entex_experiment_json_dir = self.raw_data_dir / "experiment_jsons"
        self.entex_experiment_table_fpath = self.raw_data_dir / "experiment_report_2025_5_24_19h_59m.tsv"
        self.entex_experiment_tsv_fpath = self.raw_data_dir / "entex_experiment_info.tsv"
        self.entex_file_tsv_fpath = self.raw_data_dir / "entex_file_info.tsv"
        self.entex_proc_meta_df_fpath = self.proc_data_dir / "entex_proc_meta_info.pkl"
        self.entex_merged_meta_fpath = self.proc_data_dir / "entex_merged_meta.pkl"
        self.entex_proc_meta_fpath = self.proc_data_dir / "entex_proc_meta.pkl"
        self.chrom_size_hg38_fpath = self.raw_data_dir / "GRCh38_EBV.chrom.sizes.tsv"
        
        # Non-ENTEx external data paths (derived from raw data dir)
        self.encode_blacklist_bed_fpath = self.raw_data_dir / "ENCFF356LFX.bed"
        self.raw_gencode_gtf_fpath = self.raw_data_dir / "gencode.v29.basic.annotation.gtf"
        self.gc_track_bw_fpath = self.raw_data_dir / "hg38.gc5Base.bw"
        self.mappability_single_track_bw_fpath = self.raw_data_dir / "k24.Unique.Mappability.bb"
        self.mappability_multi_track_bw_fpath = self.raw_data_dir / "k24.Umap.MultiTrackMappability.bw"
        self.gc_dir_fpath = self.proc_data_dir / "gc_track"
        
        # External scripts paths (derived from project root)
        self.external_script_dir = self.project_root / "scripts" / "external"
        self.fetch_chrom_sizes_fpath = self.external_script_dir / "fetchChromSizes.sh"
        self.bedgraph_to_bigwig_linux_fpath = self.external_script_dir / "bedGraphToBigWig"

    def get_entex_experiment_json_fpath(
        self,
        accession: str,
    ) -> Path:
        """Get the path to an ENCODE experiment JSON file.

        Args:
            accession (str): The ENCODE accession of the experiment.

        Returns:
            Path: The path to the experiment JSON file.

        """
        return self.entex_experiment_json_dir / f"{accession}.json"

    def get_entex_accession_fpaths(
        self,
        accession: str,
    ) -> list[Path]:
        """Get the paths to all files matching an accession.

        Args:
            accession (str): The ENCODE accession of the experiment.

        Returns:
            list[Path]: The paths to all files matching the accession.

        """
        dirs_to_check = [self.entex_raw_file_dir, self.entex_proc_file_dir]
        all_matches = []
        for d in dirs_to_check:
            all_matches += list(d.glob(f"**/*{accession}*"))

        # Filter out directories before returning.
        return [f for f in all_matches if f.is_file()]
