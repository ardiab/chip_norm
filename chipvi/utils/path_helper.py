"""Path management for the chipvi package."""

from __future__ import annotations

from pathlib import Path


class PathHelper:
    """Path helper for the chipvi package."""

    BASE_PACKAGE_DIR = Path(__file__).parent.parent.parent

    # Data (small files)
    BASE_DATA_DIR = BASE_PACKAGE_DIR / "data"
    RAW_DATA_DIR = BASE_DATA_DIR / "raw"
    PROC_DATA_DIR = BASE_DATA_DIR / "processed"

    # Data (large files)
    # ENTEX_FILE_DIR = Path("/scratch/adiab/entex_files/")
    ENTEX_FILE_DIR = BASE_DATA_DIR / "entex_files"

    # ENTEx large files (bam, bed, etc)
    ENTEX_RAW_FILE_DIR = ENTEX_FILE_DIR / "raw"
    ENTEX_PROC_FILE_DIR = ENTEX_FILE_DIR / "proc"

    # ENTEx metadata
    ENTEX_EXPERIMENT_JSON_DIR = RAW_DATA_DIR / "experiment_jsons"
    ENTEX_EXPERIMENT_TABLE_FPATH = RAW_DATA_DIR / "experiment_report_2025_5_24_19h_59m.tsv"
    ENTEX_EXPERIMENT_TSV_FPATH = RAW_DATA_DIR / "entex_experiment_info.tsv"
    ENTEX_FILE_TSV_FPATH = RAW_DATA_DIR / "entex_file_info.tsv"
    ENTEX_PROC_META_DF_FPATH = PROC_DATA_DIR / "entex_proc_meta_info.pkl"
    # https://www.encodeproject.org/annotations/ENCSR636HFF/
    ENTEX_MERGED_META_FPATH = PROC_DATA_DIR / "entex_merged_meta.pkl"
    ENTEX_PROC_META_FPATH = PROC_DATA_DIR / "entex_proc_meta.pkl"
    CHROM_SIZE_HG38_FPATH = RAW_DATA_DIR / "GRCh38_EBV.chrom.sizes.tsv"

    # Non-ENTEx external data
    ENCODE_BLACKLIST_BED_FPATH = RAW_DATA_DIR / "ENCFF356LFX.bed"
    RAW_GENCODE_GTF_FPATH = RAW_DATA_DIR / "gencode.v29.basic.annotation.gtf"
    GC_TRACK_BW_FPATH = RAW_DATA_DIR / "hg38.gc5Base.bw"
    MAPPABILITY_SINGLE_TRACK_BW_FPATH = RAW_DATA_DIR / "k24.Unique.Mappability.bb"
    MAPPABILITY_MULTI_TRACK_BW_FPATH = RAW_DATA_DIR / "k24.Umap.MultiTrackMappability.bw"
    GC_DIR_FPATH = PROC_DATA_DIR / "gc_track"

    # External scripts
    EXTERNAL_SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts" / "external"
    FETCH_CHROM_SIZES_FPATH = EXTERNAL_SCRIPT_DIR / "fetchChromSizes.sh"
    BEDGRAPH_TO_BIGWIG_LINUX_FPATH = EXTERNAL_SCRIPT_DIR / "bedGraphToBigWig"

    @classmethod
    def get_entex_experiment_json_fpath(
        cls,
        accession: str,
    ) -> Path:
        """Get the path to an ENCODE experiment JSON file.

        Args:
            accession (str): The ENCODE accession of the experiment.

        Returns:
            Path: The path to the experiment JSON file.

        """
        return Path(cls.ENTEX_EXPERIMENT_JSON_DIR) / f"{accession}.json"

    @classmethod
    def get_entex_accession_fpaths(
        cls,
        accession: str,
    ) -> list[Path]:
        """Get the paths to all files matching an accession.

        Args:
            accession (str): The ENCODE accession of the experiment.

        Returns:
            list[Path]: The paths to all files matching the accession.

        """
        dirs_to_check = [Path(cls.ENTEX_RAW_FILE_DIR), Path(cls.ENTEX_PROC_FILE_DIR)]
        all_matches = []
        for d in dirs_to_check:
            all_matches += list(d.glob(f"**/*{accession}*"))

        # Filter out directories before returning.
        return [f for f in all_matches if f.is_file()]
