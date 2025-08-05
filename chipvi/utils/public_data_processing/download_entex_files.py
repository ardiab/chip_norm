"""Functions for downloading and processing BAM files."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from chipvi.utils.path_helper import PathHelper
from chipvi.utils.tmux import run_in_tmux

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def _download_url_list(url_list_fpath: str) -> dict[str, list[str]]:
    """Download files specified by a text file containing URLs and return a dictionary of the downloaded files.

    Args:
        url_list_fpath (str): The path to a text file containing URLs to download.

    Returns:
        dict[str, list[str]]: A dictionary containing lists of downloaded files, keyed by file extension.

    """
    # Download the files.
    os.system(
        f"xargs -L 1 curl -O -J -L --output-dir {PathHelper.ENTEX_RAW_FILE_DIR} < {url_list_fpath}",
    )

    # Create a dictionary of the downloaded files, keyed by file extension.
    # This dictionary will be used to find and post-process all downloaded BAM files.
    file_dict = {}
    with Path(url_list_fpath).open("r") as f:
        all_urls = f.read().split("\n")
    downloaded_accessions = [url.split("/")[-1].split(".")[0] for url in all_urls]
    downloaded_accessions = [a for a in downloaded_accessions if a != ""]

    for a in downloaded_accessions:
        matches = PathHelper.get_entex_accession_fpaths(a)
        if len(matches) != 1:
            msg = f"Expected 1 file matching {a}, found {len(matches)} ({matches})."
            raise OSError(msg)
        # Cast the path to a string to get the full extension. Path objects don't appear to stem
        # properly when an extension has more than 1 dot (e.g. ".bed.gz").
        extension = ".".join(str(matches[0]).split("/")[-1].split(".")[1:])
        if extension not in file_dict:
            file_dict[extension] = []
        file_dict[extension].append(matches[0])

    return file_dict


def _split_accessions_for_parallel_processing(
    accession_list: list[str],
    file_type_list: list[str],
    n_splits: int,
) -> list[list[str]]:
    """Create n_splits lists of accessions from a single list of accessions.

    Every split will contain roughly the same number of accessions per extension.

    Args:
        accession_list (list[str]): A list of accessions.
        file_type_list (list[str]): A list of each corresponding accession's file type.
        n_splits (int): The number of splits to create.

    Returns:
        list[list[str]]: A list of n_splits lists of accessions.

    """
    # Create a dict to split accessions by file extension
    extension_dict = {}
    for accession, file_type in zip(accession_list, file_type_list):
        if file_type not in extension_dict:
            extension_dict[file_type] = []
        extension_dict[file_type].append((accession, file_type))

    # Split each list of accessions by extension into n_splits
    for file_type, file_type_accession_list in extension_dict.items():
        extension_dict[file_type] = np.array_split(file_type_accession_list, n_splits)

    # Combine the splits for each extension into a list of splits
    split_lists = list(extension_dict.values())

    return [np.concatenate([split_list[i] for split_list in split_lists]) for i in range(n_splits)]


def _write_url_list_to_file(
    accession_list: list[str],
    file_type_list: list[str],
    fpath: str,
) -> None:
    """Write a list of URLs to a text file.

    Args:
        accession_list (list[str]): A list of accessions.
        file_type_list (list[str]): A list of each corresponding accession's file type.
        fpath (str): The path to the text file to which the URLs will be written.

    """
    # This mapping from file type to extension was determined experimentally.
    file_type_extension_map = {
        "bam": "bam",
        "bed": "bed.gz",
        "bedpe": "bedpe.gz",
        "bigBed": "bigBed",
        "bigWig": "bigWig",
        "gff": "gff.gz",
        "gtf": "gtf.gz",
        "hic": "hic",
        "pairs": "pairs.gz",
        "starch": "starch",
        "tar": "tar.gz",
        "tsv": "tsv",
        "vcf": "vcf.gz",
    }

    # Any file can be downloaded from the ENCODE portal using the following URL template.
    url_template = "https://www.encodeproject.org/files/{}/@@download/{}.{}"
    accession_urls = []
    for accession, file_type in zip(accession_list, file_type_list):
        accession_extension = file_type_extension_map[file_type]
        accession_url = url_template.format(accession, accession, accession_extension)
        accession_urls.append(accession_url)

    with Path(fpath).open("w") as out_f:
        out_f.write("\n".join(accession_urls))


def download_and_process_entex_files(
    accession_list: list[str],
    file_type_list: list[str],
    n_splits: int,
    overwrite_existing: bool = False,
    filter_downloaded: bool = False,
    print_only: bool = False,
) -> None:
    """Download a list of ENTEx files and convert any BAM files to numpy arrays.

    If n_splits > 1, create slurm batch jobs to download and process the files in parallel.

    Args:
        accession_list (list[str]): A list of accessions.
        file_type_list (list[str]): A list of each corresponding accession's file type.
        n_splits (int): The number of splits to create.
        job_time (str, optional): The time limit for each slurm job. Defaults to '12:00:00'.
        job_memory (str, optional): The memory limit for each slurm job. Defaults to '64G'.
        overwrite_existing (bool, optional): Whether to overwrite existing files. Defaults to False.
        filter_downloaded (bool, optional): Whether to filter out already downloaded files. Defaults to False.

    Raises:
        IOError: An error is raised if any of the files in the URL list are already downloaded.

    """
    downloaded_accessions = []
    for accession in accession_list:
        matches = PathHelper.get_entex_accession_fpaths(accession)
        if len(matches) > 0:
            downloaded_accessions.append(accession)

    if len(downloaded_accessions) > 0 and not overwrite_existing and not filter_downloaded:
        msg = (
            "Some specified accessions have already been downloaded, and both overwrite and filter flags are False. "
            "Exiting."
        )
        raise OSError(msg)

    if filter_downloaded and overwrite_existing:
        msg = "At most one of the filter and overwrite flags can be True."
        raise OSError(msg)
    if len(downloaded_accessions) > 0 and filter_downloaded:
        # Create sets for efficient lookup
        downloaded_set = set(downloaded_accessions)
        # Filter out downloaded accessions and their corresponding file types
        filtered_pairs = [
            (acc, ft)
            for acc, ft in zip(accession_list, file_type_list)
            if acc not in downloaded_set
        ]
        logger.info(
            "Filtered %i/%i accessions (%i remaining).",
            len(accession_list) - len(downloaded_set),
            len(accession_list),
            len(filtered_pairs),
        )
        # Unzip the filtered pairs back into separate lists
        if filtered_pairs:
            accession_list, file_type_list = zip(*filtered_pairs)
        else:
            msg = "No accessions to download after filtering."
            raise OSError(msg)
        logger.warning("Filtering downloaded files.")
    elif len(downloaded_accessions) > 0 and overwrite_existing:
        logger.warning("Ignoring existing files and potentially overwriting.")

    if print_only:
        return

    # Single-process mode.
    if n_splits == 1:
        logger.info("Downloading and processing all files in a single process.")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        split_list_txt_fpath = Path(PathHelper.ENTEX_RAW_FILE_DIR) / f"url_list_{timestamp}.txt"
        _write_url_list_to_file(accession_list, file_type_list, split_list_txt_fpath)
        process_downloaded_files(split_list_txt_fpath)
    # Batch mode (call this script multiple times in parallel, once per split).
    else:
        accession_splits = _split_accessions_for_parallel_processing(
            accession_list, file_type_list, n_splits
        )
        logger.info("Splitting accessions into %s splits.", n_splits)
        logger.info("Accession splits: %s", accession_splits)
        job_info_dir = PathHelper.ENTEX_RAW_FILE_DIR / "job_info"
        if job_info_dir.is_dir():
            msg = f"Job info directory {job_info_dir} already exists"
            raise FileExistsError(msg)
        job_info_dir.mkdir()

        commands = []

        for i, split_list in enumerate(accession_splits):
            split_list_txt_fpath = job_info_dir / f"url_list_{i}.txt"
            # split_list is a list of tuples (accession, file_type).
            accessions, file_type_list = zip(*split_list)
            _write_url_list_to_file(
                accession_list=accessions,
                file_type_list=file_type_list,
                fpath=split_list_txt_fpath,
            )

            # This will need to be updated to use the new script
            command = f"python scripts/download_data.py --url_list_fpath {split_list_txt_fpath}"
            commands.append(command)

        run_in_tmux("download_and_process_entex", commands=commands)


def process_bam_file(bam_fpath: Path) -> None:
    """Convert a BAM file to a BED file containing the number of fragments per bin and the average MAPQ value per bin.

    Args:
        bam_fpath (str): The path to the BAM file to process.

    """
    t1 = time.time()
    if not bam_fpath.name.endswith(".bam"):
        msg = f"Expected BAM file, got {bam_fpath}."
        raise ValueError(msg)

    # The BAM file will be in the raw data directory (PathHelper.ENTEX_RAW_FILE_DIR)
    bam_fname = bam_fpath.name.replace(".bam", "")
    # The processed BAM file will be in the processed data directory (PathHelper.ENTEX_PROC_FILE_DIR)
    out_dirpath = PathHelper.ENTEX_PROC_FILE_DIR / bam_fname
    binned_bed_fpath = out_dirpath / f"{bam_fname}_binned_200.bed"
    out_dirpath.mkdir()

    # Call a bash script which processes the BAM file into a BED file that contains:
    # 1. The maximum number of fragments aligned to any position in each bin
    # 2. The weighted average MAPQ value of all fragments aligned to each bin
    # The script ("convert_bam_to_binned_bed.sh") takes the following arguments:
    # 1. The path to the original BAM file
    # 2. The path to the output BED file
    # 3. The path to the fetch_chrom_sizes script
    convert_bam_to_binned_bed_script_fpath = Path(__file__).parent / "process_entex_bam.sh"
    convert_bam_to_binned_bed_command = (
        f"bash {convert_bam_to_binned_bed_script_fpath} "
        f"{bam_fpath} "
        f"{binned_bed_fpath} "
        f"{PathHelper.FETCH_CHROM_SIZES_FPATH} "
        f"hg38 "
        f"200"
    )
    logger.info("Running command: %s", convert_bam_to_binned_bed_command)
    os.system(convert_bam_to_binned_bed_command)
    logger.info("Finished processing %s in %.3f seconds", bam_fpath, time.time() - t1)


def process_downloaded_files(url_list: str) -> None:
    """Download the URLs in the given list and convert any BAM files to numpy arrays.

    Args:
        url_list (str): The path to a text file containing URLs to download.

    """
    downloaded_file_dict = _download_url_list(url_list)
    for i, bam_fpath in enumerate(downloaded_file_dict.get("bam", [])):
        t1 = time.time()
        process_bam_file(bam_fpath)
        logger.info(
            "Processed BAM %i/%i in %.2f seconds.",
            i + 1,
            len(downloaded_file_dict["bam"]),
            time.time() - t1,
        )
