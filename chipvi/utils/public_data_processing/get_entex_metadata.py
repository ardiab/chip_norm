"""Download metadata jsons for ENTEx experiments.

JSONs contain information that is not available in the file and experiment spreadsheets
which can be downloaded from the ENCODE portal.
"""

import contextlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from chipvi.utils.path_helper import PathHelper


def download_json_files(exp_df: pd.DataFrame) -> list[str]:
    """Download the ENTEx metadata JSON files for the experiments specified in the input DataFrame.

    Args:
        exp_df (pd.DataFrame): A pandas DataFrame containing the experiments to download the
            metadata for. This is the experiment table that is downloaded from the ENCODE portal.

    Returns:
        list[str]: A list of paths to the downloaded JSON files.

    """
    json_fpaths = []

    if PathHelper.ENTEX_EXPERIMENT_JSON_DIR.exists():
        msg = f"Directory {PathHelper.ENTEX_EXPERIMENT_JSON_DIR} already exists."
        raise FileExistsError(msg)
    PathHelper.ENTEX_EXPERIMENT_JSON_DIR.mkdir()

    for exp_accession in tqdm(exp_df["Accession"].unique()):
        out_path = PathHelper.ENTEX_EXPERIMENT_JSON_DIR / f"{exp_accession}.json"
        json_url = f"https://www.encodeproject.org/experiments/{exp_accession}/?format=json"
        exp_json = requests.get(json_url).json()

        with out_path.open("w") as f:
            json.dump(exp_json, f)

        json_fpaths.append(str(out_path))

    return json_fpaths


class ENTExJSONParser:
    """Class for parsing the ENTEx metadata JSON files."""

    # TODO: Write a summary of the ENTEx JSON files for future reference.
    @classmethod
    def parse(cls, json_fpath: str) -> pd.DataFrame:
        """Parse the ENTEx metadata JSON file into a pandas DataFrame.

        Each row in the returned DataFrame corresponds to a single file in the experiment.
        Note that not all files will be BAM files.

        Args:
            json_fpath (str): The path to the experiment metadata JSON.

        Raises:
            ValueError: Samples from more than one donor were found in the experiment.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the metadata of each file in the experiment.

        """
        # Note: It seems that some experiments merge reads produced by different machines/platforms
        # into the same alignments file.
        # Example: https://www.encodeproject.org/experiments/ENCSR595BPR/
        # ['ENCFF855IEX', 'ENCFF451SFM', 'ENCFF503FQQ', 'ENCFF291HME', 'ENCFF053GAI', 'ENCFF121YSO',
        #  'ENCFF948BFT', 'ENCFF559CAG'] come from Illumina HiSeq 2500, and
        # ['ENCFF062OPB', 'ENCFF089PDM', 'ENCFF869FUF', 'ENCFF098OSY', 'ENCFF230WDC', 'ENCFF199GME',
        #  'ENCFF309HWC', 'ENCFF604HAJ'] come from Illumina NextSeq 500.
        with Path(json_fpath).open("r") as f:
            meta_dict = json.load(f)

        antibody_map = cls.get_antibody_map(meta_dict)
        platform_map = cls.get_platform_map(meta_dict)
        keys = [
            "accession",
            "date_created",
            ("submitted_by", "lab"),
            "file_format",
            "file_format_type",
            "output_type",
            "dataset",
            "assembly",
            "status",
            "biological_replicates",
            "technical_replicates",
            "biological_replicates_formatted",
            "donors",
            "assay_title",
            "assay_term_name",
            "target",
            "preferred_default",
            ("platform", "term_name"),
            ("replicate", "technical_replicate_number"),
            ("replicate", "antibody"),
            ("replicate", "replication_type"),
            "derived_from",
        ]

        data = []
        for f in meta_dict["files"]:
            file_dict = {}
            for k in keys:
                with contextlib.suppress(KeyError):
                    key_formatted = k if isinstance(k, str) else f"{k[0]}/{k[1]}"
                    val_formatted = f[k] if isinstance(k, str) else f[k[0]][k[1]]
                    file_dict[key_formatted] = val_formatted

            file_lab = f["lab"]["@id"]
            if file_lab == "/labs/encode-processing-pipeline/":
                file_dict["lab"] = None
            else:
                file_dict["lab"] = file_lab

            # The "analyses" field in the experiment JSON contains information on the ENCODE
            # pipelines that were used to generate the reads in the experiment.
            # The two file-specific ways that I found to get this information are
            # "pipeline_award_rfas" and "analysis_title".
            # Note that not all files have associated analyses. The files which do not have any
            # appear to mainly be raw reads (e.g. fastq files), but other file types can also
            # be missing analyses in the JSON.
            file_analyses = f["analyses"]
            encode_version = None
            analysis_title = None
            if len(file_analyses) > 0:
                encode_versions = []
                analysis_titles = []
                for a in file_analyses:
                    with contextlib.suppress(KeyError):
                        encode_versions += a["pipeline_award_rfas"]
                    with contextlib.suppress(KeyError):
                        analysis_titles.append(a["title"])
                # There should be at most one value for encode_version in the analyses field.
                if len(encode_versions) > 0:
                    if len(set(encode_versions)) != 1:
                        msg = (
                            f"Experiment {meta_dict['accession']} does not have exactly one encode "
                            f"version (found {encode_versions})."
                        )
                        raise ValueError(msg)
                    encode_version = encode_versions[0]
                # There should be at most one value for analysis_title in the analyses field.
                if len(analysis_titles) > 0:
                    if len(set(analysis_titles)) != 1:
                        msg = (
                            f"Experiment {meta_dict['accession']} does not have exactly one "
                            f"analysis title."
                        )
                        raise ValueError(msg)
                    analysis_title = analysis_titles[0]
            else:
                encode_version = None

            # If both encode version and analysis title are available, make sure they reference the
            # same ENCODE version.
            # Note that some files appear to be associated with custom analyses that do not specify
            # the ENCODE version, and such files are excluded from this check.
            if (
                (encode_version is not None)
                and (analysis_title is not None)
                and ("lab custom" not in analysis_title.lower())
                and (encode_version != analysis_title.split(" ")[0])
            ):
                msg = (
                    f"Experiment {meta_dict['accession']} has encode version {encode_version} and "
                    f"analysis title {analysis_title}."
                )
                raise ValueError(msg)

            file_dict["encode_version"] = encode_version
            file_dict["analysis_title"] = analysis_title

            file_supersedes = f.get("supersedes", [])
            file_supersedes = ",".join(file_supersedes) if len(file_supersedes) > 0 else None
            file_superseded_by = f.get("superseded_by", [])
            file_superseded_by = (
                ",".join(file_superseded_by) if len(file_superseded_by) > 0 else None
            )
            file_dict["supersedes"] = file_supersedes
            file_dict["superseded_by"] = file_superseded_by

            data.append(file_dict)

        meta_df = pd.DataFrame(data)
        donors = np.concatenate(meta_df["donors"].values).flatten()
        # There should be exactly one donor specified in the experiment's metadata.
        if len(set(donors)) != 1:
            msg = f"Experiment {meta_dict['accession']} does not have exactly one donor."
            raise ValueError(msg)
        donor = donors[0]
        meta_df["donor"] = donor

        meta_df["experiment"] = meta_dict["accession"]
        meta_df["assay_term_name"] = meta_dict["assay_term_name"]
        meta_df["biosample_summary"] = meta_dict["simple_biosample_summary"]

        meta_df["biosample_term_name"] = meta_dict["biosample_ontology"]["term_name"]
        biosample_term_name_synonyms = "|".join(meta_dict["biosample_ontology"]["synonyms"])
        if biosample_term_name_synonyms == "":
            biosample_term_name_synonyms = None
        meta_df["biosample_term_name_synonyms"] = biosample_term_name_synonyms
        biosample_organ_systems = "|".join(meta_dict["biosample_ontology"]["organ_slims"])
        if biosample_organ_systems == "":
            biosample_organ_systems = None
        meta_df["biosample_organ_systems"] = biosample_organ_systems
        meta_df["possible_controls"] = "|".join(
            d["accession"] for d in meta_dict["possible_controls"]
        )
        # Get the antibody and platform metadata used to generate the reads associated with
        # each file in the experiment (note that in the case of e.g. biological replicates,
        # there can be multiple alignments which reference different reads that have different
        # antibodies or sequencing platforms).
        for idx, row in meta_df.iterrows():
            file_antibodies = []
            file_platforms = []
            for rep in row["technical_replicates"]:
                biol_rep, tech_rep = rep.split("_")
                biol_rep, tech_rep = int(biol_rep), int(tech_rep)

                with contextlib.suppress(KeyError):
                    file_antibodies.append(antibody_map[biol_rep][tech_rep]["antibody"])
                with contextlib.suppress(KeyError):
                    file_platforms += platform_map[biol_rep][tech_rep]

            meta_df.loc[idx, "antibody"] = ",".join(sorted(set(file_antibodies)))
            meta_df.loc[idx, "platform"] = ",".join(sorted(set(file_platforms)))

        # TODO: Move this comment block to ENTExMetaHelper when it is ready.
        # Note that the majority of experiments will only have one set of files that are part of
        # ENCODE4. However, I found 2 experiments that had more than one non-archived set of
        # alignments (part of ENCODE4, but different versions of it). These experiments are:
        # /experiments/ENCSR164POT/
        # /experiments/ENCSR888RBQ/
        # It is possible that similar cases exist for other file types.
        # In such cases, files from the latest ENCODE4 version will be used.
        return meta_df

    @classmethod
    def get_antibody_map(cls, meta_dict: dict) -> dict:
        """Extract information on the antibodies used in the experiment.

        The output is a dictionary structured as follows:
        {biological_replicate_number: {technical_replicate_number: {antibody: ab, biosample: bs,
        library: lib}}}

        Args:
            meta_dict (dict): The experiment JSON loaded as a dictionary.

        Raises:
            ValueError: A technical replicate has multiple entries in the dictionary.

        Returns:
            dict: A dictionary containing the information on the antibodies used in the experiment,
                broken down by biological and technical replicate number.

        """
        # The experiment JSON contains a list of all the files associated with the experiment.
        # Each file is associated with a set of replicates, and we want to keep track of which
        # replicates are associated with which antibodies and biosamples.
        antibody_map = defaultdict(dict)

        for rep_dict in meta_dict["replicates"]:
            with contextlib.suppress(KeyError):
                # Each replicate is associated with one antibody and one biosample.
                ab = rep_dict["antibody"]["accession"]
                bs = rep_dict["library"]["biosample"]["@id"]
                bs = bs.split("/")[-2]
                lib = rep_dict["library"]["@id"]
                lib = lib.split("/")[-2]
                tech_rep_num = rep_dict["technical_replicate_number"]
                biol_rep_num = rep_dict["biological_replicate_number"]
                if tech_rep_num in antibody_map[biol_rep_num]:
                    msg = f"tech_rep_num {tech_rep_num} is already in antibody_map"
                    raise ValueError(msg)
                antibody_map[biol_rep_num][tech_rep_num] = {
                    "antibody": ab,
                    "biosample": bs,
                    "library": lib,
                }

        return antibody_map

    @classmethod
    def get_platform_map(cls, meta_dict: dict) -> dict:
        """Extract information on the sequencing platforms used to generate the reads in the experiment.

        The output is a dictionary structured as follows:
        {biological_replicate_number: {technical_replicate_number: [platform1, platform2, ...]}}

        Args:
            meta_dict (dict): The experiment JSON loaded as a dictionary.

        Returns:
            dict: A dictionary containing the sequencing platforms used in the experiment, broken
                down by biological and technical replicate number.

        """
        platform_map = defaultdict(lambda: defaultdict(list))
        for file_dict in meta_dict["files"]:
            with contextlib.suppress(KeyError):
                if "platform" in file_dict:
                    platform = file_dict["platform"]["title"]
                    tech_rep_num = file_dict["replicate"]["technical_replicate_number"]
                    biol_rep_num = file_dict["replicate"]["biological_replicate_number"]

                    platform_map[biol_rep_num][tech_rep_num].append(platform)

        return platform_map


def format_metadata_df(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Format the metadata DataFrame.

    Args:
        metadata_df (pd.DataFrame): The metadata DataFrame.

    Returns:
        pd.DataFrame: The formatted metadata DataFrame.

    """
    metadata_df["target"] = metadata_df["target"].apply(
        lambda t: t.split("/")[2].replace("-human", "") if pd.notna(t) else np.nan
    )
    # Filter out archived experiments, experiments from older genome assemblies, and experiments
    # that were released in older ENCODE versions.

    metadata_df = metadata_df[
        (metadata_df["status"] == "released")
        & (metadata_df["assembly"] == "GRCh38")
        & (metadata_df["encode_version"] == "ENCODE4")
    ]
    # Because we are using the non-archived files from the latest ENCODE version, the filtered
    # files should not be superseded by any other files.
    # The superseded_by field is parsed from a file-specific field in the ENTEx experiment
    # metadata JSON files.
    if metadata_df["superseded_by"].notna().any():
        msg = (
            f"Found {metadata_df['superseded_by'].notna().sum()} files that are "
            f"superseded by other files after filtering."
        )
        raise ValueError(msg)

    return metadata_df


def process_entex_metadata() -> None:
    """Process ENTEx metadata and save to pickle file."""
    # The first row contains the link used to download the experiment table.
    experiment_df = pd.read_csv(PathHelper.ENTEX_EXPERIMENT_TABLE_FPATH, sep="\t", skiprows=1)
    # json_fpaths = download_json_files(exp_df=experiment_df)
    json_fpaths = list(PathHelper.ENTEX_EXPERIMENT_JSON_DIR.glob("*.json"))
    exp_meta_dfs = [ENTExJSONParser.parse(str(json_fpath)) for json_fpath in json_fpaths]
    metadata_df = pd.concat(exp_meta_dfs)
    metadata_df = format_metadata_df(metadata_df)
    metadata_df.to_pickle(PathHelper.ENTEX_PROC_META_DF_FPATH)
