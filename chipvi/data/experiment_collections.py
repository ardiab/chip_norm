from __future__ import annotations

import pandas as pd

from chipvi.utils.path_helper import PathHelper

META_DF = pd.read_pickle(PathHelper.ENTEX_PROC_META_DF_FPATH)
META_DF["donor_biosample"] = META_DF["donor"] + "_" + META_DF["biosample_term_name"]


class CTCFSmallCollection:
    """A collection of CTCF experiments with multiple alignments per donor per tissue."""

    @classmethod
    def get_alignment_accessions(cls) -> list[str]:
        """Get the accessions of the CTCF alignments for the CTCF small collection.

        Returns:
            list[str]: The accessions of the CTCF alignments for the CTCF small collection.

        """
        ctcf_alignments = META_DF[
            (META_DF["target"] == "CTCF") & (META_DF["output_type"] == "alignments")
        ]
        ctcf_alignments_by_donor = ctcf_alignments.groupby(["donor", "biosample_term_name"]).size()
        ctcf_alignments_by_donor_filtered = (
            ctcf_alignments_by_donor[ctcf_alignments_by_donor >= 2]
            .reset_index()
            .groupby("donor")
            .head(2)
        )

        donor_biosample_term_name_pairs = [
            f"{donor}_{biosample_term_name}"
            for donor, biosample_term_name in zip(
                ctcf_alignments_by_donor_filtered["donor"],
                ctcf_alignments_by_donor_filtered["biosample_term_name"],
            )
        ]
        ctcf_alignments_filtered = ctcf_alignments[
            ctcf_alignments["donor_biosample"].isin(donor_biosample_term_name_pairs)
        ]

        return ctcf_alignments_filtered["accession"].to_numpy()

    @classmethod
    def get_control_accessions(cls) -> list[str]:
        """Get the accessions of the control ChIP-seq alignments for the CTCF small collection.

        Returns:
            list[str]: The accessions of the control ChIP-seq alignments for the CTCF small
                collection.

        """
        alignment_accessions = cls.get_alignment_accessions()
        possible_controls = META_DF[META_DF["accession"].isin(alignment_accessions)][
            "possible_controls"
        ].to_numpy()
        if not set(possible_controls).issubset(META_DF["experiment"].values):
            missing_controls = set(possible_controls) - set(META_DF["experiment"].values)
            msg = f"Some possible controls are not in the metadata dataframe: {missing_controls}"
            raise ValueError(msg)

        control_df = META_DF[
            (META_DF["experiment"].isin(possible_controls))
            & (META_DF["output_type"] == "alignments")
        ]
        if (control_df["assay_title"] != "Control ChIP-seq").any():
            msg = "Some possible controls are not control ChIP-seq."
            raise ValueError(msg)

        return control_df["accession"].to_numpy()

    @classmethod
    def get_alignment_control_pairs(cls, pair_replicate: bool = False) -> list[tuple[str, ...]]:
        """Get the alignment and control pairs for the CTCF small collection.

        Returns:
            list[tuple[str, str]]: The alignment and control pairs for the CTCF small collection.
        """
        pairs = []

        alignment_accessions = cls.get_alignment_accessions()
        control_accessions = cls.get_control_accessions()

        alignment_df = META_DF[META_DF["accession"].isin(alignment_accessions)]
        if alignment_df.shape[0] != len(alignment_accessions):
            msg = f"Alignment DF shape: {alignment_df.shape}, alignment accessions: {len(alignment_accessions)}"
            raise ValueError(msg)

        for idx, row in alignment_df.iterrows():
            possible_controls = row["possible_controls"]
            if "," in possible_controls:
                msg = f"Possible controls is a list: {possible_controls}"
                raise ValueError(msg)
            control_df = META_DF[
                (META_DF["experiment"] == possible_controls)
                & (META_DF["output_type"] == "alignments")
            ]
            if not set(control_df["accession"].values).issubset(control_accessions):
                msg = (
                    f"Some possible controls are not in the metadata dataframe: {possible_controls}"
                )
                raise ValueError(msg)

            pairs.append((row["accession"], control_df["accession"].values[0]))

        if pair_replicate:
            paired_reps = []
            for donor_biosample, donor_biosample_df in alignment_df.groupby(["donor_biosample"]):
                donor_biosample_accessions = donor_biosample_df["accession"].values
                if len(donor_biosample_accessions) != 2:
                    msg = f"Donor biosample {donor_biosample} has {len(donor_biosample_accessions)} alignments, expected 2"
                    raise ValueError(msg)
                r1_pair = [p for p in pairs if p[0] == donor_biosample_accessions[0]]
                r2_pair = [p for p in pairs if p[0] == donor_biosample_accessions[1]]
                if len(r1_pair) != 1 or len(r2_pair) != 1:
                    msg = f"Donor biosample {donor_biosample} has {len(r1_pair)} r1 pairs and {len(r2_pair)} r2 pairs"
                    raise ValueError(msg)
                paired_reps.append((r1_pair[0][0], r1_pair[0][1], r2_pair[0][0], r2_pair[0][1]))

            return paired_reps

        return pairs


class H3K27me3SmallCollection:
    """A collection of H3K27me3 experiments with multiple alignments per donor per tissue."""

    @classmethod
    def get_alignment_accessions(cls) -> list[str]:
        """Get the accessions of the H3K27me3 alignments for the H3K27me3 small collection.

        Returns:
            list[str]: The accessions of the H3K27me3 alignments for the H3K27me3 small collection.

        """
        h3k27me3_alignments = META_DF[
            (META_DF["target"] == "H3K27me3") & (META_DF["output_type"] == "alignments")
        ]
        h3k27me3_alignments_by_donor = h3k27me3_alignments.groupby(
            ["donor", "biosample_term_name"],
        ).size()
        h3k27me3_alignments_by_donor_filtered = (
            h3k27me3_alignments_by_donor[h3k27me3_alignments_by_donor >= 2]
            .reset_index()
            .groupby("donor")
            .head(2)
        )

        donor_biosample_term_name_pairs = [
            f"{donor}_{biosample_term_name}"
            for donor, biosample_term_name in zip(
                h3k27me3_alignments_by_donor_filtered["donor"],
                h3k27me3_alignments_by_donor_filtered["biosample_term_name"],
            )
        ]
        h3k27me3_alignments_filtered = h3k27me3_alignments[
            h3k27me3_alignments["donor_biosample"].isin(donor_biosample_term_name_pairs)
        ]

        return h3k27me3_alignments_filtered["accession"].to_numpy()

    @classmethod
    def get_control_accessions(cls) -> list[str]:
        """Get the accessions of the control ChIP-seq alignments for the H3K27me3 small collection.

        Returns:
            list[str]: The accessions of the control ChIP-seq alignments for the H3K27me3 small
                collection.

        """
        alignment_accessions = cls.get_alignment_accessions()
        possible_controls = META_DF[META_DF["accession"].isin(alignment_accessions)][
            "possible_controls"
        ].to_numpy()
        if not set(possible_controls).issubset(META_DF["experiment"].values):
            missing_controls = set(possible_controls) - set(META_DF["experiment"].values)
            msg = f"Some possible controls are not in the metadata dataframe: {missing_controls}"
            raise ValueError(msg)

        control_df = META_DF[
            (META_DF["experiment"].isin(possible_controls))
            & (META_DF["output_type"] == "alignments")
        ]
        if (control_df["assay_title"] != "Control ChIP-seq").any():
            msg = "Some possible controls are not control ChIP-seq."
            raise ValueError(msg)

        return control_df["accession"].to_numpy()

    @classmethod
    def get_alignment_control_pairs(cls, pair_replicate: bool = False) -> list[tuple[str, ...]]:
        """Get the alignment and control pairs for the H3K27me3 small collection.

        Returns:
            list[tuple[str, str]]: The alignment and control pairs for the H3K27me3 small collection.
        """
        pairs = []

        alignment_accessions = cls.get_alignment_accessions()
        control_accessions = cls.get_control_accessions()

        alignment_df = META_DF[META_DF["accession"].isin(alignment_accessions)]
        if alignment_df.shape[0] != len(alignment_accessions):
            msg = f"Alignment DF shape: {alignment_df.shape}, alignment accessions: {len(alignment_accessions)}"
            raise ValueError(msg)

        for idx, row in alignment_df.iterrows():
            possible_controls = row["possible_controls"]
            if "," in possible_controls:
                msg = f"Possible controls is a list: {possible_controls}"
                raise ValueError(msg)
            control_df = META_DF[
                (META_DF["experiment"] == possible_controls)
                & (META_DF["output_type"] == "alignments")
            ]
            if not set(control_df["accession"].values).issubset(control_accessions):
                msg = (
                    f"Some possible controls are not in the metadata dataframe: {possible_controls}"
                )
                raise ValueError(msg)

            pairs.append((row["accession"], control_df["accession"].values[0]))

        if pair_replicate:
            paired_reps = []
            for donor_biosample, donor_biosample_df in alignment_df.groupby(["donor_biosample"]):
                donor_biosample_accessions = donor_biosample_df["accession"].values
                if len(donor_biosample_accessions) != 2:
                    msg = f"Donor biosample {donor_biosample} has {len(donor_biosample_accessions)} alignments, expected 2"
                    raise ValueError(msg)
                r1_pair = [p for p in pairs if p[0] == donor_biosample_accessions[0]]
                r2_pair = [p for p in pairs if p[0] == donor_biosample_accessions[1]]
                if len(r1_pair) != 1 or len(r2_pair) != 1:
                    msg = f"Donor biosample {donor_biosample} has {len(r1_pair)} r1 pairs and {len(r2_pair)} r2 pairs"
                    raise ValueError(msg)
                paired_reps.append((r1_pair[0][0], r1_pair[0][1], r2_pair[0][0], r2_pair[0][1]))

            return paired_reps

        return pairs
