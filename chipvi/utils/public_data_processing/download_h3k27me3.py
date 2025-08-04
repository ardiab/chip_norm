import itertools

from chipvi.data.experiment_collections import H3K27me3SmallCollection
from chipvi.utils.public_data_processing.download_entex_files import download_and_process_entex_files


if __name__ == "__main__":
    h3k27me3_small_collection = H3K27me3SmallCollection()
    accessions = list(itertools.chain.from_iterable(h3k27me3_small_collection.get_alignment_control_pairs()))
    file_types = ["bam"] * len(accessions)

    download_and_process_entex_files(
        accession_list=accessions,
        file_type_list=file_types,
        n_splits=6,
    )
