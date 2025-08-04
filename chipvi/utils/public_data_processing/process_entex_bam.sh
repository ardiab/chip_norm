#!/bin/bash

set -e
set -u
set -o pipefail

# --- Argument Parsing ---
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <raw_bam_fpath> <out_bed_fpath> <fetch_chrom_sizes_script> <genome_build> <bin_size>"
    echo "Example: $0 input.bam output.bed fetch_hg38.sh hg38 25"
    exit 1
fi

RAW_BAM_FPATH=$(realpath "$1")
OUT_BED_FPATH="$2"
FETCH_CHROM_SIZES_SCRIPT=$(realpath "$3")
GENOME_BUILD="$4"
BIN_SIZE="$5"

# --- Sanity check for BIN_SIZE ---
if ! [[ "$BIN_SIZE" =~ ^[0-9]+$ ]] || [ "$BIN_SIZE" -le 0 ]; then
    echo "Error: BIN_SIZE must be a positive integer. Received: '$BIN_SIZE'"
    exit 1
fi

# --- Output and Temporary File Definitions ---
OUT_DIR=$(realpath "$(dirname "$OUT_BED_FPATH")")
# Ensure output directory exists
mkdir -p "$OUT_DIR"

BAM_FNAME="$(basename "${RAW_BAM_FPATH%.*}")"

RAW_READS_TMP_BED_FPATH="${OUT_DIR}/${BAM_FNAME}_reads_tmp.bed"
RAW_READS_TMP_UNBINNED_BED_FPATH="${OUT_DIR}/${BAM_FNAME}_reads_tmp_unbinned.bed"
BINNED_READS_TMP_BINNED_BED_FPATH="${OUT_DIR}/${BAM_FNAME}_binned_reads_tmp.bed"

RAW_MAPQ_TMP_BED_FPATH="${OUT_DIR}/${BAM_FNAME}_mapq_raw_tmp.bed"
BINNED_MAPQ_TMP_BED_FPATH="${OUT_DIR}/${BAM_FNAME}_binned_mapq_tmp.bed"

BINNED_GENOME_TMP_BED_FPATH="${OUT_DIR}/${BAM_FNAME}_binned_genome_tmp.bed"

echo "--- Starting Script ---"
echo "Input BAM: $RAW_BAM_FPATH"
echo "Output BED: $OUT_BED_FPATH"
echo "Chromosome Sizes Script: $FETCH_CHROM_SIZES_SCRIPT"
echo "Genome Build: $GENOME_BUILD"
echo "Bin Size: $BIN_SIZE"
echo "Output Directory: $OUT_DIR"
echo "Temporary files will be placed in: $OUT_DIR"
echo "-------------------------"

# Use `bedtools genomecov` to convert BAM to BED. The output is a BED file with four columns:
# chromosome name, start position, end position, and depth of coverage (i.e. total
# number of aligned fragments) within the region.
# -bga: report depth in bedgraph format, including zero-depth regions (i.e. report
# signal in the whole genome even if it's zero)
# -ibam: input BAM file
echo "Step 1: Converting BAM to BEDGraph for coverage..."
bedtools genomecov -bga -ibam "$RAW_BAM_FPATH" > "$RAW_READS_TMP_BED_FPATH"
echo "Done."

# A step-by-step explanation of the command (left to right):
# 1. Fetch chromosome sizes for the specified genome build. The command returns a tab-separated file with two columns:
#    chromosome name and chromosome size.
# 2. Filter out lines containing underscores, which include non-standard chromosomes and scaffolds.
# 3. Format chromosome sizes into a BED format with three columns: chromosome name, start position (0),
#    and end position (chromosome size).
# 4. Sort the BED file containing chromosome sizes by chromosome name (bedtools can work
#    incorrectly with unsorted BED files).
# 5. Chop the BED file containing chromosome sizes into bins of size bin_size. This will
#    create a BED file with non-overlapping bins of size bin_size across the genome. However,
#    this BED file will not contain any signal values (3 columns: chromosome, start, and end).
echo "Step 2: Generating binned genome regions (size: $BIN_SIZE bp)..."
"$FETCH_CHROM_SIZES_SCRIPT" "$GENOME_BUILD" | grep -v '_*_' | awk -v FS="\t" -v OFS="\t" '{ print $1, "0", $2 }' | sort-bed - | bedops --chop "$BIN_SIZE" - > "$BINNED_GENOME_TMP_BED_FPATH"
echo "Done."

echo "Step 3: Preparing raw reads data for bin mapping..."
sort-bed "$RAW_READS_TMP_BED_FPATH" | awk -v FS="\t" -v OFS="\t" '{ print $1, $2, $3, $4, $4 }' > "$RAW_READS_TMP_UNBINNED_BED_FPATH"
echo "Done."

# 6. Call the `bedmap` command to map the BED file (which was converted from the downloaded BAM file)
#    to the binned regions, calculating maximum value for each bin (i.e. the maximum number of tags
#    that overlap within the region).
#    Pass the binned BED file as the first argument and the raw BED file as the second argument.
# 7. The raw BED file was processed by sorting it and formatting it to have five columns (it seems that
#    bedmap requires the raw BED file to have five columns: chromosome, start, end, name, and score,
#    and the BED file that was converted from the BAM is missing the name column).
# 8. Redirect the final output to the specified binned BED file path.
echo "Step 4: Calculating maximum reads per bin..."
bedmap --echo --max --delim '\t' "$BINNED_GENOME_TMP_BED_FPATH" "$RAW_READS_TMP_UNBINNED_BED_FPATH" > "$BINNED_READS_TMP_BINNED_BED_FPATH"
echo "Done."


# Run another command very similar to the previous one, but this time calculating bin-specific
# mean MAPQ values instead of reads per bin. Add column 5 twice, because bedmap does not seem
# to work properly with 4 columns.
# Note on MAPQ read interval: The awk command below calculates read end position based on sequence length ($10 from SAM).
# This is an approximation and might not perfectly reflect the aligned span on the reference for reads
# with indels or complex CIGAR strings. For higher precision, consider `bedtools bamtobed` followed by
# a method to join MAPQ scores.
echo "Step 5: Preparing MAPQ data from BAM..."
samtools view "$RAW_BAM_FPATH" | awk -v FS="\t" -v OFS="\t" '{print $3, ($4-1), ($4+length($10)-1), $5, $5}' | sort-bed - > "$RAW_MAPQ_TMP_BED_FPATH"
echo "Done."

echo "Step 6: Calculating mean MAPQ per bin..."
bedmap --echo --wmean --delim '\t' "$BINNED_GENOME_TMP_BED_FPATH" "$RAW_MAPQ_TMP_BED_FPATH" > "$BINNED_MAPQ_TMP_BED_FPATH"
echo "Done."

# The steps above create 2 BED files which should have the same number of lines. Check that this is the case,
# and create a new BED file in which the MAPQ column is pasted as the 5th column.
# The final columns of the output file are then:
# 1. Chromosome
# 2. Start
# 3. End
# 4. Reads
# 5. MAPQ
echo "Step 7: Combining results and generating final output BED file..."
if [ "$(wc -l < "$BINNED_READS_TMP_BINNED_BED_FPATH")" -eq "$(wc -l < "$BINNED_MAPQ_TMP_BED_FPATH")" ]; then
    paste "$BINNED_READS_TMP_BINNED_BED_FPATH" "$BINNED_MAPQ_TMP_BED_FPATH" | cut -f 1,2,3,4,8 > "$OUT_BED_FPATH"
    echo "Successfully created output file: $OUT_BED_FPATH"
else
    echo "Error: Files $BINNED_READS_TMP_BINNED_BED_FPATH and $BINNED_MAPQ_TMP_BED_FPATH have a different number of lines."
    echo "Output file not created due to inconsistency."
    # Consider exiting with an error code if this happens
    # exit 1
fi
echo "Done."

# Remove all the intermediate generated files
echo "Step 8: Cleaning up temporary files..."
rm "$RAW_READS_TMP_BED_FPATH"
# rm $BINNED_READS_TMP_BED_FPATH # This was an erroneous line, variable not used in current flow for this filename.
rm "$BINNED_MAPQ_TMP_BED_FPATH"
rm "$BINNED_GENOME_TMP_BED_FPATH"
rm "$RAW_READS_TMP_UNBINNED_BED_FPATH"
rm "$RAW_MAPQ_TMP_BED_FPATH"
rm "$BINNED_READS_TMP_BINNED_BED_FPATH"
# IMPORTANT: DO NOT remove the original input BAM file unless explicitly intended.
# rm $RAW_BAM_FPATH # This line was removed.
echo "Done."

echo "--- BAM processing complete ---"