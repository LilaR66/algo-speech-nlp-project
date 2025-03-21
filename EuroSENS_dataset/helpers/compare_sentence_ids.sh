#!/bin/bash
# ========================================================
# Script: compare_sentence_ids.sh
# Description:
#   This script compares two text files and identifies 
#   sentence IDs that are unique to each file.
#
# Usage:
#   ./compare_sentence_ids.sh <file1.txt> <file2.txt>
#
# Requirements:
#   - The input files must contain sentence IDs in the format "Sentence ID=<number>".
#   - The script uses grep, sort, uniq, and comm for processing.
#
# Functionality:
#   1. Checks if two input files are provided as arguments.
#   2. Extracts all unique "Sentence ID=" values from both files.
#   3. Compares the IDs and outputs:
#      - IDs present in file1 but not in file2.
#      - IDs present in file2 but not in file1.
#   4. Cleans up temporary files after processing.
#
# Example:
#   chmod +x compare_sentence_ids.sh
#   ./compare_sentence_ids.sh file1.txt file2.txt
#
# Author: Lila Roig
# ========================================================

#  Checks if two input files are provided as arguments.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file1.txt> <file2.txt>"
    exit 1
fi

file1="$1"
file2="$2"

# Extracts all unique "Sentence ID=" values from both files. and store them in temporary files
grep -o "Sentence ID=[0-9]\+" "$file1" | sort | uniq > ids_file1.txt
grep -o "Sentence ID=[0-9]\+" "$file2" | sort | uniq > ids_file2.txt

# Find IDs present in file1 but not in file2.
echo "Sentences in $file1 but NOT in $file2:"
comm -23 ids_file1.txt ids_file2.txt

echo "=============================================="

# Find IDs present in file2 but not in file1. 
echo "Sentences in $file2 but NOT in $file1:"
comm -13 ids_file1.txt ids_file2.txt

# Cleans up temporary files after processing.
rm ids_file1.txt ids_file2.txt


