#!/bin/bash
# ========================================================
# Script: most_frequent_word_sense.sh
# Description: 
#   This script extracts and identifies the most frequent 
#   word-sense pair from an XML file containing annotations.
# 
# Note: a word-sense pair is of the form: 'faire_bn:00088736v'
#
# Usage:
#   ./most_frequent_word_sense.sh <eurosense_file.xml>
#
# Requirements:
#   - The input file must be an XML file containing <annotation> tags.
#   - The script uses grep, sed, awk, and sort to process the file.
#
# Functionality:
#   1. Checks if an XML file is provided as an argument.
#   2. Extracts lemma and sense from <annotation> tags.
#   3. Counts occurrences of each word-sense pair.
#   4. Outputs the most frequent word-sense pair.
#
# How to Use: 
#   1. Make the script executable:
# chmod +x most_frequent_word_sense.sh
#   2. Run the script with an XML file:
# ./most_frequent_word_sense.sh eurosense_fr_verbs.xml
#
#
# Author: Lila Roig
# ========================================================

# Check if an XML file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <eurosense_file.xml>"
    exit 1
fi

INPUT_FILE="$1"

echo "Processing file: $INPUT_FILE ..."

# Extract lemma and sense from annotation tags
# - Uses grep to extract annotation lines
# - Uses sed and awk to extract and count word-sense pairs
MOST_FREQUENT=$(grep "<annotation " "$INPUT_FILE" | \
    sed -E 's/.*lemma="([^"]+)".*>(bn:[0-9]+[a-z])<\/annotation>.*/\1 \2/' | \
    sort | uniq -c | sort -nr | head -n 1)

echo "Most frequent word-sense pair:"
echo "$MOST_FREQUENT"


