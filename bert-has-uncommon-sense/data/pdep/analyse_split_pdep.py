"""
=============================================================
Script: analyse_split_pdep.py
Description:
    This script extracts word-sense pairs from a CoNLL-U file 
    and compares sense distributions between training and test datasets.

Usage:
    python analyse_split_pdep.py

Requirements:
    - Input files must be in CoNLL-U format.
    - The script uses regular expressions and dictionary operations for analysis.

Functionality:
    1. Extracts word-sense pairs from CoNLL-U formatted files.
    2. Stores extracted senses in separate text files.
    3. Loads and structures the extracted data as {word: {senses}} dictionaries.
    4. Compares the vocabulary overlap between training and test sets.
    5. Identifies words that share or differ in sense distributions between datasets.
    6. Outputs summary statistics and examples of discrepancies.

Example:
    python analyse_split_pdep.py

=============================================================
"""

import re

def extract_senses(input_file, output_file):
    """
    Extract words and their associated senses from a conllu file.
    """
    sense_pattern = re.compile(r"(Sense=[0-9]+\([0-9a-z-]+\))")  # Regex to capture "Sense=X(Y)"
    
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            columns = line.strip().split("\t")  # Split by tab
            if len(columns) >= 10 and sense_pattern.search(columns[-1]):  # Check if last column contains "Sense="
                word = columns[1]  # Extract the actual word
                sense = sense_pattern.search(columns[-1]).group(1)  # Extract the sense annotation
                outfile.write(f"{word} {sense}\n")  # Save "word Sense=X(Y)"

# Run for both train and test files
extract_senses("pdep_train.conllu", "train_senses_python.txt")
extract_senses("pdep_test.conllu", "test_senses_python.txt")

print("Extraction completed")


# =======================
# COMPARISON FUNCTIONS
# =======================

def load_sense_file(filepath):
    """
    Load a file into a dictionary: {word: {set of senses}}.
    """
    word_sense_dict = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            word, sense = line.strip().split(" ", 1)  # Split word and sense
            if word not in word_sense_dict:
                word_sense_dict[word] = set()
            word_sense_dict[word].add(sense)  # Add sense to the word
    return word_sense_dict


# Load both train and test sense files
train_senses = load_sense_file("train_senses_python.txt")
test_senses = load_sense_file("test_senses_python.txt")


def compare_word_sets(train_dict, test_dict):
    """
    Compare the words in train and test.
    """
    train_words = set(train_dict.keys())
    test_words = set(test_dict.keys())

    common_words = train_words & test_words  # Intersection: words in both train & test
    only_in_train = train_words - test_words  # Words only in train
    only_in_test = test_words - train_words  # Words only in test

    print("\nWord Overlap Analysis")
    print(f"Total words in Train: {len(train_words)}")
    print(f"Total words in Test: {len(test_words)}")
    print(f"Words in both Train & Test: {len(common_words)}")
    print(f"Words only in Train: {len(only_in_train)}")
    print(f"Words only in Test: {len(only_in_test)}")

    return common_words


def compare_sense_sets(train_dict, test_dict, common_words):
    """
    Compare senses of common words between train and test.
    """
    sense_mismatch = {}  # Words with different senses in train/test
    sense_match = 0  # Counter for words with exactly the same senses

    for word in common_words:
        train_senses = train_dict[word]
        test_senses = test_dict[word]

        if train_senses == test_senses:
            sense_match += 1
        else:
            sense_mismatch[word] = (train_senses, test_senses)

    print("\nSense Overlap Analysis")
    print(f"Words with identical senses in Train & Test: {sense_match}")
    print(f"Words with different senses in Train & Test: {len(sense_mismatch)}")

    # Display 10 examples of words with different senses
    if len(sense_mismatch) > 0:
        print("\nExamples of words with different senses:")
        for word, (train_s, test_s) in list(sense_mismatch.items())[:10]:
            print(f"🔹 {word}:")
            print(f"   - Train Senses: {train_s}")
            print(f"   - Test Senses: {test_s}")

    return sense_mismatch


# =======================
# RUN THE ANALYSIS
# =======================

common_words = compare_word_sets(train_senses, test_senses)
sense_mismatch = compare_sense_sets(train_senses, test_senses, common_words)

print("\nAnalysis Completed!")
