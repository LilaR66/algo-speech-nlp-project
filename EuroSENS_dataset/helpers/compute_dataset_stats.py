"""
===================================================================
EuroSense Train/Test Label & Lemma Frequency Analysis
===================================================================
This script processes **EuroSense XML datasets** to compute:
   - Frequency statistics of **(lemma + sense ID) labels**.
   - Frequency statistics of **lemmas**.
   - **Overlap analysis** between Train and Test datasets.

**Features:**
   - Efficient parsing of large XML files using `iterparse()`.
   - Extracts and counts **(lemma + sense ID) labels** and **lemmas**.
   - Saves **frequency statistics** in `.tsv` files.
   - Computes **overlap statistics** between Train and Test datasets.
   - Works with **any XML dataset** containing `<annotation>` elements.

===================================================================
USAGE:
===================================================================
Run the script from the terminal:

    python compute_dataset_stats.py eurosense_train.xml eurosense_test.xml train_stats test_stats

Arguments:
   - `eurosense_train.xml` -> Path to the **Train XML dataset**.
   - `eurosense_test.xml` -> Path to the **Test XML dataset**.
   - `train_stats` -> Directory to save **Train statistics**.
   - `test_stats` -> Directory to save **Test statistics**.

===================================================================
OUTPUT:
===================================================================
**Generated Frequency Files:**
   - `train_stats/label_frequencies.tsv`  -> (lemma + sense ID) frequencies for Train.
   - `train_stats/lemma_frequencies.tsv`  -> Lemma frequencies for Train.
   - `test_stats/label_frequencies.tsv`   -> (lemma + sense ID) frequencies for Test.
   - `test_stats/lemma_frequencies.tsv`   -> Lemma frequencies for Test.

**Generated Overlap Statistics:**
   - `overlap_stats.txt` -> Overlap analysis between Train and Test.

===================================================================
EXAMPLE OUTPUT:
===================================================================
If the Train and Test datasets contain different sets of labels and lemmas,
the script will output something like:

    Statistics saved in: train_stats
      - Label frequency file: train_stats/label_frequencies.tsv
      - Lemma frequency file: train_stats/lemma_frequencies.tsv

    Statistics saved in: test_stats
      - Label frequency file: test_stats/label_frequencies.tsv
      - Lemma frequency file: test_stats/lemma_frequencies.tsv

    OVERLAP STATISTICS
      - Total (Lemma + Sense ID) labels in Train: 12045
      - Total (Lemma + Sense ID) labels in Test: 4521
      - Labels common in both Train & Test: 3210
      - Labels in Test but NOT in Train: 1311

      - Total lemmas in Train: 7890
      - Total lemmas in Test: 3120
      - Lemmas common in both Train & Test: 2456
      - Lemmas in Test but NOT in Train: 664

    Overlap statistics saved in 'overlap_stats.txt'

Author: Lila Roig
"""



import xml.etree.ElementTree as ET
import argparse
from collections import Counter
import os

def dataset_stats(xml_file: str, output_dir: str):
    """
    Computes frequency statistics for labels (lemma + sense ID) and lemmas from an XML dataset.
    Saves results in TSV files.

    Args:
        xml_file (str): Path to the XML dataset.
        output_dir (str): Directory to save frequency files.

    Returns:
        labels (Counter): Frequency dictionary of (lemma + sense IDs).
        lemmas (Counter): Frequency dictionary of lemmas.
    """
    labels = Counter()
    lemmas = Counter()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Parse XML file efficiently
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if elem.tag == "annotation":  # Process each annotation
            sense_id = elem.text.strip()  # Extract sense ID
            lemma = elem.attrib.get("lemma", "").strip()  # Extract lemma
            
            if lemma and sense_id:
                label = f"{lemma}_{sense_id}"  # Combine lemma + sense ID as the label
                labels[label] += 1  # Count (lemma + sense ID) occurrences
                lemmas[lemma] += 1  # Count lemma occurrences

        elem.clear()  # Free memory

    # Save label frequencies to TSV file
    label_file = os.path.join(output_dir, "label_frequencies.tsv")
    with open(label_file, "w", encoding="utf-8") as f:
        for label, freq in sorted(labels.items(), key=lambda x: -x[1]):
            f.write(f"{label}\t{freq}\n")
    
    # Save lemma frequencies to TSV file
    lemma_file = os.path.join(output_dir, "lemma_frequencies.tsv")
    with open(lemma_file, "w", encoding="utf-8") as f:
        for lemma, freq in sorted(lemmas.items(), key=lambda x: -x[1]):
            f.write(f"{lemma}\t{freq}\n")

    print(f"Statistics saved in: {output_dir}")
    print(f"  - Label frequency file: {label_file}")
    print(f"  - Lemma frequency file: {lemma_file}")

    return labels, lemmas


def compute_overlap(train_labels, test_labels, train_lemmas, test_lemmas):
    """
    Computes and prints the overlap statistics between train and test datasets.

    Args:
        train_labels (Counter): Train set label frequencies.
        test_labels (Counter): Test set label frequencies.
        train_lemmas (Counter): Train set lemma frequencies.
        test_lemmas (Counter): Test set lemma frequencies.
    """
    # Compute label overlap
    common_labels = set(train_labels.keys()) & set(test_labels.keys())
    missing_labels = set(test_labels.keys()) - set(train_labels.keys())

    # Compute lemma overlap
    common_lemmas = set(train_lemmas.keys()) & set(test_lemmas.keys())
    missing_lemmas = set(test_lemmas.keys()) - set(train_lemmas.keys())

    print("\nOVERLAP STATISTICS")
    print(f"  - Total (Lemma + Sense ID) labels in Train: {len(train_labels)}")
    print(f"  - Total (Lemma + Sense ID) labels in Test: {len(test_labels)}")
    print(f"  - Labels common in both Train & Test: {len(common_labels)}")
    print(f"  - Labels in Test but NOT in Train: {len(missing_labels)}")

    print(f"\n  - Total lemmas in Train: {len(train_lemmas)}")
    print(f"  - Total lemmas in Test: {len(test_lemmas)}")
    print(f"  - Lemmas common in both Train & Test: {len(common_lemmas)}")
    print(f"  - Lemmas in Test but NOT in Train: {len(missing_lemmas)}")

    # Save overlap stats to a file
    with open("overlap_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"OVERLAP STATISTICS\n")
        f.write(f"  - Total (Lemma + Sense ID) labels in Train: {len(train_labels)}\n")
        f.write(f"  - Total (Lemma + Sense ID) labels in Test: {len(test_labels)}\n")
        f.write(f"  - Labels common in both Train & Test: {len(common_labels)}\n")
        f.write(f"  - Labels in Test but NOT in Train: {len(missing_labels)}\n\n")

        f.write(f"  - Total lemmas in Train: {len(train_lemmas)}\n")
        f.write(f"  - Total lemmas in Test: {len(test_lemmas)}\n")
        f.write(f"  - Lemmas common in both Train & Test: {len(common_lemmas)}\n")
        f.write(f"  - Lemmas in Test but NOT in Train: {len(missing_lemmas)}\n")

    print("\n Overlap statistics saved in 'overlap_stats.txt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute (lemma + sense ID) and lemma frequencies from train and test XML datasets.")
    parser.add_argument("xml_train_file", type=str, help="Path to the train XML dataset.")
    parser.add_argument("xml_test_file", type=str, help="Path to the test XML dataset.")
    parser.add_argument("output_train_dir", type=str, help="Directory to save the train frequency files.")
    parser.add_argument("output_test_dir", type=str, help="Directory to save the test frequency files.")
    args = parser.parse_args()

    # Process train and test datasets
    train_labels, train_lemmas = dataset_stats(args.xml_train_file, args.output_train_dir)
    test_labels, test_lemmas = dataset_stats(args.xml_test_file, args.output_test_dir)

    # Compute overlap statistics
    compute_overlap(train_labels, test_labels, train_lemmas, test_lemmas)



# How to use:
# python compute_dataset_stats.py eurosense_train.xml eurosense_test.xml train_stats test_stats
