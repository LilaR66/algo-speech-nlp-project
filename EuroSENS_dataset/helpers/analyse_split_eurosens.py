"""
===================================================================
Label Overlap Analysis for EuroSense XML Datasets
===================================================================
This script analyzes **label distributions and sentence counts** 
in XML datasets, specifically comparing **Train and Test sets**.

===================================================================
KEY FEATURES:
===================================================================
- **Extracts unique sense labels (`sense_id`)** from annotations.
- **Counts the total number of sentences** in each dataset.
- **Computes label overlap** between Train and Test sets.
- **Calculates percentage of shared labels** between datasets.
- **Optimized for large XML files** using streaming (`iterparse()`).

===================================================================
PROCESSING DETAILS:
===================================================================
- The script parses XML **incrementally** to prevent memory overload.
- It tracks **sentence counts** and **sense labels** found in `<annotation>` tags.
- Overlap between Train and Test labels is **measured and reported**.

- **Outputs the following statistics:**
  - Total sentences in Train & Test.
  - Total unique labels (sense IDs) in each set.
  - Number of shared labels between Train & Test.
  - **Percentage of overlap** in label distribution.

===================================================================
HOW TO RUN:
===================================================================
Simply execute:
    python analyze_split_eurosens.py

Example usage:
    results = analyze_split_eurosens("train.xml", "test.xml")

===================================================================
OUTPUT:
===================================================================
The script prints and returns a dictionary with:
{
    "train_sentence_count": <int>,
    "test_sentence_count": <int>,
    "train_label_count": <int>,
    "test_label_count": <int>,
    "overlap_label_count": <int>,
    "overlap_percentage": <float>
}


Author: Lila Roig
"""

import collections
import xml.etree.ElementTree as ET

def get_labels_and_sentences_from_xml(xml_file: str):
    """
    Extracts all unique labels and counts sentences from an XML dataset.
    """
    labels = set()
    sentence_count = 0
    
    # Parse XML file
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if elem.tag == "sentence":  # Count sentences
            sentence_count += 1
        if elem.tag == "annotation":  # Extract labels (sense_id)
            sense_id = elem.text  # Extract label (sense_id)
            if sense_id:
                labels.add(sense_id)
        elem.clear()  # Free memory
    
    return labels, sentence_count

def analyze_label_overlap(train_file: str, test_file: str):
    """
    Analyzes label distribution, sentence counts, and overlap between Train and Test sets.
    """
    train_labels, train_sentences = get_labels_and_sentences_from_xml(train_file)
    test_labels, test_sentences = get_labels_and_sentences_from_xml(test_file)
    
    overlap_labels = train_labels.intersection(test_labels)
    
    print(f"Total sentences in Train: {train_sentences}")
    print(f"Total sentences in Test: {test_sentences}")
    print(f"Total labels in Train: {len(train_labels)}")
    print(f"Total labels in Test: {len(test_labels)}")
    print(f"Labels common in both Train & Test: {len(overlap_labels)}")
    print(f"Percentage of overlap: {len(overlap_labels) / len(train_labels) * 100:.2f}%")
    
    return {
        "train_sentence_count": train_sentences,
        "test_sentence_count": test_sentences,
        "train_label_count": len(train_labels),
        "test_label_count": len(test_labels),
        "overlap_label_count": len(overlap_labels),
        "overlap_percentage": len(overlap_labels) / len(train_labels) * 100
    }

# Example usage:
results = analyze_label_overlap("eurosense_fr_verbs_train.xml", "eurosense_fr_verbs_test.xml")
