
"""
===================================================================
EuroSense Train/Test Splitter
===================================================================
This script processes a EuroSense XML dataset containing sense-annotated 
sentences and splits it into **Train** and **Test** sets.

**Key Features:**
   - Maintains a balanced ratio of frequent and rare words in Train and Test.
   - Ensures words can be present in both Train and Test.
   - Processes large files efficiently using streaming XML parsing (`iterparse()`).
   - Customizable Train/Test split ratio (default: 80% Train, 20% Test).
   - **Optional Filtering:**
       - **Multi-word Anchor Filtering**: Removes annotations where the anchor contains multiple words.
       - **Low-Frequency Label Filtering**: Removes labels that appear fewer than 6 times in the dataset.
                                 Ensures all labels in the test set appear at least 5 times in the train.

===================================================================
**Usage:**
===================================================================
1 **Standard Usage (No Filtering):**
    ```
    python create_eurosens_splits.py eurosense_fr_verbs.xml \
           eurosense_fr_verbs_train.xml eurosense_fr_verbs_test.xml --train_ratio 0.8 --dataset_ratio 0.1
    ```
    - Splits the dataset with an **80/20 Train/Test split**.
    - Uses **10% of the original dataset**.

2 **Alternative: Fixing Number of Annotations Instead of Ratio**
    ```
    python create_eurosens_splits.py eurosense_fr_verbs.xml \
           eurosense_fr_verbs_train.xml eurosense_fr_verbs_test.xml --train_ratio 0.8 --num_annotations 10000
    ```
    - Uses exactly **10,000 annotations** across Train/Test.
    - Keeps an **80/20 split**.

3 **Enable Multi-word Anchor Filtering**
    ```
    python create_eurosens_splits.py eurosense_fr_verbs.xml \
           eurosense_fr_verbs_train.xml eurosense_fr_verbs_test.xml --train_ratio 0.8 --dataset_ratio 0.1 \
           --filter_multiword_anchors
    ```
    - Removes **annotations where the anchor contains multiple words**.

4 **Enable Low-Frequency Label Filtering**
    ```
    python create_eurosens_splits.py eurosense_fr_verbs.xml \
           eurosense_fr_verbs_train.xml eurosense_fr_verbs_test.xml --train_ratio 0.8 --dataset_ratio 0.1 \
           --filter_frequent_labels
    ```
    - Removes **labels that appear less than 6 times**.

5 **Apply Both Filters Simultaneously**
    ```
    python create_eurosens_splits.py eurosense_fr_verbs.xml \
           eurosense_fr_verbs_train.xml eurosense_fr_verbs_test.xml --train_ratio 0.8 --dataset_ratio 0.1 \
           --filter_multiword_anchors --filter_frequent_labels
    ```
    - Removes both **multi-word anchors** and **low-frequency labels**.

===================================================================
**Arguments:**
===================================================================
    - `input_file`: The original EuroSense XML file.
    - `train_output`: Output file for the Train set.
    - `test_output`: Output file for the Test set.
    - `--train_ratio`: (Optional) Ratio of Train data (default: `0.8` for 80% Train / 20% Test).
    - `--dataset_ratio`: (Optional) Ratio of the original dataset to use (default: `1.0` = full dataset).
    - `--num_annotations`: (Optional) Use a **fixed number of annotations** instead of a dataset ratio.
    - `--filter_multiword_anchors`: (Optional) Removes **multi-word anchors** (default: `False`).
    - `--filter_frequent_labels`: (Optional) Removes **labels appearing less than 6 times** (default: `False`).

===================================================================
Author: Lila Roig
"""


import xml.etree.ElementTree as ET  # Import XML parsing library
import random  # Import random module for shuffling and sampling
import collections  # Import collections module for counting occurrences
import numpy as np  # Import numpy for quartile calculations
from typing import List, Tuple  # Import type hints for better readability
import argparse


def parse_eurosense(xml_file: str) -> List[dict]:
    """
    Parses the XML file and extracts sentences with their annotations using iterparse.
    """
    sentences = []  # Initialize an empty list to store sentences
    
    # Stream through the XML file efficiently using iterparse
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if elem.tag == "sentence":  # Check if the element is a sentence
            sentence_id = elem.get("id")  # Get sentence ID attribute
            text_element = elem.find("text[@lang='fr']")  # Find French text
            text = text_element.text if text_element is not None else ""  # Extract text, or set empty string if not found
            annotations = []  # Initialize a list to store annotations
            
            # Extract annotations (lemma and sense_id) from each sentence
            for annotation in elem.findall("annotations/annotation"):
                lemma = annotation.get("lemma")  # Extract lemma
                sense_id = annotation.text  # Extract sense ID

                '''
                annotations.append((lemma, sense_id))  # Store as a tuple
                '''
                ### NEW ###
                anchor = annotation.get("anchor", lemma)  # Use lemma as fallback if anchor is missing
                coherence_score = annotation.get("coherenceScore", "0.0")  # Default score if missing
                nasari_score = annotation.get("nasariScore", "--")  # Default nasari score
                type_ = annotation.get("type", "BABELFY")  # Default annotation type
                
                annotations.append((lemma, sense_id, anchor, coherence_score, nasari_score, type_))
                ### NEW ###
                

            if annotations:  # Ensure the sentence has at least one valid annotation
                sentences.append({  # Append structured sentence data to list
                    "id": sentence_id,
                    "text": text,
                    "annotations": annotations
                })
            
            # Clear element to free memory
            elem.clear()
    
    return sentences  # Return the parsed sentences


def get_label_frequency(sentences: List[dict]) -> dict:
    """
    Computes the frequency of each label in the dataset.
    """
    label_counts = collections.Counter()
    for sentence in sentences:
        for label in sentence["annotations"]:
            label_counts[label] += 1
    return label_counts


def filter_frequent_labels(sentences: List[dict], min_occurrences: int = 6) -> List[dict]:
    """
    Filters out labels that appear less than `min_occurrences` times.
    Ensures only sentences with at least one valid annotation are retained.
    """
    label_counts = get_label_frequency(sentences)  # Get label frequencies
    filtered_sentences = []  # Initialize list to store filtered sentences
    removed_count = 0  # Track how many sentences are removed
    
    print(f"Number of sentences before filtering: {len(sentences)}")
    for sentence in sentences:
        filtered_annotations = [label for label in sentence["annotations"] if label_counts[label] >= min_occurrences]
        if filtered_annotations:  # Ensure sentence retains at least one annotation
            filtered_sentences.append({
                "id": sentence["id"],
                "text": sentence["text"],
                "annotations": filtered_annotations
            })
        else:
            removed_count += 1  # Count removed sentences
    
    return filtered_sentences  # Return filtered sentences


def filter_multi_word_anchors(sentences: List[dict]) -> List[dict]:
    """
    Filters out annotations where the `anchor` contains multiple words.
    If a sentence has no remaining annotations after filtering, it is removed.

    Args:
        sentences (List[dict]): List of sentences with annotations.

    Returns:
        List[dict]: Filtered list of sentences with only valid single-word anchors.
    """
    filtered_sentences = []
    removed_sentences = 0
    removed_annotations = 0

    for sentence in sentences:
        filtered_annotations = [annot for annot in sentence["annotations"] if len(annot[2].split()) == 1]

        if filtered_annotations:  # Keep the sentence only if it has valid annotations
            filtered_sentences.append({
                "id": sentence["id"],
                "text": sentence["text"],
                "annotations": filtered_annotations
            })
        else:
            removed_sentences += 1  # Track removed sentences

        removed_annotations += len(sentence["annotations"]) - len(filtered_annotations)

    print(f"Removed {removed_annotations} multi-word annotations.")
    print(f"Removed {removed_sentences} sentences with no remaining annotations.")

    return filtered_sentences


def compute_quartiles(label_counts: dict) -> Tuple[int, int, int]:
    """
    Computes the first (Q1), second (Q2/median), and third (Q3) quartiles for label frequency distribution.
    """
    frequencies = np.array(list(label_counts.values()))  # Convert counts to a numpy array
    q1 = np.percentile(frequencies, 25)  # First quartile (25th percentile)
    q2 = np.percentile(frequencies, 50)  # Median (50th percentile)
    q3 = np.percentile(frequencies, 75)  # Third quartile (75th percentile)
    return int(q1), int(q2), int(q3)


def sample_dataset(sentences: List[dict], dataset_ratio: float = 1, num_annotations: int = None) -> List[dict]:
    """
    Reduces the dataset size while maintaining a balanced distribution among less frequent, middle frequent, and frequent labels.
    - If `dataset_ratio` is provided, samples a proportion of the dataset.
    - If `num_annotations` is provided, samples exactly `num_annotations` annotations across sentences.
    """
    label_counts = get_label_frequency(sentences)  # Get label frequencies
    q1, q2, q3 = compute_quartiles(label_counts)  # Compute quartiles

    # Separate sentences based on label frequency quartiles
    frequent, middle_frequent, less_frequent = [], [], []
    for s in sentences:
        if any(label_counts[label] < q1 for label in s["annotations"]):
            less_frequent.append(s)
        elif any(q1 <= label_counts[label] <= q3 for label in s["annotations"]):
            middle_frequent.append(s)
        else:
            frequent.append(s)

    # CASE 1 : sample by ratio
    # ----------------------------
    # If sampling by dataset ratio (default behavior)
    if num_annotations is None:
        # Compute sample sizes
        sample_size = int(len(sentences) * sample_ratio)

        sampled_sentences = []

        # Assign the least frequent sentences first
        if sample_size > 0:
            sampled_less_frequent = random.sample(less_frequent, min(len(less_frequent), sample_size // 3))
            sampled_middle = random.sample(middle_frequent, min(len(middle_frequent), sample_size // 3))
            sampled_frequent = random.sample(frequent, min(len(frequent), sample_size // 3))
        else:
            print("sample size insufficient")
        
        # Merge all sampled sentences, ensuring uniqueness
        sampled_sentences = sampled_frequent + sampled_middle + sampled_less_frequent
    

    # CASE 2 : sample by number of annotations
    # ----------------------------
    # If sampling by number of annotations
    else:
        sampled_sentences = []
        sampled_annotations = 0
        sentence_map = {}

        # Shuffle sentences for random sampling
        all_sentences = frequent + middle_frequent + less_frequent
        random.shuffle(all_sentences)

        for s in all_sentences:
            if sampled_annotations >= num_annotations:
                break  # Stop when we reach the required number of annotations

            sentence_id = s["id"]
            annotations_to_add = []

            for annotation in s["annotations"]:
                if sampled_annotations < num_annotations:
                    annotations_to_add.append(annotation)
                    sampled_annotations += 1
                else:
                    break

            if annotations_to_add:
                sentence_map[sentence_id] = {
                    "id": s["id"],
                    "text": s["text"],
                    "annotations": annotations_to_add,
                }
        
        sampled_sentences = list(sentence_map.values())  # Convert dict back to list
       

    random.shuffle(sampled_sentences)  # Shuffle to maintain randomness
    return sampled_sentences  # Return sampled dataset


def split_train_test(sentences: List[dict], train_ratio: float = 0.8) -> Tuple[List[dict], List[dict]]:
    """
    Splits the dataset into Train (80%) and Test (20%) while keeping a balance among less frequent, middle frequent, and frequent labels.
    """
    label_counts = get_label_frequency(sentences)  # Get label frequencies
    q1, q2, q3 = compute_quartiles(label_counts)  # Compute quartiles

    # Create sets to track sentence IDs to prevent overlap
    train_ids, test_ids = set(), set()


    # Assign each sentence to only one category in order of priority: frequent → middle → less frequent
    frequent, middle_frequent, less_frequent = [], [], []
    for s in sentences:
        if any(label_counts[label] < q1 for label in s["annotations"]):
            less_frequent.append(s)
        elif any(q1 <= label_counts[label] <= q3 for label in s["annotations"]):
            middle_frequent.append(s)
        else:
            frequent.append(s)
    
    # Split while maintaining balance and ensuring no duplicate sentence IDs
    train_frequent = frequent[:int(len(frequent) * train_ratio)]
    test_frequent = frequent[int(len(frequent) * train_ratio):]
    train_middle = middle_frequent[:int(len(middle_frequent) * train_ratio)]
    test_middle = middle_frequent[int(len(middle_frequent) * train_ratio):]
    train_less_frequent = less_frequent[:int(len(less_frequent) * train_ratio)]
    test_less_frequent = less_frequent[int(len(less_frequent) * train_ratio):]

    # Merge train and test sets
    train_set = train_frequent + train_middle + train_less_frequent
    test_set = test_frequent + test_middle + test_less_frequent
    
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set  # Return train and test sets


def write_xml(sentences: List[dict], output_file: str):
    """
    Writes a set of annotated sentences to an XML file.
    """
    root = ET.Element("dataset")  # Create root XML element
    
    for sentence in sentences:
        sentence_el = ET.SubElement(root, "sentence", id=sentence["id"])  # Create sentence element
        text_el = ET.SubElement(sentence_el, "text", lang="fr")  # Create text element
        text_el.text = sentence["text"]  # Set text content
        annotations_el = ET.SubElement(sentence_el, "annotations")  # Create annotations container
        
        # Add annotations to the XML structure
        for lemma, sense_id, anchor, coherence_score, nasari_score, type_ in sentence["annotations"]:
            annotation_el = ET.SubElement(annotations_el, "annotation", lang="fr", 
                                         lemma=lemma, 
                                         anchor=anchor, 
                                         coherenceScore=coherence_score, 
                                         nasariScore=nasari_score, 
                                         type=type_)  # Create annotation element
            annotation_el.text = sense_id  # Set sense ID text
    
    tree = ET.ElementTree(root)  # Create XML tree
    tree.write(output_file, encoding="utf-8", xml_declaration=True)  # Write to file
    print(f"File {output_file} successfully generated.")

# ======================================================================================
#                                MAIN EXECUTION BLOCK
# ======================================================================================
if __name__ == "__main__":
    """
    Main execution script for splitting EuroSense XML into Train and Test datasets.
    """
    # Parse command-line arguments (train-test split ratio, dataset size, etc.)
    parser = argparse.ArgumentParser(description="Split EuroSense dataset into Train/Test")
    parser.add_argument("input_file", type=str, help="Path to the original EuroSense XML dataset")
    parser.add_argument("train_output", type=str, help="Output file for Train dataset")
    parser.add_argument("test_output", type=str, help="Output file for Test dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of dataset used for training (default 80%)")
    parser.add_argument("--dataset_ratio", type=float, default=1.0, help="Proportion of dataset to use (default 100%)")
    parser.add_argument("--num_annotations", type=int, default=None, help="Number of annotations to use accross sentences (default None)")
    parser.add_argument("--filter_multiword_anchors", action="store_true",
                        help="Enable filtering of multi-word anchors (default: False)")
    parser.add_argument("--filter_frequent_labels", action="store_true",
                        help="Enable filtering of labels that appear less than 6 times (default: False)")
    args = parser.parse_args()

    sentences = parse_eurosense(args.input_file)
    print(f"Total sentences parsed before filtering: {len(sentences)}")


    # Apply frequent label filtering if enabled
    if args.filter_frequent_labels:
        print("Applying label frequency filtering...")
        filtered_sentences = filter_frequent_labels(sentences)
        print(f"Total sentences after filtering (appear fewer than 6 times):{len(filtered_sentences)}")
        print(f"Removed sentences: {len(sentences) - len(filtered_sentences)}")
    else:
        print("Skipping label frequency filtering.")
        filtered_sentences = sentences

    # Apply multi-word anchor filtering if enabled
    if args.filter_multiword_anchors:
        print("Applying multi-word anchor filtering...")
        filtered_sentences = filter_multi_word_anchors(sentences)
        print(f"Total sentences after filtering: {len(filtered_sentences)}")
    else:
        print("Skipping multi-word anchor filtering.")
        filtered_sentences = sentences


    sampled_sentences = sample_dataset(filtered_sentences, args.dataset_ratio, args.num_annotations )
    title = 'dataset ratio of' + str(args.dataset_ratio*100)+"%" if args.num_annotations is None else  'number of annotations = '+str(args.num_annotations) 
    #print(f"Number of sentences after balanced sampling ({args.dataset_ratio*100}%): {len(sampled_sentences)}")
    print(f"Number of sentences after balanced sampling : {len(sampled_sentences)} ({title})")

    train_set, test_set = split_train_test(sampled_sentences)
    print(f"Train: {len(train_set)} sentences, Test: {len(test_set)} sentences")
    print(f"Train sentences + Test sentences : {len(train_set) + len(test_set)}")

    # Check if the split is correct
    assert len(train_set) + len(test_set) == len(sampled_sentences), \
        f"Mismatch! Train + Test = {len(train_set) + len(test_set)}, but sampled = {len(sampled_sentences)}"

    write_xml(train_set, args.train_output)
    write_xml(test_set, args.test_output)
    print(f"Done! Train and Test sets saved at:\n  - Train: {args.train_output}\n  - Test: {args.test_output}")
