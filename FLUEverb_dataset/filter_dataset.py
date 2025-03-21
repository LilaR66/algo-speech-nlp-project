"""
This script filters an XML dataset and its corresponding .gold.key.txt file to keep only sentences containing verbs. 
All other parts of the dataset, such as non-verb instances and sentences without verbs are removed.


What It Does
------------
    Parses an XML file containing WSD-annotated sentences.
    Keeps only the <sentence> elements that contain at least one <instance> annotated as a verb (pos="V").
    Collects the IDs of the retained verb instances.
    Filters the corresponding .gold.key.txt file to keep only the lines matching those verb instance IDs.
    Saves two new output files:
        A filtered XML file containing only verb sentences.
        A filtered .gold.key.txt file with only labels for the remaining verb instances.

Example of usage:
----------------
    python filter_dataset.py 


Inside filter_dataset.py, make sure you define the file paths:
    xml_input = "train/wiktionary-190418.data.xml"
    xml_output = "train_small/wiktionary-190418.filtered_verbs.xml"
    gold_input = "train/wiktionary-190418.gold.key.txt"
    gold_output = "train_small/wiktionary-190418.filtered_verbs.gold.key.txt"

filter_verbs_from_xml(xml_input, xml_output, gold_input, gold_output)

Output
------
    wiktionary-190418.filtered_verbs.xml: contains only sentences with at least one verb instance.
    wiktionary-190418.filtered_verbs.gold.key.txt: contains only the gold labels for those verbs.

Author: Lila Roig
"""

import xml.etree.ElementTree as ET

def filter_verbs_from_xml(xml_path, output_xml_path, gold_key_path, output_gold_key_path):
    """
    Filters sentences containing verbs from an XML file and updates the corresponding .gold.key.txt file.

    Args:
        xml_path (str): Path to the input XML file.
        output_xml_path (str): Path to save the filtered output XML file.
        gold_key_path (str): Path to the input .gold.key.txt file.
        output_gold_key_path (str): Path to save the filtered .gold.key.txt file.
    """
    # Load the XML tree structure
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # A set to store instance IDs of verbs
    verb_instance_ids = set()

    # Iterate over all sentences within texts
    for text in root.findall("text"):
        for sentence in text.findall("sentence"):
            # Flag to check if a verb is found in the sentence
            verb_found = False
            for instance in sentence.findall("instance"):
                if instance.attrib.get("pos") == "V":  # Check if the instance is a verb
                    verb_instance_ids.add(instance.attrib["id"])  # Store its ID
                    verb_found = True
            
            # Remove the sentence if it does not contain any verbs
            if not verb_found:
                text.remove(sentence)

    # Save the updated XML file with only sentences containing verbs
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    # ---- Filtering the gold.key.txt file ----
    with open(gold_key_path, "r", encoding="utf-8") as infile, open(output_gold_key_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            instance_id = line.split()[0]  # Extract the instance ID (e.g., d000.s044.t000)
            if instance_id in verb_instance_ids:  # Keep only IDs that exist in the filtered XML file
                outfile.write(line)

# ---- Running the script ----
xml_input = "train/wiktionary-190418.data.xml"
xml_output = "train_small/wiktionary-190418.filtered_verbs.xml"
gold_input = "train/wiktionary-190418.gold.key.txt"
gold_output = "train_small/wiktionary-190418.filtered_verbs.gold.key.txt"

filter_verbs_from_xml(xml_input, xml_output, gold_input, gold_output)
print("Filtering complete: XML file and gold.key.txt file updated with only verbs.")
