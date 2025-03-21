"""
===================================================================
EuroSense French Sentence Filter & POS-Based Splitter
===================================================================
This script processes the EuroSense dataset (`eurosense.v1.0.high-precision.xml`)
and extracts only **French (`lang="fr"`) sentences** along with their annotations.

Each extracted sentence is categorized based on its **Part of Speech (POS)**
from the BabelNet ID (`bn:XXXXXXXXXL` where `L` is the POS type).

The sentences are stored in **separate XML files** for each POS type:
- **Nouns (`n`)** -> `eurosense_fr_nouns.xml`
- **Verbs (`v`)** -> `eurosense_fr_verbs.xml`
- **Adjectives (`a`)** -> `eurosense_fr_adjectives.xml`
- **Adverbs (`r`)** -> `eurosense_fr_adverbs.xml`
- **Prepositions (`p`)** -> `eurosense_fr_prepositions.xml` (if they exist)
- **Conjunctions (`c`)** -> `eurosense_fr_conjunctions.xml` (if they exist)
- **Adjective Satellites (`s`)** -> `eurosense_fr_adj_satellites.xml` (if they exist)

The script is optimized for **large datasets (~4GB)** and processes the XML **in streaming mode** (`iterparse()`).
It prints **progress updates every 100,000 sentences** to track execution.

===================================================================
HOW TO RUN:
===================================================================
Simply execute:
    python filter_eurosense_french_by_pos.py

If you want to specify a custom input file:
    python filter_eurosense_french_by_pos.py 
===================================================================

Author: Lila Roig
"""

import xml.etree.ElementTree as ET  

# Define file paths
input_file = "eurosense_fr.v1.0.high-precision.xml"  # Filtered dataset containing only French sentences

# Define output files for each POS type
output_files = {
    "n": "eurosense_fr_nouns.xml",
    "v": "eurosense_fr_verbs.xml",
    "a": "eurosense_fr_adjectives.xml",
    "r": "eurosense_fr_adverbs.xml",
    #"p": "eurosense_fr_prepositions.xml",  # If prepositions exist
    #"c": "eurosense_fr_conjunctions.xml",  # If conjunctions exist
    #"s": "eurosense_fr_adj_satellites.xml"  # If adjective satellites exist
}

# Open output files in binary write mode
output_handles = {pos: open(filename, "wb") for pos, filename in output_files.items()}

# Write XML headers to each file
for out in output_handles.values():
    out.write(b'<?xml version="1.0" encoding="utf-8"?>\n<corpus source="europarl">\n')

# Initialize counters for tracking progress
sentence_count = 0  # Total number of sentences processed
french_sentence_count = 0  # Number of French sentences extracted
pos_counts = {pos: 0 for pos in output_files}  # Count sentences per POS type

# Parse the XML file in streaming mode to avoid memory overload
for event, elem in ET.iterparse(input_file, events=("start", "end")):  
    # Check if we have reached the end of a <sentence> element
    if event == "end" and elem.tag == "sentence":  
        sentence_count += 1  # Increment total sentence count

        # Find the French version of the text within the <text> tags
        text_fr = elem.find("text[@lang='fr']")  

        # Process only sentences that contain French text
        if text_fr is not None:  
            french_sentence_count += 1  # Increment count of French sentences

            # Dictionary to store sentences categorized by POS type
            sentences_by_pos = {pos: None for pos in output_files}
            annotations_by_pos = {pos: [] for pos in output_files}  # Store annotations per POS

            # Iterate through annotations and classify them by POS type
            for annotation in elem.findall("annotations/annotation[@lang='fr']"):
                babelnet_id = annotation.text.strip()
                if babelnet_id.startswith("bn:") and len(babelnet_id) == 12:  
                    pos_letter = babelnet_id[-1]  # Extract the last character (POS)

                    # Ensure the POS type exists in the predefined mapping
                    if pos_letter in output_files:
                        annotations_by_pos[pos_letter].append(annotation)

            # Write sentences only to the correct POS file if it has a valid annotation
            for pos, annotations in annotations_by_pos.items():
                if annotations:  # If there are valid annotations for this POS
                    sentence_elem = ET.Element("sentence", id=elem.attrib["id"])
                    sentence_elem.append(text_fr)  # Add the French sentence text

                    annotations_elem = ET.Element("annotations")
                    for annotation in annotations:
                        annotations_elem.append(annotation)  # Add only matching POS annotations

                    sentence_elem.append(annotations_elem)

                    # Write the sentence to the corresponding POS file
                    output_handles[pos].write(ET.tostring(sentence_elem, encoding="utf-8"))
                    pos_counts[pos] += 1  # Increment count for this POS type

        # Free memory after processing each sentence
        elem.clear()  

        # Display progress every 100,000 sentences
        if sentence_count % 100000 == 0:
            print(f"Processed {sentence_count:,} sentences... ({french_sentence_count:,} French sentences extracted)")
            for pos, count in pos_counts.items():
                print(f"  - {pos.upper()} Sentences: {count:,}")

# Close the root <corpus> tag in each output file
for out in output_handles.values():
    out.write(b"\n</corpus>")
    out.close()

# Print final summary after processing all sentences
print(f"\nProcessing complete! Total sentences processed: {sentence_count:,}")
print(f"Total French sentences extracted: {french_sentence_count:,}")
for pos, count in pos_counts.items():
    print(f"{pos.upper()} Sentences: {count:,}")




# Ancien code 
'''
# Import the built-in XML parsing module
import xml.etree.ElementTree as ET  

# Define file paths
input_file = "eurosense_fr.v1.0.high-precision.xml"  # Path to the original EuroSense dataset (large file, ~4GB)

# Define output files for each POS type
output_files = {
    "n": "eurosense_fr_nouns.xml",
    "v": "eurosense_fr_verbs.xml",
    "a": "eurosense_fr_adjectives.xml",
    "r": "eurosense_fr_adverbs.xml",
    "p": "eurosense_fr_prepositions.xml",  # If prepositions exist
    "c": "eurosense_fr_conjunctions.xml",  # If conjunctions exist
    "s": "eurosense_fr_adj_satellites.xml"  # If adjective satellites exist
}

# Open output files in binary write mode
output_handles = {pos: open(filename, "wb") for pos, filename in output_files.items()}

# Write XML headers to each file
for out in output_handles.values():
    out.write(b'<?xml version="1.0" encoding="utf-8"?>\n<corpus source="europarl">\n')

# Initialize counters for tracking progress
sentence_count = 0  # Total number of sentences processed
french_sentence_count = 0  # Number of French sentences extracted
pos_counts = {pos: 0 for pos in output_files}  # Count sentences per POS type

# Parse the XML file in streaming mode to avoid memory overload
for event, elem in ET.iterparse(input_file, events=("start", "end")):  
    
    # Check if we have reached the end of a <sentence> element
    if event == "end" and elem.tag == "sentence":  
        sentence_count += 1  # Increment total sentence count

        # Find the French version of the text within the <text> tags
        text_fr = elem.find("text[@lang='fr']")  

        # If a French sentence is found, process it
        if text_fr is not None:  
            french_sentence_count += 1  # Increment French sentence count

            # Dictionary to store sentences per POS type
            sentences_by_pos = {pos: None for pos in output_files}

            # Create a new <sentence> element with the same ID as the original
            sentence_fr = ET.Element("sentence", id=elem.attrib["id"])  
            sentence_fr.append(text_fr)  # Add the extracted French text

            # Categorize annotations by POS type
            for annotation in elem.findall("annotations/annotation[@lang='fr']"):
                babelnet_id = annotation.text.strip()
                if babelnet_id.startswith("bn:") and len(babelnet_id) == 12: 
                    pos_letter = babelnet_id[-1]  # Extract POS type from last character

                    # Ensure we only process known POS
                    if pos_letter in output_files:
                        # Create a new sentence element for this POS if not already created
                        if sentences_by_pos[pos_letter] is None:
                            sentences_by_pos[pos_letter] = ET.Element("sentence", id=elem.attrib["id"])
                            sentences_by_pos[pos_letter].append(text_fr)

                        # Add only the relevant annotations to each POS file
                        annotations_pos = ET.Element("annotations")

                        # *** Only add the correct POS annotations ***
                        if pos_letter == "v":  # VERBS FILE: ONLY ADD VERB ANNOTATIONS
                            if annotation.text.strip().endswith("v"):
                                annotations_pos.append(annotation)
                        else:
                            annotations_pos.append(annotation)  # Keep all relevant POS in their files

                        # Ensure we only write sentences that contain at least one relevant annotation
                        if len(annotations_pos) > 0:
                            sentences_by_pos[pos_letter].append(annotations_pos)

            # Write each filtered sentence to its respective output file
            for pos, sentence in sentences_by_pos.items():
                if sentence is not None:
                    output_handles[pos].write(ET.tostring(sentence, encoding="utf-8"))
                    pos_counts[pos] += 1  # Increment counter for this POS type

        # Clear the processed element from memory to prevent RAM overload
        elem.clear()  

        # Print progress update every 100,000 sentences
        if sentence_count % 100000 == 0:
            print(f"Processed {sentence_count:,} sentences... ({french_sentence_count:,} French sentences extracted)")
            for pos, count in pos_counts.items():
                print(f"  - {pos.upper()} Sentences: {count:,}")

# Close the root <corpus> tag in each output file
for out in output_handles.values():
    out.write(b"\n</corpus>")
    out.close()

# Print final summary after processing all sentences
print(f"\nProcessing complete! Total sentences processed: {sentence_count:,}")
print(f"Total French sentences extracted: {french_sentence_count:,}")
for pos, count in pos_counts.items():
    print(f"{pos.upper()} Sentences: {count:,}")
'''
