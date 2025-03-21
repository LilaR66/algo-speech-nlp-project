"""
===================================================================
EuroSense French Sentence Filter & Anchor Validation
===================================================================
This script processes the **EuroSense dataset (`eurosense.v1.0.high-precision.xml`)**
to extract only **French (`lang="fr"`) sentences** along with their annotations.
It also validates annotation anchors by ensuring they are present in the sentence 
and meet quality criteria.

===================================================================
KEY FEATURES:
===================================================================
- **Extracts only French sentences** from the EuroSense dataset.
- **Filters out sentences with no valid French annotations**.
- **Validates annotation anchors**:
    - Ensures anchors appear in the corresponding sentence.
    - Removes anchors that are empty, numeric, or punctuation-only.
- **Logs anchor mismatches and invalid cases** for review.
- **Optimized for large datasets (~4GB) with streaming XML parsing**.

===================================================================
PROCESSING DETAILS:
===================================================================
- The script processes **each sentence** incrementally to prevent memory overload.
- It **normalizes** text and tokenizes words to improve anchor detection.
- It prints **progress updates every 100,000 sentences**.

- **Sentences are removed if:**
  - They contain **no valid French annotations** after filtering.
- **Annotations are removed if:**
  - The anchor is **not found** in the sentence (case of anchor mismatch: 
    the anchor uses a different language compared to the sentence)
  - The anchor is **empty, numeric, or contains only punctuation**.

===================================================================
HOW TO RUN:
===================================================================
Simply execute:
    python filter_eurosense_french.py

If you want to specify a different dataset location, modify `input_file`.

===================================================================
OUTPUT FILES:
===================================================================
- **Filtered French dataset** → `eurosense_fr.v1.0.high-precision.xml`
- **Mismatch log file** → `anchor_mismatch_log.txt`
  (Contains removed annotations due to mismatched or invalid anchors)

  
Author: Lila Roig
"""

# Version avec gestion des mismatch, fichier de log, suppression anchors invalides
# et si aucune annotation apres filtrage suppression de la phrase 

import xml.etree.ElementTree as ET  
import re  # Import regex for improved tokenization

# Define file paths
input_file = "EuroSense/eurosense.v1.0.high-precision.xml"  # Original dataset (~4GB)
output_file = "eurosense_fr.v1.0.high-precision.xml"  # Filtered dataset
log_file_path = "anchor_mismatch_log.txt"  # Log file for mismatches

# Initialize counters to track processing progress
sentence_count = 0  # Total sentences processed
french_sentence_count = 0  # Number of extracted French sentences
filtered_out_count = 0  # Sentences removed due to lack of valid French annotations
anchor_mismatch_count = 0  # Annotations removed due to missing anchors
invalid_anchor_count = 0  # Annotations removed due to being empty, numeric, or punctuation-only

# Open log file for writing mismatch cases
log_file = open(log_file_path, "w", encoding="utf-8")

# Improved tokenization function for better word boundary detection
def tokenize(text):
    """
    Tokenization function that extracts words while keeping apostrophes.
    It removes all punctuation except apostrophes and hyphens.
    Example:
        - "c'est mon ami." -> ["c'est", "mon", "ami"]
        - "mon ami" -> ["mon", "ami"]
        - "New-York" -> ["New-York"]
    """
    return re.findall(r"\b\w+(?:[-']\w+)*\b", text.lower())

# Function to check if an anchor is valid (supports French letters with accents)
def is_valid_anchor(anchor):
    """
    Check if an anchor is valid:
    - It must contain at least one French letter (including accents).
    - It must not be empty.
    """
    return bool(re.search(r'[a-zA-Zàâçéèêëîïôûùæœ]', anchor))  # Accepts French letters with accents

# Open the output file in binary write mode ("wb")
with open(output_file, "wb") as out:  
    # Write the XML header and open the <corpus> tag
    out.write(b'<?xml version="1.0" encoding="utf-8"?>\n<corpus source="europarl">\n')  


    # Process the XML file incrementally to avoid memory overflow
    for event, elem in ET.iterparse(input_file, events=("start", "end")):  
        
        # Process sentences when the closing tag </sentence> is encountered
        if event == "end" and elem.tag == "sentence":  
            sentence_count += 1  # Increment total sentence count

            # Extract the French version of the text
            text_fr = elem.find("text[@lang='fr']")  

            # Check if the sentence contains French text
            if text_fr is not None and text_fr.text:  # Ensure text_fr is not None
                sentence_text = text_fr.text.strip().lower()  # Normalize sentence
                sentence_tokens = tokenize(sentence_text)  # Tokenize sentence
                
                # Create a new <sentence> element with the same ID as the original
                sentence_fr = ET.Element("sentence", id=elem.attrib["id"])  
                sentence_fr.append(text_fr)  # Add the extracted French text

                # Create a new <annotations> element to store only valid French annotations
                annotations_fr = ET.Element("annotations")  
                mismatch_detected = False  # Track if any mismatch occurs

                # Extract and store only valid French annotations
                for annotation in elem.findall("annotations/annotation[@lang='fr']"):  
                    anchor_text = annotation.attrib.get("anchor", "").strip().lower()  # Normalize anchor text
                    anchor_tokens = tokenize(anchor_text)  # Tokenize the anchor phrase

                    # Skip multi-word anchors
                    '''
                    if len(anchor_tokens) > 1:
                        continue  # Skip this annotation
                    '''

                    # Check if the anchor appears as an exact sequence in the sentence (FIRST STEP)
                    found = False
                    for i in range(len(sentence_tokens) - len(anchor_tokens) + 1):
                        if sentence_tokens[i:i + len(anchor_tokens)] == anchor_tokens:
                            found = True
                            break  # Stop searching after first match
                    
                    if not found:  # If the anchor is not in the sentence, we log and remove it
                        anchor_mismatch_count += 1
                        mismatch_detected = True  

                        # Log the issue
                        log_file.write(f"MISMATCH - Sentence ID={elem.attrib['id']}\n")
                        log_file.write(f"Anchor: {anchor_text} NOT found in sentence.\n")
                        log_file.write(f"Sentence: {text_fr.text.strip()}\n")
                        log_file.write("="*80 + "\n")
                        continue  # Skip this annotation (do not check if it's valid)

                    # Now we check if the anchor is valid (SECOND STEP)
                    if not is_valid_anchor(anchor_text):
                        invalid_anchor_count += 1  # Count invalid anchors

                        # Log invalid anchor
                        log_file.write(f"INVALID ANCHOR - Sentence ID={elem.attrib['id']}\n")
                        log_file.write(f"Anchor: '{anchor_text}' is invalid (empty, numeric, or punctuation-only).\n")
                        log_file.write(f"Sentence: {text_fr.text.strip()}\n")
                        log_file.write("="*80 + "\n")
                        continue  # Skip this annotation

                    # If it passes both checks, we keep it
                    annotations_fr.append(annotation)  

                # If no valid French annotations are found, skip the sentence
                if len(annotations_fr) == 0:
                    filtered_out_count += 1
                    elem.clear()  # Free memory
                    continue  # Move to the next sentence

                # Add valid French annotations to the sentence
                sentence_fr.append(annotations_fr)  

                # Write the validated sentence to the output file
                out.write(ET.tostring(sentence_fr, encoding="utf-8"))  
                french_sentence_count += 1  # Increment count of extracted sentences

            else:
                filtered_out_count += 1  # Count missing text cases

            # Free memory after processing each sentence
            elem.clear()  

            # Display progress every 100,000 sentences
            if sentence_count % 100000 == 0:
                print(f"Processed {sentence_count:,} sentences... ({french_sentence_count:,} French sentences extracted, {filtered_out_count:,} skipped)")
                print(f"  - {anchor_mismatch_count:,} annotations removed due to anchor mismatch.")
                print(f"  - {invalid_anchor_count:,} invalid anchors removed (empty, punctuation-only, numeric).")

    # Close the <corpus> tag to ensure valid XML structure
    out.write(b"\n</corpus>")  
    log_file.close()  # Close log file

# Print final processing summary
print("\nProcessing complete")
print(f"Total sentences processed: {sentence_count:,}")
print(f"Total French sentences extracted: {french_sentence_count:,}")
print(f"Total sentences skipped (no valid French annotation): {filtered_out_count:,}")
print(f"Total annotations removed due to anchor mismatch: {anchor_mismatch_count:,}")
print(f"Total invalid anchors removed (empty, punctuation-only, numeric): {invalid_anchor_count:,}")
print(f"\nMismatch log saved in: {log_file_path}")

