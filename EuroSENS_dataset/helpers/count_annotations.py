"""
===================================================================
EuroSense XML Annotation Counter
===================================================================
This script counts the total number of `<annotation>` elements 
in an XML dataset, such as EuroSense.

**Features:**
   - Efficient parsing of large XML files using `iterparse()`.
   - Counts all `<annotation>` tags without loading the full file into memory.
   - Works with any XML dataset containing annotation elements.

===================================================================
USAGE:
===================================================================
Run the script from the terminal:

    python count_annotations.py eurosense_fr_verbs.xml

Replace `eurosense_fr_verbs.xml` with your XML file.

===================================================================
OUTPUT EXAMPLE:
===================================================================
If the file contains 150,000 annotations, the script prints:

    Total number of annotations: 150000

Author: Lila Roig
"""

import xml.etree.ElementTree as ET
import argparse

def count_annotations(xml_file: str) -> int:
    """
    Counts the total number of annotations in an XML dataset.
    """
    annotation_count = 0
    
    # Efficiently parse XML without loading everything into memory
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if elem.tag == "annotation":  # Count each annotation element
            annotation_count += 1
        elem.clear()  # Free memory

    return annotation_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count the number of annotations in a EuroSense XML file.")
    parser.add_argument("xml_file", type=str, help="Path to the XML file")
    args = parser.parse_args()

    total_annotations = count_annotations(args.xml_file)
    print(f"Total number of annotations: {total_annotations}")


# HOW to use
# python count_annotations.py eurosense_fr_verbs.xml
