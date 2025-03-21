# Author: Lila Roig

import os
import xml.etree.ElementTree as ET
import numpy as np
import re
from typing import Dict, Iterable, List, Any, Literal
import time  ### NEW ###

from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.fields import TextField, SpanField, LabelField, ArrayField
from bssp.common.embedder_model import EmbedderModelPredictor


def lemma_from_label(label: str) -> str:
    """Extracts the lemma from a given label (e.g., 'attendre_bn:00082820v' → 'attendre')"""
    return label.split("_")[0]


def tokenize_text(text):
    """
    Tokenization function that extracts words while keeping apostrophes.
    It removes all punctuation except apostrophes and hyphens.
    Example:
        - "c'est mon ami." -> ["c'est", "mon", "ami"]
        - "mon ami" -> ["mon", "ami"]
        - "New-York" -> ["New-York"]
    """
    return re.findall(r"\b\w+(?:[-']\w+)*\b|&apos;", text.lower())


@DatasetReader.register("eurosense")
class EuroSenseReader(DatasetReader):
    """
    A DatasetReader for the EuroSense dataset.

    - Processes XML files in streaming mode (`ET.iterparse()`).
    - Supports both **embedding-based** and **non-embedding** modes.

    The dataset should be **pre-filtered** to contain only French sentences.
    """

    def __init__(
        self,
        split: Literal["train", "test", "all"],
        token_indexers: Dict[str, TokenIndexer] = None,
        embedding_predictor: EmbedderModelPredictor = None,
        use_embeddings: bool = True,
        **kwargs,
    ):
        """
        :param split: "train", "test", or "all".
        :param token_indexers: Dictionary mapping token types to indexers.
        :param embedding_predictor: Model for extracting embeddings (if needed).
        :param use_embeddings: Whether to load embeddings (default: False).
        """
        super().__init__(**kwargs)
        self.split = split
        self.token_indexers = token_indexers or {"tokens": TokenIndexer()}
        self.embedding_predictor = embedding_predictor if use_embeddings else None

    def text_to_instance(
        self, tokens: List[str], span_start: int, span_end: int, label: str, embeddings: np.ndarray = None
    ) -> Instance:
        """
        Converts raw text and sense annotations into an AllenNLP Instance.

        :param tokens: Tokenized sentence.
        :param span_start: Index of the annotated word.
        :param span_end: Index of the annotated word.
        :param label: Sense label (formatted as `lemma_bnID`).
        :param embeddings: Optional precomputed embeddings.
        """
        tokens = [Token(t) for t in tokens]
        text_field = TextField(tokens, self.token_indexers)
        lemma_span_field = SpanField(span_start, span_end, text_field)
        label_field = LabelField(label)
        lemma_field = LabelField(lemma_from_label(label), label_namespace="lemma_labels")

        fields = {
            "text": text_field,
            "label_span": lemma_span_field,
            "label": label_field,
            "lemma": lemma_field,
        }

        if self.embedding_predictor:
            fields["span_embeddings"] = ArrayField(embeddings[span_start:span_end + 1, :])

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the EuroSense dataset from an XML file.

        Args:
            file_path (str): Path to the dataset XML file.

        Yields:
            AllenNLP Instance objects, each containing a tokenized sentence and verb sense information.
        """

        ### NEW ### - Initialization of tracking variables
        total_sentences = sum(1 for event, elem in ET.iterparse(file_path, events=("end",)) if elem.tag == "sentence")
        processed_sentences = 0
        start_time = time.time()
        progress_step = total_sentences // 20  # Display a message every 5% (1/20)
        print(f"DEBUG: Total sentences in dataset: {total_sentences}")
        ### NEW ###


        for event, elem in ET.iterparse(file_path, events=("end",)):
            if elem.tag == "sentence":

                text_elem = elem.find("text[@lang='fr']")
                if text_elem is None:
                    elem.clear()
                    continue

                text = text_elem.text.strip()
                tokens = tokenize_text(text)  # Properly tokenize text
                annotations = []

                # Extract all annotated words and their senses
                for annotation in elem.findall("annotations/annotation"):
                    bn_id = annotation.text.strip()  # BabelNet sense ID
                    lemma = annotation.attrib.get("lemma", "").strip()
                    anchor = annotation.attrib.get("anchor", "").strip()  # The actual word in the sentence
                    annotations.append((bn_id, lemma, anchor))

                # Process annotations
                for bn_id, lemma, anchor in annotations:
                    label = f"{lemma}_{bn_id}"  # Format: "lemma_bn:XXXXXX"
                    span_start, span_end = self._find_word_span(tokens, anchor)

                    # Compute embeddings if enabled
                    embeddings = None
                    if self.embedding_predictor:
                        embeddings = np.array(self.embedding_predictor.predict(tokens)["embeddings"])

                    yield self.text_to_instance(tokens, span_start, span_end, label, embeddings)

                elem.clear()

                ### NEW ### - Update and display progression 
                processed_sentences += 1
                if processed_sentences % progress_step == 0:  # Display each 5% progression
                    elapsed_time = time.time() - start_time  # Time spend in secondes 
                    processing_speed = processed_sentences / elapsed_time  # Sentences processed per second
                    remaining_sentences = total_sentences - processed_sentences
                    estimated_remaining_time = remaining_sentences / processing_speed  # Estimation of the remaing time in seconds

                    elapsed_minutes = int(elapsed_time // 60)  # Minutes passed
                    elapsed_seconds = int(elapsed_time % 60)  # Seconds passed

                    remaining_minutes = int(estimated_remaining_time // 60)  # Remaining minutes
                    remaining_seconds = int(estimated_remaining_time % 60)  # Remaining secondes

                    percent_done = (processed_sentences / total_sentences) * 100

                    print(f"DEBUG: {processed_sentences}/{total_sentences} sentences processed "
                        f"({percent_done:.1f}%) - Time elapsed: {elapsed_minutes} min {elapsed_seconds} sec "
                        f"- Estimated remaining time: {remaining_minutes} min {remaining_seconds} sec")
                ### NEW ### 

    def _find_word_span(self, tokens: List[str], anchor: str) -> (int, int):
        """
        Finds the position of an `anchor` word in the tokenized text.
        Supports both single-word and multi-word anchors.

        :param tokens: Tokenized sentence.
        :param anchor: The exact word or phrase found in the XML under `anchor`.
        :return: (start_index, end_index)
        """
        anchor_tokens = tokenize_text(anchor)  # Tokenize the anchor in the same way as the sentence
        anchor_len = len(anchor_tokens)

        tokens_lower = [t.lower() for t in tokens]
        anchor_tokens_lower = [t.lower() for t in anchor_tokens]

        # Try to find the full sequence in the tokenized sentence
        for i in range(len(tokens) - anchor_len + 1):
            if tokens_lower[i:i + anchor_len] == anchor_tokens_lower:
                return i, i + anchor_len - 1  # Start and end span of the multi-word expression
            
        print(f"WARNING: Anchor '{anchor}' not found in tokens: {tokens}")
        return 0, 0  # Default if not found



'''
# Examples of usage - check if everything works well 
#*************************************************

# Define the dataset file path
dataset_name = "eurosense_fr_verbs_train_10per.xml"
dataset_path = "/Users/lila/Documents/Master_MVA/COURS/S2/AlgoForSpeechRecognition/projet/bert-has-uncommon-sense-deep-seek/data/eurosens/" + dataset_name

# Initialize the dataset reader
reader = EuroSenseReader(
    split="train",
    use_embeddings=True,  # Set to True if embeddings are required
)

# -----------------------------------------------
# Inspect the first French sentence in the dataset
# -----------------------------------------------
# Parse the XML file
tree = ET.parse(dataset_path)
root = tree.getroot()

# Find the first sentence that contains French text
for sentence in root.findall("sentence"):
    text_elem = sentence.find("text[@lang='fr']")
    if text_elem is not None:
        print("\nFirst sentence in the dataset (raw format):")
        print(f"Sentence ID: {sentence.get('id')}")
        print(f"Original text: {text_elem.text.strip()}")

        print("\nFound annotations:")
        for annotation in sentence.findall("annotations/annotation"):
            bn_id = annotation.text.strip()  # BabelNet sense ID
            lemma = annotation.attrib.get("lemma", "").strip()  # Associated lemma
            anchor = annotation.attrib.get("anchor", "").strip()  # Annotated word
            coherence_score = annotation.attrib.get("coherenceScore", "N/A")  # Coherence score
            nasari_score = annotation.attrib.get("nasariScore", "N/A")  # NASARI score (if available)

            print(f"   - BabelNet ID: {bn_id}")
            print(f"   - Lemma: {lemma}")
            print(f"   - Anchor (annotated word): {anchor}")
            print(f"   - Coherence Score: {coherence_score}")
            print(f"   - Nasari Score: {nasari_score}\n")

        break  # Stop after processing the first sentence

# -----------------------------------------------
# Extract and display the first complete sentence
# -----------------------------------------------
# Re-parse the XML file to find the first full sentence
tree = ET.parse(dataset_path)
root = tree.getroot()
first_sentence = None

for sentence in root.findall("sentence"):
    first_sentence = sentence
    break  # Take only the first sentence

# Display the raw content of the first sentence (without filtering)
if first_sentence is not None:
    print("\n**Raw content of the first sentence**")
    print(ET.tostring(first_sentence, encoding="unicode"))
else:
    print("No sentence found in the XML file.")

print("-------------------")

# -----------------------------------------------
# Display the first five sentences and their annotations
# -----------------------------------------------

# Parse the XML file again
tree = ET.parse(dataset_path)
root = tree.getroot()

# Iterate through the first five sentences
for i, sentence in enumerate(root.findall("sentence")):
    text_elem = sentence.find("text[@lang='fr']")
    if text_elem is None:
        continue  # Skip if no French text is found

    text = text_elem.text.strip()
    print(f"\n**Sentence {i+1}**")
    print(f"Text: {text}")

    annotations = sentence.findall("annotations/annotation")
    if not annotations:
        print("No annotation found for this sentence.")
        continue

    print("**Found annotations**:")
    for annotation in annotations:
        lang = annotation.attrib.get("lang", "??")
        anchor = annotation.attrib.get("anchor", "??")
        lemma = annotation.attrib.get("lemma", "??")
        bn_id = annotation.text.strip()
        print(f"  -  Language: {lang} | Anchor: {anchor} | Lemma: {lemma} | BabelNet ID: {bn_id}")

    if i == 4:  # Stop after processing five sentences
        break

print("------------------------------")

# -----------------------------------------------
# Detect missing anchors in the text
# -----------------------------------------------

# Parse the XML file once again
tree = ET.parse(dataset_path)
root = tree.getroot()

# Initialize error count
error_count = 0

# Iterate through all sentences
for sentence in root.findall("sentence"):
    text_elem = sentence.find("text[@lang='fr']")
    if text_elem is None:
        continue

    text = text_elem.text.strip()
    tokens = text.split()  # Simple tokenization for testing

    # Check each annotation
    for annotation in sentence.findall("annotations/annotation"):
        anchor = annotation.attrib.get("anchor", "??")
        lemma = annotation.attrib.get("lemma", "??")
        bn_id = annotation.text.strip()

        # Verify if the anchor is present in the sentence
        if anchor.lower() not in text.lower():
            print("\n**Anchor not found in the sentence!** ")
            print(f"Text: {text}")
            print(f"Anchor (annotated word): {anchor}")
            print(f"Lemma: {lemma} | BabelNet ID: {bn_id}")
            error_count += 1

    if error_count >= 10:  # Display only the first 10 errors
        break

print(f"\nTotal number of errors found: {error_count}\n")

print("-----------------------------------")

# -----------------------------------------------
# Print an example instance
# -----------------------------------------------
# Load dataset into AllenNLP instances

print("**Read all dataset...**")
dataset = list(reader.read(dataset_path))

print("**Print the first elements of the formatted dataset**\n")
print(dataset[0])
print(dataset[1])
print(dataset[2])
'''
