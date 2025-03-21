# Author: Lila Roig

import os
from glob import glob
from typing import Dict, Iterable, List, Literal, Any

from allennlp.common.logging import logger
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.fields import ArrayField, LabelField, SpanField, TextField

from bssp.common.embedder_model import EmbedderModelPredictor

from bs4 import BeautifulSoup as BS
import numpy as np
from tqdm import tqdm

import re

def lemma_from_label(label: str) -> str:
    """
    Extracts the lemma from a given label.
    
    Expected label format:
        - "with_1(1)" → Returns "with"
        - "aboutir_3(2)" → Returns "aboutir"
    
    Args:
        label (str): The full label containing the lemma and sense information.
    
    Returns:
        str: The extracted lemma.
    """
    return label.split("_")[0]  # Extract everything before the first "_"
        

def extract_label(label: str) -> str:
    """
    Extracts the appropriate label format

    ******
    Note: Understanding the label format:
    Example: `__ws_2_aboutir__verb__1`
    
    - `__ws_`   -> Standard prefix indicating a Word Sense (ws)
    - `2`       -> Sense cluster (Cluster ID): Groups similar meanings together to facilitate disambiguation.
                    In some disambiguation frameworks, senses are grouped into clusters.
                    These clusters group closely related senses together, making disambiguation easier.
    - `aboutir` -> Verb lemma: The base form of the word.
    - `verb`    -> Grammatical category (verb in this case).
    - `1`       -> Sense number within the cluster: Specifies the exact meaning of the verb in its group.
                    Represents the exact sense of the verb within its assigned cluster.
    ******

    Args
        label (str): Expected label formats for the FLUE dataset. Ex: "__ws_2_aboutir__verb__1" 

    Returns:
        str: The extracted label correctly formatted. Ex: returns "aboutir_2(1)"
    """
    parts = label.split("_")  # Example input: "__ws_2_aboutir__verb__1"
    if len(parts) >= 9:
        cluster = parts[3] # Sense cluster (e.g., '2')
        lemma = parts[4]  # Verb lemma (e.g., 'aboutir')
        fine_sense = parts[-1]  # Fine-grained sense (e.g., '1')

        # If fine_sense contains letters (like "1b"), preserve them
        if re.search(r"\d+[a-z]?", fine_sense):
            return f"{lemma}_{cluster}({fine_sense})"  # e.g., "aboutir_2(1b)"
        
        return f"{lemma}_{cluster}({fine_sense})"  # Default: "aboutir_2(1)"
    
    raise ValueError(f"Unexpected label format: {label}")

@DatasetReader.register("flue_verb") #clres_flue
class FlueVerbReader(DatasetReader): #class ClresFlueReader
    """
    A custom DatasetReader for reading FLUE verb dataset in XML format and associating 
    verb instances with their correct sense labels from a .gold.key.txt file.
    
    This class can be configured to use either:
    - Only the sense cluster (`2`)
    - Both the sense cluster and specific sense (`2_1`)
    """

    def __init__(
        self,
        split: Literal["train", "test", "all"],
        token_indexers: Dict[str, TokenIndexer] = None,
        embedding_predictor: EmbedderModelPredictor = None,
        **kwargs,
    ):
        """
        Initializes the reader with dataset split type, token indexers, and label extraction method.

        Args:
            split (Literal["train", "test", "all"]): Dataset split type.
            token_indexers (Dict[str, TokenIndexer], optional): Token indexers for AllenNLP.
            embedding_predictor: Optional predictor for embeddings.
        """
        super().__init__(**kwargs)
        self.split = split
        self.token_indexers = token_indexers or {"tokens": TokenIndexer()}
        self.embedding_predictor = embedding_predictor
      
    def load_gold_labels(self, gold_labels_path: str) -> Dict[str, str]:
        """
        Loads the gold standard labels from the .gold.key.txt file.

        Args:
            gold_labels_path (str): Path to the .gold.key.txt file containing sense labels.

        Returns:
            Dict[str, str]: A dictionary mapping instance IDs to their corresponding sense labels.
        """

        # Initialize an empty dictionary to store the mapping of instance IDs to their labels
        # The key is an instance ID (e.g., d000.s000.t000).
        # The value is the gold label (e.g., __ws_2_aboutir__verb__1).
        gold_labels = {}

        # Open the .gold.key.txt file for reading
        with open(gold_labels_path, "r", encoding="utf-8") as f:  
            # Iterate over each line in the file
            for line in f:
                # Remove any leading/trailing whitespace and split the line into parts using spaces as delimiters
                parts = line.strip().split()

                # Ensure the line contains exactly two elements (ID + label) before processing
                if len(parts) == 2:
                    # Assign the first element as the key : parts[0] -> Instance ID (e.g., d000.s000.t000).
                    # and the second element  as the value in the dictionary: parts[1] -> Sense label (e.g., __ws_2_aboutir__verb__1).
                    gold_labels[parts[0]] = parts[1]

        # Return the dictionary containing the instance-to-label mappings
        return gold_labels

    def text_to_instance(self, tokens: List[str], span_start: int, span_end: int, label: str, embeddings: np.ndarray = None) -> Instance:
        """
        Creates an AllenNLP instance from tokenized text and its associated verb sense label.

        Args:
            tokens (List[str]): The list of words in the sentence.
            span_start (int): Index of the target verb in the sentence.
            span_end (int): Index of the target verb (usually the same as span_start).
            label (str): The sense label extracted from .gold.key.txt.

        Returns:
            Instance: An AllenNLP `Instance` containing text, target verb span, and label fields.
        """
        tokens = [Token(t) for t in tokens]  # Convert raw words into AllenNLP Token objects
        text_field = TextField(tokens, self.token_indexers)  # Store the full sentence
        lemma_span_field = SpanField(span_start, span_end, text_field)  # Mark the verb position
        label_field = LabelField(label)  # Store the label as a classification target

        lemma = lemma_from_label(label) #Extract the lemma usinf the lemma_from_label function
        lemma_field = LabelField(lemma, label_namespace="lemma_labels")

        fields = {"text": text_field, "label_span": lemma_span_field, "label": label_field, "lemma": lemma_field}
        if self.embedding_predictor: # If embeddings are available, add them
            fields["span_embeddings"] = ArrayField(embeddings[span_start : span_end + 1, :])
        return Instance(fields)
    


    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads an XML file and extracts sentences containing annotated verb instances.

        Args:
            file_path (str): Path to the directory containing the XML files.

        Yields:
            AllenNLP Instance objects, each containing a tokenized sentence and verb sense information.
        """
        xml_file_paths = sorted(glob(file_path + os.sep + "*.xml")) # Find all XML files in the directory

        print("xml_file_paths:", xml_file_paths)  
 
        gold_labels_file_paths =  sorted(glob(file_path + os.sep + "*.gold.key.txt"))   

        print("gold_labels_file_paths:", gold_labels_file_paths) 

        if len(xml_file_paths) != len(gold_labels_file_paths):
            raise ValueError(f"There should be the same number of files for the XML files in {xml_file_paths} and the gold.key.txt in {gold_labels_file_paths}") 

        for i, xml_file_path in enumerate(xml_file_paths):  # loop over all XML files founded, and also over all gold.key.txt files, which should be the same number as XML files
            
            gold_labels = self.load_gold_labels(gold_labels_file_paths[i]) # load gold labels 

            print(f"[{i}/{len(xml_file_paths)}] Processing", xml_file_path)

            with open(xml_file_path, "r", encoding="utf-8") as f: 
                soup = BS(f.read(), "xml")  # Parse XML content using BeautifulSoup

            # Iterate over each sentence in the XML
            # Retrieves what is inside <sentence...>  </sentence> tag 
            for sentence in tqdm(soup.find_all("sentence")): 
                tokens = []  # Store sentence words
                verb_index = None  # The index position of the verb in the sentence
                verb_label = None  # The correct sense label for the verb

                # Iterate through each word in the sentence
                # Iterate over each word (<wf> or <instance>) in the sentence 
                for i, word in enumerate(sentence.find_all(["wf", "instance"])):
                    tokens.append(word.text)  # Store the word's text 

                    # If this is a verb instance (annotated target)
                    if word.name == "instance": 
                        verb_index = i  # Mark verb position 
                        instance_id = word["id"]  # Get unique ID for this verb occurrence

                        # Retrieve correct sense label from .gold.key.txt
                        if instance_id in gold_labels:   
                            verb_label = extract_label(gold_labels[instance_id]) 
                        else:
                            logger.warn(f"Warning: No label found for {instance_id}, skipping.")
                            continue  # Skip instances that have no label

                # Generate embeddings if a predictor is provided
                embeddings = None
                if self.embedding_predictor:
                    embeddings = np.array(self.embedding_predictor.predict(tokens)["embeddings"],  dtype=np.float16) # dtype=np.float16: because otherwise this is too long to run

                # Create an instance if a verb was found with a valid label
                if verb_index is not None and verb_label is not None:
                    yield self.text_to_instance(tokens, verb_index, verb_index, verb_label, embeddings=embeddings) 
                    # What will be sent to AllenNLP:
                    # Instance({
                    #    "text": ["Il", "rend", "hommage", "aboutissent", "traité"],
                    #    "label_span": (3, 3),
                    #    "label": "2",
                    #    "span_embeddings": None  # (or embeddings if available)
                    #})



'''
# Examples of usage - checks that everything works well
# ********************************************

# CASE 1
# ----------------------
reader = FlueVerbReader( 
    split="train", #
    #gold_labels_path="DATA_DIR/test/FSE-1.1.gold.key.txt",
    #label_type="full"  # Keeps only '2'
)


instances = list(reader._read("/Users/lila/Documents/Master_MVA/COURS/S2/AlgoForSpeechRecognition/projet/bert-has-uncommon-sense-deep-seek/data/flueverb/train_small"))

# Print an example instance
print(instances[0])
print(instances[1])
print(instances[2])


# CASE 1
# ----------------------
reader = FlueVerbReader( 
    split="train", #
    gold_labels_path="DATA_DIR/train/wiktionary-190418.gold.key.txt",
    label_type="group"  # Keeps only '2'
)

instances = list(reader._read("DATA_DIR/train/wiktionary-190418.data.xml"))

# Print an example instance
print(instances[0])



# CASE 2
# ----------------------
reader = FlueVerbReader(
    split="train",
    gold_labels_path="DATA_DIR/train/wiktionary-190418.gold.key.txt",
    label_type="full"  # Keeps '2_1'
)

instances = list(reader._read("DATA_DIR/train/wiktionary-190418.data.xml"))

# Print an example instance
print(instances[0])

'''
