"""
This file is used to check how the labels in the CLRES dataset are formatted by the ClresConlluReader class in bssp/clres/dataset_reader.py. 
We use this so that we can create the file bssp/flue/dataset_reader.py and correctly format the labels in the FlueVerbReader class for the FLUE dataset
following the CLRES format.
To run this code, this file has to be placed in the main bert-has-uncommon-sense directory. 

Author: Lila Roig
"""

import os
import unittest
from allennlp.data import DatasetReader
from bssp.clres.dataset_reader import ClresConlluReader # Make sure to import your dataset reader
from typing import List


class TestClresConlluReader(unittest.TestCase):
    """
    Unit test for the ClresConlluReader class using real PDEP test data.
    """

    def setUp(self):
        """
        Set up the test by defining paths to the real dataset.
        """
        # Ensure this path points to your real .conllu dataset
        self.test_file = "/Users/lila/Documents/Master_MVA/COURS/S2/AlgoForSpeechRecognition/projet/bert-has-uncommon-sense-deep-seek/data/pdep/pdep_test.conllu"

        # Load dataset reader
        self.reader = ClresConlluReader(split="test")

    def test_read_instances(self):
        """
        Test if ClresConlluReader correctly reads and processes a CoNLL-U file.
        """
        # Read instances from the dataset
        instances = list(self.reader._read(self.test_file))

        # Print a few instances to inspect their structure
        print(f"Total instances loaded: {len(instances)}")
        for i, instance in enumerate(instances[:5]):  # Print only first 5 samples
            print(f"Instance {i+1}:")
            print(instance)

        # Validate that instances are correctly structured
        self.assertGreater(len(instances), 0, "No instances were loaded!")

        for instance in instances:
            # Check if "text" field exists
            self.assertIn("text", instance.fields, "Missing 'text' field in instance!")
            text_field = instance.fields["text"]
            self.assertTrue(isinstance(text_field, type(instances[0].fields["text"])), "Invalid text field type!")

            # Check if "label" exists and follows expected format
            self.assertIn("label", instance.fields, "Missing 'label' field in instance!")
            label_field = instance.fields["label"]
            label_str = label_field.label
            self.assertTrue("_" in label_str, f"Unexpected label format: {label_str}")

            # Ensure "label_span" correctly marks a single verb in the sentence
            self.assertIn("label_span", instance.fields, "Missing 'label_span' field!")
            span_field = instance.fields["label_span"]
            span_start, span_end = span_field.span_start, span_field.span_end
            self.assertEqual(span_start, span_end, f"Span field is not a single-word span: {span_start}, {span_end}")

        print("Test passed: All instances correctly structured!")


if __name__ == "__main__":
    unittest.main()
