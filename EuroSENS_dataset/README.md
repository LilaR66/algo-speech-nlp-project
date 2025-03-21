# EuroSENS Dataset Setup and Preprocessing

This file explains how to download and setup the EuroSENS dataset to reproduce the experiments.\
*Author: Lila Roig*

---
## STEP 1: Download EuroSENS dataset

**Paper of EuroSENS dataset**:  
[https://aclanthology.org/P17-2094.pdf](https://aclanthology.org/P17-2094.pdf)  

**Link to download EuroSENS dataset** *(accessed in February 2025)*:  
[http://lcl.uniroma1.it/eurosense/](http://lcl.uniroma1.it/eurosense/)

We chose to download the **"EuroSense - High-precision [ tar.gz: 3.7 GB ]"** version (since it is smaller and more precise).

The file can now be unzipped and placed in the folder:  
`EuroSENS_dataset/`

The obtained file is named:  
`eurosense.v1.0.high-precision.xml`

---
## (optional) STEP 2: Understanding the structure of the EuroSENS files

See the README of EuroSENS available here *(accessed in February 2025)*:  
[http://lcl.uniroma1.it/eurosense/](http://lcl.uniroma1.it/eurosense/)

Each EuroSense file contains a list of annotated sentences in XML format:

- `<sentence>` tag:  
  Each sentence has a unique ID and contains multiple language versions.

- `<text>` tag:  
  Each sentence version is stored in a `<text>` tag with a `lang` attribute indicating the ISO language code.  
  **Example**:  
  ```xml
  <text lang="fr">Enfin, Monsieur le Président, M. Santer a parlé d'une nouvelle approche.</text>
  ```  
  French sentences use `lang="fr"` and can be easily extracted.

- `<annotations>` tag:  
  This section lists disambiguated words, each with:
  - The **anchor** (word in the sentence)  
  - The **lemma** (base form)  
  - The **BabelNet sense ID** (`bn:XXXXXXX`)  
  - A **coherenceScore** (higher = more reliable)  
  - The **annotation type** (e.g., BABELFY or NASARI)  

  **Example**:
  ```xml
  <annotation lang="fr" type="BABELFY" anchor="parlé" lemma="parler" coherenceScore="0.1414">bn:00090943v</annotation>
  ```

Although the README does not explicitly state the part of speech (POS), it can be inferred from the **BabelNet ID**, where the final letter indicates the POS:

- `n` → Noun (e.g., `bn:00017517n`)  
- `v` → Verb (e.g., `bn:00090943v`)  
- `a` → Adjective (e.g., `bn:00023456a`)  
- `r` → Adverb (e.g., `bn:00056789r`)

Thus, both annotations and word categories can be extracted from the XML structure and BabelNet IDs.

---
## STEP 3: Filter EuroSENS dataset to retain only French annotations

As our focus is on **French**, the dataset is filtered to retain only French annotations.  
The EuroSENS dataset also contains errors such as:

- Mislabeled languages (e.g., sentence in French but anchor in another language)  
- Empty anchors  
- Punctuation-only anchors  

This step also **cleans** the dataset to ensure reliability.

**run**:  
```
conda activate bhus
cd folder/of/the/file
python filter_eurosense_french.py
```

**Hard-coded in** `filter_eurosense_french.py` **we have**:  
```python
input_file    = "eurosense.v1.0.high-precision.xml"     # Original dataset (~4GB)
output_file   = "eurosense_fr.v1.0.high-precision.xml"  # Filtered dataset
log_file_path = "anchor_mismatch_log.txt"               # Log file for mismatches
```

---
## STEP 4: Divide the French EuroSENS dataset into 4 subsets (verbs, nouns, adverbs, adjectives) 

The French EuroSENS dataset is divided into **4 subsets** (*verbs, nouns, adverbs, and adjectives*)  
to compare **CamemBERT-base**’s performance across word types.  
A **combined dataset** containining all verbs, nouns, adverbs and adjectives is retained for reference.

**run**:  
```
conda activate bhus
cd path/to/your/file
python filter_eurosense_french_by_pos.py
```

**Hard-coded in** `filter_eurosense_french_by_pos.py` **we have**:  
```python
input_file = "eurosense_fr.v1.0.high-precision.xml"  # Filtered dataset containing only French sentences
output_files = {  # Define output files for each POS type
    "n": "eurosense_fr_nouns.xml",
    "v": "eurosense_fr_verbs.xml",
    "a": "eurosense_fr_adjectives.xml",
    "r": "eurosense_fr_adverbs.xml",
}
```

**(Optional)** Check that the most frequent word-sense pair has a reasonably high frequency in `file_name.xml`:

**run**:
```
conda activate bhus
cd path/to/your/file/helpers
chmod +x most_frequent_word_sense.sh
./most_frequent_word_sense.sh file_name.xml  # e.g. replace file_name.xml by eurosense_fr_nouns.xml
```

---
## STEP 5: Subsample the datasets and split them into train/test sets

Given the **large dataset size** and limited computational resources,  
**stratified subsampling** based on **quartiles** is applied to obtain a smaller dataset  
with a **balanced representation** of frequent and rare words.  
=> **80,000 annotations** are selected for each subsample.

Then, each subsampled dataset is **split into**:
- **Training set** (80%)  
- **Test set** (20%)  
Both sets retain a mix of frequent and rare labels.


This script also supports **optional filtering**:

- Remove annotations where the anchor contains **multiple words**  
- Discard **labels appearing fewer than 6 times** in the subsample  

This ensures that all test labels appear **at least 5 times** in training,  
matching the logic in `main.py`, which ignores test labels with fewer than 5 training examples.

**run to create train/test splits for** `eurosense_fr_adverbs.xml`:
```
conda activate bhus
cd path/to/your/file/
python create_eurosens_splits.py eurosense_fr_adverbs.xml \                                                     
        eurosense_fr_adverbs_train.xml eurosense_fr_adverbs_test.xml \
        --train_ratio 0.8 --num_annotations 80000
```

**Another example: with filters, for** `eurosense_fr_nouns.xml`:
```
conda activate bhus
cd path/to/your/file/
python create_eurosens_splits.py eurosense_fr_nouns.xml \                                                     
        eurosense_fr_nouns_train.xml eurosense_fr_nouns_test.xml \
        --train_ratio 0.8 --num_annotations 80000 \
        --filter_multiword_anchors --filter_frequent_labels
```

**Another example: for the whole French dataset**:
```
conda activate bhus
cd path/to/your/file/
python create_eurosens_splits.py eurosense_fr.v1.0.high-precision.xml \                                                     
        eurosense_fr_all_train.xml eurosense_fr_all_test.xml \
        --train_ratio 0.8 --num_annotations 80000
```

**(Optional)**: To gain insight into the contents of the train/test splits:

**run**:  
```
conda activate bhus
cd path/to/your/file/helpers
python compute_dataset_stats.py eurosense_fr_nouns_train.xml eurosense_fr_nouns_test.xml train_stats test_stats
```

**Or run**:  
```
conda activate bhus
cd path/to/your/file/helpers
python analyze_split_eurosens.py  # file paths are hard-coded in the script
```


---

## STEP 6: Recap

The creation of the train/test files for **EuroSENS** is done.

After running **STEP 1**, we should obtain the file:  
- `eurosense.v1.0.high-precision.xml`

After running **STEP 3**, we should obtain the file:  
- `eurosense_fr.v1.0.high-precision.xml`

After running **STEP 4**, we should obtain the files:  
- `eurosense_fr_nouns.xml`  
- `eurosense_fr_verbs.xml`  
- `eurosense_fr_adjectives.xml`  
- `eurosense_fr_adverbs.xml`

After running **STEP 4** for all four datasets *(nouns, verbs, adjectives, and adverbs)*,  
as well as for the full French dataset — **both with and without multi-word anchor filtering** —  
you should obtain the files listed in the table below.

Make sure to **move each file into its corresponding folder** under  
`bert-has-uncommon-sense/data/` as indicated, so the rest of the project can run correctly.

| **Source File**                       | **Train File**                      | **Test File**                      | **Anchor Filter**          | **Destination Folder**                                               |
|--------------------------------------|-------------------------------------|------------------------------------|-----------------------------|-----------------------------------------------------------------------|
| `eurosense_fr_nouns.xml`             | `eurosense_fr_nouns_train.xml`      | `eurosense_fr_nouns_test.xml`      | multi-word anchors kept     | `bert-has-uncommon-sense/data/eurosens_nouns`                        |
| `eurosense_fr_nouns.xml`             | `eurosense_fr_nouns_train_swa.xml`  | `eurosense_fr_nouns_test_swa.xml`  | multi-word anchors removed  | `bert-has-uncommon-sense/data/eurosens_nouns_singleWordAnchorOnly`  |
| `eurosense_fr_verbs.xml`             | `eurosense_fr_verbs_train.xml`      | `eurosense_fr_verbs_test.xml`      | multi-word anchors kept     | `bert-has-uncommon-sense/data/eurosens_verbs`                        |
| `eurosense_fr_verbs.xml`             | `eurosense_fr_verbs_train_swa.xml`  | `eurosense_fr_verbs_test_swa.xml`  | multi-word anchors removed  | `bert-has-uncommon-sense/data/eurosens_verbs_singleWordAnchorOnly`  |
| `eurosense_fr_adjectives.xml`        | `eurosense_fr_adjectives_train.xml` | `eurosense_fr_adjectives_test.xml` | multi-word anchors kept     | `bert-has-uncommon-sense/data/eurosens_adjectives`                  |
| `eurosense_fr_adjectives.xml`        | `eurosense_fr_adjectives_train_swa.xml` | `eurosense_fr_adjectives_test_swa.xml` | multi-word anchors removed  | `bert-has-uncommon-sense/data/eurosens_adjectives_singleWordAnchorOnly` |
| `eurosense_fr_adverbs.xml`           | `eurosense_fr_adverbs_train.xml`    | `eurosense_fr_adverbs_test.xml`    | multi-word anchors kept     | `bert-has-uncommon-sense/data/eurosens_adverbs`                      |
| `eurosense_fr_adverbs.xml`           | `eurosense_fr_adverbs_train_swa.xml`| `eurosense_fr_adverbs_test_swa.xml`| multi-word anchors removed  | `bert-has-uncommon-sense/data/eurosens_adverbs_singleWordAnchorOnly`|
| `eurosense_fr.v1.0.high-precision.xml`| `eurosense_fr_all_train.xml`        | `eurosense_fr_all_test.xml`        | multi-word anchors kept     | `bert-has-uncommon-sense/data/eurosens_all`                          |
| `eurosense_fr.v1.0.high-precision.xml`| `eurosense_fr_all_train_swa.xml`    | `eurosense_fr_all_test_swa.xml`    | multi-word anchors removed  | `bert-has-uncommon-sense/data/eurosens_all_singleWordAnchorOnly`     |
