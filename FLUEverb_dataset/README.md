title: "FLUE Dataset Setup and Preprocessing"
author: Lila Roig
date: 2025-03-21
---
# FLUE Dataset Setup and Preprocessing

This file explains how to download and set up the **FLUE dataset** to reproduce the experiments.

## STEP 1: Download the FLUE dataset

**FLUE paper**:  
[https://arxiv.org/abs/1912.05372](https://arxiv.org/abs/1912.05372)

Go to the **GitHub page for FLUE Verbs** *(accessed February 2025)*:  
[https://github.com/getalp/Flaubert/tree/master/flue/wsd/verbs](https://github.com/getalp/Flaubert/tree/master/flue/wsd/verbs)

As mentioned in the README, download the data from *(accessed February 2025)*:  
[http://www.llf.cnrs.fr/dataset/fse/](http://www.llf.cnrs.fr/dataset/fse/)

You will obtain a file named:  
`FSE-1.1-10_12_19.tar.gz`

Place it into the folder:  
`FLUEverb_dataset/`

Unzip the file, and place its contents into a directory of your choice  
(we’ll refer to it as `FSE_DIR`).  
You can now delete the original `.tar.gz` file since you no longer need it.

Download the `prepare_data.py` script and the `modules/` directory (including all its contents)  
from the FLUE verbs GitHub page *(accessed February 2025)*: 
[https://github.com/getalp/Flaubert/tree/master/flue/wsd/verbs](https://github.com/getalp/Flaubert/tree/master/flue/wsd/verbs)

Place them into the directory:  
`FLUEverb_dataset/`

**run**:  
```
conda activate bhus
conda install -c anaconda lxml
cd path/to/FLUEverb_dataset
python prepare_data.py --data "path/to/FLUEverb_dataset/FSE_DIR" --output DATA_DIR_
```

This generates several directories and files:

- `DATA_DIR_test`: contains the test data  
- `DATA_DIR_train`: contains the training data  
- `DATA_DIR_targets`: a file containing the target labels

**Finally**, reorganize the file structure by creating a `DATA_DIR/` directory containing:

```
DATA_DIR/test/FSE-1.1.data.xml
DATA_DIR/test/FSE-1.1.gold.key.txt
DATA_DIR/train/wiktionary-190418.data.xml
DATA_DIR/train/wiktionary-190418.gold.key.txt
DATA_DIR/targets
```

You can now safely delete the original `FSE_DIR` directory.

---

## (Optional) STEP 2: Understanding the structure of the FLUE files

The dataset includes:

- XML files containing sentences with annotated **target verbs**  
- `.gold.key.txt` files providing **gold-standard sense labels** for each verb occurrence  
- A `targets` file listing the specific verbs to be disambiguated


### 1) **Training Data — `DATA_DIR/train/`**

#### a) `wiktionary-190418.data.xml`

Contains sentences with verbs annotated for the **Word Sense Disambiguation (WSD)** task.

**Head of the file**:
```xml
<?xml version="1.0" ?>
<corpus lang="fr" source="frenchsemeval">              
  <text id="d000">
    <sentence id="d000.s000">
        <wf lemma="il" pos="CL">Il</wf>
        <wf lemma="rendre" pos="V">rend</wf>
        <wf lemma="hommage" pos="N">hommage</wf>
        <instance fine_pos="V" id="d000.s000.t000" lemma="aboutir" pos="V">aboutissent</instance>
        <wf lemma="traité" pos="N">traité</wf>
    </sentence>
  </text>
</corpus>
```

**Structure details**:

- The file starts with a `<corpus>` tag containing all sentences.  
- Each sentence appears within a `<sentence>` tag, nested under `<text>` (representing a document).  
- Regular words are wrapped in `<wf>` tags, which include attributes:
  - `lemma`, `pos`, and optionally `fine_pos`  
- **Target verbs** for disambiguation are wrapped in `<instance>` tags, which include:
  - `lemma`: base form of the verb  
  - `id`: unique identifier  
  - **Text content**: the verb form in the sentence  

**Example**:  
In the sentence *"Il rend hommage ... aboutissent traité."*

- Target verb: `aboutissent` (lemma: `aboutir`)  
- Instance ID: `d000.s000.t000`  
- This ID is used to retrieve the gold label from `FSE-1.1.gold.key.txt`

#### b) `wiktionary-190418.gold.key.txt`

Maps each **instance ID** from the XML file to its **gold sense label**.

**Example format**:
```
d000.s000.t000 __ws_2_aboutir__verb__1
```

Where:
- `__ws_2_`: sense cluster group  
- `aboutir__verb__1`: lemma and sense ID within the group

**Note**: The **training set** is general-purpose and includes multiple parts of speech (verbs, nouns, adjectives, etc.),  
whereas the **test set focuses only on verbs**.

### 2) **Test Data — `DATA_DIR/test/`**

#### a) `FSE-1.1.data.xml`  
Similar structure to the training XML, but includes only **test sentences with annotated verbs**.

#### b) `FSE-1.1.gold.key.txt`  
Contains **gold labels** for the test set.  
Used to **evaluate model performance**.

### 3) **Target Verb List — `DATA_DIR/targets/`**

Contains a list of verbs to be disambiguated.

**Example entries**:
```
investir__V
payer__V
alimenter__V
```

Used to filter relevant verbs from the XML files.  
**Not used in our project.**


## STEP 3: Filter training set to keep only verb annotations

Since the **training set contains multiple word types** while the **test set includes only verbs**,  
we filter the training set to **retain only verbs** in order to **reduce computational costs**.

**run**:
```
conda activate bhus
cd path/to/your/file
python filter_dataset.py
```

Inside `filter_dataset.py`, make sure the following file paths are correctly defined:

```python
xml_input  = "train/wiktionary-190418.data.xml"
xml_output = "train_small/wiktionary-190418.filtered_verbs.xml"
gold_input = "train/wiktionary-190418.gold.key.txt"
gold_output = "train_small/wiktionary-190418.filtered_verbs.gold.key.txt"
```

We now have a filtered **`train_small`** dataset containing **only verbs**.


## STEP 4: Move the files to the correct folder

We can now move the following files:

From:  
`train_small/wiktionary-190418.filtered_verbs.xml`  
`train_small/wiktionary-190418.filtered_verbs.gold.key.txt`

To:  
`bert-has-uncommon-sense/data/flueverb/train_small/`


And the following files:

From:  
`test/FSE-1.1.data.xml`  
`test/FSE-1.1.gold.key.txt`

To:  
`bert-has-uncommon-sense/data/flueverb/test/`

**FLUE is now ready for WSD.**

## (Optional) Note:

The script `test_clres_conllu_reader.py` was used to inspect how labels are formatted  
in the **CLRES (PDEP)** dataset by the `ClresConlluReader` class.  

It helps replicate the same formatting for the **FLUE** dataset  
by guiding the implementation of the `FlueVerbReader` class.

To run the script, place it in the root directory:  
`bert-has-uncommon-sense/`
