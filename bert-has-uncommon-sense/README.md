# Running and Modifying the Experiments

This file explains how to run the experiments to replicate the results.
*Author: Lila Roig*

**Paper on which our project is based:**
 https://arxiv.org/abs/2109.09780

**Original code on which our project is based** - before all the modifications we added *(accessed January 2025)*: 
https://github.com/lgessler/bert-has-uncommon-sense

---

## STEP 1: Download the GitHub Repo

```bash
cd path/to/your/folder/
git clone "  "
cd "bert-has-uncommon-sense"
git submodule init   # Initialize sub-modules 
git submodule update # Download and synchronise sub-module files
```

---
## STEP 2: Setup working environment  

Create and activate a Conda environment:

```
conda create --name bhus python=3.8
conda activate bhus
```

Install project dependencies in this order:

```
conda install -c conda-forge tokenizers==0.12.1
pip install allennlp==2.10.1 allennlp_models==2.10.1
pip install pandas click bs4
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

**Note**: If you're unsure about any specific package version, you can refer to the `requirements.txt` file for guidance. However, we do **not** recommend installing directly from it using `pip install -r requirements.txt`, as the steps above are sufficient to properly set up the environment.

---
## STEP 3: Download datasets

**PDEP (CLRES) Dataset**:  
Already included in the project’s Git repository — no additional download is required.

**OntoNotes 5.0 Dataset**:  
Must be downloaded from the Linguistic Data Consortium (LDC):

- Visit `https://cemantix.org/data/ontonotes.html`, which redirects to the LDC.
- Go to the official LDC page: `https://catalog.ldc.upenn.edu/LDC2013T19`
- You need to create an LDC account to access the data.

**STREUSLE Dataset**:  
This dataset should already be available if you initialized the project’s Git submodules. No further download is needed.

**FLUE Dataset**:  
Please refer to the README of the folder `FLUEverb_dataset/`

**EuroSENS Dataset**:  
Please refer to the README of the folder `EuroSENS_dataset/`

---
## STEP 4: Run experiments

### Option 1 – Run step by step from the command line (Recommended)

1) *(Optional)* Fine-tune the model on a small number of STREUSLE instances:

```
python main.py finetune --num_insts 100 distilbert-base-cased models/distilbert-base-cased_100.pt
```

2) Run a trial using the fine-tuned model on the PDEP dataset (aliased as `clres`):

```
python main.py trial \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    --override-weights models/distilbert-base-cased_100.pt \
    clres
```

**If we did not fine-tune a model:**

```
python main.py trial \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    clres
```

3) Summarize and analyze results by bucket:

```
python main.py summarize \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    --override-weights models/distilbert-base-cased_100.pt \
    clres
```

**If we did not fine-tune a model:**

```
python main.py summarize \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    clres
```

**Note**: It's normal for the script to appear stuck on `"reading split train"`. Computing contextual embeddings can take over an hour on CPU, depending on the dataset size.

### Option 2 – Run one experiment end-to-end using a script

run: 
```
sh one_experiment.sh
```

You can modify this script to run any specific experiment.

### Option 3 – Run all experiments** *(Not recommended if you have insufficient computational resources)*

run
```
sh script/all_experiments.sh
```
---
## Overview of the modifications

Below is a list of all the modifications made to the original code from the paper.

### **Modified Files from the Original Code**  
All changes to the original code are clearly marked with `### NEW ###` or `### MODIF ###` comments.

- `bert-has-uncommon-sense/main.py`  
- `bert-has-uncommon-sense/bssp/common/config.py`  
- `bert-has-uncommon-sense/bssp/common/reading.py`  
- `bert-has-uncommon-sense/bssp/common/nearest_neighbor_models.py`

### **New Files (Not Present in the Original Code)**  
These files were added to extend or adapt the original implementation.

> **Note**: If any of the files listed below are not found in the specified location, please refer to the README files in the `EuroSENS_dataset` and `FLUEverb_dataset` directories.  
> They explain how to generate these files and where to place them properly.

- `bert-has-uncommon-sense/one_exeriment.sh`  
- `bert-has-uncommon-sense/bssp/eurosens/dataset_reader.py`  
- `bert-has-uncommon-sense/bssp/flue/dataset_reader.py`  
 <br>
- `bert-has-uncommon-sense/data/flueverb/train_small/wiktionary-190418.filtered_verbs.xml`  
- `bert-has-uncommon-sense/data/flueverb/train_small/wiktionary-190418.filtered_verbs.gold.key.txt`  
- `bert-has-uncommon-sense/data/flueverb/test/FSE-1.1.data.xml`  
- `bert-has-uncommon-sense/data/flueverb/test/FSE-1.1.gold.key.txt`  
 <br>
- `bert-has-uncommon-sense/data/eurosens_adjectives/eurosense_fr_adjectives_train.xml`  
- `bert-has-uncommon-sense/data/eurosens_adjectives/eurosense_fr_adjectives_test.xml`  
- `bert-has-uncommon-sense/data/eurosens_adjectives_singleWordAnchorOnly/eurosense_fr_adjectives_train_swa.xml`  
- `bert-has-uncommon-sense/data/eurosens_adjectives_singleWordAnchorOnly/eurosense_fr_adjectives_test_swa.xml`  
//
- `bert-has-uncommon-sense/data/eurosens_adverbs/eurosense_fr_adverbs_train.xml`  
- `bert-has-uncommon-sense/data/eurosens_adverbs/eurosense_fr_adverbs_test.xml`  
- `bert-has-uncommon-sense/data/eurosens_adverbs_singleWordAnchorOnly/eurosense_fr_adverbs_train_swa.xml`  
- `bert-has-uncommon-sense/data/eurosens_adverbs_singleWordAnchorOnly/eurosense_fr_adverbs_test_swa.xml`  
//
- `bert-has-uncommon-sense/data/eurosens_nouns/eurosense_fr_nouns_train.xml`  
- `bert-has-uncommon-sense/data/eurosens_nouns/eurosense_fr_nouns_test.xml`  
- `bert-has-uncommon-sense/data/eurosens_nouns_singleWordAnchorOnly/eurosense_fr_nouns_train_swa.xml`  
- `bert-has-uncommon-sense/data/eurosens_nouns_singleWordAnchorOnly/eurosense_fr_nouns_test_swa.xml`  
//
- `bert-has-uncommon-sense/data/eurosens_verbs/eurosense_fr_verbs_train.xml`  
- `bert-has-uncommon-sense/data/eurosens_verbs/eurosense_fr_verbs_test.xml`  
- `bert-has-uncommon-sense/data/eurosens_verbs_singleWordAnchorOnly/eurosense_fr_verbs_train_swa.xml`  
- `bert-has-uncommon-sense/data/eurosens_verbs_singleWordAnchorOnly/eurosense_fr_verbs_test_swa.xml`  
//
- `bert-has-uncommon-sense/data/eurosens_all/eurosense_fr_all_train.xml`  
- `bert-has-uncommon-sense/data/eurosens_all/eurosense_fr_all_test.xml`  
- `bert-has-uncommon-sense/data/eurosens_all_singleWordAnchorOnly/eurosense_fr_all_train_swa.xml`  
- `bert-has-uncommon-sense/data/eurosens_all_singleWordAnchorOnly/eurosense_fr_all_test_swa.xml`  
//
- `EuroSENS_dataset/filter_eurosens_french.py`  
- `EuroSENS_dataset/filter_eurosens_french_by_pos.py`  
- `EuroSENS_dataset/create_eurosens_splits.py`  
- `EuroSENS_dataset/helpers/compute_dataset_stats.py`  
- `EuroSENS_dataset/helpers/analyse_split_eurosens.py`  
- `EuroSENS_dataset/helpers/count_annotations.py`  
- `EuroSENS_dataset/helpers/most_frequent_word_sense.sh`  
- `EuroSENS_dataset/helpers/compare_sentence_ids.sh`  
//
- `FLUEverb_dataset/prepare_data.py`  
- `FLUEverb_dataset/modules/`  
- `FLUEverb_dataset/filter_dataset.py`  
- `FLUEverb_dataset/test_clres_conllu_reader.py`  
- `FLUEverb_dataset/DATA_DIR/`
