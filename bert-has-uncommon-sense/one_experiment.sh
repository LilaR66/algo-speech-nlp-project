###############################################################
# NEW SCRIPT created by Lila to run ONE experiment of our choice.
# This script is heavily inspired by the script all_experiments.sh 
###############################################################

#!/bin/sh

# This shell script automates the execution of word sense disambiguation (WSD) experiments using various BERT-based models 
# on two different corpora: OntoNotes and CLREs.

# Main Steps:
# 1. Environment Setup:
#    - Activates the `bhus` conda environment to ensure all dependencies are available.

# 2. Baseline Evaluation:
#    - Runs a trial and summarizes the results using the "bert-base-cased" model with a "baseline" metric at layer 7.

# 3. Model Testing:
#    - Iterates over a list of pre-trained transformer models (`bert-base-cased`, `distilbert-base-cased`, `roberta-base`, etc.).
#    - Sets the appropriate **BERT layer** (layer **5** for distilled models, **11** for others).

# 4. Experiment Execution:
#    - Runs trials and summarization for each model on the corpus using the "cosine" metric.
#    - If a fine-tuned model with a specific number of training instances (`100`, `250`, `500`, etc.) is needed:
#      - Checks if the fine-tuned weights exist (`models/{BERT_MODEL}_{NUM_INSTS}.pt`).
#      - If missing, trains the model using `python main.py finetune`.
#      - Runs the trial and summarizes results using the fine-tuned model.

# 5. Results Collection:
#    - The script iterates over all corpora (`ontonotes` and `clres`) and all model configurations, ensuring all trials are run.

# Usage:
# Simply execute this script in a terminal:
# sh script.sh
# Ensure that the `bhus` environment and necessary dependencies are installed.

# ---- Start of Script ----




# Activate the conda environment containing necessary dependencies
eval "$(conda shell.bash hook)" 
# Switch to the `bhus` environment before running the experiments
conda activate bhus 

# Iterate over the two datasets: OntoNotes and CLReS
for CORPUS in "flue"; do ### POSSIBLE OPTIONS: "ontonotes" "clres" et NEW: "flue"
  
  ###### I commented this section since I already run the baseline 
  # Run baseline trials and summarization using BERT-base-cased
  # python main.py trial --embedding-model "bert-base-cased" --metric "baseline" --query-n 1 --bert-layer 7 "$CORPUS" 
  # python main.py summarize --embedding-model "bert-base-cased" --metric "baseline" --query-n 1 --bert-layer 7 "$CORPUS"
  ######

  # Iterate over different transformer models
  for BERT_MODEL in "camembert-base" ; do ### POSSIBLE OPTIONS: "bert-base-cased" "distilbert-base-cased" "roberta-base" "distilroberta-base" "albert-base-v2" "xlnet-base-cased" "gpt2"  et NEW: "camembert-base" 

    # Determine which layer to use for embedding extraction
    # DistilBERT and DistilRoBERTa have only 6 layers, so we use the 5th (indexing from 0)
    # Other models typically have 12 layers, so we use the 11th
    if [ "$BERT_MODEL" == "distilbert-base-cased" ] || [ "$BERT_MODEL" == "distilroberta-base" ]; then
      BERT_LAYER=5
    else
      BERT_LAYER=11
    fi
    # Print the model and selected layer for logging/debugging
    echo "$BERT_MODEL $BERT_LAYER"

    # Iterate over different numbers of fine-tuning instances (0 means no fine-tuning)
    # NUM_INSTS is the number of examples we fine-tune our model on 
    for NUM_INSTS in 0 ; do  ### POSSIBLE OPTIONS: 0 100 250 500 1000 2500
      if [ "$NUM_INSTS" -eq "0" ];
      then
        # If NUM_INSTS is 0, run trials and summarization **without fine-tuning**
        python main.py trial --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER "$CORPUS"
        python main.py summarize --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER "$CORPUS"
      else
        # Define the model fine-tuning weights file path
        WEIGHTS_LOCATION="models/${BERT_MODEL}_${NUM_INSTS}.pt"
        # Check if fine-tuned weights already exist, if not, run the fine-tuning step
        if [ ! -f "$WEIGHTS_LOCATION" ]; then
          python main.py finetune "$BERT_MODEL" "$WEIGHTS_LOCATION"
        fi
        # Run the trial with fine-tuned model weights
        python main.py trial --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER --override-weights "$WEIGHTS_LOCATION" "$CORPUS"
        # Summarize the results for the fine-tuned model
        python main.py summarize --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER --override-weights "$WEIGHTS_LOCATION" "$CORPUS"
      fi
    done
  done
done


  