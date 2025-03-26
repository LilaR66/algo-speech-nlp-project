#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate bhus

for CORPUS in "ontonotes" "clres"; do
  
  for BERT_MODEL in "bert-base-cased" "distilbert-base-cased" "roberta-base" "distilroberta-base" "albert-base-v2" "xlnet-base-cased" "gpt2"; do
    if [ "$BERT_MODEL" = "distilbert-base-cased" ] || [ "$BERT_MODEL" = "distilroberta-base" ]; then
      LAYERS=$(seq 0 5)  # Distilled models (6 layers)
    else
      LAYERS=$(seq 0 11) # Standard models (12 layers)
    fi

    for BERT_LAYER in $LAYERS; do
      echo "Processing: $BERT_MODEL Layer: $BERT_LAYER on Corpus: $CORPUS"
      python main.py trial --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer "$BERT_LAYER" "$CORPUS"
      python main.py summarize --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer "$BERT_LAYER" "$CORPUS"
    done
  done
done