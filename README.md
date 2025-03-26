# PROJECT: BERT Has Uncommon Sense: *Similarity Ranking for Word Sense BERTology*

This project was completed as part of the **MVA Master 2024–2025** program.

- **Students**: Abdellah Rebaine, David Arrustico, Lila Roig  

### Paper Reference

We based our work on the following paper:

**“BERT Has Uncommon Sense: Similarity Ranking for Word Sense BERTology”**  
*Gessler and Schneider, 2021*  
[https://aclanthology.org/2021.blackboxnlp-1.43/](https://aclanthology.org/2021.blackboxnlp-1.43/)

### Original Codebase

The original code used in the paper is available here *(accessed February 2025)*:  
[https://github.com/lgessler/bert-has-uncommon-sense](https://github.com/lgessler/bert-has-uncommon-sense)

---
## Summary of the paper 
For our project, we focus on the paper *BERT Has Uncommon Sense: Similarity Ranking for Word Sense BERTology*, which examines how well contextualized word embedding models like BERT capture different word senses, particularly rare ones. Instead of using a traditional word sense disambiguation (WSD) approach, the authors propose a similarity ranking method based on cosine similarity between word embeddings in context. Performance is evaluated using Mean Average Precision (MAP) on the top 50 results. The study tests several pre-trained models—BERT, RoBERTa, DistilBERT, ALBERT, XLNet, and GPT-2—on two English sense-annotated corpora: OntoNotes 5.0 (covering nouns and verbs) and PDEP (Pattern Dictionary of English Prepositions). Results show that while all models outperform random ranking, they vary significantly, especially for rare senses. BERT achieves the best overall performance, while RoBERTa, despite excelling on other benchmarks, struggles with rare sense representation. The study reveals key differences between similar models and questions whether standard NLP benchmarks truly reflect their ability to capture lexical semantics.

---
## Project Overview and Experimental Contributions

Our work is organized into four main stages: 

### 1) Reproducing Original Results 
**Contributor:** *David Arrustico, Abdellah Rebaine, Lila Roig* \
We began by reproducing the results of Gessler & Schneider (2021), using the original codebase:
- Evaluated multiple transformer models: BERT, RoBERTa, DistilBERT, ALBERT, XLNet, and GPT-2.
- Tested on OntoNotes 5.0 and PDEP datasets.
- Compared model performance at different fine-tuning levels (0, 100, 250, 500, 1000, 2500 examples), within our computational constraints.

### 2) Extending to French Models and Datasets 
**Contributor:** *Lila Roig* \
This phase extended the original experiments to French-language resources:
- Adapted the code to support CamemBERT-base, a transformer model specifically trained for French.
- Integrated two French WSD datasets: FLUE and EuroSENS, ensuring compatibility with the existing pipeline.
- Ran new experiments to evaluate performance on French contextual word sense disambiguation.

> Lila began by reproducing experiments on the CLRES dataset, comparing her results with those obtained by Abdellah and David. She then attempted to adapt the method from the original paper to more recent models such as DeepSeek and Mistral. However, this proved to be a major challenge due to the fact that the entire pipeline (from data preprocessing to model evaluation) relies heavily on the latest version of the AllenNLP library, which is not compatible with newer tokenizers like LlamaTokenizer. Lila explored and tested numerous workarounds to overcome this incompatibility — including substituting tokenizers, manually adjusting tokenization outputs, and attempting to decouple specific components of the pipeline from AllenNLP. Despite these efforts, no solution proved effective without requiring a deep and extensive refactoring of the entire codebase. Given the time constraints she ultimately decided to abandon this line of investigation and explore a more feasible alternative.

> Following this, Lila focused on extending the project to work with French data. She chose a model supported by AllenNLP — CamemBERT-base — and applied it to two new French WSD datasets. This extension required significant adaptation of the existing codebase. Specifically, she made modifications to the four main scripts and developed 15 new scripts to handle various aspects of the new pipeline: cleaning and preprocessing the French datasets, formatting them to match AllenNLP’s expected input structure, integrating them into the pipeline, and addressing several technical challenges such as the handling of multi-word expressions in embeddings and the management of empty bins in the evaluation phase. She then ran 24 experiments on the two French corpora and analyzed the results in detail. All of the work — including code changes, new scripts, and results — is fully documented ont this GitHub repository. The README files in the "bert-has-uncommon-sense" folder includes clear instructions on how to set up the environment and reproduce the experiments, along with a detailed summary of all changes made.

### 3) Exploring Alternative Similarity Metrics
**Contributor:** *Abdellah Rebaine* \
The original paper relied solely on cosine similarity for comparing contextual embeddings. In this extension:
- We tested Euclidean distance as an alternative to evaluate the robustness of model performance across metrics.
- This paves the way for future inclusion of other distance or classification-based approaches.

### 4) Analyzing Embedding Layers 
**Contributor:** *David Arrustico* \
Investigated the assumption that the final layer gives the most meaningful word representation.
We reran the experiments using embeddings from the first layer, as well as several intermediate layers, instead of the final one.
This allowed us to track how representational quality evolves throughout the network’s depth, and whether deeper always means better.
Findings offer insight into how much information is encoded at each level of the transformer.
