## DeepText LM Pretrain
Train a BERT language model using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) tasks. There are two versions of BERT that can be pretrained, regular BERT and exBERT.

exBERT is based on the [paper](https://www.aclweb.org/anthology/2020.findings-emnlp.129.pdf) with origial code available [here](https://github.com/cgmhaicenter/exBERT). This version refactors a lot of the Pytorch / Huggingface [transformer](https://github.com/huggingface/transformers) modules to clean it up, apply inheritance from original HuggingFace BERT modules so that the changes to the BERT modules from the latest transformers library are taken into account, and added compatibility of training the exBERT model with the `transformers.Trainer` class.

#### Setup for regular BERT pretraining
1. Clone the repo.
2. `cd /path/to/repo/`
3. `conda env create -f environment.yml`
4. `conda activate deeptext_lm_pretrain`
5. Update the parameters (e.g., dataset paths) in settings.yml.
6. `bash run_bert_pretraining.sh`

#### Setup for exBERT pretraining
1. Clone the repo.
2. `cd /path/to/repo/`
3. `conda env create -f environment.yml`
4. `conda activate deeptext_lm_pretrain`
5. Update the parameters in run_exbert_create_vocab.sh
6. `bash run_exbert_create_vocab.sh`
7. Update the parameters in settings_exbert.yml.
8. `bash run_exbert_pretraining.sh`
