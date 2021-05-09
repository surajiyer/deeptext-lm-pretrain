python -m lm_pretrain.src.exBert.create_vocabulary \
 --src_txt_glob_path "/home/suraj_iyer/lab/DeepText/data/01_raw/*.txt" \
 --tokenizer_name GroNLP/bert-base-dutch-cased \
 --dest_tokenizer_name /home/suraj_iyer/lab/DeepText/data/02_intermediate/exbert_tokenizer