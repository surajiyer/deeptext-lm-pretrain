NOW=`date +"%Y%m%d_%H%M"`
python -m lm_pretrain.src --now "$NOW" --method "bert" --settings "settings.yml" || exit
