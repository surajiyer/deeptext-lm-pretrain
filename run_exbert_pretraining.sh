NOW=`date +"%Y%m%d_%H%M"`
python -m lm_pretrain.src --now "$NOW" --method "exbert" --settings "settings_exbert.yml" || exit
