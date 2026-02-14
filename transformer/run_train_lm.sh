#!/bi/bash
nohup python -u train_lm.py --dataset wikitext --epochs 3 --snapshot_interval 1000 2>&1 | tee train_lm_12feb26_1746.log&

