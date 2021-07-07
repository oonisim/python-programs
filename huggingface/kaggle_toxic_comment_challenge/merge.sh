#!/usr/bin/env bash
files=($(ls | grep  '^[0-9][0-9]*.*\.ipynb' | sort -g))

#--------------------------------------------------------------------------------
# Notebook has been split into sections. Merge them into toxic_comment_analysis.ipynb
#  01_setup.ipynb
#  02_analysis.ipynb 
#  03_model_setup.ipynb 
#  04_model_toxic.ipynb 
#  05_model_severe.ipynb 
#  06_model_obscene.ipynb 
#  07_model_threat.ipynb 
#  08_model_insult.ipynb 
#  09_model_identity.ipynb 
#  10_model_history.ipynb 
#  11_evaluation.ipynb
#--------------------------------------------------------------------------------
echo "generating the merged notebook as toxic_comment_classification.ipynb"
nbmerge "${files[@]}" > toxic_comment_classification.ipynb

#--------------------------------------------------------------------------------
# Generate a analysis notebook for a specific category
#--------------------------------------------------------------------------------
echo "generating the analysis notebook as toxic_comment_analysis.ipynb"
nbmerge 01_setup.ipynb 02_analysis.ipynb > toxic_comment_analysis.ipynb

#--------------------------------------------------------------------------------
# Generate a training notebook for a specific category
#--------------------------------------------------------------------------------
CATEGORIES=("toxic" "severe" "obscene" "threat" "insult" "identity")
for category in "${CATEGORIES[@]}"
do
  echo "generating a model training notebook for $category"
  target=$(find -maxdepth 1 -type f -name "*_model_${category}.ipynb" -printf '%P\n'  | grep ${category})
  nbmerge 01_setup.ipynb 03_model_setup.ipynb ${target} > toxic_comment_${category}.ipynb
done
