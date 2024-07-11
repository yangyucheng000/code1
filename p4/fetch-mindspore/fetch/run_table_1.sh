#!/bin/bash
# table 1: comparison on cross-validation of the paper

# datasets=("airfoil" "Bikeshare_DC" "Housing_Boston" "house_kc" "Openml_586" "Openml_589" "Openml_607" "Openml_616" "Openml_618" "Openml_620" "Openml_637" "adult" "amazon_employee" "default_credit_card" "credit_a" "fertility_Diagnosis" "german_credit_24" "hepatitis" "ionosphere" "lymphography" "megawatt1" "messidor_features" "PimaIndian" "spambase" "SPECTF" "winequality-red" "winequality-white")
datasets=("airfoil" "Bikeshare_DC" "Housing_Boston" "winequality-red" "winequality-white")
# datasets=("spambase" "PimaIndian" "messidor_features" "megawatt1" "lymphography" "ionosphere" "hepatitis" "german_credit_24" "fertility_Diagnosis" "credit_a" "default_credit_card" "amazon_employee" "adult" "Openml_637" "Openml_620" "Openml_618" "Openml_616" "Openml_607" "Openml_589" "Openml_586" "house_kc" "Housing_Boston" "Bikeshare_DC")
for dataset in "${datasets[@]}"
do
  python /tmp/code/fetch-mindspore/fetch/main_attention.py --file_name $dataset --cuda 3 --seed 0 --steps_num 3 --epochs 300 --episodes 24 &
  # python main_attention.py --file_name $dataset --cuda 4 --seed 1 --steps_num 3 --epochs 300 --episodes 24 &
  # python main_attention.py --file_name $dataset --cuda 5 --seed 2 --steps_num 3 --epochs 300 --episodes 24 &
  # python main_attention.py --file_name $dataset --cuda 6 --seed 3 --steps_num 3 --epochs 300 --episodes 24 &
  # python main_attention.py --file_name $dataset --cuda 7 --seed 4 --steps_num 3 --epochs 300 --episodes 24 &
  wait
done
