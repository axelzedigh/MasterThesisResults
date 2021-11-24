#!/bin/bash
# This script will run rank tests and insert into db using python.
# Following variables should be fed to the python-script:

# MASTER_THESIS_RESULTS=$HOME/MasterThesisResults
# PYTHONPATH=$MASTER_THESIS_RESULTS

DATABASE_MAIN="main.db"
DATABASE_TMP_1="tmp_1.db"
DATABASE_TMP_2="tmp_2.db"
DATABASE_TMP_3="tmp_3.db"
RUNS=105
TEST_DATASET_ID=1 # 1 - wang2021, 2 - zedigh2021
TRAINING_DATASET_ID=1
ENVIRONMENT_ID=1
DISTANCE=15
TRAINING_MODEL_ID=1
KEYBYTE=0
EPOCH=65
ADDITIVE_NOISE_METHOD_IDS=('None' 1 2 3 4 5 6 7 8 9 10 11)
DENOISING_METHOD_IDS=('None' 1 2)
TRACE_PROCESS_ID=4

case_1(){
DEVICES=(10 6 8 9 10)
ADDITIVE_NOISE_METHOD_IDS=('None' 1 2 3 4 5)
DENOISING_METHOD_ID='None'
DISTANCES=(15)

for DISTANCE in "${DISTANCES[@]}"; do
  for DEVICE in "${DEVICES[@]}"; do
    for ADDITIVE_NOISE_METHOD_ID in "${ADDITIVE_NOISE_METHOD_IDS[@]}"; do
      python3 test_models__termination_point_to_db.py $DATABASE_TMP_1 $RUNS $TEST_DATASET_ID $TRAINING_DATASET_ID $ENVIRONMENT_ID $DISTANCE $DEVICE $TRAINING_MODEL_ID $KEYBYTE $EPOCH "$ADDITIVE_NOISE_METHOD_ID" $DENOISING_METHOD_ID $TRACE_PROCESS_ID
    done
  done
done
}

case_2(){
ADDITIVE_NOISE_METHOD_IDS=(6 7 8 9 10 11)
DENOISING_METHOD_ID='None'
DISTANCES=(5 10 15)
for DISTANCE in "${DISTANCES[@]}"; do
  for DEVICE in "${DEVICES[@]}"; do
    for ADDITIVE_NOISE_METHOD_ID in "${ADDITIVE_NOISE_METHOD_IDS[@]}"; do
      python3 test_models__termination_point_to_db.py $DATABASE_TMP_1 $RUNS $TEST_DATASET_ID $TRAINING_DATASET_ID $ENVIRONMENT_ID $DISTANCE $DEVICE $TRAINING_MODEL_ID $KEYBYTE $EPOCH "$ADDITIVE_NOISE_METHOD_ID" $DENOISING_METHOD_ID $TRACE_PROCESS_ID
    done
  done
done
}

case_3(){
for element in Hydrogen Helium Lithium Beryllium
do
#  echo "Element: $element"
  for yeah in yo ya ye
  do
    echo "$yeah $element"
  done
done
}


if [[ $1 -eq 1 ]]
then
  case_1
elif [[ $1 -eq 2 ]]
then
  case_2
elif [[ $1 -eq 3 ]]
then
  case_3
else
  echo "Something is wrong"
fi
