#!/bin/bash
# This script will run rank tests and insert into db using python.
# Following variables should be fed to the python-script:

# MASTER_THESIS_RESULTS=$HOME/MasterThesisResults
# PYTHONPATH=$MASTER_THESIS_RESULTS

DATABASE_1="main.db"
DATABASE_2="main_2.db"
DATABASE_3="main_3.db"
DATABASE_4="main_4.db"
RUNS=20
TEST_DATASET_ID=1
TRAINING_DATASET_ID=1
ENVIRONMENT_ID=1
DISTANCE=15
DEVICE=(8 9 10)
TRAINING_MODEL_ID=1
KEYBYTE=0
EPOCH=65
ADDITIVE_NOISE_METHOD_ID=('None' 1 2 3 4 5 6 7 8 9 10 11)
DENOISING_METHOD_ID=('None' 1 2)

case_1(){
  DEVICE=9
  ADDITIVE_NOISE_METHOD_ID=4
  DENOISING_METHOD_ID="None"
  python3 test_models__termination_point_to_db.py $DATABASE_2 $RUNS $TEST_DATASET_ID $TRAINING_DATASET_ID $ENVIRONMENT_ID $DISTANCE $DEVICE $TRAINING_MODEL_ID $KEYBYTE $EPOCH $ADDITIVE_NOISE_METHOD_ID $DENOISING_METHOD_ID
}

case_2(){
  DEVICE=10
  ADDITIVE_NOISE_METHOD_ID=4
  DENOISING_METHOD_ID="None"
  python3 test_models__termination_point_to_db.py $DATABASE_2 $RUNS $TEST_DATASET_ID $TRAINING_DATASET_ID $ENVIRONMENT_ID $DISTANCE $DEVICE $TRAINING_MODEL_ID $KEYBYTE $EPOCH $ADDITIVE_NOISE_METHOD_ID $DENOISING_METHOD_ID
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
