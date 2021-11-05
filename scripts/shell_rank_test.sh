#!/bin/bash
# This script will run rank tests using python script
# Following variables should be fed to the python-script:
#
# RUNS=100
# TEST_TRACE_PATH="master-thesis/path/to/trace/"
# DISTANCE=[5,10,15]
# DEVICES=[6,7,8,9,10]
# EPOCH=[20,...,100]
# MODEL=[
  #"no_noise",
  #"gaussian_noise_04", "gaussian_noise_03",
  #"rayleigh_noise_0138","rayleigh_noise_0276"
  #"collected_noise_25", "collected_noise_50", "collected_noise_75",
  #]

#E.g.
#RUN=10;TEST_TRACE="/master-thesis/previous_testing_traces";
#DISTANCE=15;DEVICES=6;EPOCH=65;MODEL="GWN_04";
# python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL

case_1(){
  RUN=30;TEST_TRACE="master-thesis/previous_testing_traces";
  DISTANCE=15;DEVICES=10;EPOCH=65;MODEL="no_noise";
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL

}

case_2(){
  RUN=20;TEST_TRACE="master-thesis/previous_testing_traces";
  DISTANCE=15;DEVICES=6;EPOCH=65;MODEL="rayleigh_noise_0138";
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL

}

case_3(){
  RUN=30;TEST_TRACE="master-thesis/previous_testing_traces";
  DISTANCE=15;DEVICES=6;EPOCH=65;MODEL="rayleigh_noise_0138";
  python3 RankTestScript.py $RUN $TEST_TRACE $DISTANCE $DEVICES $EPOCH $MODEL
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
