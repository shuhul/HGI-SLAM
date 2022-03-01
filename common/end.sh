#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/root/HGI_SLAM/"
export TESTING_DATASET_PATH="/root/MONO_LONG" 
python3 /root/HGI_SLAM/common/end.py $TESTING_DATASET_PATH "m"