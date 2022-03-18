#!/bin/bash
export DISPLAY=:0.0
export DATA_PATH=/root/SP_Data
export EXPER_PATH=/root/SP_Experiment
export PYTHONPATH="${PYTHONPATH}:/root/HGI_SLAM/"
python3 test.py