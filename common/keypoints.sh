#!/bin/bash
export DISPLAY=:0.0
export DATA_PATH=/root/SP_Data
export EXPER_PATH=/root/SP_Experiment
export PYTHONPATH="${PYTHONPATH}:/root/salgan/scripts/"
export PYTHONPATH="${PYTHONPATH}:/root/HGI_SLAM/"
export THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.5,optimizer_including=cudnn
python3 /root/HGI_SLAM/superpoint/test.py
python3 /root/HGI_SLAM/salgan/test.py