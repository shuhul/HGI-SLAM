#!/bin/bash
export DISPLAY=:0.0
export PYTHONPATH="${PYTHONPATH}:/root/salgan/scripts/"
export PYTHONPATH="${PYTHONPATH}:/root/HGI_SLAM/"
export THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.5,optimizer_including=cudnn
python3 test.py