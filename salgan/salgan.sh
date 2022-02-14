#!/bin/bash 
export PYTHONPATH="${PYTHONPATH}:/root/salgan/scripts/"
export PYTHONPATH="${PYTHONPATH}:/root/HGI_SLAM/"
export THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.5,optimizer_including=cudnn
# THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.5,optimizer_including=cudnn python3 /root/salgan/scripts/03-predict.py
# feh /root/salgan/saliency/i112.jpg
python3 predictor.py /root/ORB_FR1