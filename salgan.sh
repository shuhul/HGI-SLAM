#!/bin/bash 
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.5,optimizer_including=cudnn python3 /root/salgan/scripts/03-predict.py
feh /root/salgan/saliency/i112.jpg
