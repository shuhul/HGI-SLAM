#!/bin/bash
export DISPLAY=:0.0
export DATA_PATH=/root/SP_Data
export EXPER_PATH=/root/SP_Experiment
python3 extractor.py sp_v6 $DATA_PATH/HPatches/i_pool/1.ppm $DATA_PATH/HPatches/i_pool/6.ppm
