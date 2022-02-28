#!/bin/bash
cd /root/ORB_SLAM2/
cd build
cmake .. 
make
cd ..
cd Examples/Monocular/
./mono_tum ../../Vocabulary/ORBvoc.txt TUM3.yaml $1 $2 $3
