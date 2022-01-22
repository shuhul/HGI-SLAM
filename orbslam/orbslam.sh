#!/bin/bash
cd /root/ORB_SLAM2/
cd build
cmake .. 
make
cd ..
cd Examples/Monocular/
./mono_tum ../../Vocabulary/ORBvoc.txt TUM1.yaml /root/ORB_FR1
