#!/bin/bash
cd /root/ORB_SLAM2/
./build.sh
cd Examples/Monocular/
./mono_tum ../../Vocabulary/ORBvoc.txt TUM1.yaml /root/ORB_FR1
