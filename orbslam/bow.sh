#!/bin/bash
cd /root/ORB_SLAM2/
cd Thirdparty/HGI_DBoW2/
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j