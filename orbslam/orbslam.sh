#!/bin/bash
cd /root/ORB_SLAM2/
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
# ./build.sh
cd ..
cd Examples/Monocular/
./mono_tum ../../Vocabulary/ORBvoc.txt TUM1.yaml /root/ORB_FR1

# cd /root/ORB_SLAM2/build/CMakeFiles/
# make -j mono_tum.dir/build.make
# cd /root/ORB_SLAM2/Examples/Monocular/
# ./mono_tum ../../Vocabulary/ORBvoc.txt TUM1.yaml /root/ORB_FR1

