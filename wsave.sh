#!/bin/bash
cd /root/HGI_SLAM/
git add .
git reset "*.obj"
git commit -m "$1"
git push