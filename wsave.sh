#!/bin/bash
cd /root/HGI_SLAM/
git add .
git reset superpoint/saved/descriptor_list.obj
git reset salgan/saved/descriptor_list.obj
git commit -m "$1"
git push