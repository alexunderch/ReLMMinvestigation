#!/bin/bash
apt update
apt install software-properties-common
apt-get install -y git\
                   libglu1-mesa-dev\
                   libgl1-mesa-dev\
                   libosmesa6-dev\
                   xvfb\
                   ffmpeg\
                   curl\
                   patchelf\
                   libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig\
                   wget\
                   unzip

git clone https://github.com/charlesjsun/ReLMM.git
git clone https://github.com/alexunderch/ReLMMinvestigation.git

python install_mujoco.py

cd ReLMM
rm setup.py 
cp ../ReLMMinvestigation/impl/errata/setup.py setup.py 
rm softlearning/environments/gym/locobot/urdf/locobot_description.urdf 
cp ../ReLMMinvestigation/impl/errata/locobot_description.urdf softlearning/environments/gym/locobot/urdf/locobot_description.urdf 
rm -r others/*
cp ../ReLMMinvestigation/impl/reproduction/*(.) others/

pip install -e .


