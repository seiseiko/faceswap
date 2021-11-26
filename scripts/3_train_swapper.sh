#!/usr/bin/env bash
source env.sh
cd swapper

$DFD_PYTHON "train_Disney.py" \
    -si "../$DFD_WORKSPACE/data_source/crop_img" \
    -sm "../$DFD_WORKSPACE/data_source/face_mask" \
    -ti "../$DFD_WORKSPACE/data_target/crop_img" \
    -tm "../$DFD_WORKSPACE/data_target/face_mask" \
    -di "../$DFD_WORKSPACE/data_dbw/crop_img" \
    -dm "../$DFD_WORKSPACE/data_dbw/face_mask" \
    -wi "../$DFD_WORKSPACE/data_wyj/crop_img" \
    -wm "../$DFD_WORKSPACE/data_wyj/face_mask" \
    -m "../$DFD_WORKSPACE/model" \
    -b "[64, 64, 64, 32, 32, 16, 8, 4, 2]" \
    -e "[30, 30, 30, 75, 100, 100, 125, 150, 175]" \
    -lr "[1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 3e-5, 3e-5, 3e-5, 3e-5]" \
    -r 1024 \
    -cl 2 \
    -n 4 \
    -M 5000 \
    --cuda