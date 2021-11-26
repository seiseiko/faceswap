#!/usr/bin/env bash
source env.sh
cd video_editor

$DFD_PYTHON "video_to_img.py" \
    -f "../$DFD_WORKSPACE/source.mp4" \
    -r "0"\
    -o "../$DFD_WORKSPACE/data_source/origin"

$DFD_PYTHON "video_to_img.py" \
    -f "../$DFD_WORKSPACE/target.mp4" \
    -r "0"\
    -o "../$DFD_WORKSPACE/data_target/origin"

$DFD_PYTHON "video_to_img.py" \
    -f "../$DFD_WORKSPACE/dbw.mp4" \
    -r "0"\
    -o "../$DFD_WORKSPACE/data_dbw/origin"

$DFD_PYTHON "video_to_img.py" \
    -f "../$DFD_WORKSPACE/wyj.mp4" \
    -r "0"\
    -o "../$DFD_WORKSPACE/data_wyj/origin"