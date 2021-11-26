#!/usr/bin/env bash
source env.sh
cd video_editor

$DFD_PYTHON "img_to_video.py" \
    -f "../$DFD_WORKSPACE/result.mp4" \
    -r "25"\
    -i "../$DFD_WORKSPACE/data_target/merge_result"
