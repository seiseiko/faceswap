#!/usr/bin/env bash
source env.sh
cd swapper

$DFD_PYTHON "run_Disney.py" \
    -i "../$DFD_WORKSPACE/data_target/face_img" \
    -m "../$DFD_WORKSPACE/model" \
    -o "../$DFD_WORKSPACE/data_target/swap_face" \
    -r 1024 \
    -cl 2 \
    -sid 0 \
    -tid 1 \
    --cuda