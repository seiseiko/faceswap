#!/usr/bin/env bash
source env.sh
cd postprocessor

$DFD_PYTHON "postprocessor.py" \
    -f "../$DFD_WORKSPACE/data_target/swap_face" \
    -m "../$DFD_WORKSPACE/data_target/face_mask" \
    -b "../$DFD_WORKSPACE/data_target/origin" \
    -t "../$DFD_WORKSPACE/data_target/trans_matrix" \
    -o "../$DFD_WORKSPACE/data_target/merge_result"
