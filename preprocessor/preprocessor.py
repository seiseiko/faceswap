# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml
import os
import numpy as np
from tqdm import tqdm

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool
import glob

from preprocessor_utils import normalize, fill_mask_mouth

def main(args):
    face_resolution = int(args.resolution)
    cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    label_landmark = np.load("weights/label.npy")
    face_verts_idx = np.load("weights/face_verts_idx.npy")
    face_tri = np.load("weights/face_tri.npy")
    
    tddfa = TDDFA(gpu_mode=True, **cfg)
    face_boxes = FaceBoxes()

    img_folder = glob.glob(os.path.join(args.img_folder, '*'))

    for img_file in tqdm(img_folder, ascii=True, desc="preprocessor"):
        img = cv2.imread(img_file)

        boxes = face_boxes(img)
        param_lst, roi_box_lst = tddfa(img, boxes)

        vertices_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0:1]
        landmark_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0:1]

        vertices_lst[0] = vertices_lst[0][:, face_verts_idx]
        face_mask = render(img, vertices_lst, face_tri, alpha=1, with_bg_flag=False)
        face_mask[np.where(face_mask>0)] = 1
        face_mask = fill_mask_mouth(face_mask)
        face_img = face_mask * img

        face_img, face_mask, crop_img, norm_mats = normalize(label_landmark, landmark_lst[0].T, face_resolution, face_img, face_mask, img)


        cv2.imwrite(os.path.join(args.mask_folder, os.path.basename(img_file)), face_mask*255)
        cv2.imwrite(os.path.join(args.face_folder, os.path.basename(img_file)), face_img)
        cv2.imwrite(os.path.join(args.crop_folder, os.path.basename(img_file)), crop_img)
        np.save(os.path.join(args.trans_matrix_folder, os.path.basename(img_file).replace("png", "npy")), norm_mats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-i', '--img_folder', type=str, default='../workspace/data_source/origin')
    parser.add_argument('-m', '--mask_folder', type=str, default='../workspace/data_source/face_mask')
    parser.add_argument('-f', '--face_folder', type=str, default='../workspace/data_source/face_img')
    parser.add_argument('-c', '--crop_folder', type=str, default='../workspace/data_source/crop_img')
    parser.add_argument('-t', '--trans_matrix_folder', type=str, default='../workspace/data_source/trans_matrix')
    parser.add_argument('-r', '--resolution', type=str, default='512')
    args = parser.parse_args()
    main(args)
