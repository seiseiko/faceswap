import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import glob
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

seamlessArgsList = []


def seamlessProcess(i):

    global seamlessArgsList

    args, img_file = seamlessArgsList[i]
    swapped_face = cv2.imread(img_file)
    background = cv2.imread(os.path.join(
        args.background_folder, os.path.basename(img_file)))
    mask = cv2.imread(os.path.join(args.mask_folder, os.path.basename(
        img_file)), cv2.IMREAD_GRAYSCALE)/255.0
    rot_mat, ali_mat, trs_mat = np.load(os.path.join(
        args.transform_matrix_folder, os.path.basename(img_file).replace("png", "npy")))
    rect_mask = np.ones_like(mask)

    rot_mat = np.linalg.inv(np.vstack((rot_mat, np.array([0, 0, 1]))))
    alitrs_mat = np.linalg.inv(
        np.vstack((ali_mat + trs_mat, np.array([0, 0, 1]))))

    mask = cv2.warpAffine(
        mask, alitrs_mat[0:2, :], (background.shape[1], background.shape[0]))
    mask = cv2.warpAffine(
        mask, rot_mat[0:2, :], (background.shape[1], background.shape[0]))
    mask = np.expand_dims(mask, 2)
    mask = (mask > 0).astype(np.uint8)

    rect_mask = cv2.warpAffine(
        rect_mask, alitrs_mat[0:2, :], (background.shape[1], background.shape[0]))
    rect_mask = cv2.warpAffine(
        rect_mask, rot_mat[0:2, :], (background.shape[1], background.shape[0]))
    rect_mask = cv2.merge([rect_mask for _ in range(3)])
    rect_mask = (rect_mask > 0).astype(np.uint8)
    rect_mask = 1 - rect_mask

    rect = cv2.boundingRect(mask)
    reference_point = (rect[0]+rect[2]//2, rect[1]+rect[3]//2)
    mask = cv2.merge([mask for _ in range(3)])

    swapped_face = cv2.warpAffine(
        swapped_face, alitrs_mat[0:2, :], (background.shape[1], background.shape[0]))
    swapped_face = cv2.warpAffine(
        swapped_face, rot_mat[0:2, :], (background.shape[1], background.shape[0]))

    swapped_face = swapped_face*mask
    merged_img = cv2.seamlessClone(swapped_face, background, mask*255, (int(
        reference_point[0]), int(reference_point[1])), cv2.NORMAL_CLONE)

    # merged_img = background*rect_mask + swapped_face

    cv2.imwrite(os.path.join(args.merged_face_folder,
                             os.path.basename(img_file)), merged_img)


def main(args):

    global seamlessArgsList

    img_folder = glob.glob(os.path.join(args.swapped_face_folder, '*'))
    pbar = tqdm(total=len(img_folder))
    for img_file in img_folder:
        seamlessArgsList.append((args, img_file))
    with Pool(processes=cpu_count()) as pool:
        for _ in pool.imap_unordered(seamlessProcess, range(len(seamlessArgsList))):
            pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merging swapped faces to origin image')
    parser.add_argument('-f', '--swapped_face_folder',
                        type=str, default='../workspace/data_target/swap_face')
    parser.add_argument('-m', '--mask_folder', type=str,
                        default='../workspace/data_target/face_mask')
    parser.add_argument('-b', '--background_folder', type=str,
                        default='../workspace/data_target/origin')
    parser.add_argument('-t', '--transform_matrix_folder',
                        type=str, default='../workspace/data_target/trans_matrix')
    parser.add_argument('-o', '--merged_face_folder', type=str,
                        default='../workspace/data_target/merge_result')
    args = parser.parse_args()
    main(args)
