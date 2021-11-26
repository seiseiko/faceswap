import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import glob
import cv2
from tqdm import tqdm
from deepfakes_pytorch import *
from deepfakes_pytorch.utils.visualize import visualize_output


def main(args):
    device = "cuda" if args.cuda else "cpu"
    cur_res = args.resolution
    common_level = args.common_level
    source = args.source_id
    target = args.target_id
    num_person = 2

    print('==========Current Resolution: {}x{}=========='.format(cur_res, cur_res))

    face_encoder = Encoder(cur_res, common_level).to(device)
    face_decoders = []
    for i in range(num_person):
        face_decoders.append(
            Decoder(cur_res, common_level).to(device))

    face_encoder.eval()
    for i in range(num_person):
        face_decoders[i].eval()

    Logger = LogModel(cur_res, person_num=num_person,
                      saving_path=args.model_folder)
    _ = Logger.load_model(face_encoder, face_decoders)

    img_folder = glob.glob(os.path.join(args.target_face_folder, '*'))
    for img_file in tqdm(img_folder, ascii=True, desc="swapping"):
        img = cv2.imread(img_file).astype(np.float32)
        input_face = (torch.from_numpy(cv2.resize(
            img, (cur_res, cur_res))/255.0)).to(device).permute(2, 1, 0).unsqueeze(0)

        swapped_face = face_decoders[args.source_id](
            face_encoder(input_face, 1), 1)
        swapped_face = swapped_face.squeeze(0).permute(2, 1, 0)
        swapped_face = np.clip(swapped_face.cpu().detach().numpy(), 0, 1)

        cv2.imwrite(os.path.join(args.target_out_folder,
                                 os.path.basename(img_file)), swapped_face*255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The swapping process of trained Disney model')
    parser.add_argument('-i', '--target_face_folder', type=str,
                        default='../workspace/data_target/face_img')
    parser.add_argument('-o', '--target_out_folder', type=str,
                        default='../workspace/data_target/swap_face')
    parser.add_argument('-m', '--model_folder', type=str,
                        default='../workspace/model')
    parser.add_argument('-r', '--resolution', type=int, default=512)
    parser.add_argument('-cl', '--common_level', type=int, default=2)
    parser.add_argument('-sid', '--source_id', type=int, default=0)
    parser.add_argument('-tid', '--target_id', type=int, default=1)
    parser.add_argument('--cuda', action="store_true")
    args = parser.parse_args()
    main(args)
