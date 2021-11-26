import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
# import cv2
from pathlib import Path
import os
import argparse
import visdom
from tqdm import tqdm
from deepfakes_pytorch import *
from torchvision import transforms
import deepfakes_pytorch.loss_function.pytorch_ssim as pytorch_ssim
from deepfakes_pytorch.loss_function import MaskLoss, LossCnt
from deepfakes_pytorch.utils.visualize import visualize_output


def main(args):
    num_epoch = args.epoch
    batch_size = args.batch_size
    learning_rate = eval(args.learning_rate)
    device = "cuda" if args.cuda else "cpu"
    target_res = args.resolution
    common_level = args.common_level
    num_person = args.num_person
    max_img_num = args.max_data_size

    # load datas
    image_paths = []
    mask_paths = []

    image_paths.append(get_image_list(args.source_img_folder, max_img_num))
    mask_paths.append(get_image_list(args.source_mask_folder, max_img_num))
    image_paths.append(get_image_list(args.target_img_folder, max_img_num))
    mask_paths.append(get_image_list(args.target_mask_folder, max_img_num))

    # seed = 0
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    # -------------------the different loss functions----------------------
    criterion_SSIM = pytorch_ssim.SSIM(window_size=11)

    criterion_MSE = MaskLoss(device=device).to(device)
    # ----------------------------------------------------------------------

    level = int(np.log2(target_res))-2
    cur_res = target_res
    print('==========Current Resolution: {}x{}=========='.format(cur_res, cur_res))

    data_transforms = transforms.Compose([
        TransformDeepfakes(warp=False, transform=True, flip=True,
                           is_border_replicate=True)
    ])

    datasets = []
    for i in range(num_person):
        datasets.append(
            FaceData(image_paths[i], mask_paths[i], cur_res, data_transforms))

    super_set = MergeDataSet(datasets)
    dataloader = DataLoader(super_set, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    face_encoder = Encoder(cur_res, common_level).to(device)
    face_decoders = []
    for i in range(num_person):
        face_decoders.append(
            Decoder(cur_res, common_level).to(device))

    Logger = LogModel(cur_res, person_num=num_person,
                      saving_path=args.model_folder)
    _ = Logger.load_model(face_encoder, face_decoders)

    face_encoder.train()
    for i in range(num_person):
        face_decoders[i].train()

    all_param = list(face_encoder.parameters())
    for i in range(num_person):
        all_param += list(face_decoders[i].parameters())
    AE_optimizer = torch.optim.Adam(params=all_param,
                                    lr=learning_rate, betas=(0.5, 0.999))

    visdom_visualize = visdom.Visdom(env="refine")
    last_count = 0
    count = 0
    loss_count = 0

    for epoch in tqdm(range(num_epoch)):
        last_count = count
        cur_alpha = 1
        for k, imgs in enumerate(dataloader):
            total_loss = 0
            for i in range(num_person):
                # -----------------------------------------------------------------------------------------------
                image_in = imgs[i]['rgb'].to(
                    device)*imgs[i]['mask'].to(device)
                # image_in = imgs[i]['rgb'].to(encoder_device)

                # image_label = imgs[i]['rgb_label'].to(encoder_device)*imgs[i]['mask_label'].to(maskloss_device)
                # if i == 1:
                #     visualize_output("test1", image_in.cpu())
                # if i == 0:
                #     visualize_output("test0", image_in.cpu())
                #     visualize_output("test_label", image_label.cpu())
                img_out = face_decoders[i](
                    face_encoder(image_in, cur_alpha), cur_alpha)  # 3*128*128

                # -----------------------------------------------------------------------------------------------
                loss_SSIM = criterion_SSIM(img_out.to(
                    device)*imgs[i]['mask'].to(device), imgs[i]['rgb'].to(device)*imgs[i]['mask'].to(device))
                loss_MSE = criterion_MSE(imgs[i]['mask'].to(
                    device), img_out.to(device), imgs[i]['rgb'].to(device))

                total_loss += 10*(loss_SSIM+loss_MSE)

                # ======================Visualization===========================
                if (k % 20 == 0) & (i == 0):
                    visualize_output("0_out", img_out.cpu())
                if (k % 20 == 0) & (i == 1):
                    visualize_output("1_out", img_out.cpu())
                    swap_0 = face_decoders[0](face_encoder(
                        imgs[1]['rgb_label'].to(device)*imgs[1]['mask_label'].to(device), cur_alpha), cur_alpha)
                    visualize_output("swap0_out", swap_0.cpu())
                # ==============================================================

            AE_optimizer.zero_grad()
            total_loss.backward()
            AE_optimizer.step()

            count += 1
            loss_count += total_loss.item()

        visdom_visualize.line(X=[count], Y=[loss_count/(count-last_count)], win="loss_{}".format(
            cur_res), update="append", opts={'title': "loss_{}_".format(cur_res)})

        print('Loss is {}'.format(loss_count/(count-last_count)))
        loss_count = 0
        print('Alpha is {}'.format(cur_alpha))

        # Logger_debug = LogModel(cur_res, model_name="progressive_Disney_epoch_{}".format(
        #     epoch), person_num=num_person)
        # Logger_debug.log_model(num_epoch[level], face_encoder.state_dict(), [
        #     decoder_i.state_dict() for decoder_i in face_decoders])

        Logger.log_model(num_epoch, face_encoder.state_dict(), [
                         decoder_i.state_dict() for decoder_i in face_decoders])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The training process of Disney swapping model')
    parser.add_argument('-si', '--source_img_folder', type=str,
                        default='../workspace/data_source/crop_img')
    parser.add_argument('-sm', '--source_mask_folder',
                        type=str, default='../workspace/data_source/face_mask')
    parser.add_argument('-ti', '--target_img_folder', type=str,
                        default='../workspace/data_target/crop_img')
    parser.add_argument('-tm', '--target_mask_folder',
                        type=str, default='../workspace/data_target/face_mask')
    parser.add_argument('-m', '--model_folder', type=str,
                        default='../workspace/model')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=2)
    parser.add_argument('-e', '--epoch', type=int,
                        default=300)
    parser.add_argument('-lr', '--learning_rate', type=str,
                        default="1e-5")
    parser.add_argument('-r', '--resolution', type=int,
                        default=1024)
    parser.add_argument('-cl', '--common_level', type=int,
                        default=2)
    parser.add_argument('-n', '--num_person', type=int,
                        default=2)
    parser.add_argument('-M', '--max_data_size', type=int,
                        default=5000)
    parser.add_argument('--cuda', action="store_true")
    args = parser.parse_args()
    main(args)
