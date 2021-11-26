# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime : 2020-01-03 10:13:43
@LastEditors  : yanxinhao
@Description: 
'''
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
import numpy as np
import torch
from .visualize import umeyama
from .warp import gen_warp_params, warp_by_params


class MergeDataSet(Dataset):
    def __init__(self, datasets):
        super(MergeDataSet, self).__init__()
        self.datasets = datasets

    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]

    def __len__(self):
        total_len = 0
        for dataset in self.datasets:
            total_len = max(total_len, len(dataset))
        return total_len


class FaceData(Dataset):
    def __init__(self, path_list, mask_path_list, image_shape=256, transform=None):
        if not isinstance(path_list, list):
            raise ValueError("the path_list must be a list")
        self.transform = transform
        self.images = []
        for image_path in tqdm(path_list):
            image = cv2.imread(
                str(image_path), cv2.IMREAD_UNCHANGED)
            # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            if image.shape[0] != image_shape:
                image = cv2.resize(
                    image, (image_shape, image_shape), cv2.INTER_CUBIC)
                # cv2.imshow("test", image)
                # cv2.waitKey(0)
            self.images.append(image)

        self.masks = []
        for mask_path in tqdm(mask_path_list):
            mask = cv2.imread(
                str(mask_path), cv2.IMREAD_GRAYSCALE)
            # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            if mask.shape[0] != image_shape:
                mask = cv2.resize(
                    mask, (image_shape, image_shape), cv2.INTER_CUBIC)
                # cv2.imshow("test", mask.reshape(128,128,1)*image)
                # cv2.waitKey(0)
            self.masks.append(mask)
        print("there are {} images in the dataset".format(len(self.images)))

    def __getitem__(self, index):
        if index >= len(self.images):
            index = np.random.randint(len(self.images))
        sample = self.images[index]
        sample_mask = self.masks[index]

        rgb_a = {}

        warped_image, target_image = sample,sample
        warped_mask, target_mask = sample_mask, sample_mask

        # rgb = torch.from_numpy(
        #         np.array(sample)).type(dtype=torch.float)/255
        # rgb_a['rgb']=rgb[:, :, :3].transpose(0, 2)

        if self.transform:
            warped_image, target_image, warped_mask, target_mask = self.transform([sample, sample_mask])
        warped_image_tensor = torch.from_numpy(
            np.array(warped_image)).type(dtype=torch.float)/255.0
        img = warped_image_tensor.transpose(0, 2)
        # mask = warped_image_tensor[:, :, 3].unsqueeze(2).transpose(0, 2)
        mask = torch.from_numpy(np.array(warped_mask)).type(dtype=torch.float).unsqueeze(2).transpose(0, 2)/255.0
        # label : target_image
        target_image_tensor = torch.from_numpy(
            np.array(target_image)).type(dtype=torch.float)/255.0
        img_label = target_image_tensor.transpose(0, 2)
        # mask_label = target_image_tensor[:, :, 3].unsqueeze(2).transpose(0, 2)
        mask_label = torch.from_numpy(np.array(target_mask)).type(dtype=torch.float).unsqueeze(2).transpose(0, 2)/255.0

        mask[mask < 0.5] = 0.0
        mask[mask >= 0.5] = 1.0
        mask = np.clip(mask, 0, 1)
        mask_label[mask_label < 0.5] = 0.0
        mask_label[mask_label >= 0.5] = 1.0
        mask_label = np.clip(mask_label, 0, 1)

        rgb_a['rgb'] = img
        rgb_a['mask'] = mask
        rgb_a['rgb_label'] = img_label
        rgb_a['mask_label'] = mask_label

        return rgb_a

    def __len__(self):
        return len(self.images)


class FlipHorizontally(object):
    def __init__(self):
        super().__init__()

    def __call__(self, image):
        if np.random.randint(2):
            return cv2.flip(image, 1)
        else:
            return image


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle
        super().__init__()

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        # cos = np.abs(M[0, 0])
        # sin = np.abs(M[0, 1])

        # # compute the new bounding dimensions of the image
        # nW = int((h * sin) + (w * cos))
        # nH = int((h * cos) + (w * sin))

        # # adjust the rotation matrix to take into account translation
        # M[0, 2] += (nW / 2) - cX
        # M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)

    def __call__(self, images):
        angle = np.random.randint(-self.angle, self.angle)
        out = []
        for image in images:
            out.append(self.rotate_bound(image, angle))
        return out


class RandomWarp(object):
    def __init__(self, res=256):
        super().__init__()
        self.res = res

    def random_warp_rev(self, image, res):
        # assert image.shape == (256, 256, 6)
        if image.shape[0] != res:
            image = cv2.resize(image, (256, 256))
        res_scale = 256//64
        assert res_scale >= 1, f"Resolution should be >= 64. Recieved {res}."

        interp_param = 80 * res_scale
        interp_slice = slice(interp_param//10, 9*interp_param//10)
        dst_pnts_slice = slice(0, 65*res_scale, 16*res_scale)

        rand_coverage = 256/2  # random warping coverage
        rand_scale = np.random.uniform(5., 6.2)  # random warping scale

        range_ = np.linspace(128-rand_coverage, 128+rand_coverage, 5)
        mapx = np.broadcast_to(range_, (5, 5))
        mapy = mapx.T
        mapx = mapx + np.random.normal(size=(5, 5), scale=rand_scale)
        mapy = mapy + np.random.normal(size=(5, 5), scale=rand_scale)
        interp_mapx = cv2.resize(mapx, (interp_param, interp_param))[
            interp_slice, interp_slice].astype('float32')
        interp_mapy = cv2.resize(mapy, (interp_param, interp_param))[
            interp_slice, interp_slice].astype('float32')
        warped_image = cv2.remap(
            image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[dst_pnts_slice, dst_pnts_slice].T.reshape(-1, 2)
        mat = umeyama(src_points, dst_points, True)[0:2]
        target_image = cv2.warpAffine(image, mat, (256, 256))

        warped_image = cv2.resize(warped_image, (res, res))
        target_image = cv2.resize(target_image, (res, res))
        return warped_image, target_image

    def __call__(self, image):
        return self.random_warp_rev(image, self.res)


class TransformDeepfakes(object):
    def __init__(self, rotation_range=[-10, 10], scale_range=[-0.1, 0.1], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05],
                 warp=True, transform=True, flip=True, is_border_replicate=True):
        super().__init__()
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.tx_range = tx_range
        self.ty_range = ty_range
        self.flip = flip
        self.warp = warp
        self.transform = transform
        self.is_border_replicate = is_border_replicate

    # def __call__(self, image):
    #     transform_params = gen_warp_params(
    #         image, self.flip, self.rotation_range, self.scale_range, self.tx_range, self.ty_range)
    #     image_bgra = warp_by_params(params=transform_params, img=image, warp=self.warp,
    #                                 transform=self.transform, flip=self.flip, is_border_replicate=self.is_border_replicate)
    #     if transform_params['flip']:
    #         image = cv2.flip(image, 1)
    #     return image_bgra, image

    def __call__(self, image_mask):
        image, mask = image_mask
        transform_params = gen_warp_params(
            image, self.flip, self.rotation_range, self.scale_range, self.tx_range, self.ty_range)
        image_bgra = warp_by_params(params=transform_params, img=image, warp=self.warp,
                                    transform=self.transform, flip=self.flip, is_border_replicate=self.is_border_replicate)
        mask_bgra = warp_by_params(params=transform_params, img=mask, warp=self.warp,
                                    transform=self.transform, flip=self.flip, is_border_replicate=self.is_border_replicate)
        if transform_params['flip']:
            image_bgra = cv2.flip(image_bgra, 1)
            mask_bgra = cv2.flip(mask_bgra, 1)
        return image_bgra, image, mask_bgra, mask

