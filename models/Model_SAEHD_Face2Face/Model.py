# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 21:00:25
@LastEditors: yanxinhao
@Description: 
'''
from ..Model_SAEHD import Model
import multiprocessing
import operator
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase


class ModelFace2Face(Model):

    # override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW" if len(
            devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        self.resolution = resolution = self.options['resolution']
        self.face_type = {'h': FaceType.HALF,
                          'mf': FaceType.MID_FULL,
                          'f': FaceType.FULL,
                          'wf': FaceType.WHOLE_FACE,
                          'head': FaceType.HEAD}[self.options['face_type']]

        eyes_prio = self.options['eyes_prio']
        archi = self.options['archi']
        is_hd = 'hd' in archi
        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        self.pretrain = self.options['pretrain']
        if self.pretrain_just_disabled:
            self.set_iter(0)

        self.gan_power = gan_power = self.options['gan_power'] if not self.pretrain else 0.0

        masked_training = self.options['masked_training']
        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        models_opt_on_gpu = False if len(
            devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device == '/CPU:0'

        input_ch = 3
        bgr_shape = nn.get4Dshape(resolution, resolution, input_ch)
        mask_shape = nn.get4Dshape(resolution, resolution, 1)
        self.model_filename_list = []

        with tf.device('/CPU:0'):
            # Place holders on CPU
            self.warped_src = tf.placeholder(nn.floatx, bgr_shape)
            self.warped_dst = tf.placeholder(nn.floatx, bgr_shape)

            self.target_src = tf.placeholder(nn.floatx, bgr_shape)
            self.target_dst = tf.placeholder(nn.floatx, bgr_shape)

            self.target_srcm_all = tf.placeholder(nn.floatx, mask_shape)
            self.target_dstm_all = tf.placeholder(nn.floatx, mask_shape)

        # Initializing model classes
        model_archi = nn.DeepFakeArchi(
            resolution, mod='uhd' if 'uhd' in archi else None)

        with tf.device(models_opt_device):
            if 'df' in archi:
                self.encoder = model_archi.Encoder(
                    in_ch=input_ch, e_ch=e_dims, is_hd=is_hd, name='encoder')
                encoder_out_ch = self.encoder.compute_output_channels(
                    (nn.floatx, bgr_shape))

                self.inter = model_archi.Inter(
                    in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, is_hd=is_hd, name='inter')
                inter_out_ch = self.inter.compute_output_channels(
                    (nn.floatx, (None, encoder_out_ch)))

                self.decoder_src = model_archi.Decoder(
                    in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, is_hd=is_hd, name='decoder_src')
                self.decoder_dst = model_archi.Decoder(
                    in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, is_hd=is_hd, name='decoder_dst')

                self.model_filename_list += [[self.encoder,     'encoder.npy'],
                                             [self.inter,       'inter.npy'],
                                             [self.decoder_src, 'decoder_src.npy'],
                                             [self.decoder_dst, 'decoder_dst.npy']]

            elif 'liae' in archi:
                self.encoder = model_archi.Encoder(
                    in_ch=input_ch, e_ch=e_dims, is_hd=is_hd, name='encoder')
                encoder_out_ch = self.encoder.compute_output_channels(
                    (nn.floatx, bgr_shape))

                self.inter_AB = model_archi.Inter(
                    in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, is_hd=is_hd, name='inter_AB')
                self.inter_B = model_archi.Inter(
                    in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims*2, is_hd=is_hd, name='inter_B')

                inter_AB_out_ch = self.inter_AB.compute_output_channels(
                    (nn.floatx, (None, encoder_out_ch)))
                inter_B_out_ch = self.inter_B.compute_output_channels(
                    (nn.floatx, (None, encoder_out_ch)))
                inters_out_ch = inter_AB_out_ch+inter_B_out_ch
                self.decoder = model_archi.Decoder(
                    in_ch=inters_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, is_hd=is_hd, name='decoder')

                self.model_filename_list += [[self.encoder,  'encoder.npy'],
                                             [self.inter_AB, 'inter_AB.npy'],
                                             [self.inter_B, 'inter_B.npy'],
                                             [self.decoder, 'decoder.npy']]

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if 'df' in archi:
                    if model == self.inter:
                        do_init = True
                elif 'liae' in archi:
                    if model == self.inter_AB or model == self.inter_B:
                        do_init = True
            else:
                do_init = self.is_first_run()

            if not do_init:
                do_init = not model.load_weights(
                    self.get_strpath_storage_for_file(filename))

            if do_init:
                model.init_weights()

        # Initializing merge function
        with tf.device(f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):
            # with tf.device(f'/CPU:0'):
            if 'df' in archi:
                gpu_dst_code = self.inter(self.encoder(self.warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(
                    gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            elif 'liae' in archi:
                gpu_dst_code = self.encoder(self.warped_dst)
                gpu_dst_inter_B_code = self.inter_B(gpu_dst_code)
                gpu_dst_inter_AB_code = self.inter_AB(gpu_dst_code)
                gpu_dst_code = tf.concat(
                    [gpu_dst_inter_B_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis)
                gpu_src_dst_code = tf.concat(
                    [gpu_dst_inter_AB_code, gpu_dst_inter_AB_code], nn.conv2d_ch_axis)

                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder(
                    gpu_src_dst_code)
                _, gpu_pred_dst_dstm = self.decoder(gpu_dst_code)

        def AE_merge(warped_dst):
            return nn.tf_sess.run([gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst: warped_dst})

        self.AE_merge = AE_merge


Model = ModelFace2Face
