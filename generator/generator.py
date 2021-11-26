# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 11:10:51
@LastEditors: yanxinhao
@Description: 
'''
import numpy as np
import models
import os
from pathlib import Path
from merger import MergeMasked
from core.joblib import MPClassFuncOnDemand, MPFunc
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
from camera import Camera


class Generator:
    """it the main module that generate the final faceswapped image
    Inputs:
        frame_info
    Returns:
        frame_info
    """

    def __init__(self, model_class_name="SAEHD", saved_models_path="./checkpoint/model", force_gpu_idxs=None, force_model_class_name=None, cpu_only=False):

        from core.leras import nn
        nn.initialize_main_env()
        self.saved_models_path = Path(os.path.abspath(
            os.path.expanduser(saved_models_path)))
        run_on_cpu = False  # len(nn.getCurrentDeviceConfig().devices) == 0

        model = models.import_model(model_class_name)(is_training=False,
                                                      saved_models_path=self.saved_models_path,
                                                      force_gpu_idxs=force_gpu_idxs,
                                                      force_model_class_name=force_model_class_name,
                                                      cpu_only=cpu_only)

        self.xseg_256_extract_func = MPClassFuncOnDemand(XSegNet, 'extract',
                                                         name='XSeg',
                                                         resolution=256,
                                                         weights_file_root=self.saved_models_path,
                                                         place_model_on_cpu=True,
                                                         run_on_cpu=run_on_cpu)

        self.face_enhancer_func = MPClassFuncOnDemand(FaceEnhancer, 'enhance',
                                                      place_model_on_cpu=True,
                                                      run_on_cpu=run_on_cpu)

        self.predictor_func, self.predictor_input_shape, self.cfg = model.get_MergerConfig()

        # Preparing MP functions
        # self.predictor_func = MPFunc(self.predictor_func)

    def generate(self, frame_info):
        # if len(frame_info.landmarks) == 0:
        #     # print("No face is detected")
        #     return Camera.Frame(frame_info.frame.index, frame_info.frame.rgb)
        final_img = MergeMasked(self.predictor_func, self.predictor_input_shape,
                                face_enhancer_func=self.face_enhancer_func,
                                xseg_256_extract_func=self.xseg_256_extract_func,
                                cfg=self.cfg,
                                frame_info=frame_info)
        out = Camera.Frame(frame_info.frame.index, final_img)
        return out
