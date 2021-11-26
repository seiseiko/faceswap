# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 20:53:09
@LastEditors: yanxinhao
@Description: 
'''
from .extractor import Extractor
from pipeline import PipelineBase
import numpy as np
import facelib
import multiprocessing
from core import imagelib
from core import mathlib
from facelib import FaceType, LandmarksProcessor
from core.interact import interact as io
from core.joblib import Subprocessor
from core import pathex
from core.cv2ex import *
import time
from camera import CameraBase


class ExtractorMultiFace2Face(PipelineBase):
    class Worker(Extractor, PipelineBase.Worker):
        def __init__(self, image_size=256, face_type='full_face', cpu_only=False, device_idx=0,
                     max_faces_from_image=0, worker_idx=0, debug=False, warp_face=True):
            self.image_size = image_size
            self.face_type = face_type
            self.cpu_only = cpu_only
            self.device_idx = device_idx
            self.max_faces_from_image = max_faces_from_image
            self.debug = debug
            self.warp_face = warp_face
            PipelineBase.Worker.__init__(
                self, worker_idx=worker_idx, work_func=self.process)

        def _process(self, work_func):
            from core.leras import nn
            nn.initialize_main_env()
            Extractor.__init__(self, image_size=self.image_size, face_type=self.face_type, cpu_only=self.cpu_only,
                               device_idx=self.device_idx, max_faces_from_image=self.max_faces_from_image, debug=self.debug, warp_face=self.warp_face)
            # 1.initialize
            # test_data = CameraBase.Frame(
            #     index=-1, rgb=np.zeros((1080, 1920, 3), dtype=np.uint8))
            # work_func(test_data)
            self.c2s.put("success")

            # 2.main loop
            while True:
                if not self.s2c.empty():
                    image_expression, image_bgr = self.s2c.get()
                    try:
                        frame_info_expression = work_func(image_expression)
                        if len(frame_info_expression.final_faces) == 0:
                            frame_info_expression = None
                        frame_info_bgr = work_func(image_bgr)
                        out = (frame_info_expression, frame_info_bgr)
                    except:
                        print("something wrong in extractor")
                        frame_info_bgr = Extractor.Data()
                        frame_info_bgr.frame = image_bgr
                        out = (None, frame_info_bgr)
                    self.c2s.put(out)
                else:
                    time.sleep(0.001)

    def __init__(self, image_size=256, face_type='full_face', cpu_only=False,
                 device_idx=0, max_faces_from_image=0, get_input_func=None, num_workers=4,
                 buffer_size=100, debug=False):
        self.image_size = image_size
        self.face_type = face_type
        self.cpu_only = cpu_only
        self.device_idx = device_idx
        self.max_faces_from_image = max_faces_from_image
        self.get_input_func = get_input_func
        self.num_workers = num_workers
        self.debug = debug
        PipelineBase.__init__(self, get_input_func=get_input_func,
                              num_workers=num_workers, buffer_size=buffer_size)

    def generate_workers(self):
        # for worker_idx in range(self.num_workers):
        for worker_idx in [0, 1]:
            yield self.worker_cls(image_size=self.image_size, face_type=self.face_type, cpu_only=self.cpu_only, device_idx=worker_idx,
                                  max_faces_from_image=self.max_faces_from_image, worker_idx=worker_idx, debug=self.debug)

    def on_initialize(self):
        self.worker_cls = ExtractorMultiFace2Face.Worker

    @classmethod
    def get_input_from_camera(cls, camera_expression, camera_bgr):
        image_expression = camera_expression.capture()
        image_bgr = camera_bgr.capture()
        if image_expression is None or image_bgr is None:
            return
        return image_expression, image_bgr
