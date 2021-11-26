# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 15:10:33
@LastEditors: yanxinhao
@Description: 
'''
import cv2
import numpy as np
import time
import multiprocessing
from .camera_base import CameraBase
from player import CalculateFPS


class Camera(CameraBase):
    def __init__(self, buffer_size=10, camera_path="0"):
        super().__init__(buffer_size, frame_cls=CameraBase.Frame, camera_path=camera_path)

    def _start(self, frames, available):
        self.cameraCapture = cv2.VideoCapture(int(self.camera_path))

        # initialize
        success, frame = self.cameraCapture.read()
        frame_index = 0
        if success:
            print("camera initialized")
            frame = self.frame_cls(frame_index, frame)
            frames.put(frame)
            available.put("OK")

        cal_fps = CalculateFPS()
        # main loop
        while success:
            success, rgb = self.cameraCapture.read()
            if frames.qsize() == self.buffer_size:
                # print("pop frame unused")
                frames.get()
            frame = self.frame_cls(frame_index, rgb)
            frames.put(frame)
            frame_index += 1
            try:
                # time per frame
                t_pf = cal_fps.cal_t_pf()
                self.display(frame=frame, show_fps=True,
                             t_pf=t_pf, name_image="kinect")
            except:
                time.sleep(0.001)

    def destructor(self):
        self.cap_thread.join()
        self.cameraCapture.release()
