# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-03 20:45:29
@LastEditors: yanxinhao
@Description: 
'''
import cv2
import numpy as np
import time
import multiprocessing
from .camera_base import CameraBase
from player import CalculateFPS
import Camera


class CameraBLMG(CameraBase):
    def __init__(self, buffer_size=10, camera_path="0", resolution_in=(1920, 1080), resolution_out=(1280, 720)):
        self.resolution_in = resolution_in
        self.resolution_out = resolution_out
        super().__init__(buffer_size, frame_cls=CameraBase.Frame, camera_path=camera_path)

    def _start(self, frames, available):
        self.cameraCapture = Camera.Camera(int(self.camera_path))

        self.cameraCapture.StartCapture()
        pts, rgb = self.cameraCapture.GetFrame()
        available.put("OK")
        cal_fps = CalculateFPS()
        print("camera initialized, the camera is  : ", int(self.camera_path))
        # main loop
        while True:
            # if frames.qsize() == self.buffer_size:
            #     print("pop frame unused")
            #     frames.pop()
            pts, rgb = self.cameraCapture.GetFrame()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
            rgb = cv2.resize(rgb, dsize=self.resolution_out)
            # print(pts)
            frame = self.frame_cls(pts, rgb)
            frames.put(frame)
            t_pf = cal_fps.cal_t_pf()
            self.display(frame, True, t_pf=t_pf)
            # cv2.imshow("camera", rgb)
            # cv2.waitKey(1)

    def destructor(self):
        self.cap_thread.join()
        self.cameraCapture.release()
