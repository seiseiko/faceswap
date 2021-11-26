# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 19:45:40
@LastEditors: yanxinhao
@Description: 
'''
import time
import cv2
from .camera_base import CameraBase
from player import CalculateFPS


class CameraFile(CameraBase):
    def __init__(self, buffer_size=2, video_path=''):
        self.video_path = video_path
        super().__init__(buffer_size, frame_cls=CameraBase.Frame)

    def _start(self, frames, available):
        self.cameraCapture = cv2.VideoCapture(self.video_path)

        # initialize
        success, frame = self.cameraCapture.read()
        frame_index = 0
        if success:
            frame = self.frame_cls(frame_index, frame)
            frames.put(frame)
            available.put("OK")

        cal_fps = CalculateFPS()
        # main loop
        while success:
            success, rgb = self.cameraCapture.read()
            while frames.qsize() == self.buffer_size:
                # frames.get()
                time.sleep(0.01)
            frame = self.frame_cls(frame_index, rgb)
            frames.put(frame)
            frame_index += 1
            try:
                # time per frame
                t_pf = cal_fps.cal_t_pf()
                self.display(frame=frame, show_fps=True,
                             t_pf=t_pf, name_image="video")
                time.sleep(0.01)
            except:
                time.sleep(0.001)

    def destructor(self):
        self.cap_thread.kill()
        self.cameraCapture.release()
