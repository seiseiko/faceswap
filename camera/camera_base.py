# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 15:10:08
@LastEditors: yanxinhao
@Description: 
'''

import cv2
import numpy as np
import multiprocessing
from player import CalculateFPS


class CameraBase:
    class Frame:
        def __init__(self, index, rgb):
            self.index = index
            self.rgb = rgb

    def __init__(self, buffer_size=10, frame_cls=None, camera_path=None):
        self.available = multiprocessing.Queue()
        self.frames = multiprocessing.Queue()
        self.buffer_size = buffer_size
        self.frame_cls = frame_cls
        self.camera_path = camera_path
        self.cap_thread = multiprocessing.Process(
            target=self._start, args=(self.frames, self.available,))

    def start_capture(self):
        self.cap_thread.start()
        if self.available.get() is "OK":
            return True

    def _start(self, frames, available):
        raise NotImplementedError

    def capture(self):
        if not self.frames.empty():
            frame = self.frames.get()
        else:
            # frame = self.frame_cls(-1, None)
            # frame.rgb = None
            frame = None
        return frame

    def display(self, frame, show_fps=False, t_pf=1.0, name_image="camera"):
        image = frame.rgb.copy()
        if show_fps:
            time_pf = "time per frame : "+str(t_pf)
            cv2.putText(image, str(time_pf), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            fps = "fps :"+str(1/t_pf)
            cv2.putText(image, str(fps), (50, 80),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            index_str = "index : "+str(frame.index)
            cv2.putText(image, str(index_str), (50, 110),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(name_image, image)
        cv2.waitKey(1)

    def destructor(self):
        self.cap_thread.join()

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        self.destructor()
