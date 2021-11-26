# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-03 20:57:01
@LastEditors: yanxinhao
@Description: 
'''
from camera import CameraBLMGTest, Camera
from config import Config
from player import Displayer
import time
import cv2


class TestCamera:
    def __init__(self):
        self.camera = CameraBLMGTest(
            buffer_size=Config.camera["camera_buffer_size"], camera_path=Config.camera["camera_path"])
        self.displayer = Displayer(get_input_func=self.camera.capture)

    def start(self):
        self.camera.start_capture()
        self.displayer.start()


class TestCamera_1:
    def __init__(self):
        self.camera = Camera(
            buffer_size=Config.camera["camera_buffer_size"], camera_path=Config.camera["camera_path"])

    def start(self):
        self.camera.start_capture()
        time.sleep(0.01)
        while True:
            res = self.camera.capture()
            if res.rgb is None:
                continue
            cv2.imshow("test", res.rgb)
            cv2.waitKey(1)


test_camera = TestCamera_1()
test_camera.start()

while True:
    time.sleep(1)
