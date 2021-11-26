# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-05-20 15:23:13
@LastEditors: yanxinhao
@Description: 
'''
import cv2
import time
from camera import Camera, CameraFile
from extractor import ExtractorMulti
from generator import GeneratorMulti
from merger import Merger
from player import Player, Displayer
from config import Config


class DetectDemo:
    def __init__(self, config_camera, config_extractor, debug):

        if config_camera['filecamera']:
            self.camera = CameraFile(buffer_size=config_camera['camera_buffer_size'],
                                     video_path=config_camera['camera_path'])
        else:
            self.camera = Camera(
                buffer_size=config_camera['camera_buffer_size'])

        self.extractor = ExtractorMulti(
            get_input_func=self.camera.capture,
            device_idx=config_extractor['device_idx'],
            num_workers=config_extractor['num_workers'],
            buffer_size=config_extractor['buffer_size_extractor'], debug=debug)

        def get_input():
            data = self.extractor.get_result()
            if data is None:
                return
            return data.frame

        # self.player = Player(get_input_func=get_input)
        self.player = Displayer(get_input_func=get_input)

    def start(self):
        self.camera.start_capture()
        self.extractor.start()
        self.player.start()
        while True:
            time.sleep(1)
        self.camera.destructor()
        self.extractor.destructor()


demo = DetectDemo(Config.camera, Config.extractor, Config.debug)
demo.start()
