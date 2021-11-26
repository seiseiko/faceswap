# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-05-14 18:41:04
@LastEditors: yanxinhao
@Description: 
'''
import time
from camera import Camera, CameraFile
from extractor import ExtractorMulti
from generator import GeneratorMulti
from merger import Merger
from player import Player
import cv2


class FakerDemo:
    def __init__(self, config_camera, config_extractor, config_generator):

        if config_camera['filecamera']:
            self.camera = CameraFile(buffer_size=config_camera['camera_buffer_size'],
                                     video_path=config_camera['filepath'])
        else:
            self.camera = Camera(buffer_size=config_camera['camera_buffer_size'])

        self.extractor = ExtractorMulti(
            get_input_func=self.camera.capture,
            device_idx=config_extractor['device_idx'],
            num_workers=config_extractor['num_workers'],
            buffer_size=config_extractor['buffer_size_extractor'])
        self.generator = GeneratorMulti(
            get_input_func=self.extractor.get_result,
            force_model_class_name=config_generator['model_name'],
            num_workers=config_generator['num_workers'],
            buffer_size=config_generator['buffer_size_generator'])
        self.player = Player(get_input_func=self.generator.get_result)

    def start(self):
        self.camera.start_capture()
        self.extractor.start()
        self.generator.start()
        self.player.start()
        while True:
            time.sleep(1)
        self.camera.destructor()
        self.camera.destructor()
        self.extractor.destructor()

