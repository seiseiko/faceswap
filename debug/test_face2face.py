# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 18:22:37
@LastEditors: yanxinhao
@Description: 
'''
import time
from config import Config
from camera import Camera, CameraFile
from extractor import ExtractorMulti
from generator import GeneratorMultiFace2Face, GeneratorFace2Face
from player import Player, Displayer


class TestFace2Face:
    def __init__(self, debug):
        self.camera_expression = CameraFile(
            buffer_size=Config.camera['camera_buffer_size'], video_path=Config.camera['video_path_expression'])
        self.camera_bgr = CameraFile(
            buffer_size=Config.camera['camera_buffer_size'], video_path=Config.camera['video_path_bgr'])

        self.extractor_dst = ExtractorMulti(
            get_input_func=self.camera_expression.capture,
            device_idx=Config.extractor['device_idx'],
            num_workers=Config.extractor['num_workers'],
            buffer_size=Config.extractor['buffer_size_extractor'], debug=debug)
        self.extractor_src = ExtractorMulti(
            get_input_func=self.camera_bgr.capture,
            device_idx=Config.extractor['device_idx'],
            num_workers=Config.extractor['num_workers'],
            buffer_size=Config.extractor['buffer_size_extractor'], debug=debug)

        def get_input_generator():
            return GeneratorFace2Face.get_input_from_extractor(self.extractor_dst, self.extractor_src)

        self.generator_face2face = GeneratorMultiFace2Face(
            get_input_func=get_input_generator, force_model_class_name=Config.generator[
                'model_name'],
            num_workers=Config.generator['num_workers'],
            buffer_size=Config.generator['buffer_size_generator'])

        self.player = Displayer(
            get_input_func=self.generator_face2face.get_result)

    def start(self):
        self.camera_expression.start_capture()
        self.camera_bgr.start_capture()
        self.extractor_dst.start()
        self.extractor_src.start()
        self.generator_face2face.start()
        self.player.start()
        while True:
            time.sleep(1)


test = TestFace2Face(debug=Config.debug)
test.start()
