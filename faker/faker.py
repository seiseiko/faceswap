# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-04-30 16:55:01
@LastEditors: yanxinhao
@Description: 
'''
import cv2
import time
from camera import Camera, CameraFile
from extractor import Extractor
from generator import Generator
from merger import Merger


class Faker:
    def __init__(self):
        self.camera = CameraFile()
        self.extractor = Extractor()
        self.generator = Generator()

    def start(self):
        self.camera.start_capture()
        time_avg = 0.
        index = 0
        while True:
            start_time = time.time()
            frame = self.camera.capture()
            if frame is None:
                time.sleep(0.1)
                continue
            frame_info = self.extractor.process(frame)
            # end_time = time.time()
            # print('time per frame in extract:', str(end_time-start_time))

            # start_time = time.time()
            fake_image = self.generator.generate(frame_info)
            end_time = time.time()
            index += 1
            time_avg += (end_time-start_time)/10.0
            if index == 10:
                print('time per frame in extract:', time_avg)
                index = 0
                time_avg = 0.
            # cv2.imshow("fake", fake_image)
            # cv2.imshow("face", frame_info.final_faces[0])
            cv2.imshow("real", frame)
            cv2.waitKey(1)
        self.camera.destructor()
