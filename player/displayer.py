# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-03 21:35:49
@LastEditors: yanxinhao
@Description: 
'''
import time
import cv2
import multiprocessing
from .fps_cal import CalculateFPS


class Displayer:
    def __init__(self, get_input_func, show_fps=True):
        self.show_fps = show_fps
        self.get_input_func = get_input_func
        self.displayer = multiprocessing.Process(
            target=self.run, args=(self.get_input_func,))

    def start(self):
        self.displayer.start()

    def run(self, get_input_func):

        cal_fps = CalculateFPS()
        while True:
            frame = get_input_func()
            if frame is not None:
                frame.t_pf = cal_fps.cal_t_pf()
                self.display(frame)
            else:
                time.sleep(0.001)

    def display(self, frame):
        image = frame.rgb
        if self.show_fps:
            time_pf = "time per frame : "+str(frame.t_pf)
            cv2.putText(image, str(time_pf), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            fps = "fps :"+str(1/frame.t_pf)
            cv2.putText(image, str(fps), (50, 80),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            index_str = "index : "+str(frame.index)
            cv2.putText(image, str(index_str), (50, 110),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        # print(fps)
        cv2.imshow("fake", image)
        cv2.waitKey(1)

    def destructor(self):
        self.displayer.kill()
