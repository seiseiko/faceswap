# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 09:33:01
@LastEditors: yanxinhao
@Description: 
'''
import cv2
import time
import numpy as np
from pipeline import PipelineBase
from multiprocessing import Manager
import multiprocessing
from .fps_cal import CalculateFPS


class Player(PipelineBase):
    class FrameDisplayer:
        def __init__(self, index, rgb):
            self.index = index
            self.rgb = rgb
            self.t_pf = None

    class Worker(PipelineBase.Worker):
        def __init__(self, worker_idx, show_fps=True):
            self.show_fps = show_fps
            super().__init__(
                worker_idx=worker_idx, work_func=self.display)

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

        def _process(self, work_func):
            cal_fps = CalculateFPS()
            while True:
                if not self.s2c.empty():
                    # time per frame
                    t_pf = cal_fps.cal_t_pf()

                    # print('fps is :', t_pf)
                    data = self.s2c.get()
                    data.t_pf = t_pf
                    self.c2s.put(work_func(data))
                else:
                    time.sleep(0.001)

    def __init__(self, get_input_func=None, num_workers=1, buffer_size=1):
        self.background = Player.FrameDisplayer(
            0, np.zeros((1080, 1920, 3)))
        '''
        super PipelineBase
        '''
        self.get_input_func = get_input_func
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.buffer = Manager().list()

        self.pipeline_module_part = multiprocessing.Process(
            target=self.run, args=(self.buffer, self.buffer_size, self.get_input_func,))
        self.work_func = None
        self.on_initialize()

    def generate_workers(self):
        for worker_idx in range(self.num_workers):
            yield self.worker_cls(worker_idx)

    def on_initialize(self):
        self.worker_cls = Player.Worker

    def run(self, buffer, buffer_size, get_input_func):
        workers = []
        for worker in self.generate_workers():
            workers.append(worker)
        print("initialized player")
        while True:

            # get data and put it into buffer
            for worker in workers:
                try:
                    data = get_input_func()
                    if data is None:
                        continue

                    """ it is a bad idea"""
                    # if len(buffer) != 0 and data.index < buffer[0].index:
                    #     continue
                except:
                    continue
                buffer.append(data)

            # put the frame of display into s2c queue
            for worker in workers:
                try:
                    frame = self.frame_dispatcher()  # buffer[-1]
                    if frame is None:
                        continue
                except:
                    continue
                worker.s2c.put(frame)

    def frame_dispatcher(self):
        buffer_length = len(self.buffer)
        # print("buffer_length in player : ", buffer_length)
        # if buffer_length <= self.buffer_size:
        if buffer_length == 0:
            return
        else:
            index = 0
            for i in range(len(self.buffer)):
                if self.buffer[index].index > self.buffer[i].index:
                    index = i
            return self.buffer.pop(index)
