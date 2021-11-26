# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-05-20 12:09:00
@LastEditors: yanxinhao
@Description: 
'''

import time


class CalculateFPS:
    def __init__(self, length_avg=10):
        self.start_t = time.time()
        self.end_t = time.time()
        self.index = 0
        self.length_avg = length_avg
        # time per frame
        self.t_pf = 1.0

    def cal_t_pf(self):
        if self.index == self.length_avg:
            self.end_t = time.time()
            self.t_pf = (self.end_t-self.start_t)/self.length_avg
            self.start_t = time.time()
            self.index = 0
        else:
            self.index += 1
        return self.t_pf
