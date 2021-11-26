# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-05-12 15:32:03
@LastEditors: yanxinhao
@Description: 
'''
from faker import FakerDemo
from core import osex
from core.leras import nn
from config import Config
if __name__ == "__main__":
    faker = FakerDemo(Config.camera, Config.extractor, Config.generator)
    faker.start()
