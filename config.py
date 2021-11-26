# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 20:59:07
@LastEditors: yanxinhao
@Description: 
'''


class Config:
    debug = False
    camera = {
        'filecamera': False,
        'camera_buffer_size': 4,
        'camera_path': "0",
        'video_path_bgr': "./data_src.mp4",
        'video_path_expression': "./data_dst.mp4"
    }
    extractor = {
        'num_workers': 2,
        'buffer_size_extractor': 10,
        'device_idx': 0
    }
    generator = {
        'num_workers': 2,
        'buffer_size_generator': 10,
        'model_name': "wangyan_SAEHD"
    }
