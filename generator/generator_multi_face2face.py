# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 21:02:30
@LastEditors: yanxinhao
@Description: 
'''
import time
from .generator_face2face import GeneratorFace2Face
from pipeline import PipelineBase


class GeneratorMultiFace2Face(PipelineBase):
    class Worker(GeneratorFace2Face, PipelineBase.Worker):
        def __init__(self, model_class_name, saved_models_path, force_gpu_idxs, force_model_class_name, cpu_only, worker_idx):
            self.model_class_name = model_class_name
            self.saved_models_path = saved_models_path
            self.force_gpu_idxs = force_gpu_idxs
            self.force_model_class_name = force_model_class_name
            self.cpu_only = cpu_only
            PipelineBase.Worker.__init__(
                self, worker_idx=worker_idx, work_func=self.generate)

        def _process(self, work_func):
            from core.leras import nn
            nn.initialize_main_env()
            GeneratorFace2Face.__init__(self, model_class_name=self.model_class_name, saved_models_path=self.saved_models_path,
                                        force_gpu_idxs=self.force_gpu_idxs, force_model_class_name=self.force_model_class_name, cpu_only=self.cpu_only)

            # 1.initialize
            # test_data = Extractor.Data()
            # work_func(test_data)
            self.c2s.put("success")

            # 2.main loop
            while True:
                if not self.s2c.empty():
                    frame_info_expression, frame_info_bgr = self.s2c.get()
                    try:
                        self.c2s.put(
                            work_func(frame_info_expression, frame_info_bgr))
                    except:
                        # print("something wrong in generator")
                        self.c2s.put(frame_info_bgr.frame)
                else:
                    time.sleep(0.001)

    def __init__(self, model_class_name="SAEHD_Face2Face", saved_models_path="./checkpoint/model",
                 force_gpu_idxs=None, force_model_class_name=None, cpu_only=False, get_input_func=None, num_workers=4,
                 buffer_size=100):
        self.model_class_name = model_class_name
        self.saved_models_path = saved_models_path
        self.force_gpu_idxs = force_gpu_idxs
        self.force_model_class_name = force_model_class_name
        self.cpu_only = cpu_only
        PipelineBase.__init__(
            self, get_input_func=get_input_func, buffer_size=buffer_size, num_workers=num_workers)

    def generate_workers(self):
        # for worker_idx in range(self.num_workers):
        for worker_idx in [1, 1]:
            yield self.worker_cls(model_class_name=self.model_class_name, saved_models_path=self.saved_models_path,
                                  force_gpu_idxs=[worker_idx], force_model_class_name=self.force_model_class_name, cpu_only=self.cpu_only, worker_idx=worker_idx)

    def on_initialize(self):
        self.worker_cls = GeneratorMultiFace2Face.Worker
