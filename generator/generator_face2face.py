# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime: 2020-06-04 20:50:46
@LastEditors: yanxinhao
@Description:
'''
from extractor import ExtractorMulti
from .generator import Generator
from merger import MergeMaskedFace2Face
from player import Player


class GeneratorFace2Face(Generator):
    def __init__(self, model_class_name="SAEHD_Face2Face", saved_models_path="./checkpoint/model", force_gpu_idxs=None, force_model_class_name=None, cpu_only=False):
        super().__init__(model_class_name=model_class_name, saved_models_path=saved_models_path,
                         force_gpu_idxs=force_gpu_idxs, force_model_class_name=force_model_class_name, cpu_only=cpu_only)

    def generate(self, frame_info_expression, frame_info_bgr):
        print("generate new fake face")
        if frame_info_expression is None:
            print("frame_info_expression in generator is None")
            return frame_info_bgr.frame
        final_img = MergeMaskedFace2Face(self.predictor_func, self.predictor_input_shape,
                                         face_enhancer_func=self.face_enhancer_func,
                                         xseg_256_extract_func=self.xseg_256_extract_func,
                                         cfg=self.cfg,
                                         frame_info_expression=frame_info_expression, frame_info_bgr=frame_info_bgr)
        out = Player.FrameDisplayer(frame_info_bgr.frame.index, final_img)
        return out

    @classmethod
    def get_input_from_extractor(cls, extractor_dst, extractor_src):
        frame_info_target = extractor_dst.get_result()
        frame_info_src = extractor_src.get_result()
        if frame_info_src is None or frame_info_target is None:
            return
        return frame_info_target, frame_info_src
