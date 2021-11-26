
import numpy as np
import facelib
import multiprocessing
from core import imagelib
from core import mathlib
from facelib import FaceType, LandmarksProcessor
from core.interact import interact as io
from core.joblib import Subprocessor
from core import pathex
from core.cv2ex import *


class Extractor:
    """it the module which extracts the face info (rect,landmark,etc..)
    Inputs:
        CameraBase.Frame

    Returns:
        frame_info
    """
    class Data(object):
        def __init__(self, rects=None, landmarks=None, landmarks_accurate=True, final_output_files=None):
            self.rects = rects or []
            self.rects_rotation = 0
            self.landmarks_accurate = landmarks_accurate
            self.landmarks = landmarks or []
            self.final_output_files = final_output_files or []
            self.faces_detected = 0
            self.final_faces = []
            self.frame = None

    def __init__(self, image_size=256, face_type='full_face', cpu_only=False, device_idx=0, max_faces_from_image=0, debug=False, warp_face=False):

        from core.leras import nn
        self.image_size = image_size
        self.cpu_only = cpu_only
        self.device_idx = device_idx
        self.warp_face = warp_face
        self.debug = debug
        # device_config = nn.DeviceConfig.GPUIndexes( force_gpu_idxs or nn.ask_choose_device_idxs(choose_only_one=detector=='manual', suggest_all_gpu=True) ) \
        #         if not cpu_only else nn.DeviceConfig.CPU()

        if self.cpu_only:
            device_config = nn.DeviceConfig.CPU()
            place_model_on_cpu = True
        else:
            device_config = nn.DeviceConfig.GPUIndexes([self.device_idx])
            place_model_on_cpu = device_config.devices[0].total_mem_gb < 4

        nn.initialize(device_config)
        self.face_type = FaceType.fromString(face_type)
        self.max_faces_from_image = max_faces_from_image

        # loading models
        self.rects_extractor = facelib.S3FDExtractor(
            place_model_on_cpu=place_model_on_cpu)
        self.landmarks_extractor = facelib.FANExtractor(landmarks_3D=self.face_type >= FaceType.HEAD,
                                                        place_model_on_cpu=place_model_on_cpu)

    def process(self, frame):
        frame_info = Extractor.Data()
        frame_info.frame = frame
        # rect stage
        frame_info = self.rects_stage(
            frame_info, frame.rgb, self.max_faces_from_image, self.rects_extractor)

        # landmark stage
        frame_info = self.landmarks_stage(frame_info, frame.rgb, extract_from_dflimg=False,
                                          landmarks_extractor=self.landmarks_extractor, rects_extractor=self.rects_extractor)

        # debug :just output the frame with rectangular and landmarks
        # if self.debug:
        #     frame_info.frame.rgb = self.plot_face(frame, frame_info)
        if self.warp_face:
            frame_info = self.final_stage(frame_info, frame.rgb)
        return frame_info

    def rects_stage(self, data,
                    image,
                    max_faces_from_image,
                    rects_extractor,
                    ):
        h, w, c = image.shape
        if min(h, w) < 128:
            # Image is too small
            data.rects = []
        else:
            # for rot in ([0, 90, 270, 180]):
            # for realtime system ,we only detect the face with rot=0
            for rot in ([0]):
                if rot == 0:
                    rotated_image = image
                elif rot == 90:
                    rotated_image = image.swapaxes(0, 1)[:, ::-1, :]
                elif rot == 180:
                    rotated_image = image[::-1, ::-1, :]
                elif rot == 270:
                    rotated_image = image.swapaxes(0, 1)[::-1, :, :]
                rects = data.rects = rects_extractor.extract(
                    rotated_image, is_bgr=True)
                if len(rects) != 0:
                    data.rects_rotation = rot
                    break
            if max_faces_from_image != 0 and len(data.rects) > 1:
                data.rects = data.rects[0:max_faces_from_image]
        return data

    def landmarks_stage(self, data,
                        image,
                        extract_from_dflimg,
                        landmarks_extractor,
                        rects_extractor,
                        ):
        h, w, ch = image.shape

        if data.rects_rotation == 0:
            rotated_image = image
        elif data.rects_rotation == 90:
            rotated_image = image.swapaxes(0, 1)[:, ::-1, :]
        elif data.rects_rotation == 180:
            rotated_image = image[::-1, ::-1, :]
        elif data.rects_rotation == 270:
            rotated_image = image.swapaxes(0, 1)[::-1, :, :]

        data.landmarks = landmarks_extractor.extract(rotated_image, data.rects, rects_extractor if (
            not extract_from_dflimg and data.landmarks_accurate) else None, is_bgr=True)
        if data.rects_rotation != 0:
            for i, (rect, lmrks) in enumerate(zip(data.rects, data.landmarks)):
                new_rect, new_lmrks = rect, lmrks
                (l, t, r, b) = rect
                if data.rects_rotation == 90:
                    new_rect = (t, h-l, b, h-r)
                    if lmrks is not None:
                        new_lmrks = lmrks[:, ::-1].copy()
                        new_lmrks[:, 1] = h - new_lmrks[:, 1]
                elif data.rects_rotation == 180:
                    if lmrks is not None:
                        new_rect = (w-l, h-t, w-r, h-b)
                        new_lmrks = lmrks.copy()
                        new_lmrks[:, 0] = w - new_lmrks[:, 0]
                        new_lmrks[:, 1] = h - new_lmrks[:, 1]
                elif data.rects_rotation == 270:
                    new_rect = (w-b, l, w-t, r)
                    if lmrks is not None:
                        new_lmrks = lmrks[:, ::-1].copy()
                        new_lmrks[:, 0] = w - new_lmrks[:, 0]
                data.rects[i], data.landmarks[i] = new_rect, new_lmrks

        return data

    def final_stage(self, frame_info, frame):
        rects = frame_info.rects
        landmarks = frame_info.landmarks
        for rect, image_landmarks in zip(rects, landmarks):

            if image_landmarks is None:
                continue

            rect = np.array(rect)

            image_to_face_mat = LandmarksProcessor.get_transform_mat(
                image_landmarks, self.image_size, self.face_type)

            face_image = cv2.warpAffine(
                frame, image_to_face_mat, (self.image_size, self.image_size), cv2.INTER_LANCZOS4)
            face_image_landmarks = LandmarksProcessor.transform_points(
                image_landmarks, image_to_face_mat)

            landmarks_bbox = LandmarksProcessor.transform_points(
                [(0, 0), (0, self.image_size-1), (self.image_size-1, self.image_size-1), (self.image_size-1, 0)], image_to_face_mat, True)

            rect_area = mathlib.polygon_area(np.array(rect[[0, 2, 2, 0]]).astype(
                np.float32), np.array(rect[[1, 1, 3, 3]]).astype(np.float32))
            landmarks_area = mathlib.polygon_area(landmarks_bbox[:, 0].astype(
                np.float32), landmarks_bbox[:, 1].astype(np.float32))

            # get rid of faces which umeyama-landmark-area > 4*detector-rect-area
            if self.face_type <= FaceType.FULL_NO_ALIGN and landmarks_area > 4*rect_area:
                continue
            frame_info.final_faces.append(face_image)
        return frame_info

    def plot_face(self, frame, frame_info):
        image_debug = frame.rgb.copy()
        rects = frame_info.rects
        landmarks = frame_info.landmarks
        # plot rect
        for rect in rects:
            cv2.rectangle(image_debug, tuple(rect[:2]), tuple(
                rect[2:]), color=(0, 0, 255), thickness=3)
        # for landmark in landmarks:
        #     for point in landmark:
        #         cv2.circle(image_debug, tuple(map(lambda x: int(x), point)), radius=5,
        #                    color=(0, 0, 255), thickness=3)
        # plot landmarks
        return image_debug
