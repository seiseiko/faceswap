import os
import cv2
import argparse
from tqdm import tqdm
def main(args):
    videoCapture = cv2.VideoCapture(args.video_fp)
    ori_fps = videoCapture.get(cv2.CAP_PROP_FPS)
    usr_fps = ori_fps if (int(args.frame_rate) == 0 or int(args.frame_rate) > ori_fps) else int(args.frame_rate)
    ori_time_break, usr_time_break = 1/ori_fps, 1/usr_fps

    success, frame = videoCapture.read()
    
    time_break, img_count = 0, 0
    with tqdm(total=int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while success:
            time_break += ori_time_break
            if time_break >= usr_time_break:
                cv2.imwrite(os.path.join(args.out_folder, "{}.png".format(str(img_count).zfill(5))), frame)
                time_break -= usr_time_break
                img_count += 1
            pbar.update(1)
            success, frame = videoCapture.read()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-f', '--video_fp', type=str, default='../workspace/source.mp4')
    parser.add_argument('-r', '--frame_rate', type=str, default='0')
    parser.add_argument('-o', '--out_folder', type=str, default='../workspace/data_source/origin')
    args = parser.parse_args()
    main(args)
