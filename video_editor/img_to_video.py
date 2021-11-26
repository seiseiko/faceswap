import os
import cv2
import argparse
import glob
from tqdm import tqdm


def main(args):
    input_folder = glob.glob(os.path.join(args.input_folder, '*'))
    label_frame = cv2.imread(os.path.join(args.input_folder, '00000.png'))
    output = cv2.VideoWriter(args.video_fp, cv2.VideoWriter_fourcc(
        *'mp4v'), float(args.frame_rate), (label_frame.shape[1], label_frame.shape[0]))

    with tqdm(total=int(len(input_folder))) as pbar:
        for i in range(len(input_folder)):
            frame = cv2.imread(os.path.join(
                args.input_folder, "{}.png".format(str(i).zfill(5))))
            output.write(frame)
            pbar.update(1)
    output.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Combine generated frames into a video')
    parser.add_argument('-f', '--video_fp', type=str,
                        default='../workspace/result.mp4')
    parser.add_argument('-r', '--frame_rate', type=str, default='30')
    parser.add_argument('-i', '--input_folder', type=str,
                        default='../workspace/data_target/merge_result')
    args = parser.parse_args()
    main(args)
