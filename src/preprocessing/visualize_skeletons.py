import cv2 as cv
import numpy as np
from PIL import Image
import math
import os
import jsonlines as json
import argparse

limbs_list = [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14]
stick_width = 6
colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
          [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]]


def process_one_image(description_file_name, output_dir):
    with json.open(description_file_name) as reader:
        num_limb = int(len(limbs_list) / 2)
        limbs = np.array(limbs_list).reshape((num_limb, 2))
        limbs = limbs.astype(np.int)
        canvas = np.float32(Image.new('RGB', (1920, 1080), (0, 0, 0)))
        for person_info in reader:
            for part in range(1, max(limbs_list)):
                cv.circle(canvas, (int(person_info[part][2]), int(person_info[part][1])), 3, (0, 0, 0), -1)
            for idx, l in enumerate(limbs):
                cur_canvas = canvas.copy()
                Y = np.array([person_info[l[0]][1], person_info[l[1]][1]])
                X = np.array([person_info[l[0]][2], person_info[l[1]][2]])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stick_width), int(angle), 0, 360, 1)
                cv.fillConvexPoly(cur_canvas, polygon, colors[idx])
                canvas = canvas * 0.4 + cur_canvas * 0.6  # for transparency

    print(os.path.join(output_dir, os.path.splitext(os.path.basename(description_file_name))[0] + '.jpg'))
    # cv.imshow('img', canvas)
    cv.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(description_file_name))[0] + '.jpg'), canvas)
    # cv.waitKey()


def process_batch(files_dir, output_dir):
    for file in os.listdir(files_dir):
        process_one_image(os.path.join(files_dir, file), output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('poses_files_dir',
                        help="root directory for the whole data set's txt poses files")
    parser.add_argument('output_dir',
                        help='root directory for output images')

    args = parser.parse_args()
    for label in os.listdir(args.poses_files_dir):
        class_dir = os.path.join(args.output_dir, label)
        if not os.path.exists(class_dir): os.mkdir(class_dir)
        process_batch(os.path.join(args.poses_files_dir, label), os.path.join(args.output_dir, label))


        # process_one_image('/home/moniczka/sample/yimGp0XUcEE_150_0210.txt')
