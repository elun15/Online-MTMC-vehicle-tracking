'''
################################
#    Pre-process AICC19 data   #
################################
'''

import os
import cv2
import time
import numpy as np


#offset = dict()
offset= {'train': {'1': 0,
                '2': 1.640,
                '3': 2.049,
                '4': 2.177,
                '5': 2.235},
                '10': 8.715,
                '11': 8.457,
                '12': 5.879,
                '13': 0,
                '14': 5.042,
                '15':8.492,
                '16': 0,
                '17': 14.318,
                '18': 29.955,
                '19': 26.979,
                '20': 25.905,
                '21': 39.973,
                '22': 49.422,
                '23': 45.716,
                 '24': 50.853,
                '25': 50.263,
                '26': 70.450,
                '27': 85.097,
                '28': 100.110,
                '29': 125.788,
                '30': 124.319,
                '31': 125.033,
                '32': 125.199,
                '33': 150.893,
                '34': 140.218,
                '35': 165.568,
                '36': 170.797,
                '37': 170.567,
                '38': 175.426,
                '39': 175.644,
                '40': 175.838},
        }




         'test': {'6': 0,
                '7': 0.061,
                '8': 0.421,
                '9': 0.660},
         'train': { ,
         'train':{
         'test':{ 'c010': 0,
                'c016': 0,
                'c017': 0,
                'c018': 0,
                'c019': 0,
                'c020': 0,
                'c021': 0,
                'c022': 0,
                'c023': 0,
                'c024': 0,
                'c025': 0,
                'c026': 0,
                'c027': 0,
                'c028': 0,
                'c029': 0,
                'c033': 0,
                'c034': 0,
                'c035': 0,
                'c036': 0}}

max_frame={'S01': 2110,
           'S02': 2110,
           'S03': 2422,
           'S04': 710,
           'S05': 4299,
           }


def  process(mode,offset):
    # Current root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Original dataset directory
    dataset_dir = os.path.join('/home/vpu/Datasets/AICC19', mode)

    scenarios = os.listdir(dataset_dir)
    for s in scenarios:

        cameras = os.listdir(os.path.join(dataset_dir, s))

        for c in cameras:

            tStart = time.time()
            print('Processing ' + s + ' ' + c + ' with offset = ' + str(offset[s][c]))

            vdo_dir = os.path.join(dataset_dir, s, c, 'vdo.avi')
            video = cv2.VideoCapture(vdo_dir)

            num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video.get(cv2.CAP_PROP_FPS)
            h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

            blank_image = np.zeros((h, w, 3), np.uint8)

            output_dir = os.path.join(dataset_dir, s, c, 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            offset_frames = int(round((offset[s][c]) * fps))
            frame_counter = 1

            # If offset > 0 : fill with blank images at the begining of the sequence
            if offset_frames > 0:
                for f in range(1, offset_frames + 1):
                    frame_name = os.path.join(output_dir, str(frame_counter).zfill(6) + ".jpg")
                    cv2.imwrite(frame_name, blank_image)

                    frame_counter += 1

            # Read video file and save image frames
            while video.isOpened():

                ret, frame = video.read()
                frame_name = os.path.join(output_dir, str(frame_counter).zfill(6) + ".jpg")

                # print(video.get(cv2.CAP_PROP_POS_FRAMES))

                if not ret:
                    print("End of video file.")
                    a = 1
                    break
                cv2.imwrite(frame_name, frame)
                frame_counter += 1

            if frame_counter < max_frame[s]:
                # Fill at the end with black frames to reach max number of frames
                for f in range(frame_counter, max_frame[s] + 1):
                    frame_name = os.path.join(output_dir, str(frame_counter).zfill(6) + ".jpg")
                    cv2.imwrite(frame_name, blank_image)
                    frame_counter += 1

            tEnd = time.time()
            print("It cost %f sec" % (tEnd - tStart))


if __name__ == '__main__':

    file = 'test'

    process(file,offset)
