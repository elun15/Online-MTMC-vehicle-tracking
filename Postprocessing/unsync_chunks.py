'''
################################
#    Pre-process AICC19 data   #
################################
'''

import os
import cv2
import time
import numpy as np



offset= {'S01': {'c001': 0,
                'c002': 1.640,
                'c003': 2.049,
                'c004': 2.177,
                'c005': 2.235},
         'S02': {'c006': 0,
                'c007': 0.061,
                'c008': 0.421,
                'c009': 0.660},
         'S03': { 'c010': 8.715,
                 'c011': 8.457,
                 'c012': 5.879,
                 'c013': 0,
                 'c014': 5.042,
                 'c015':8.492},
         'S04':{ 'c016': 0,
                'c017': 14.318,
                'c018': 29.955,
                'c019': 26.979,
                'c020': 25.905,
                'c021': 39.973,
                'c022': 49.422,
                'c023': 45.716,
                'c024': 50.853,
                'c025': 50.263,
                'c026': 70.450,
                'c027': 85.097,
                'c028': 100.110,
                'c029': 125.788,
                'c030': 124.319,
                'c031': 125.033,
                'c032': 125.199,
                'c033': 150.893,
                'c034': 140.218,
                'c035': 165.568,
                'c036': 170.797,
                'c037': 170.567,
                'c038': 175.426,
                'c039': 175.644,
                'c040': 175.838},
         'S05':{ 'c010': 0,
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


def  process(file,offset, scenario, dataset_dir, output_file,i,fi,fo):

    res = np.loadtxt(file,delimiter=' ')

    cameras = os.listdir(os.path.join(dataset_dir, scenario))

    for c in cameras:

        tStart = time.time()
        print('Processing ' + scenario + ' ' + c + ' with offset = ' + str(offset[scenario][c]))

        vdo_dir = os.path.join(dataset_dir, scenario, c, 'vdo.avi')
        video = cv2.VideoCapture(vdo_dir)

        # num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)

        offset_frames = int(round((offset[scenario][c]) * fps))

        c_rows = res[:,0]  == int(c[1:])
        res[c_rows, 2] = res[c_rows, 2] - offset_frames

        tEnd = time.time()
        print("It cost %f sec" % (tEnd - tStart))

    np.savetxt(output_file, res, delimiter=',')

    os.makedirs('/home/elg/Repositories/AICityChallengeTrack/amilan-motchallenge-devkit/res/AIC20/v'+ str(i)+ '_run/'+ str(fi)+'_'+str(fo),exist_ok=True)
    np.savetxt('/home/elg/Repositories/AICityChallengeTrack/amilan-motchallenge-devkit/res/AIC20/v'+ str(i)+ '_run/'+ str(fi)+'_'+str(fo)+'/S02.txt', res, delimiter=',')

if __name__ == '__main__':

    scenario = 'S02'

    w_size = 2110
    max_f = 2110
    frames_start = np.arange(1, max_f, w_size)
    frames_start = np.arange(1, max_f, w_size)

    frames_end = frames_start + (w_size - 1)
    # frames_start = frames_start[(frames_end < max_f)]
    # frames_end = frames_end[(frames_end < max_f)]
    # frames_end = [25, 50, 100, 250, 500, 1000, 1500, 2110]
    frames_end =1000 + np.asarray([25, 50, 100, 250, 500, 1000, 1500, 2110])
    frames_end = frames_end[(frames_end < max_f)]

    for f in range(0, len(frames_end)):
        # f_start = frames_start[f]
        f_start = 1000

        f_end = frames_end[f]

        for i in ['147']:

            results_file = './../v'+str(i) + '/' + str(f_start)  + '_' +  str(f_end) + '/S02.txt'
            unsync_file = './../v'+str(i) + '/' + str(f_start)  + '_' +  str(f_end) + '/S02_unsync.txt'

            dataset_dir = '/home/elg/Datasets/AIC20/validation'

            process(results_file,offset, scenario, dataset_dir, unsync_file,i,f_start, f_end)

    # v = 'v164'
    #
    # results_file = './../ablation_study/' + scenario + '/' + v + '.txt'
    # unsync_file = './../ablation_study/' + scenario + '/' + v + '_unsync.txt'
    #
    # dataset_dir = '/home/vpu/Datasets/AIC20/validation'
    #
    # process(results_file, offset, scenario, dataset_dir, unsync_file, v)