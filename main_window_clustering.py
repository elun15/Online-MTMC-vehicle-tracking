'''
################################
#  Cluster-based MTMC tracking #
################################
'''

# Python modules
import os
import sys
import cv2
import math
import time
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
#import torchvision.models
import torch

import torch.nn.functional as F
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

from thirdparty.dtw import accelerated_dtw
from thirdparty.dtw import dtw

# Own modules
import preprocessing_data
import sct
import camera
import dataset
from network import resnet_elg
import colors
import display
import features
from thirdparty import sklearn_dunn
import clustering
import tracking

class mtmc():
    def __init__(self, dataset_dir, sct_tracker):
        self.dataset_root_dir = dataset_dir
        self.sct_tracker = sct_tracker

        self.max_frame = {'S01': 2132,
                          'S02': 2110,
                          'S03': 2422,
                          'S04': 710,
                          'S05': 4299,
                          'S06': 2001
                          }

        self.offset = {'S01': {'c001': 0,
                               'c002': 1.640,
                               'c003': 2.049,
                               'c004': 2.177,
                               'c005': 2.235},
                       'S02': {'c006': 0,
                               'c007': 0.061,
                               'c008': 0.421,
                               'c009': 0.660},
                       'S03': {'c010': 8.715,
                               'c011': 8.457,
                               'c012': 5.879,
                               'c013': 0,
                               'c014': 5.042,
                               'c015': 8.492},
                       'S04': {'c016': 0,
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
                       'S05': {'c010': 0,
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
                               'c036': 0},
                       'S06': {'c041': 0,
                               'c042': 0,
                               'c043': 0,
                               'c044': 0,
                               'c045': 0,
                               'c046': 0}}



        self.colors = colors.distinguishable_colors()

        self.preprocess_flag = False
        self.display = True
        self.dist_th = 0.0002
        self.global_tracks  = list(list())

        self.global_tracks.append(list())

    # frame ,time, cam_id ,SCT_id ,latitude ,longitude,  start_x, start_y ,
    # end_x, end_y, start_time, end_time, left, top, width, heigth
    # def __init__(self, scene):


if __name__ == '__main__':

    contador_total_escritura = 0
    contador_total_sct = 0
    display_flag = False
    '''
    Train set: S01, S03, S04
    Test set: S02, S05
    '''
    dataset_dir = '/home/vpu/Datasets/AIC20'
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')
    # sct_tracker = 'mtsc_tc_ssd512'
    sct_tracker = 'mtsc_tnt_mask_rcnn'
    set = ['test']  # ['test' 'train']

    l2_norm = lambda x, y: (x - y) ** 2



    # Initialize global mtmc class
    mtmc = mtmc(dataset_dir, sct_tracker)

    # Inicialize cam class
    cam = camera.camera(os.path.join(mtmc.dataset_root_dir, 'calibration'))

    # Dataset class
    aicc = dataset.dataset()

    # Display class
    display = display.display(mtmc.display)

    # Features class
    net = resnet_elg.resnet101(pretrained=True)
    model_name ='reid.pth'
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/' + model_name)
    feat = features.features(aicc, net, 'appearance_only')  # appearance_only , distance_only
    feat.load_model(model_path, model_name)

    # Tracking class
    track = tracking.tracking(mtmc)


    # Pre-processing needs to be executed only once after downloading the AICC19 dataset

    # if mtmc.preprocess_flag:
    #     for s in set:
    #         print('Preprocessing data from ' + s + 'set' + '\n')
    #         preprocess_data.process(set, mtmc.offset)

    # Load Single Camera Tracking data

    # Initialize sct strucure
    sct = sct.sct(mtmc)

    print('Loading SCT and homographies...')
    # For each set (train/test/both)
    for st in set:

        # Set directory
        set_dir = os.path.join(mtmc.dataset_root_dir, st)
        scenarios = os.listdir(set_dir)

        # For each scenario in the set
        for s in scenarios:

            # Create new data dictionary in sct class
            sct.new(s)

            # Fill it with sct data: e.g.  sct.data[scene][camera] -> [ndarray]
            sct.load(st, s, mtmc.offset)

            # Load homography matrices
            cameras = os.listdir(os.path.join(mtmc.dataset_root_dir, st, s))

            for c in cameras:
                cam.load_homography_matrix(c)

    print('Done.')

    # MTMC - Main Loop

    # For each set (train/test/both)
    for st in set:

        # Set directory
        set_dir = os.path.join(mtmc.dataset_root_dir, st)
        scenarios = os.listdir(set_dir)
        scenarios.sort()

        # Results file
        file_results = os.path.join(results_dir, s, 'results_clustering.txt')

        f_id = open(file_results, 'w+')
        # Scenarios
        for s in scenarios:

            cameras = os.listdir(os.path.join(mtmc.dataset_root_dir, st, s))
            cameras.sort()

            # Frames
            for f in range(1,  mtmc.max_frame[s] + 1):

                mtmc.global_tracks.append(list())

                # Create empty dictionary for this frame sct
                sct_f = sct.new_frame_data()

                # Create empty dictionary for this window sct
                sct_f_w = sct.new_frame_w_data()

                # Window size
                W = 11

                # Cameras
                for c in cameras:

                    print('Processing ' + str(s) + ' frame ' + str(f) + ' camera ' + str(c))

                    frame_img = Image.open(os.path.join(mtmc.dataset_root_dir, st, s, c, 'img', '%06d.jpg' % f))
                    # display.show_frame(frame_img,c)

                    sct_array = np.array(sct.data[s][c])
                    # Get SCT data at current frame
                    sct_f_data = sct_array[sct_array[:, 0] == f, :]

                    # Get SCT data within all frames in the window
                    first_time = True
                    frame_img_w = {}

                    # Limit frames in the window
                    min_f_w = f - math.floor(W / 2)
                    if min_f_w < 1:
                        min_f_w = 1

                    max_f_w = f + math.floor(W / 2)
                    if max_f_w >  mtmc.max_frame[s] + 1:
                        max_f_w = mtmc.max_frame[s] + 1


                    for f_w in range(min_f_w, max_f_w):

                        # Store images from each frame in the window
                        img = Image.open(os.path.join(mtmc.dataset_root_dir, st, s, c, 'img', '%06d.jpg' % f_w))
                        frame_img_w[f_w] = img

                        # Store sct data
                        if first_time:
                            sct_f_w_data = sct_array[sct_array[:, 0] == f_w, :]
                            first_time = False
                        else:
                            aux = sct_array[sct_array[:, 0] == f_w, :]

                            if aux.shape.__len__() == 1:
                                aux = np.expand_dims(aux, axis=0)

                            sct_f_w_data = np.concatenate((sct_f_w_data, aux), axis=0)

                    # Fill sct_f_w dictionary with current window information
                    for i in range(sct_f_w_data.shape[0]):

                        sct_f_w['id_cam'].append(int(c[-3:]))
                        sct_f_w['f'].append(int(sct_f_w_data[i][0]))
                        sct_f_w['id'].append(int(sct_f_w_data[i][1]))

                        x = int(round(sct_f_w_data[i][2]))
                        y = int(round(sct_f_w_data[i][3]))
                        w = int(round(sct_f_w_data[i][4]))
                        h = int(round(sct_f_w_data[i][5]))
                        sct_f_w['x'].append(x)
                        sct_f_w['y'].append(y)
                        sct_f_w['w'].append(w)
                        sct_f_w['h'].append(h)

                        # draw bbox
                        # display.draw_bbox(x, y, w, h)

                        # Crop bbox
                        bbox_img = transforms.functional.crop(frame_img_w[int(sct_f_w_data[i][0])], y, x, h, w)

                        # Base of the bounding box to projection
                        bx = round(x + round(w / 2))
                        by = round(y + h)
                        xw, yw = cam.apply_homography_image_to_world(bx, by, cam.homography_matrix[c])
                        sct_f_w['xw'].append(xw)
                        sct_f_w['yw'].append(-yw)  # IMPORTANT: changed sign to positive coordinate

                        # DISPLAY WINDOW TRAJECTORIES
                        if display_flag:
                            plt.plot(xw, yw, 'o', color=mtmc.colors.list[int(c[-3:])], ms=8)
                            plt.text(xw, yw, str(int(sct_f_w_data[i][1])), fontsize=18)
                        # Feature extraction

                        # Get a square bbox to not to change the aspect ratio
                        # square_bbox = aicc.square(bbox_img,frame_img, x, y)
                        bbox_padded = aicc.pad(bbox_img, (100, 100, 100))

                        # plt.figure()
                        # plt.imshow(bbox_padded)
                        bbox_padded_tensor = aicc.data_transform(bbox_padded)

                        features_np = feat.extract(bbox_padded_tensor)
                        sct_f_w['features'].append(features_np)


                    # Fill sct_f dictionary with current frame information
                    for i in range(sct_f_data.shape[0]):

                        sct_f['id_cam'].append(int(c[-3:]))
                        sct_f['id'].append(int(sct_f_data[i][1]))

                        x = int(round(sct_f_data[i][2]))
                        y = int(round(sct_f_data[i][3]))
                        w = int(round(sct_f_data[i][4]))
                        h = int(round(sct_f_data[i][5]))
                        sct_f['x'].append(x)
                        sct_f['y'].append(y)
                        sct_f['w'].append(w)
                        sct_f['h'].append(h)

                        # draw bbox
                        # display.draw_bbox(x, y, w, h)

                        # Crop bbox
                        bbox_img = transforms.functional.crop(frame_img, y, x, h, w)

                        # Base of the bounding box to projection
                        bx = round(x + round(w / 2))
                        by = round(y + h)
                        xw, yw = cam.apply_homography_image_to_world(bx, by, cam.homography_matrix[c])
                        sct_f['xw'].append(xw)
                        sct_f['yw'].append(-yw)  #IMPORTANT: changed sign to positive coordinate

                        # Feature extraction

                        # Get a square bbox to not to change the aspect ratio
                        #square_bbox = aicc.square(bbox_img,frame_img, x, y)
                        bbox_padded = aicc.pad(bbox_img,(100, 100, 100))

                        # plt.figure()
                        # plt.imshow(bbox_padded)
                        bbox_padded_tensor = aicc.data_transform(bbox_padded)
                        features_np = feat.extract(bbox_padded_tensor)

                        sct_f['features'].append(features_np)



                if display_flag:
                    plt.show(block=False)

                # Display
                # if f == 1:
                #     fig =  plt.figure()
                #     ax = fig.gca()

                num_det_f = sct_f['id_cam'].__len__()
                num_tracks_w = sct_f_w['id_cam'].__len__()

                if num_det_f != 0:

                    # # Display
                    # for i in range(num_det_f):
                    #
                    #     plt.plot(sct_f['xw'][i], sct_f['yw'][i], 'o', color=mtmc.colors.list[sct_f['id_cam'][i]], ms=8)
                    #     # plt.text(np.array(sct_f['xw'][i]), np.array(sct_f['yw'][i]), str(sct_f['id_cam'][i]) + ' ' + str(int(sct_f['id'][i])))
                    #     if f % 1 == 0:
                    #         plt.text(np.array(sct_f['xw'][i]), np.array(sct_f['yw'][i]), str(int(sct_f['id'][i])),
                    #                  fontsize=18)
                    #     # ax.add_artist(plt.Circle((sct_f['xw'][i], sct_f['yw'][i]), mtmc.dist_th, color='#000033', alpha=0.5, fill=False))
                    #
                    # if f % 1 == 0:
                    #     # ax.add_artist(plt.Circle((sct_f['xw'][i], sct_f['yw'][i]), mtmc.dist_th, color='#000033', alpha=0.5, fill=False))
                    #     plt.title('Frame ' + str(f))
                    #     plt.show(block=False)
                    #     plt.close('all')
                    #     fig = plt.figure()
                    #     ax = fig.gca()


                    #  Initialize clustering class. New clusters structure each frame
                    clust = clustering.clustering(mtmc)

                    # Extract trajectories corresponding to the detections at the current frame

                    # Matrix pairs_id_cam_f with colum 1 = ids_cam and colum 2 = ids (at current frame)
                    pairs_id_cam_f = (np.concatenate((np.expand_dims(np.array(sct_f['id_cam']), axis=0),
                                                          np.expand_dims(np.array(sct_f['id']), axis=0)), axis=0)).T


                    id_cam_f = (np.expand_dims(np.array(sct_f['id_cam']), axis=0)).T
                    ids_f = (np.expand_dims(np.array(sct_f['id']), axis=0)).T

                    # Matrix pairs_id_cam_f_w with colum 1 = ids_cam and colum 2 = ids (at current windowl )
                    pairs_id_cam_f_w = (np.concatenate((np.expand_dims(np.array(sct_f_w['id_cam']), axis=0),
                                                      np.expand_dims(np.array(sct_f_w['id']), axis=0)), axis=0)).T

                    id_cam_f_w = (np.expand_dims(np.array(sct_f_w['id_cam']), axis=0)).T
                    ids_f_w = (np.expand_dims(np.array(sct_f_w['id']), axis=0)).T

                    feature_descriptor_trajectories = list()
                    xw_trajectories = list()
                    yw_trajectories = list()

                    for p in range(0, num_det_f):

                        pair = np.expand_dims(pairs_id_cam_f[p,:], axis = 0)

                        id_cam = id_cam_f[p]
                        id = ids_f[p]

                        id_cam_matches = np.equal(id_cam_f_w, id_cam)
                        id_matches = np.equal(ids_f_w, id)

                        matches = id_cam_matches & id_matches

                        # matches_int = np.zeros(matches.size)
                        # matches_int[np.where(np.squeeze(matches) == True)] = 1
                        # matches_int = matches_int.T

                        # All xw yw positions of the trajectory
                        xw_matches_w = np.expand_dims(np.array(sct_f_w['xw']), axis=1)[matches]
                        xw_trajectories.append(xw_matches_w)
                        yw_matches_w = np.expand_dims(np.array(sct_f_w['yw']), axis=1)[matches]
                        yw_trajectories.append(yw_matches_w)
                        # Feature descriptor by averaging all de features vectors
                        features_matches_w = np.expand_dims(np.array(sct_f_w['features']), axis=1)[matches]
                        feature_descriptor_w = np.mean(features_matches_w, axis = 0)
                        feature_descriptor_trajectories.append(feature_descriptor_w)

                        # Here, we use L2 norm as the element comparison distance

                        # d, cost_matrix, acc_cost_matrix, path = dtw(l, l2_norm)
                        # d, cost_matrix, acc_cost_matrix, path = dtw(xw_matches_w, xw_matches_w , l2_norm)


                    dist_spatial = np.zeros((num_det_f, num_det_f))
                    # Set diagonal to 1 to avoid zeros
                    dist_spatial = dist_spatial + (np.eye(dist_spatial.shape[0]))
                    for i in range(num_det_f):
                        for j in range(num_det_f):

                            # Only half of the symmetric matrix
                            if (i != j) & (i > j):
                                d_xw, cost_matrix, acc_cost_matrix, path = accelerated_dtw(xw_trajectories[i], xw_trajectories[j], l2_norm)
                                d_yw, cost_matrix, acc_cost_matrix, path = accelerated_dtw(yw_trajectories[i], yw_trajectories[j], l2_norm)

                                dist_spatial[i, j] = math.sqrt(d_xw + d_yw)
                                dist_spatial[j,i] = math.sqrt(d_xw + d_yw)


                    # If there are some close detections and more than 1 camera
                    if  ((np.unique(sct_f['id_cam'])).size > 1): #(sum(sum(dist_flag)) != 0) and #dist_flag = (dist_spatial < mtmc.dist_th) * 1

                        # Perform clustering
                        dist_features = pairwise_distances(np.array(feature_descriptor_trajectories),
                                                           np.array(feature_descriptor_trajectories),
                                                           metric='euclidean')
                        # dist_features_norm = (F.softmax(torch.from_numpy(dist_features), dim=1)).numpy()

                        # Apply restrictions
                        if feat.characteristic == 'distance_only':
                            restricted_dist_features, association_matrix = feat.apply_restrictions(dist_spatial, dist_spatial, sct_f, mtmc.dist_th,feat.characteristic)

                        elif feat.characteristic == 'appearance_only':
                            restricted_dist_features, association_matrix = feat.apply_restrictions(dist_features,  dist_spatial, sct_f, mtmc.dist_th,feat.characteristic)

                        else: # both

                            restricted_dist_features, association_matrix = feat.apply_restrictions(dist_features, dist_spatial, sct_f,  mtmc.dist_th,feat.characteristic)

                        # Compute hierarchical clustering
                        idx, optimal_clusters = clust.compute_clusters(restricted_dist_features, association_matrix)

                    else: # All detections are alone, no need to cluster

                        optimal_clusters = num_det_f
                        idx = np.array(range(0, optimal_clusters))

                    # Fill clust.clusters structure
                    # plt.figure()

                    for cl in range(optimal_clusters):

                        # Initialize empty structure of the cluster
                        clust.clusters_frame.append(clust.new_cluster())

                        # Extract  detection in each    cluster
                        det_in_cluster = np.where(idx == cl)[0]

                        # Plot detections in cluster
                        # clust.display_detections_cluster(sct_f,det_in_cluster,cl)

                        # Get centroid of the cluster, mean position of every detectionin the cluster
                        mean_xw = np.mean((np.array(sct_f['xw']))[det_in_cluster])
                        mean_yw = np.mean((np.array(sct_f['yw']))[det_in_cluster])

                        clust.clusters_frame[-1]['xw'] = mean_xw
                        clust.clusters_frame[-1]['yw'] = mean_yw

                        # Plot centroid
                        # clust.display_centroid_cluster(mean_xw, mean_yw, cl)


                        for d in range(det_in_cluster.__len__()):
                            idx_det = det_in_cluster[d]
                            clust.clusters_frame[-1]['det'].append( clust.new_detection())
                            clust.clusters_frame[-1]['det'][-1]['x'] = sct_f['x'][idx_det]
                            clust.clusters_frame[-1]['det'][-1]['y'] = sct_f['y'][idx_det]
                            clust.clusters_frame[-1]['det'][-1]['w'] = sct_f['w'][idx_det]
                            clust.clusters_frame[-1]['det'][-1]['h'] = sct_f['h'][idx_det]
                            clust.clusters_frame[-1]['det'][-1]['id_cam'] = sct_f['id_cam'][idx_det]
                            clust.clusters_frame[-1]['det'][-1]['features'] = sct_f['features'][idx_det]


                    # plt.show(block = False)
                    # plt.close('all')


                    for i in range(optimal_clusters):

                        for det in range(clust.clusters_frame[i]['det'].__len__()):

                            arg1 = clust.clusters_frame[i]['det'][det]['id_cam']
                            arg2 = clust.clusters_frame.index(clust.clusters_frame[i]) + 1
                            arg3 = f
                            arg4 = clust.clusters_frame[i]['det'][det]['x']
                            arg5 = clust.clusters_frame[i]['det'][det]['y']
                            arg6 = clust.clusters_frame[i]['det'][det]['w']
                            arg7 = clust.clusters_frame[i]['det'][det]['h']

                            f_id.write("%d %d %d %d %d %d %d -1 -1\n" % (arg1, arg2, arg3, arg4, arg5, arg6, arg7))



                '''
                # CLUSTERS - TRACKS   ASSOCIATION
                track.predict_new_locations()

                track.cluster_track_assignment(clust.clusters_frame, 1)

                # Update each assigned track with the corresponding detection.It calls the correct method of vision.KalmanFilter to correct the location estimate.
                #  Next, it stores the new bounding box, and increases the age of the track and the total  visible count by 1.
                #  Finally, the function sets the invisible count to 0.

                track.update_assigned_tracks(clust.clusters_frame)

                #Mark each unassigned track as invisible and increase its age by 1

                track.update_unassigned_tracks()

                # Delete tracks that have been invisible for too many frames

                track.delete_lost_tracks()

                # Create new tracks from unassigned detections. Assume that any unassigned detection is a start of a new track.
                # In practice you can use other cues to eliminate nnoisy detections such as size, location, or appearance

                track.create_new_tracks_KF(clust.clusters_frame)

                track.save_global_tracking_data(clust.clusters_frame,f,mtmc.global_tracks)

                # DNo(k) = indexDN(squareform(vec), labels(:, k), 'euclidean');


                plt.close('all')

                # ESCRITURA RESULTADOS

                if track.updated_flag:

                    num_tracks_f = mtmc.global_tracks[f].__len__()
                    for i in range(num_tracks_f):

                        for det in range(mtmc.global_tracks[f][i]['det'].__len__()):

                            arg1 = mtmc.global_tracks[f][i]['det'][det]['id_cam']
                            arg2 = mtmc.global_tracks[f][i]['id']
                            arg3 = f
                            arg4 = mtmc.global_tracks[f][i]['det'][det]['x']
                            arg5 = mtmc.global_tracks[f][i]['det'][det]['y']
                            arg6 = mtmc.global_tracks[f][i]['det'][det]['w']
                            arg7 = mtmc.global_tracks[f][i]['det'][det]['h']

                            f_id.write("%d %d %d %d %d %d %d -1 -1\n" % (arg1, arg2, arg3, arg4, arg5, arg6, arg7))
            f_id.close()

            a=1
            '''

            f_id.close()






#
# bbox_replicated_tensor = aicc.data_transform(Image.fromarray(bbox_replicated))
# bbox_replicated_tensor = torch.unsqueeze(bbox_replicated_tensor, dim=0)
# net.cuda()
# preds = net(bbox_replicated_tensor.cuda())
# predictions_replicated = preds.topk(k=1)
# #
# bbox_img_tensor = aicc.data_transform(Image.fromarray(bbox_img))
# bbox_img_tensor = torch.unsqueeze(bbox_img_tensor, dim=0)
# net.cuda()
# preds = net(bbox_img_tensor.cuda())
# predictions_res_norm = preds.topk(k=1)
#
# square_bbox_tensor = aicc.data_transform(Image.fromarray(square_bbox))
# square_bbox_tensor = torch.unsqueeze(square_bbox_tensor, dim=0)
# net.cuda()
# preds = net(square_bbox_tensor.cuda())
# predictions_square_bbox_norm = preds.topk(k=1)