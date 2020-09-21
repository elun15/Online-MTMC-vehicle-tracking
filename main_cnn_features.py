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
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
# import torchvision.models
import torch

import torch.nn.functional as F
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

import onnx

# Own modules
from preprocessing_data import preprocess_data
import sct
import camera
import dataset
from network import resnet_elg
from network import joined_network
from network import net_id_classifier
from network import vgg
import colors
import display
import features
from thirdparty import sklearn_dunn
import clustering
import tracking
import torchfile
import torchvision.transforms as transforms


# from torch.utils.serialization import load_lua

class mtmc():
    def __init__(self, dataset_dir, sct_tracker):
        self.dataset_root_dir = dataset_dir
        self.sct_tracker = sct_tracker

        self.max_frame = {'S01': 2132,
                          'S02': 2110,
                          'S03': 2422,
                          'S04': 710,
                          'S05': 4299,
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
                               'c036': 0}}

        self.colors = colors.distinguishable_colors()

        self.preprocess_flag = False
        self.display = True
        self.dist_th = 0.00008
        self.global_tracks = list(list())

        self.global_tracks.append(list())

    # frame ,time, cam_id ,SCT_id ,latitude ,longitude,  start_x, start_y ,
    # end_x, end_y, start_time, end_time, left, top, width, heigth
    # def __init__(self, scene):


if __name__ == '__main__':

    contador_total_escritura = 0
    contador_total_sct = 0
    '''
    Train set: S01, S03, S04
    Test set: S02, S05
    '''
    dataset_dir = '/home/vpu/Datasets/AIC20'
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    sct_tracker = 'mtsc_tc_ssd512'
    # sct_tracker = 'mtsc_tc_mask_rcnn'
    set = ['validation']  # ['test' 'train']

    # Initialize global mtmc class
    mtmc = mtmc(dataset_dir, sct_tracker)

    # Inicialize cam class
    cam = camera.camera(os.path.join(mtmc.dataset_root_dir, 'calibration'))

    # Dataset class
    aicc = dataset.dataset()

    # Display class
    display = display.display(mtmc.display)

    ### LOAD NET
    # model_cls = 'ResNet50_AICC19_classifier_2048_best.pth.tar'
    # model= 'ResNet50_AIC20_veri_classifier_best_focal_triplet.pth.tar'  # triplet loss
    # model= 'ResNet50_AIC20_veri_focal_classifier_best.pth.tar'  # focal loss
    # model = 'ResNet50_AIC20_veri_focal_imaug_classifier_best.pth.tar'
    model = "ResNet50_AIC20_veri_classifier_focalimaugfreeze6_best.pth.tar"

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/' + model)

    net = net_id_classifier.net_id_classifier('ResNet50', 903)
    weights = torch.load(model_path)['state_dict']
    net.load_state_dict(weights, strict=True)

    net.cuda()
    net.eval()


    # Features class
    # net = resnet_elg.resnet50(pretrained=True)
    # model_name = 'reid.pth.pth' #reid , net_last resnet50_optimizer_120
    # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/' + model_name)

    feat = features.features(aicc, net, 'both')  # appearance_only , distance_only
    # feat.load_model(model_path, model_path)


    # # Tracking class
    track = tracking.tracking(mtmc)


    # Pre-processing needs to be executed only once after downloading the AICC19 dataset
    if mtmc.preprocess_flag:
        for s in set:
            print('Preprocessing data from ' + s + 'set' + '\n')
            preprocess_data.process(set, mtmc.offset)

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
        for s in ['S02']:

            # Create new data dictionary in sct class
            sct.new(s)

            # Fill it with sct data: e.g.  sct.data[scene][camera] -> [ndarray]
            sct.load(st, s, mtmc.offset, flag_filter_size=True)

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
        file_results = os.path.join(results_dir, s, 'v38.txt')

        f_id = open(file_results, 'w+')

        # Scenarios
        for s in ['S02']:

            cameras = os.listdir(os.path.join(mtmc.dataset_root_dir, st, s))
            cameras.sort()
            tic = time.time()

            # Frames
            for f in range(1,mtmc.max_frame[s] + 1):
                # print(['Frame ' + str(f)])


                mtmc.global_tracks.append(list())

                # Create empty dictionary for this frame sct
                sct_f = sct.new_frame_data()

                # Cameras
                for c in cameras:

                    # print('Processing ' + str(s) + ' frame ' + str(f) + ' camera ' + str(c))

                    frame_img = Image.open(os.path.join(mtmc.dataset_root_dir, st, s, c, 'img', '%06d.jpg' % f))
                   # display.show_frame(frame_img,c)

                    sct_array = np.array(sct.data[s][c])
                    sct_f_data = sct_array[sct_array[:, 0] == f, :]

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
                        #display.draw_bbox(x, y, w, h)

                        # Crop bbox
                        bbox_img = transforms.functional.crop(frame_img, y, x, h, w)

                        # Get a square bbox to not to change the aspect ratio
                        # square_bbox = aicc.square(bbox_img,frame_img, x, y)
                        # bbox_padded = aicc.pad(bbox_img, (0, 0, 0))

                        bbox_img_norm = aicc.data_transform((bbox_img))
                        sct_f['bbox'].append(bbox_img_norm)

                        # Base of the bounding box to projection
                        bx = round(x + round(w / 2))
                        by = round(y + h)
                        xw, yw = cam.apply_homography_image_to_world(bx, by, cam.homography_matrix[c])
                        sct_f['xw'].append(xw)
                        sct_f['yw'].append(-yw)  # IMPORTANT: changed sign to positive coordinate

                        # Feature extraction

                        # plt.figure()
                        # plt.imshow(bbox_padded)

                        features_np = feat.extract(bbox_img_norm)
                        sct_f['features'].append(features_np)

                # Display

                # if f == 1:
                #     fig =  plt.figure()
                #     ax = fig.gca()

                num_det_f = sct_f['id_cam'].__len__()

                if num_det_f != 0:

                    # Display
                    # for i in range(num_det_f):
                    #
                    #     plt.plot(sct_f['xw'][i],sct_f['yw'][i],'o', color = mtmc.colors.list[sct_f['id_cam'][i]], ms = 8)
                    #     # plt.text(np.array(sct_f['xw'][i]), np.array(sct_f['yw'][i]), str(sct_f['id_cam'][i]) + ' ' + str(int(sct_f['id'][i])))
                    #     if f%5 == 0:
                    #         plt.text(np.array(sct_f['xw'][i]), np.array(sct_f['yw'][i]), str(int(sct_f['id'][i])), fontsize = 18)
                    #     # ax.add_artist(plt.Circle((sct_f['xw'][i], sct_f['yw'][i]), mtmc.dist_th, color='#000033', alpha=0.5, fill=False))
                    #
                    # if f % 40 == 0:
                    #
                    #     # ax.add_artist(plt.Circle((sct_f['xw'][i], sct_f['yw'][i]), mtmc.dist_th, color='#000033', alpha=0.5, fill=False))
                    #     plt.title('Frame ' + str(f))
                    #     plt.show(block = False)
                    #     plt.close('all')
                    #     fig = plt.figure()
                    #     ax = fig.gca()

                    # Clustering mode

                    # Spatial distance
                    xy = np.transpose(np.stack((np.array(sct_f['xw']), np.array(sct_f['yw'])), axis=0))

                    dist_spatial = pairwise_distances(xy, xy,
                                                       metric='euclidean')  # dist2 = pdist(xy,metric= metric)  #euclidean cosine cityblock

                    # Set diagonal to 1 to avoid zeros
                    dist_spatial = dist_spatial + (np.eye(dist_spatial.shape[0]))

                    # Flag matrix with 1 when sct detections are closer than threshold
                    dist_flag = (dist_spatial < mtmc.dist_th) * 1

                    # norm = normalize(dist, norm='l2', axis = 0, copy = True, return_norm = False)

                    #  Initialize clustering class. New clusters structure each frame
                    clust = clustering.clustering(mtmc)

                    # If there are some close detections and more than 1 camera
                    if (sum(sum(dist_flag)) != 0) and ((np.unique(sct_f['id_cam'])).size > 1):

                        # Perform clustering using features

                        features_all = np.array(sct_f['features'])
                        dist_features = pairwise_distances(features_all, features_all, metric='euclidean') #CITIBLOCK PROBAR
                        # dist_features_norm = (F.softmax(torch.from_numpy(dist_features), dim=1)).numpy()
                        #
                        if feat.characteristic == 'distance':
                            restricted_dist_features, association_matrix = feat.apply_restrictions(dist_spatial,
                                                                                                          dist_spatial,
                                                                                                          sct_f,
                                                                                                          mtmc.dist_th,
                                                                                                          feat.characteristic)
                            idx, optimal_clusters = clust.compute_clusters(restricted_dist_features, association_matrix)


                        elif feat.characteristic == 'appearance':
                            restricted_dist_features, association_matrix = feat.apply_restrictions(
                                dist_features, dist_spatial, sct_f, mtmc.dist_th, feat.characteristic)

                            idx, optimal_clusters = clust.compute_clusters(restricted_dist_features, association_matrix)

                        else:

                            # Clustering
                            restricted_dist_features, association_matrix = feat.apply_restrictions(
                                dist_features, dist_spatial, sct_f, mtmc.dist_th, feat.characteristic)
                            idx, optimal_clusters = clust.compute_clusters(restricted_dist_features, association_matrix)



                    else:  # All detections are alone, no need to cluster

                        optimal_clusters = num_det_f
                        idx = np.array(range(0, optimal_clusters))
                        association_matrix = np.array([])
                        dist_features = []

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
                            clust.clusters_frame[-1]['det'].append(clust.new_detection())
                            new_w = round(sct_f['w'][idx_det] + sct_f['w'][idx_det] * 0)
                            new_h = round(sct_f['h'][idx_det] + sct_f['h'][idx_det] * 0)
                            # c_x = sct_f['x'][idx_det] + round(sct_f['w'][idx_det] / 2 )
                            # c_y = sct_f['y'][idx_det] + round(sct_f['h'][idx_det] / 2 )
                            clust.clusters_frame[-1]['det'][-1]['x'] = sct_f['x'][idx_det] + round(sct_f['w'][idx_det] / 2 ) - round(new_w / 2)
                            clust.clusters_frame[-1]['det'][-1]['y'] = sct_f['y'][idx_det] + round(sct_f['h'][idx_det] / 2 ) - round(new_h / 2)
                            clust.clusters_frame[-1]['det'][-1]['w'] = new_w
                            clust.clusters_frame[-1]['det'][-1]['h'] = new_h
                            clust.clusters_frame[-1]['det'][-1]['id_cam'] = sct_f['id_cam'][idx_det]
                            clust.clusters_frame[-1]['det'][-1]['id_global'] = int(idx_det)


                            # clust.clusters_frame[-1]['det'][-1]['features'] = sct_f['features'][idx_det]

                    # plt.show(block = False)
                    # plt.close('all')




                # CLUSTERS - TRACKS   ASSOCIATION
                track.predict_new_locations()
                # track.display_tracks()
                # plt.title('Frame ' + str(f))
                # plt.show(block=False)

                track.cluster_track_assignment(clust.clusters_frame, 1)

                # Update each assigned track with the corresponding detection.It calls the correct method of vision.KalmanFilter to correct the location estimate.
                #  Next, it stores the new bounding box, and increases the age of the track and the total  visible count by 1.
                #  Finally, the function sets the invisible count to 0.

                track.update_assigned_tracks(clust.clusters_frame)

                #Mark each unassigned track as invisible and increase its age by 1

                track.update_unassigned_tracks()

                # Delete tracks that have been invisible for too many frames

                track.delete_lost_tracks()

                track.check_unassigned_clusters(clust.clusters_frame, association_matrix, dist_features, dist_spatial)

                # Create new tracks from unassigned detections. Assume that any unassigned detection is a start of a new track.
                # In practice you can use other cues to eliminate nnoisy detections such as size, location, or appearance


                track.create_new_tracks_KF(clust.clusters_frame)

                track.save_global_tracking_data(clust.clusters_frame,f,mtmc.global_tracks,cam)

                # DNo(k) = indexDN(squareform(vec), labels(:, k), 'euclidean');


                # plt.close('all')

                # ESCRITURA RESULTADOS

                if track.updated_flag:

                    num_tracks_f = mtmc.global_tracks[f].__len__()
                    for i in range(num_tracks_f):

                        for det in range(mtmc.global_tracks[f][i]['det'].__len__()):

                            new_w = round(mtmc.global_tracks[f][i]['det'][det]['w'] + mtmc.global_tracks[f][i]['det'][det]['w'] * 0.6)
                            new_h = round(mtmc.global_tracks[f][i]['det'][det]['h'] + mtmc.global_tracks[f][i]['det'][det]['h'] * 0.6)



                            arg1 = mtmc.global_tracks[f][i]['det'][det]['id_cam']
                            arg2 = mtmc.global_tracks[f][i]['id']
                            arg3 = f
                            arg4 = mtmc.global_tracks[f][i]['det'][det]['x'] + round(mtmc.global_tracks[f][i]['det'][det]['w'] / 2) - round(new_w / 2)
                            arg5 = mtmc.global_tracks[f][i]['det'][det]['y'] + round(mtmc.global_tracks[f][i]['det'][det]['h'] / 2) - round(new_h / 2)
                            arg6 = new_w
                            arg7 = new_h
                            '''
                            arg1 = mtmc.global_tracks[f][i]['det'][det]['id_cam']
                            arg2 = mtmc.global_tracks[f][i]['id']
                            arg3 = f
                            arg4 = mtmc.global_tracks[f][i]['det'][det]['x']
                            arg5 = mtmc.global_tracks[f][i]['det'][det]['y']
                            arg6 = mtmc.global_tracks[f][i]['det'][det]['w']
                            arg7 = mtmc.global_tracks[f][i]['det'][det]['h']
                            '''

                            f_id.write("%d %d %d %d %d %d %d -1 -1\n" % (arg1, arg2, arg3, arg4, arg5, arg6, arg7))
            f_id.close()

            a=1
            toc = time.time()
            print( toc - tic, 'sec Elapsed' )



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