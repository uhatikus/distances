#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####################
# Aleksandr Ukhatov #
#####################

# imports 
import os
import sys
import cv2
import ntpath
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import collections
import struct
import glob
import read_model
import multiprocessing as mp

from mpl_toolkits.mplot3d import Axes3D
from shutil import copyfile
from multiprocessing import Pool


# In[2]:


class ResultError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


# In[3]:


# classes for MASKs RCNN
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# In[33]:


# find matches between to given images
# inputs are image names
def match_imgs(img1, img2_name):
    
#     MIN_MATCH_COUNT = 70

    # read images 
#     img1 = cv2.imread(img1_name,0) # queryImage
    img2 = cv2.imread(img2_name,0) # trainImage

    # Initiate SIFT detector
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    # return number of good matches
    return len(good)


# In[34]:


# get target frames, with target object/car   
def get_target_frames(project_dir, img1_name, folder_to="target", folder_from="cropped"):
    # init constants and lists 
    MIN_MATCH_COUNT = 70
    delta = 6
    MIN_GOOD_IMAGES_IN_NEIGHBUORS = 2
    
    # make folder for target images
    os.system("mkdir " + project_dir + folder_to)
    # get name of target/query image 
    img1_name = ntpath.dirname(project_dir[:-1]) + "/" + img1_name
    # get names of images which might become target
    cropped_imgs =  sorted(glob.glob(project_dir + folder_from + "/*.jpg"))
    
    # use multiprocessing 
    pool = mp.Pool(mp.cpu_count())
    # find number of matches of all images with target/query image
    img1 = cv2.imread(img1_name,0) # queryImage
    goodness = [pool.apply(match_imgs, args=(img1, img)) for img in cropped_imgs]
    
    # find relevant images, with target on them
    good_images = list(map(lambda x: int(x > MIN_MATCH_COUNT), goodness))
    
#     print(good_images)
#     print(goodness)

    # for each image decide copy it or not
    images_n = len(good_images)
    if images_n == 0:
        raise ResultError("There is no target car in the input videos!")
    for i in range(images_n):
        if sum(good_images[max(0, i-(delta)):min(images_n, i+(delta+3))]) > MIN_GOOD_IMAGES_IN_NEIGHBUORS:
            # copy of true
            copyfile(cropped_imgs[i], project_dir + folder_to + "/" + ntpath.basename(cropped_imgs[i]))
            
    # copy target file to find cluster with turget 
    copyfile(img1_name, project_dir + folder_to + "/" + "000.jpg")



# In[6]:


# crop frames to have only usefull area in each image
def crop_frames(labeled_points, project_dir, folder_to="cropped", folder_from="images"):
    # init lists
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    
    # maken folder for cropped images 
    os.system("mkdir " + project_dir + folder_to)
    
    # for every image go through labeled points 
    # and find rectangular parameters of usefull are
    for key, labeled_points_on_img in labeled_points.items():
        
        # init local minimums and maximums for x and y
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0
        
        # print current image name
        print(key)
        
        # get mask of current image
        masks = labeled_points_on_img['masks']
        # combine masks to get one mask, which reflects 
        # the most important parts of current image
        mask = masks[:, :, 0]
        for j in range(masks.shape[2]-1):
            mask = np.logical_or(mask, masks[:, :, j+1])
        
        # get local minimums and maximums for x and y
        for k in range(masks.shape[0]):
            for l in range(masks.shape[1]):
                if (mask[k,l] != 0):
                    min_x = min(k, min_x)
                    max_x = max(k, max_x)
                    min_y = min(l, min_y)
                    max_y = max(l, max_y)
                    
        # save obtained results
        x1.append(min_x)
        y1.append(min_y)
        x2.append(max_x)
        y2.append(max_y)
                       
    # get initial image files
    image_files = sorted(glob.glob(project_dir + folder_from + "/*.jpg"))

    # obtain cropped image of each initial image                     
    for i, image_file in enumerate(image_files):
        # read image
        image = skimage.io.imread(image_file)
        # crop image
        cropped = image[x1[i]:x2[i], y1[i]:y2[i]]
        # save cropped image
        skimage.io.imsave(project_dir + folder_to + "/" + str(i+1).zfill(3) + ".jpg", cropped) 


# In[7]:


def video_to_frames(video_files, pr_dir=0):
    # init constants
    scale = 0.5
    ms = 200
    place = "data/projects/"

    # init project directory
    video_name = ((ntpath.basename(video_files[0])).split("."))[0]
    os.system("mkdir " + place +  video_name)
    os.system("mkdir "+ place + video_name + "/images")
    # save current project directory name
    project_dir = os.getcwd() + "/" + place + video_name + "/" 
    if pr_dir:
        return project_dir
    
    # get frames from every video
    count = 1
    for video_file in video_files:
        count_v = 1
        # read image
        vidcap = cv2.VideoCapture(video_file)
        # read first image
        success, image = vidcap.read()
        while success:
            print('Read a new frame #' + str(count))
            # resize image
            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            # save image
            cv2.imwrite(project_dir + "images/" + video_name + "_frame_" + str(count).zfill(3) + ".jpg", image)     # save frame as JPEG file      
            # read new image
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count_v*ms))
            success, image = vidcap.read()
            # increase counts
            count += 1  
            count_v += 1
            # limitations
            if count > 100 or count_v > 50:
                break 
    print("Obtained " + str(count-1) + " images")
    # return current project directory
    return project_dir


# In[8]:


# get model for MASK RCNN
def get_model(IMAGE_DIR):
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Import Mask RCNN
    sys.path.append(os.path.join(ROOT_DIR, "code/"))  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "code/coco/"))  # To find local version
    import coco
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/")
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "MASK_RCNN_models/mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        raise ResultError("Can not find COCO_MODELL")
    #     utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    # config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    
    return model


# In[9]:


# Object Detection  
# Label points on images 
def label_points(IMAGE_DIR, model):
 
    # init dictinary for labeled points
    labeled_points = {}
    
    # for each image apply MASK RCNN
    lst = sorted(glob.glob(IMAGE_DIR + "*.jpg"))
    for image_file in lst:
        print(image_file)
        # read image
        image = skimage.io.imread(image_file)

        # Run detection
        labeled_points[image_file] = (model.detect([image], verbose=1))[0]
        
        # visualization of MASKS
        
#         from mrcnn import visualize
#         results = labeled_points[image_file]
#         visualize.display_instances(image, results['rois'], results['masks'], results['class_ids'], class_names, results['scores'])
    return labeled_points


# In[10]:


# combine information
# from MASK RCNN (labeled_points) 
# and SfM/COLMAP (images, points3D)
def combine(project_dir, labeled_points, images, points3D):

        
    ###########################
    # Combining
    ###########################
    
    print("number of images: " + str(len(images)))
    print("number of points before removing useless points: " + str(len(points3D)))
    
    # init dictionary for points3D IDs and points3D classes
    points3D_on_images = {}
    points3D_on_images_classes = {}
    # init dictionaries for each point3D
    for point3D_id in points3D:
        points3D_on_images[point3D_id] = {}
        points3D_on_images_classes[point3D_id] = {}
    
    # process combining for each image
    for i in images:
        
        # get class ids and masks for current image
        labeled_point = labeled_points[project_dir + images[i].name]
        class_ids = labeled_point['class_ids']
        masks = labeled_point['masks']
        masks = masks.astype(int)   
        
        # combine all masks of current image 
        masks_n = masks.shape[2]
        mask = masks[:, :, 0]
        for j in range(masks_n):
            mask = mask + np.logical_xor(mask, masks[:, :, j]*((j+1)*masks[:, :, j]))
        
        # for points3D detected by SfM find corresponding label 
        for j, point3D_id in enumerate(images[i].point3D_ids):
            if (point3D_id == -1):
                continue 
            
            # get x and y of current point on current image 
            xy = images[i].xys[j]
            x = int(round(xy[0]))
            y = int(round(xy[1]))
            # get label/id
            object_id = mask[y-1][x-1]
            # save label/id if it is important
            if object_id != 0:
                points3D_on_images[point3D_id][i] = object_id
                points3D_on_images_classes[point3D_id][i] = class_ids[object_id - 1]
                
    ###########################
    # Filtering
    ###########################
    
    threshold = 5
                
    # remove irrelevant points3D     
    points3D_on_images_keys = list(points3D_on_images.keys())
    for point3D_id in points3D_on_images_keys:
        # remove points3D found less than 'threshold' times 
        if len(points3D_on_images[point3D_id]) < threshold:
            del points3D_on_images[point3D_id]
            del points3D_on_images_classes[point3D_id]
            continue
        
        # remove background points3D
        counts = np.bincount(list(points3D_on_images_classes[point3D_id].values()))
        object_class = np.argmax(counts)
        if object_class == 0:
            del points3D_on_images[point3D_id]
            del points3D_on_images_classes[point3D_id]
        else:
            # if not background point3D, assign object class to point3D
            points3D_on_images_classes[point3D_id] = object_class
    
    print("final number of points: " + str(len(points3D_on_images)))
    return points3D_on_images, points3D_on_images_classes


# In[47]:


# get clusters for points3В 
def get_clusters(points3D_on_images, images_n):
        
    ###########################
    # Getting ready
    ###########################
    
    points3D_i_to_point3D_id = []
    max_id = 0

    # get max_id among all points3D for all images
    for point3D_id in points3D_on_images:
        max_id_for_cur_point3D = max(list(points3D_on_images[point3D_id].values()))
        if max_id_for_cur_point3D > max_id:
            max_id = max_id_for_cur_point3D
    
    # get points3D relevant for clustering:
    # id to array with length max_id. Example: [0,0,0,0,1,0,0]
    # and concotanate arrays for each point
    points3D_on_images_to_cluster = np.zeros((len(points3D_on_images), images_n*max_id))
    for i, point3D_id in enumerate(points3D_on_images):
        points3D_i_to_point3D_id.append(point3D_id)
        for img_number in points3D_on_images[point3D_id]:
            points3D_on_images_to_cluster[i][(img_number - 1)* max_id + (points3D_on_images[point3D_id][img_number] - 1)] = 1
    
    ###########################
    # Clustering
    ###########################
    
    n_clusters = 5
    
    from sklearn.cluster import KMeans
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(points3D_on_images_to_cluster)
    
    cluster = {}
    for i in range(n_clusters):
        cluster[i] = []
    for i, cl in enumerate(clustering.labels_):
        cluster[cl].append(points3D_i_to_point3D_id[i])
        
    return cluster


# In[48]:


def filter_cluster(cluster, points3D):
    ###########################
    # Filtering 
    ###########################
    
    # find params (std, mean) of clusters for filtering
    cluster_points3D = []
    cluster_xyz = []
    cluster_stds_xyz = []
    cluster_means_xyz = []
    min_cluster_len = 30
    n_clusters = len(cluster)
    
    for i in range(n_clusters):
        cluster_points3D.append({})
        cluster_xyz.append([])
        for j in cluster[i]:
            cluster_points3D[i][j] = points3D[j]
            cluster_xyz[i].append(points3D[j].xyz)
        cluster_xyz[i] = np.vstack(cluster_xyz[i])
        cluster_stds_xyz.append(np.std(cluster_xyz[i], axis=0))
        cluster_means_xyz.append(np.mean(cluster_xyz[i], axis=0))
    all_xyz = []
    for i in points3D:
        all_xyz.append(points3D[i].xyz)
    all_xyz = np.vstack(all_xyz)
    all_std_xyz = np.std(all_xyz, axis=0)
    
    # remove clusters with high std
    for i in range(n_clusters):
        if sum(cluster_stds_xyz[i]/all_std_xyz) > 0.8 or len(cluster[i]) < min_cluster_len:
            del cluster[i]
            
    # if clusters have relatively small differencein mean, unite them
    for i in range(n_clusters):
        if i not in cluster:
            continue
        for j in range(n_clusters - i - 1):
            j = j + i + 1
            if j not in cluster:
                continue
            # distance between means
            if np.linalg.norm(cluster_means_xyz[i]-cluster_means_xyz[j]) < 1:
                cluster[i] = cluster[i]+cluster[j]
                del cluster[j]
    
    return cluster


# In[36]:


# get cluster with target 
# find cluster, which has the biggest number of points3D from target image
def get_target_cluster(images, cluster):
    # init some values
    target_cluster = 0
    max_num_of_points = 0
    set_a = 0
    
    # get set_a -- points3D from target image
    for i in images:
        if images[i].name == "000.jpg":
            set_a = set(images[i].point3D_ids)
    if not set_a:
        raise ResultError("Could not fint 000.jpg")
    # Check every cluster
    for i in cluster:
        set_b = set(cluster[i])
        if max_num_of_points < len(set_a & set_b):
            max_num_of_points = len(set_a & set_b)
            target_cluster = i
            
    return target_cluster


# In[13]:


# find distances between pair of clusters
def find_dist(points3D, cluster_1, cluster_2):
    # number of minimum distances to consider
    n_min_dists = 5
    min_dist = {}
    for i in range(n_min_dists):
        min_dist[i] = float("inf")
        
    # considering every pair of points3D from given clusters
    for id_1 in cluster_1:
        for id_2 in cluster_2:
            # find distances between two current points
            local_dist = np.linalg.norm(points3D[id_1].xyz-points3D[id_2].xyz)
            # update minimum distances
            for i in range(n_min_dists):
                if min_dist[i] > local_dist:
                    for j in range(n_min_dists - i - 1):
                        j = j + i
                        min_dist[j+1] = min_dist[j]
                    min_dist[i] = local_dist
                    break
    # calculate average distance among closest
    min_dist = [v for v in min_dist.values()]
    ans = sum(min_dist)/n_min_dists
    
    return ans

# find distances between all clusters
def get_distances_between_clusters(cluster):
    distances_between_clusters = {}
    for i in cluster:
        distances_between_clusters[i] = {}
    len_cl = len(cluster)
    # for every pait of cluster find distance
    for i in cluster:
        for j in cluster:
            if j > i:
                distances_between_clusters[i][j] = find_dist(points3D, cluster[i], cluster[j])
                distances_between_clusters[j][i] = distances_between_clusters[i][j]
                
    return distances_between_clusters


# In[25]:


# project directory and video to frames
ROOT_DIR = os.getcwd()
while ((list(filter(bool, ROOT_DIR.split("/"))))[-1] != "distances"):
    os.chdir('..')
    ROOT_DIR = os.getcwd()
video_files = []
# video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190816_134744.mp4")
# video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190816_135049.mp4")
video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190816_134426.mp4")
video_files.append("/Users/ukhatov/Documents/Projects/distances/data/videos/20190816_134532.mp4")

project_dir = video_to_frames(video_files)


# In[26]:


# get model
model = get_model(project_dir + "images/")
# label points for cropping
labeled_points_for_cropping = label_points(project_dir + "images/", model)
# crop
crop_frames(labeled_points_for_cropping, project_dir, "cropped", "images")
# get target frames
get_target_frames(project_dir, "target.jpg", "target")


# In[27]:


# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
# [32, 32, 31, 36, 51, 31, 55, 33, 34, 31, 23, 37, 34, 38, 42, 40, 28, 39, 28, 39, 33, 44, 32, 35, 24, 18, 25, 23, 35, 26, 23, 30, 20, 34, 30, 28, 32, 29, 25, 31, 30, 19, 32, 31, 26, 61, 48, 41, 59, 84, 158, 173, 75, 43, 25, 29, 42, 31, 46, 34, 38, 26, 31, 24, 34, 26, 34, 26, 23, 28, 47, 36, 40, 43, 22, 26, 30, 23, 37, 33, 22, 27, 27, 23, 27, 25, 32, 17, 36, 35, 27, 26, 34, 23, 33, 24, 23, 44, 23, 18, 28, 29, 63, 26, 34, 20, 21, 22, 33, 39, 67, 26, 46, 32, 47, 39, 29, 18, 22, 33, 36, 32, 31, 31, 32, 25, 35, 36, 24, 21, 42, 45, 25, 40, 23, 27, 34, 39, 34, 44, 43, 30, 51, 66, 97, 131, 282, 81, 77, 37]


# In[38]:


# Structure from Motion
os.system("colmap automatic_reconstructor --workspace_path " + project_dir + " --image_path " + project_dir + "target/")
cameras, images, points3D = read_model.read_model(path=project_dir+"sparse/0/", ext=".bin")
# label points
labeled_points = label_points(project_dir + "target/", model)


# In[49]:


cameras, images, points3D = read_model.read_model(path=project_dir+"sparse/1/", ext=".bin")
# combine
points3D_on_images, points3D_on_images_classes = combine(project_dir + "target/", labeled_points, images, points3D)
# get clusters
cluster = get_clusters(points3D_on_images, len(labeled_points))
filtered_cluster = filter_cluster(cluster, points3D)
# find distnaces 
distances_between_clusters = get_distances_between_clusters(filtered_cluster)
# get target cluster
target_cluster_num = get_target_cluster(images, filtered_cluster)


# In[55]:


print("project_dir = " + '"' + project_dir + '"')
# print(cluster)
print("clusters = " +str(filtered_cluster))
print(target_cluster_num)
print(distances_between_clusters)


# In[104]:


#printing
print(cluster)
print(target_cluster_num)
print(distances_between_clusters)


# In[52]:


# for i, a in enumerate(points3D_on_images):
# #     if not (a[0] == 29 and clusters[i]==0):
#     print(class_names[a[0]])
#     print(a)


# In[23]:


# def show(a):
#     print(class_names[a[0]])
#     print(a)
#     plt.plot(a[1:-1])
#     plt.ylim(0,20)
#     plt.show()
    
# for i, a in enumerate(points3D_on_images):
# #     if not (a[0] == 29 and clusters[i]==0):
#     show(a)


# In[26]:


# cluster = {0: [13, 18, 29, 38, 53, 57, 60, 62, 80, 89, 90, 124, 127, 131, 132, 194, 202, 210, 213, 265, 271, 273, 274, 278, 287, 356, 362, 366, 367, 369, 384, 398, 485, 497, 502, 508, 509, 571, 637, 644, 658, 666, 667, 669, 670, 676, 712, 748, 760, 807, 810, 862, 870, 923, 1051, 1063, 1123, 1164, 1183, 1188, 1219, 1221, 1222, 1371, 1430, 1455, 1485, 1773, 1787, 1897, 1990, 2058, 2062, 2074, 2181, 2319, 2339, 2658, 2702, 2705, 2724, 2964, 3037, 3092, 3138, 3587, 5505, 6880, 6929, 7524, 7803, 8108, 8112, 8139, 8160, 8329, 8466, 8598, 8626, 8642, 8649], 1: [119, 453, 821, 1666, 2262, 2459, 2497, 2669, 2982, 3021, 3029, 3057, 3197, 3241, 3269, 3271, 3287, 3288, 3302, 3306, 3308, 3316, 3320, 3322, 3346, 3347, 3353, 3375, 3436, 3439, 3462, 3468, 3472, 3477, 3585, 3636, 3644, 3699, 3713, 3757, 3767, 3909, 3921, 3990, 4012, 4079, 4139, 4175, 4178, 4179, 4226, 4233, 4240, 4261, 4283, 4288, 4294, 4385, 4445, 4459, 4472, 4495, 4511, 4540, 4559, 4579, 4609, 4626, 4657, 4688, 4837, 4839, 4855, 4940, 5158, 5175, 5176, 5189, 5222, 5411, 5416, 5512, 5547, 5747, 5798, 5799, 5801, 5804, 5806, 6014, 6052, 6055, 6060, 6064, 6065, 6068, 6069, 6078, 6080, 6081, 6082, 6095, 6099, 6247, 6282, 6464, 6497, 6694, 6714, 6744, 6757, 6761, 6763, 6764, 6765, 6767, 6768, 6770, 6772, 6775, 6783, 6785, 6799, 6802, 6804, 6810, 6879, 6882, 6885, 6892, 6895, 6903, 6905, 6908, 6910, 6917, 6930, 6931, 6933, 6935, 6937, 6942, 6947, 6948, 6949, 6959, 6960, 6961, 6965, 6967, 6970, 6973, 6975, 7022, 7034, 7112, 7114, 7116, 7311, 7660, 7682, 7697, 7701, 7704, 7705, 8325, 8624, 8640, 8852, 9028], 2: [1843, 3206, 3272, 3299, 3309, 3311, 3314, 3334, 3336, 3366, 3391, 3392, 3420, 3422, 3426, 3517, 3591, 3597, 3695, 3754, 3761, 3797, 3815, 3820, 3827, 3866, 3870, 3889, 3910, 3912, 3929, 3964, 4056, 4121, 4124, 4137, 4140, 4247, 4252, 4253, 4276, 4278, 4310, 4382, 4437, 4444, 4480, 4494, 4526, 4530, 4539, 4558, 4608, 4625, 4763, 4786, 4804, 4925, 5171, 5351, 5477, 5504, 5518, 5650, 5742, 5805, 6088, 6097, 6100, 6286, 6321, 6333, 6353, 6391, 6431, 6444, 6457, 6496, 6505, 6646, 6746, 6752, 6793, 6812, 6904, 6919, 6921, 6926, 6927, 6934, 6938, 6940, 6941, 6945, 6946, 6950, 6951, 6952, 6954, 6957, 6958, 6974, 7005, 7014, 7027, 7061, 7111, 7113, 7115, 7310, 7312, 7352, 7391, 7494, 7502, 7548, 7574, 7653, 7659, 7664, 7667, 7669, 7672, 7698, 7702, 7703, 7928, 7981, 8140, 8328, 8341, 8485, 8582, 8641, 8643, 8644, 8645, 8647, 8853, 8854, 8857, 8869, 9032, 9033, 9034, 9045, 9046], 3: [1, 5, 8, 14, 20, 71, 87, 91, 125, 135, 141, 154, 203, 207, 245, 248, 329, 332, 345, 349, 352, 353, 355, 370, 378, 380, 394, 395, 459, 468, 487, 493, 500, 531, 578, 613, 650, 653, 699, 711, 857, 868, 912, 919, 922, 934, 1028, 1057, 1131, 1163, 1240, 1408, 1434, 1483, 1490, 1802, 1803, 1855, 1869, 1895, 1907, 1918, 1931, 1935, 1979, 1982, 1983, 1984, 2027, 2031, 2040, 2054, 2069, 2071, 2086, 2093, 2121, 2132, 2151, 2153, 2178, 2180, 2188, 2233, 2254, 2275, 2287, 2324, 2326, 2350, 2479, 2496, 2501, 2503, 2513, 2515, 2517, 2522, 2523, 2525, 2528, 2530, 2532, 2534, 2545, 2550, 2563, 2568, 2575, 2581, 2626, 2630, 2633, 2639, 2667, 2729, 2740, 2752, 2797, 2925, 2932, 2985, 3010, 3034, 3172, 3187, 3255, 3291, 3315, 3454, 3469, 3578, 3655, 3829, 3961, 4026, 4052, 4102, 4234, 4237, 4295, 4296, 4318, 4341, 4343, 4357, 4364, 4384, 4393, 4394, 4422, 4424, 4465, 4578, 4596, 4643, 4665, 4781, 4825, 5034, 5272, 5414, 5417, 5418, 5424, 5429, 5435, 5454, 5462, 5519, 5527, 5533, 5540, 5559, 5606, 5617, 5620, 5632, 5655, 5663, 5665, 5671, 5707, 5749, 5754, 5758, 5767, 5769, 5781, 5993, 5997, 6002, 6011, 6013, 6034, 6054, 6057, 6063, 6070, 6071, 6072, 6073, 6074, 6076, 6086, 6090, 6091, 6096, 6098, 6101, 6105, 6110, 6151, 6201, 6248, 6314, 6408, 6462, 6500, 6513, 6587, 6695, 6711, 6758, 6769, 6778, 6784, 6788, 6801, 6806, 6809, 6883, 6894, 6907, 6912, 6913, 6918, 6964, 6966, 6971, 6972, 6976, 6991, 7004, 7016, 7087, 7098, 7137, 7183, 7284, 7301, 7308, 7309, 7359, 7377, 7399, 7410, 7415, 7432, 7463, 7466, 7470, 7478, 7488, 7507, 7514, 7589, 7628, 7663, 7665, 7666, 7668, 7684, 7696, 7700, 7722, 7745, 7802, 7828, 7866, 7875, 7922, 7923, 7925, 7930, 7950, 7959, 7968, 7982, 8014, 8136, 8138, 8159, 8161, 8169, 8170, 8171, 8179, 8228, 8230, 8326, 8327, 8360, 8361, 8405, 8465, 8488, 8529, 8612, 8623, 8625, 8650, 8660, 8668, 8680, 8699, 8720, 8778, 9029, 9030, 9031]}


# In[27]:


# cluster = {0: [8, 10, 238, 1961, 1983, 2562, 2595, 3625, 3668, 4119, 4131, 4135, 4136, 4137, 4166, 4183, 4184, 4185, 4305, 4356, 4378, 4488, 4517, 4520, 4524, 4538, 4558, 4601, 4626, 4658, 4743, 4748, 4758, 4760, 4765, 4772, 4838, 4873, 4876, 4942, 4984, 5000, 5001, 5002, 5013, 5014, 5015, 5037, 5038, 5040, 5043, 5044, 5045, 5058, 5059, 5061, 5086, 5114, 5123, 5131, 5132, 5189, 5331, 5358, 5366, 5368, 5372, 5373, 5376, 5377, 5378, 5383, 5384, 5386, 5387, 5400, 5406, 5407, 5411, 5412, 5414, 5427, 5430, 5433, 5435, 5447, 5460, 5463, 5465, 5468, 5470, 5476, 5517, 5528, 5580, 5673, 5678, 5702, 5714, 5715, 5716, 5717, 5720, 5729, 5734, 5744, 5745, 5746, 5757, 5758, 5764, 5773, 5774, 5775, 5777, 5801, 5803, 5806, 5808, 5844, 5848, 5916, 5918, 5926, 5929, 5970, 6022, 6035, 6036, 6037, 6038, 6050, 6051, 6052, 6092, 6097, 6112, 6135, 6137, 6148, 6154, 6156, 6202, 6215, 6278, 6283, 6289, 6295, 6305, 6388, 6407, 6505, 6518, 6536, 6547, 6565, 6567, 6585, 6628, 6634, 6704, 6710, 6777, 6826, 6854, 6859, 6864, 6865, 6883, 6892, 6942, 6953, 6956, 6997, 6998, 7107, 7115, 7157, 7275, 7277, 7283, 7305, 7306, 7345, 7461, 7475, 7476, 7508, 7511, 7512, 7513, 7547, 7712, 7764, 7766, 7891, 8042, 8045, 8062, 8072, 8073, 8079, 8082, 8083, 8106, 8192, 8193, 8195, 8200, 8410, 8411, 8492, 8627, 8628, 8888, 8889, 8956, 9363, 9365, 9616, 9618, 9799, 9800, 9802, 9992, 10216, 10633, 10636, 10639, 10650, 10845], 1: [40, 88, 90, 118, 126, 129, 175, 217, 224, 252, 290, 301, 375, 447, 593, 618, 852, 855, 944, 973, 1004, 1006, 1211, 1276, 1320, 1527, 1640, 1726, 1753, 1845, 1919, 2099, 2175, 2318, 2794, 2798, 3095, 3313, 3770, 3775, 4078, 4749, 4752, 5359, 6018, 6020, 6024, 6025, 6026, 6027, 6031, 6041, 6042, 6046, 6047, 6056, 6068, 6078, 6080, 6088, 6537, 11066, 11082, 11688, 11689, 11895, 12127, 12328, 12725, 13709, 13788, 14054], 2: [14, 27, 60, 86, 117, 123, 127, 131, 166, 167, 170, 174, 213, 219, 246, 248, 287, 288, 291, 292, 334, 352, 356, 379, 382, 420, 433, 449, 495, 509, 511, 518, 522, 535, 576, 578, 594, 603, 604, 668, 703, 705, 733, 832, 862, 889, 912, 919, 950, 974, 1034, 1143, 1209, 1282, 1289, 1326, 1381, 1459, 1464, 1554, 1578, 1655, 1727, 1754, 1755, 1795, 1913, 1963, 1964, 2020, 2183, 2266, 2440, 2459, 2515, 2533, 2569, 2574, 2577, 2617, 2648, 2677, 2816, 2854, 2869, 2883, 2894, 2940, 2971, 3003, 3031, 3084, 3101, 3134, 3149, 3219, 3254, 3268, 3273, 3279, 3297, 3300, 3315, 3610, 3664, 3709, 3740, 3827, 3864, 3866, 3880, 3881, 3902, 3903, 3960, 3963, 3964, 3991, 3993, 4002, 4031, 4055, 4061, 4096, 4107, 4194, 4208, 4226, 4288, 4289, 4294, 4333, 4362, 4405, 4412, 4432, 4497, 4518, 4522, 4525, 4539, 4583, 4622, 4627, 4630, 4671, 4686, 4696, 4707, 4714, 4737, 4742, 4923, 4940, 5005, 5311, 5327, 5333, 5364, 5379, 5396, 5420, 5494, 5498, 5529, 5672, 5684, 5687, 5693, 5814, 5833, 5865, 5965, 6023, 6034, 6040, 6054, 6070, 6072, 6074, 6102, 6103, 6106, 6107, 6109, 6122, 6123, 6141, 6142, 6143, 6149, 6161, 6167, 6200, 6259, 6319, 6324, 6391, 6405, 6455, 6562, 6570, 6571, 6575, 6577, 6578, 6579, 6580, 6581, 6582, 6584, 6588, 6591, 6593, 6595, 6601, 6602, 6609, 6612, 6616, 6621, 6623, 6637, 6638, 6642, 6649, 6666, 6667, 6672, 6679, 6681, 6683, 6684, 6689, 6690, 6707, 6718, 6737, 6812, 6820, 6861, 6862, 6876, 6886, 6902, 6922, 6923, 6937, 6945, 6946, 6977, 6980, 7000, 7001, 7002, 7099, 7112, 7143, 7258, 7274, 7302, 7326, 7330, 7337, 7360, 7376, 7416, 7470, 7510, 7527, 7533, 7540, 7545, 7546, 7548, 7549, 7550, 7567, 7575, 7578, 7584, 7586, 7589, 7590, 7591, 7595, 7601, 7613, 7615, 7627, 7654, 7658, 7661, 7665, 7669, 7677, 7739, 7744, 7746, 7757, 7768, 7789, 7800, 7808, 7809, 7811, 7812, 7814, 7835, 7836, 7843, 7847, 7855, 7872, 7878, 7883, 7885, 7890, 7912, 7918, 7964, 7975, 7978, 8016, 8037, 8046, 8050, 8068, 8070, 8095, 8124, 8161, 8168, 8170, 8181, 8194, 8239, 8263, 8287, 8299, 8327, 8332, 8340, 8349, 8396, 8445, 8455, 8491, 8515, 8523, 8590, 8591, 8599, 8601, 8619, 8629, 8646, 8648, 8649, 8650, 8651, 8652, 8653, 8654, 8661, 8665, 8670, 8671, 8679, 8680, 8687, 8688, 8689, 8694, 8699, 8702, 8705, 8707, 8722, 8727, 8773, 8783, 8825, 8837, 8845, 8847, 8887, 8894, 8919, 8923, 8929, 8935, 8937, 8938, 8949, 8950, 8953, 8961, 8963, 8967, 8968, 8978, 8981, 8985, 8991, 9003, 9008, 9011, 9015, 9041, 9048, 9050, 9059, 9066, 9068, 9083, 9102, 9203, 9236, 9257, 9269, 9287, 9344, 9361, 9368, 9385, 9387, 9388, 9430, 9443, 9467, 9503, 9508, 9632, 9641, 9650, 9652, 9666, 9704, 9717, 9772, 9776, 9801, 9803, 9807, 9808, 9827, 9830, 9834, 9836, 9839, 9840, 9844, 9873, 9969, 9973, 9977, 9986, 9988, 10024, 10075, 10114, 10117, 10187, 10239, 10249, 10269, 10292, 10335, 10359, 10439, 10442, 10452, 10461, 10547, 10632, 10642, 10647, 10663, 10846, 10870, 10872, 10886, 10900, 10910, 10919, 10929, 10952, 10953, 10954, 10955, 10956, 10957, 10976, 10981, 10986, 11024, 11045, 11052, 11053, 11072, 11081, 11092, 11103, 11114, 11168, 11218, 11240, 11263, 11278, 11313, 11341, 11351, 11357, 11359, 11363, 11416, 11417, 11426, 11428, 11430, 11445, 11451, 11475, 11488, 11549, 11552, 11555, 11598, 11647, 11662, 11669, 11674, 11686, 11690, 11721, 11778, 11781, 11785, 11815, 11847, 11868, 11871, 11876, 11894, 11896, 11898, 11910, 12008, 12012, 12102, 12109, 12159, 12167, 12192, 12225, 12249, 12259, 12268, 12298, 12302, 12327, 12344, 12349, 12442, 12444, 12495, 12497, 12505, 12508, 12513, 12519, 12525, 12530, 12535, 12561, 12566, 12583, 12594, 12595, 12618, 12642, 12652, 12715, 12716, 12726, 12741, 12752, 12756, 12788, 12819, 12881, 12887, 12907, 13062, 13072, 13073, 13078, 13101, 13289, 13312, 13419, 13420, 13422, 13423, 13424, 13552, 13690, 13701, 13711, 13723, 13725, 13789, 13792, 13793, 13794, 13869, 13891, 13920, 13931, 13938, 13954, 13989, 13990, 13992, 14013, 14022, 14033, 14043, 14049, 14091, 14094, 14139, 14286], 3: [25, 26, 37, 42, 49, 51, 52, 63, 65, 77, 83, 89, 94, 105, 111, 150, 152, 153, 155, 157, 203, 311, 339, 367, 369, 370, 371, 372, 388, 396, 399, 448, 497, 499, 653, 893, 986, 1197, 1252, 1488, 1511, 1518, 1716, 1717, 1970, 2002, 2151, 2244, 2539, 2739, 2886, 2887, 3296, 3301, 3593, 3677, 4203, 4307, 4309, 4323, 4334, 4367, 4388, 4519, 4531, 4533, 4551, 4554, 4559, 4565, 4636, 4710, 4719, 4745, 4782, 4812, 4851, 4877, 4881, 4945, 4965, 4988, 5048, 5053, 5065, 5070, 5334, 5393, 5399, 5680, 5681, 5682, 5692, 5761, 5787, 5819, 5863, 5872, 5973, 5975, 5976, 5977, 5979, 6014, 6032, 6048, 6062, 6064, 6076, 6084, 6090, 6153, 6221, 6237, 6329, 6516, 6545, 6713, 6834, 6857, 6870, 6930, 7116, 7211, 7278, 7767, 7824, 8044, 8047, 8199, 8214, 8247, 8249, 8409, 8414, 8479, 8885, 8886, 8893, 9018, 9155, 9168, 10011, 10871, 11911], 4: [30, 36, 48, 67, 80, 81, 108, 112, 113, 120, 144, 145, 154, 164, 177, 193, 199, 200, 206, 222, 247, 250, 254, 263, 265, 266, 267, 271, 272, 313, 323, 374, 395, 397, 402, 435, 436, 485, 557, 560, 563, 566, 567, 568, 632, 646, 649, 650, 652, 657, 663, 664, 669, 674, 675, 676, 677, 678, 679, 696, 697, 698, 720, 721, 726, 727, 759, 761, 762, 763, 766, 767, 775, 776, 782, 785, 788, 796, 833, 836, 842, 843, 845, 894, 937, 961, 968, 985, 988, 1022, 1029, 1051, 1055, 1057, 1065, 1068, 1069, 1070, 1073, 1091, 1104, 1109, 1125, 1130, 1133, 1140, 1146, 1185, 1236, 1238, 1249, 1255, 1258, 1264, 1265, 1266, 1277, 1300, 1302, 1303, 1306, 1314, 1317, 1362, 1374, 1391, 1423, 1438, 1449, 1516, 1525, 1537, 1566, 1575, 1610, 1620, 1623, 1627, 1633, 1659, 1662, 1743, 1762, 1791, 1830, 1843, 1879, 1994, 2012, 2024, 2087, 2141, 2170, 2190, 2191, 2243, 2246, 2247, 2249, 2277, 2306, 2320, 2328, 2409, 2437, 2468, 2502, 2514, 2540, 2545, 2698, 2741, 2790, 2791, 2793, 2795, 3002, 3075, 3116, 3131, 3160, 3200, 3298, 3302, 3310, 3314, 3328, 3531, 3636, 3671, 3772, 3773, 3809, 3812, 3816, 3817, 3819, 3820, 3823, 3824, 4077, 4095, 4167, 4329, 4330, 4331, 4335, 4487, 4490, 4709, 4711, 4744, 4944, 5335, 5365, 5375, 5685, 5689, 5691, 5712, 5974, 5978, 6008, 6013, 6016, 6019, 6039, 6057, 6077, 6082, 6085, 6089, 6091, 6344, 6860, 6863, 6866, 6868, 6873, 7526, 7528, 8241, 8246, 8791, 9154, 9204, 11068, 11945, 11948, 12130, 14287]}


# In[28]:


# cluster = {0: [1, 5, 8, 13, 14, 18, 20, 29, 38, 53, 57, 60, 62, 71, 80, 87, 89, 90, 91, 124, 125, 127, 131, 132, 135, 141, 154, 194, 202, 203, 207, 210, 213, 245, 248, 265, 271, 273, 274, 278, 287, 329, 332, 345, 349, 352, 353, 355, 356, 362, 366, 367, 369, 370, 378, 380, 384, 394, 395, 398, 459, 468, 485, 487, 493, 497, 500, 502, 508, 509, 531, 571, 578, 613, 637, 644, 650, 653, 658, 666, 667, 669, 670, 676, 699, 711, 712, 748, 760, 807, 810, 857, 862, 868, 870, 912, 919, 922, 923, 934, 1028, 1051, 1057, 1063, 1123, 1131, 1163, 1164, 1183, 1188, 1219, 1221, 1222, 1371, 1408, 1430, 1434, 1455, 1483, 1485, 1490, 1773, 1787, 1802, 1803, 1855, 1869, 1895, 1897, 1907, 1918, 1931, 1935, 1979, 1982, 1983, 1984, 1990, 2027, 2054, 2058, 2062, 2069, 2071, 2074, 2086, 2093, 2121, 2132, 2151, 2153, 2178, 2180, 2181, 2188, 2233, 2275, 2287, 2319, 2324, 2326, 2339, 2350, 2479, 2496, 2501, 2503, 2513, 2515, 2517, 2522, 2523, 2525, 2528, 2530, 2534, 2545, 2550, 2568, 2575, 2581, 2630, 2633, 2639, 2658, 2667, 2702, 2705, 2724, 2729, 2740, 2752, 2797, 2925, 2932, 2964, 2985, 3034, 3037, 3092, 3138, 3172, 3187, 3255, 3291, 3315, 3454, 3469, 3578, 3587, 3655, 3829, 3961, 4026, 4052, 4234, 4237, 4295, 4296, 4318, 4341, 4343, 4364, 4384, 4393, 4394, 4422, 4424, 4465, 4578, 4596, 4643, 4781, 4825, 5034, 5272, 5414, 5417, 5418, 5424, 5429, 5435, 5454, 5505, 5519, 5533, 5540, 5559, 5606, 5617, 5620, 5632, 5655, 5663, 5665, 5671, 5707, 5749, 5754, 5758, 5767, 5769, 5781, 5993, 5997, 6002, 6011, 6013, 6034, 6054, 6070, 6072, 6073, 6074, 6076, 6086, 6096, 6098, 6101, 6105, 6110, 6151, 6201, 6248, 6314, 6408, 6462, 6513, 6587, 6695, 6758, 6769, 6778, 6788, 6801, 6806, 6809, 6880, 6883, 6907, 6912, 6913, 6918, 6929, 6966, 6971, 6972, 6991, 7004, 7016, 7087, 7098, 7137, 7183, 7284, 7301, 7308, 7309, 7359, 7377, 7399, 7410, 7415, 7432, 7463, 7466, 7470, 7478, 7488, 7507, 7514, 7524, 7589, 7628, 7663, 7665, 7666, 7668, 7684, 7696, 7700, 7722, 7745, 7802, 7803, 7828, 7866, 7875, 7922, 7923, 7925, 7930, 7950, 7959, 7968, 7982, 8014, 8108, 8112, 8136, 8138, 8139, 8159, 8160, 8161, 8169, 8170, 8171, 8179, 8228, 8230, 8326, 8327, 8329, 8360, 8361, 8405, 8465, 8466, 8488, 8529, 8598, 8612, 8623, 8625, 8626, 8642, 8649, 8650, 8660, 8668, 8699, 8720, 8778, 9029, 9030, 9031], 1: [119, 453, 821, 1666, 2031, 2040, 2254, 2262, 2459, 2497, 2532, 2563, 2626, 2669, 2982, 3010, 3021, 3029, 3057, 3197, 3241, 3269, 3271, 3287, 3288, 3302, 3306, 3308, 3316, 3320, 3322, 3346, 3347, 3353, 3375, 3436, 3439, 3462, 3468, 3472, 3477, 3585, 3636, 3644, 3699, 3713, 3757, 3767, 3909, 3921, 3990, 4012, 4079, 4102, 4139, 4175, 4178, 4179, 4226, 4233, 4240, 4261, 4283, 4288, 4294, 4357, 4385, 4445, 4459, 4472, 4495, 4511, 4540, 4559, 4579, 4609, 4626, 4657, 4665, 4688, 4837, 4839, 4855, 4940, 5158, 5175, 5176, 5189, 5222, 5411, 5416, 5462, 5512, 5527, 5547, 5747, 5798, 5799, 5801, 5804, 5806, 6014, 6052, 6055, 6057, 6060, 6063, 6064, 6065, 6068, 6069, 6071, 6078, 6080, 6081, 6082, 6090, 6091, 6095, 6099, 6247, 6282, 6464, 6497, 6500, 6694, 6714, 6744, 6757, 6761, 6763, 6764, 6765, 6767, 6768, 6770, 6772, 6775, 6783, 6784, 6785, 6799, 6802, 6804, 6810, 6879, 6882, 6885, 6892, 6894, 6895, 6903, 6905, 6908, 6910, 6917, 6930, 6931, 6933, 6935, 6937, 6942, 6947, 6948, 6949, 6959, 6960, 6961, 6964, 6965, 6967, 6970, 6973, 6975, 6976, 7022, 7034, 7112, 7114, 7116, 7311, 7660, 7682, 7697, 7701, 7704, 7705, 8325, 8624, 8640, 8680, 8852, 9028], 2: [1240, 1843, 3206, 3272, 3299, 3309, 3311, 3314, 3334, 3336, 3366, 3391, 3392, 3420, 3422, 3426, 3517, 3591, 3597, 3695, 3754, 3761, 3797, 3815, 3820, 3827, 3866, 3870, 3889, 3910, 3912, 3929, 3964, 4056, 4121, 4124, 4137, 4140, 4247, 4252, 4253, 4276, 4278, 4310, 4382, 4437, 4444, 4480, 4494, 4526, 4530, 4539, 4558, 4608, 4625, 4763, 4786, 4804, 4925, 5171, 5351, 5477, 5504, 5518, 5650, 5742, 5805, 6088, 6097, 6100, 6286, 6321, 6333, 6353, 6391, 6431, 6444, 6457, 6496, 6505, 6646, 6711, 6746, 6752, 6793, 6812, 6904, 6919, 6921, 6926, 6927, 6934, 6938, 6940, 6941, 6945, 6946, 6950, 6951, 6952, 6954, 6957, 6958, 6974, 7005, 7014, 7027, 7061, 7111, 7113, 7115, 7310, 7312, 7352, 7391, 7494, 7502, 7548, 7574, 7653, 7659, 7664, 7667, 7669, 7672, 7698, 7702, 7703, 7928, 7981, 8140, 8328, 8341, 8485, 8582, 8641, 8643, 8644, 8645, 8647, 8853, 8854, 8857, 8869, 9032, 9033, 9034, 9045, 9046]}


# In[40]:





# In[52]:


distances_between_clusters


# In[144]:


distances_between_clusters


# In[142]:


distances_between_clusters


# In[140]:


distances_between_clusters


# In[151]:


0.10069333174420231/0.11198481350500883


# In[152]:


0.10765931682453683/0.11649991839028856


# In[153]:


0.11272115225288319/0.11800495335204847


# In[156]:


0.11677062059556027/0.11920898132145638


# In[32]:


print(images)


# In[57]:


print(cluster)


# In[ ]:




