import os
import sys
import cv2
import ntpath
import random
import math
import numpy as np
import skimage.io
import collections
import struct
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
from shutil import copyfile
import multiprocessing as mp
import time


global global_model

def match_imgs(img1_name, img2_name):
    MIN_MATCH_COUNT = 70

    img1 = cv2.imread(img1_name,0) # queryImage
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

    # print(len(good))
    # print(MIN_MATCH_COUNT)

    # if len(good)>MIN_MATCH_COUNT:
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()

    #     h,w = img1.shape
    #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,M)

    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # else:
    #     print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

    #     matchesMask = None

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)

    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    # plt.imshow(img3, 'gray'),plt.show()
    return len(good)



def copy_target_image(project_dir, folder_to, img):
    name = (img.split("/"))[-1]
    num = int(name[-7:-4])
    end = name[-4:]
    copyfile(img, project_dir + folder_to + "/" + name)
    
def get_target_frames(project_dir, img1_name, folder_to="target", folder_from="cropped"):
    MIN_MATCH_COUNT = 70
    delta = 6
    img1_name = "/" + "/".join(list(filter(bool, project_dir.split("/")))[:(-1)]) + "/" + img1_name
    os.system("mkdir " + project_dir + folder_to)
    cropped_imgs =  sorted(glob.glob(project_dir + folder_from + "/*.jpg"))
    images_n = len(cropped_imgs)
    good_images = []
    goodness = []
    for i in range(images_n):
        # should implement multiprocessing
        print(".", end = '')
        goodness.append(match_imgs(img1_name, cropped_imgs[i]))
        if goodness[-1] > MIN_MATCH_COUNT:
            good_images.append(1)
        else:
            good_images.append(0)
#     print(good_images)
#     print(goodness)
    for i in range(images_n):
        if sum(good_images[max(0, i-(delta)):min(images_n, i+(delta+3))]) > 2:
            copy_target_image(project_dir, folder_to, cropped_imgs[i])
    copyfile(img1_name, project_dir + folder_to + "/" + "000.jpg")


def crop_frames(labeled_points, project_dir, folder_to="cropped", folder_from="images"):
    images_n = len(labeled_points)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(images_n):
        masks = labeled_points[i]['masks']
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0
        masks_n = len(masks[0,0,:])
        masks_x = len(masks[:,0,0])
        masks_y = len(masks[0,:,0])
        mask = masks[:, :, 0]
        for j in range(masks_n-1):
            mask = np.logical_or(mask, masks[:, :, j+1])
        for k in range(masks_x):
            for l in range(masks_y):
                if (mask[k,l] != 0):
                    if (min_x > k):
                        min_x = k
                    if (max_x < k):
                        max_x = k
                    if (min_y > l):
                        min_y = l
                    if (max_y < l):
                        max_y = l
        x1.append(min_x)
        y1.append(min_y)
        x2.append(max_x)
        y2.append(max_y)
    lst = sorted(glob.glob(project_dir + folder_from + "/*.jpg"))
    count = 1
    os.system("mkdir " + project_dir + folder_to)
    for image_file in lst:
        image = skimage.io.imread(image_file)
        cropped = image[x1[count-1]:x2[count-1], y1[count-1]:y2[count-1]]
        skimage.io.imsave(project_dir + folder_to + "/" + str(count).zfill(3) + ".jpg", cropped) 
        count += 1


def resize(image, scale): 
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def video_to_frames(video_files, pr_dir=0):
    project_dir = ""
    scale = 0.5
    ms = 200
    video_name = ((ntpath.basename(video_files[0])).split("."))[0]
    os.system("mkdir data/projects/" + video_name)
    os.system("mkdir data/projects/" + video_name + "/images")
    count = 1
    for video_file in video_files:
        count_v = 1
        vidcap = cv2.VideoCapture(video_file)
        success,image = vidcap.read()
        if project_dir == "":
            project_dir = os.getcwd() + "/data/projects/" + video_name + "/"    
        if pr_dir:
            return project_dir
        while success:
            print('Read a new frame #' + str(count) + ': ', success)
            image = resize(image, scale)
            cv2.imwrite(project_dir + "images/" + video_name + "_frame_" + str(count).zfill(3) + ".jpg", image)     # save frame as JPEG file      
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count_v*ms))
            success,image = vidcap.read()
            count += 1  
            count_v += 1
            if count > 100:
                break 
    print("Obtained " + str(count-1) + " images")
    return project_dir


def plot_points3D(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0;
    X = [];
    Y = [];
    Z = [];
    n_points = len(points3D)
    #n_points = 100
    RGB = np.zeros(shape=(n_points,3));

    for point_id in points3D:
        if i >= n_points:
            break
        point = points3D[point_id]
        xyz = point.xyz
        rgb = point.rgb  

        X.append(xyz[0])
        Y.append(xyz[1])
        Z.append(xyz[2])
        RGB[i, :] = rgb/256
        i = i+1

    Xmean = np.mean(X)
    Xstd = np.std(X)
    Ymean = np.mean(Y)
    Ystd = np.std(Y)
    Zmean = np.mean(Z)
    Zstd = np.std(Z)

    ax.scatter(X, Y, Z, facecolors = RGB, s=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    t_n = 2.5
    ax.set_xlim([Xmean - t_n*Xstd, Xmean + t_n*Xstd])
    ax.set_ylim([Ymean - t_n*Ystd, Ymean + t_n*Ystd])
    ax.set_zlim([Zmean - t_n*Zstd, Zmean + t_n*Zstd])

    plt.show()
    return


def plot_points3D_after(points3D, mark_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t_n = 2.5
    i = 0;
    X = [];
    Y = [];
    Z = [];
    n_points = len(points3D)
    n_points = int(n_points/2)
    RGB = np.zeros(shape=(n_points,3));

    for point_id in points3D:
        if i >= n_points:
            break
        point = points3D[point_id]
        xyz = point.xyz
        rgb = point.rgb  

        X.append(xyz[0])
        Y.append(xyz[1])
        Z.append(xyz[2])
        RGB[i, :] = rgb/256
        if point_id in mark_list:
            RGB[i, :] = [1, 0, 0]
        else:
             RGB[i, :] *= 0
        #     RGB[i, :] += [1, 1, 1]
        #     RGB[i, :] *= -0.4
        #     RGB[i, :] += [1, 1, 1]
        i = i+1

    Xmean = np.mean(X)
    Xstd = np.std(X)
    Ymean = np.mean(Y)
    Ystd = np.std(Y)
    Zmean = np.mean(Z)
    Zstd = np.std(Z)

    # i = 0;
    # X = [];
    # Y = [];
    # Z = [];
    # for point_id in points3D:
    #     if i >= n_points:
    #         break
    #     point = points3D[point_id]
    #     xyz = point.xyz
    #     rgb = point.rgb  
    #     if xyz[0] > Xmean - t_n*Xstd and xyz[1] > Ymean - t_n*Ystd and xyz[2] > Zmean - t_n*Zstd and xyz[0] < Xmean + t_n*Xstd and xyz[1] < Ymean + t_n*Ystd and xyz[2] < Zmean + t_n*Zstd: 
    #         X.append(xyz[0])
    #         Y.append(xyz[1])
    #         Z.append(xyz[2])
    #         RGB[i, :] = rgb/256
    #         i = i+1 

    # print(i)
    # RGB = RGB[0:i, :] 

    ax.scatter(X, Y, Z, facecolors = RGB, s=1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    ax.set_xlim([Xmean - t_n*Xstd, Xmean + t_n*Xstd])
    ax.set_ylim([Ymean - t_n*Ystd, Ymean + t_n*Ystd])
    ax.set_zlim([Zmean - t_n*Zstd, Zmean + t_n*Zstd])

    plt.show()
    return

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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


def get_model(IMAGE_DIR, n_imgs):
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
        raise Exception("Can not find COCO_MODELL")
    #     utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = n_imgs//GPU_COUNT + int(bool(n_imgs % GPU_COUNT))

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

def label_points_on_image(image_file, model=None):
    if model == None:
        global global_model
        model = global_model

    start = time.time()
    print("Processing image # " + str(image_file))
    image = skimage.io.imread(image_file)

    # Run detection
    results = (model.detect([image], verbose=1))[0]
    end = time.time()
    print("time for label_points_on_image: " + str(end - start) + " seconds")
    return results

def label_points(IMAGE_DIR, model):
    # Object Detection    
    lst = sorted(glob.glob(IMAGE_DIR + "*.jpg"))
    images = []
    for image_file in lst:
        images.append(skimage.io.imread(image_file))
    labeled_points = model.detect(images, verbose=1)
    return labeled_points





def combine(labeled_points, cameras, images, points3D):
    images_n = len(images)
    print("number of images: " + str(images_n))
    points3D_n = max(points3D)
    points3D_on_images = np.zeros((points3D_n,images_n+2))
    points3D_on_images_classes = -np.ones((points3D_n,images_n))
    img_id = 0
    good = 0
    bad = 0
    for i in range(images_n):
        while True:
            try: 
                img_i = images[img_id]
            except:
                img_id += 1
            else: 
                break
        labeled_point = labeled_points[img_id-1]
        masks = labeled_point['masks']
        class_ids = labeled_point['class_ids']
        masks = masks.astype(int)   
        masks_n = len(masks[0,0,:])
        masks_1 = len(masks[:,0,0])
        masks_2 = len(masks[0,:,0])
        mask = masks[:, :, 0]
        for j in range(masks_n):
            #masks[:, :, j] = (j+1)*masks[:, :, j]
            mask = mask + np.logical_xor(mask, masks[:, :, j]*((j+1)*masks[:, :, j]))
        points2D_n = len(images[img_id].point3D_ids)
        for j in range(points2D_n):
            point3D_id = images[img_id].point3D_ids[j]
            if (point3D_id == -1):
                continue 
            xy = images[img_id].xys[j]
            x = int(round(xy[0]))
            y = int(round(xy[1]))
            try:
                points3D_on_images[point3D_id-1][i+1] = mask[y-1][x-1]
                if mask[y-1][x-1] != 0:
                    points3D_on_images_classes[point3D_id-1][i] = class_ids[mask[y-1][x-1]-1]
                if points3D_on_images[point3D_id-1][images_n+1] == 0:
                    points3D_on_images[point3D_id-1][images_n+1] = point3D_id
                good += 1
            except:
                bad += 1
        img_id += 1
    print("good: " + str(good))
    print("bad: " + str(bad))
    print("accuracy: " + str(good/(good+bad)))
    print("number of points before removing useless points: " + str(len(points3D_on_images)))
    points3D_on_images = np.copy(points3D_on_images.astype(int))
    points3D_on_images_classes = 1+points3D_on_images_classes.astype(int)
    for i in range(len(points3D_on_images)):
        counts = np.bincount(points3D_on_images_classes[i])
        summ_z = counts[0]
        if np.argmax(counts) == 0:
            counts[0] = 0
        points3D_on_images[i][0] = np.argmax(counts)-1
        if images_n-summ_z < 5:
            points3D_on_images[i][0] = -1
    points3D_on_images = points3D_on_images[(points3D_on_images[:, 0] != -1)]
    print("final number of points: " + str(len(points3D_on_images)))
    return points3D_on_images

def get_clusters(points3D_on_images):
        
    max_id = points3D_on_images[:,1:-1].max()
    points3D_on_images_to_cluster = np.zeros((len(points3D_on_images), len(points3D_on_images[0, 1:-1])*max_id))
    for i, point3D_ids in enumerate(points3D_on_images):
        for j, obj_id in enumerate(point3D_ids[1:-1]):
            if obj_id != 0:
                points3D_on_images_to_cluster[i][j * max_id + obj_id - 1] = 1
    
    n_clusters = 5
    from sklearn.cluster import KMeans
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(points3D_on_images_to_cluster)
    
    cluster = {}
    for i in range(clustering.n_clusters):
        cluster[i] = []
    for i, cl in enumerate(clustering.labels_):
        cluster[cl].append(points3D_on_images[i][-1])
    cluster_points3D = []
    cluster_xyz = []
    cluster_stds_xyz = []
    cluster_means_xyz = []
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
    for i in range(clustering.n_clusters):
        if sum(cluster_stds_xyz[i]/all_std_xyz) > 0.8 or len(cluster[i]) < 30:
            del cluster[i]
    for i in range(clustering.n_clusters):
        if i not in cluster:
            continue
        for j in range(clustering.n_clusters - i - 1):
            j = j + i + 1
            if np.linalg.norm(cluster_means_xyz[i]-cluster_means_xyz[j]) < 1:
                cluster[i] = cluster[i]+cluster[j]
                del cluster[j]
    return cluster

def get_target_cluster(images, cluster):
    target_cluster = 0
    max_num_of_points = 0
    set_a = 0
    for i in images:
        if images[i].name == "000.jpg":
            set_a = set(images[i].point3D_ids)
    for i in cluster:
        set_b = set(cluster[i])
        if max_num_of_points < len(set_a & set_b):
            max_num_of_points = len(set_a & set_b)
            target_cluster = i
    return target_cluster

def find_dist(points3D, cluster_1, cluster_2):
    n_min_dists = 5
    min_dist = {}
    for i in range(n_min_dists):
        min_dist[i] = float("inf")
    for id_1 in cluster_1:
        for id_2 in cluster_2:
            local_dist = np.linalg.norm(points3D[id_1].xyz-points3D[id_2].xyz)
            for i in range(n_min_dists):
                if min_dist[i] > local_dist:
                    for j in range(n_min_dists - i - 1):
                        j = j + i
                        min_dist[j+1] = min_dist[j]
                    min_dist[i] = local_dist
                    break
    min_dist = [v for v in min_dist.values()]
    ans = sum(min_dist)/n_min_dists
    return ans

def get_distances_between_clusters(cluster):
    distances_between_clusters = {}
    for i in cluster:
        distances_between_clusters[i] = {}
    len_cl = len(cluster)
    for i in cluster:
        for j in cluster:
            if j > i:
                distances_between_clusters[i][j] = find_dist(points3D, cluster[i], cluster[j])
                distances_between_clusters[j][i] = distances_between_clusters[i][j]
    return distances_between_clusters