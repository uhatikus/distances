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
# import matplotlib.pyplot as plt
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
    # img1_name = "/" + "/".join(list(filter(bool, project_dir.split("/")))[:(-1)]) + "/" + img1_name
    os.system("mkdir " + project_dir + folder_to)
    cropped_imgs =  sorted(glob.glob(project_dir + folder_from + "/*.jpg"))
    images_n = len(cropped_imgs)
    good_images = []
    goodness = []
    for i in range(images_n):
        print("image #" + str(i+1))
        goodness.append(match_imgs(img1_name, cropped_imgs[i]))
        if goodness[-1] > MIN_MATCH_COUNT:
            good_images.append(1)
        else:
            good_images.append(0)
    print(good_images)
    print(goodness)
    for i in range(images_n):
        if sum(good_images[max(0, i-(delta)):min(images_n, i+(delta+3))]) > 2:
            copy_target_image(project_dir, folder_to, cropped_imgs[i])


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

def video_to_frames(video_files):
    project_dir = ""
    scale = 0.5
    ms = 500
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
        while success:
            print('Read a new frame #' + str(count) + ': ', success)
            image = resize(image, scale)
            cv2.imwrite(project_dir + "images/" + video_name + "_frame_" + str(count).zfill(3) + ".jpg", image)     # save frame as JPEG file      
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count_v*ms))
            success,image = vidcap.read()
            count += 1  
            count_v += 1
            if count >= 30:
                break 
    print("Obtained " + str(count-1) + " images")
    n_imgs = count-1
    return (project_dir, n_imgs)


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
    print(images_n)
    points3D_n = max(points3D)
    points3D_on_images = np.zeros((points3D_n,images_n+2))
    points3D_on_images_classes = -np.ones((points3D_n,images_n))

    for i in range(images_n):
        labeled_point = labeled_points[i]
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
        points2D_n = len(images[i+1].point3D_ids)
        for j in range(points2D_n):
            point3D_id = images[i+1].point3D_ids[j]
            if (point3D_id == -1):
                continue 
            xy = images[i+1].xys[j]
            x = int(round(xy[0]))
            y = int(round(xy[1]))
            points3D_on_images[point3D_id-1][i+1] = mask[y-1][x-1]
            if mask[y-1][x-1] != 0:
                points3D_on_images_classes[point3D_id-1][i] = class_ids[mask[y-1][x-1]-1]
            if points3D_on_images[point3D_id-1][images_n+1] == 0:
                points3D_on_images[point3D_id-1][images_n+1] = point3D_id

    points3D_on_images_init = np.copy(points3D_on_images)

    print(len(points3D_on_images_init))
    #points3D_on_images = points3D_on_images_init[~np.all(points3D_on_images == 0, axis=1)]
    points3D_on_images = np.copy(points3D_on_images_init.astype(int))
    points3D_on_images_classes = 1+points3D_on_images_classes.astype(int)
    for i in range(len(points3D_on_images)):
    #     if i > 10:
    #         break
        counts = np.bincount(points3D_on_images_classes[i])
    #     print(points3D_on_images_classes[i])
        summ_z = counts[0]
        if np.argmax(counts) == 0:
            counts[0] = 0
        points3D_on_images[i][0] = np.argmax(counts)-1
        summ = sum(points3D_on_images[i][1:images_n])
    #     print(points3D_on_images[i])
        if len(counts) == 0 or summ_z > 10:
    #         print("hehe")
            points3D_on_images[i][0] = -1
    #     print(points3D_on_images[i])
    points3D_on_images = points3D_on_images[(points3D_on_images[:, 0] != -1)]
    print(len(points3D_on_images))