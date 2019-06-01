import os
import sys
import cv2
import ntpath
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import collections
import struct
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob


def video_to_frames(video_file):
    vidcap = cv2.VideoCapture(video_file)
    video_name = ((ntpath.basename(video_file)).split("."))[0]
    os.system("mkdir data/projects/" + video_name)
    os.system("mkdir data/projects/" + video_name + "/images")
    success,image = vidcap.read()
    count = 1
    project_dir = os.getcwd() + "/data/projects/" + video_name + "/"
    while success:
        cv2.imwrite(project_dir + "images/" + video_name + "_frame_%d.jpg" % count, image)     # save frame as JPEG file      
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))
        success,image = vidcap.read()
        print('Read a new frame #' + str(count) + ': ', success)
        count += 1  
        if count > 0:
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


def label_points(IMAGE_DIR):
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
        #raise Exception("Can not find COCO_MODELL")
        utils.download_trained_weights(COCO_MODEL_PATH)

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


    # Object Detection
    labeled_points = []
    count = 1
    for image_file in glob.glob(IMAGE_DIR + "*.jpg"):
        if count > 3:
            break
        print("Processing image # " + str(count))
        count += 1
        image = skimage.io.imread(image_file)

        # Run detection
        results = (model.detect([image], verbose=1))[0]
        labeled_points.append(results)
        # Visualize results
        # visualize.display_instances(image, results['rois'], results['masks'], results['class_ids'], class_names, results['scores'])
    return labeled_points
