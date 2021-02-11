#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os, sys
import torch, torchvision
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import multiprocessing

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog





#Variable containing path of data folder
dirc = '/working/dir'
files = sorted(os.listdir(dirc))

#Read all images in folder dirc
image_data = [cv2.imread('{}/{}'.format(dirc, i)) for i in files]

#Percentages of body parts 
head_perc = 0.25
upper_body_perc = 0.15
lower_body_perc = 0.40

#Colors used for representing people
colors = [(np.array(color) * 255).astype(np.uint8) for name, color in mcolors.BASE_COLORS.items()]
colors = [(int(color[2]), int(color[1]), int(color[0])) for color in colors]
all_imgs = []

#Predictor setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)


# ## Run Object Detection

# In[4]:


def print_stdlab(lab):
    """Print actual Lab values of color"""
    print([float(lab[0]) / 256 * 100, float(lab[1]) - 128, float(lab[2]) - 128])
    
def visualize_tracking(img, indices, center_color, instances):
    """Show image with colored bounding boxes on people"""
    img1 = np.copy(img)
    for i, ind in enumerate(indices):
        x1, y1, x2, y2 = instances.pred_boxes.tensor[ind].numpy().astype(np.uint32)
        img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), center_color[i], 2)
    cv2.imshow("image", img1)
    k = cv2.waitKey(1)
    if (k == 99):
        cv2.destroyAllWindows()
        sys.exit(0)
        
def find_dist(prev_center, new_center):
    """Find distance between two Lab color centers"""
    return 8 * np.sqrt((float(prev_center[1]) - float(new_center[1])) ** 2 + (float(prev_center[2]) - float(new_center[2])) ** 2) + 2 * abs(prev_center[0] - new_center[0])
    
def find_all_dists(cur_center, centers):
    """Find all distances between 2 lists of color centers"""
    ret = []
    for cen_i, center in enumerate(centers):
        for c_i, c in enumerate(cur_center):
            ret.append((find_dist(c, center), (cen_i, c_i)))
    return sorted(ret, key = lambda x: x[0])

def update_all_dists(all_dists, match):
    """Remove distances of centers that have already been matched"""
    return [i for i in all_dists if (i[1][0] != match[0] and i[1][1] != match[1])]

def update_cur_center(cur_center, matches, center):
    """Update current center"""
    matched = []
    for match in matches:
        matched.append(match[0])
        cur_center[match[1]] = (center[match[0]] + cur_center[match[1]]) / 2
    for i in range(len(center)):
        if i not in matched:
            cur_center.append(center[i])
    return cur_center



def find_dom_color(box, mask, image_gb, head_perc, upper_body_perc, num_clusters):
    """Find dominant color of upper body part using kmeans clustering"""
    x1, y1, x2, y2 = box
    h = y2 - y1
    upper_body_mask = np.zeros(image_gb.shape[:2], dtype = bool)
    upper_body_mask[int(y1 + h * head_perc) : int(y1 + h * (upper_body_perc + head_perc)), x1 : x2] = True
    upper_body_mask = np.logical_and(mask, upper_body_mask)
    img1 = cv2.cvtColor(image_gb, cv2.COLOR_BGR2LAB)
    
    #kmeans to find dominant color
    data = img1[upper_body_mask]
    if len(data) >= 2:
        clt = KMeans(n_clusters = num_clusters, n_jobs = 2)
        clt.fit(data)
        centers_ = clt.cluster_centers_
        temp = np.unique(clt.labels_, return_counts = True)[1]
        center = list(map(lambda a: a[1], sorted(enumerate(centers_), key = lambda a: temp[a[0]], reverse = True)))[0]
        return center
    print("Person mask is not eligible")
    return None

def fix_masks(masks, indices):
    """Remove from every mask any common pixels with another mask"""
    new_masks = []
    for ind in indices:
        mask = masks[ind]
        for ind1 in indices:
            if ind != ind1:
                mask = np.logical_and(mask, np.logical_and(masks[ind], np.logical_not(masks[ind1])))
        new_masks.append(mask)
    return new_masks

def show_masks(img, masks):
    """Helper for showing in white a mask on a given image"""
    img1 = np.copy(img)
    for mask in masks:
        img1[mask] = [255, 255, 255]   
    cv2.imshow("image", img1)
    k = cv2.waitKey(300)
    if (k == 99):
        cv2.destroyAllWindows()
        sys.exit(0)

#Setup for multiprocessing
pool = multiprocessing.Pool(8)

for i, img in enumerate(image_data):
    #Mask-RCNN output
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    
    #Gaussian Blurring
    image_gb = cv2.GaussianBlur(img, (7, 7), 0)
    
    #Indices of detected people in outputs
    indices = np.where(instances.pred_classes.numpy() == 0)[0]
    
    masks = fix_masks(instances.pred_masks.numpy(), indices)
    
    #Used only for kmeans clustering without parallelization
    """centers = []
    for ind_i, ind in enumerate(indices):
        center = find_dom_color(instances.pred_boxes.tensor[ind].numpy().astype(np.uint32), masks[ind_i], image_gb, head_perc, upper_body_perc, 2)
        centers.append(center)"""
        
    #Parallel computation of color centers    
    centers = pool.starmap(find_dom_color, [(instances.pred_boxes.tensor[indices[ind_i]].numpy().astype(np.uint32), masks[ind_i], image_gb, head_perc, upper_body_perc, 2) for ind_i in range(len(indices))])
    
    #If mask produced by nn is incomplete, skip that person
    new_indices = []
    new_centers = []
    for ci, center in enumerate(centers):
        if not center is None:
            new_indices.append(indices[ci])
            new_centers.append(center)
            
    indices = new_indices
    centers = new_centers
    
    if i == 0:
        #Initialize centers from first frame
        cur_center = centers
        cur_center_colors = colors[:len(cur_center)]
        center_colors = cur_center_colors
        
    else:
        #Calculate distances between previous and current centers
        all_dists = find_all_dists(cur_center, centers)
        matches = []
        
        #Pick minimum distance as match till no more possible matches are available
        while all_dists:
            matches.append(all_dists[0][1])
            all_dists = update_all_dists(all_dists, all_dists[0][1])
            
        #Update center
        cur_center = update_cur_center(cur_center, matches, centers)
        
        #Add more colors for visualization in case of new people detected
        if len(cur_center_colors) < len(cur_center):
            cur_center_colors.extend(colors[len(cur_center_colors) : len(cur_center)])

        #Used only for visualization
        center_colors = np.ones((len(centers), 3))
        for match in matches:
            center_colors[match[0]] = cur_center_colors[match[1]]

        marker = len(matches)
        for i, color in enumerate(center_colors):
            if (color == 1.).all():
                center_colors[i] = cur_center_colors[marker]
                marker += 1
        
    visualize_tracking(image_gb, indices, center_colors, instances)
    
    #Save results for further computation 
    all_imgs.append([image_gb, indices, center_colors, instances])
cv2.destroyAllWindows()

