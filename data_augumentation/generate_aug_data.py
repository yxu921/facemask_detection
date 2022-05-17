from cProfile import label
import numpy as np # linear algebra
from bs4 import BeautifulSoup
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json

import cv2 

import pickle as pkl
import shutil
import sys
from pathlib import PurePath

from data_aug.data_aug import *
from data_aug.bbox_util import * 

IMG_PATH = "./facemask_dataset/images/"
LABEL_PATH = "./facemask_dataset/annotations/"

IMG_FILES = [os.path.join(IMG_PATH, file) for file in sorted(os.listdir(IMG_PATH))]
LABEL_FILES = [os.path.join(LABEL_PATH, file) for file in sorted(os.listdir(LABEL_PATH))]

TRAINING_DATASET_PATH = "./training_dataset"
TESTING_DATASET_PATH = "./testing_dataset"

def generate_box(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    elif obj.find('name').text == "without_mask":
        return 3
    return 0

def read_targets_from_xml_file(image_id, label_xml_file): 
    with open(label_xml_file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'lxml')
        objects = soup.find_all('object')
        
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        
        return target

def generate_new_image_label_files(image_file, label_file, new_image_dir, new_label_dir):        
    img = np.array(Image.open(image_file).convert("RGB"))
    img_idx = image_file.split("/")[-1].lstrip("maksssksksss").rstrip(".png")
    targets = read_targets_from_xml_file(img_idx, label_file)
        
    img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), np.array(targets["boxes"], dtype=np.float32).copy())   
    
    targets["boxes"] = bboxes_.tolist()
    im = Image.fromarray(img_)
        
    new_img_file = os.path.join(new_image_dir, f"maksssksksss{img_idx}.hori_flip.png")
    new_label_file = os.path.join(new_label_dir, f"maksssksksss{img_idx}.hori_flip.json")
    
    im.save(new_img_file)
    with open(new_label_file, "w") as outfile:
        json.dump(targets, outfile)   

def generate_sheared_image_label_files(image_file, label_file, new_image_dir, new_label_dir):        
    img = np.array(Image.open(image_file).convert("RGB"))
    img_idx = image_file.split("/")[-1].lstrip("maksssksksss").rstrip(".png")
    targets = read_targets_from_xml_file(img_idx, label_file)
        
    img_, bboxes_ = RandomShear(0.2)(img.copy(), np.array(targets["boxes"], dtype=np.float32).copy())   
    
    targets["boxes"] = bboxes_.tolist()
    im = Image.fromarray(img_)
        
    new_img_file = os.path.join(new_image_dir, f"maksssksksss{img_idx}.shear.png")
    new_label_file = os.path.join(new_label_dir, f"maksssksksss{img_idx}.shear.json")
    
    im.save(new_img_file)
    with open(new_label_file, "w") as outfile:
        json.dump(targets, outfile) 

def find_matching_img_file(label_file):
    file_name = PurePath(label_file).stem
    idx = file_name.lstrip("maksssksksss")
    img_file = os.path.join(IMG_PATH, f"maksssksksss{idx}.png")
    assert os.path.exists(img_file)
    return img_file

def split_training_testing_labels():
    num_other = 0
    num_wear_mask = 0
    num_wear_mask_wrong = 0
    num_no_mask = 0

    for label_file in LABEL_FILES:
        targets = read_targets_from_xml_file(-1, label_file)
        for label in targets["labels"]:
            if label == 0:
                num_other += 1
            elif label == 1:
                num_wear_mask += 1
            elif label == 2:
                num_wear_mask_wrong += 1
            elif label == 3:
                num_no_mask += 1    
    print(f"DEBUG: num_no_mask_label = {num_no_mask}, num_wear_mask = {num_wear_mask}, num_wear_mask_wrong = {num_wear_mask_wrong}, num_other = {num_other}")
    testing_data = []
    training_data = []
    num_other_testing = 0
    num_wear_mask_testing = 0
    num_wear_mask_wrong_testing = 0
    num_no_mask_testing = 0
    had_enough_testing_data = False
    for label_file in LABEL_FILES:
        targets = read_targets_from_xml_file(-1, label_file)
        if had_enough_testing_data:
            training_data.append(label_file)
        else:
            testing_data.append(label_file)
        for label in targets["labels"]:
            if label == 0:
                num_other_testing += 1
            elif label == 1:
                num_wear_mask_testing += 1
            elif label == 2:
                num_wear_mask_wrong_testing += 1
            elif label == 3:
                num_no_mask_testing += 1   
            if num_wear_mask_testing >= 0.1 * num_wear_mask and \
                num_wear_mask_wrong_testing >= 0.1 * num_wear_mask_wrong and \
                    num_no_mask_testing >= 0.1 * num_no_mask:
                had_enough_testing_data = True
    return training_data, testing_data

def copy_training_and_testing_data(training_labels, testing_labels):
    for training_label in training_labels:
        training_image = find_matching_img_file(training_label)        
        dest = os.path.join(TRAINING_DATASET_PATH, "images", PurePath(training_image).name)
        shutil.copy(training_image, dest)
        dest = os.path.join(TRAINING_DATASET_PATH, "annotations", PurePath(training_label).name)
        shutil.copy(training_label, dest)
    for testing_label in testing_labels:
        testing_image = find_matching_img_file(testing_label)        
        dest = os.path.join(TESTING_DATASET_PATH, "images", PurePath(testing_image).name)
        shutil.copy(testing_image, dest)
        dest = os.path.join(TESTING_DATASET_PATH, "annotations", PurePath(testing_label).name)
        shutil.copy(testing_label, dest)  
    
    
def generate_aug_data():
    print("Running generate_aug_data")    
    print(f"DEBUG: img files = {len(IMG_FILES)}")
    print(f"DEBUG: label files = {len(LABEL_FILES)}")

    new_image_path = "./hori_flip_dataset/images"
    new_label_path = "./hori_flip_dataset/annotations"

    if os.path.exists(new_image_path):
        shutil.rmtree(new_image_path)
    if os.path.exists(new_label_path):
        shutil.rmtree(new_label_path)
    os.makedirs(new_image_path, exist_ok=True)
    os.makedirs(new_label_path, exist_ok=True)
    for idx in range(len(IMG_FILES)):
        generate_new_image_label_files(IMG_FILES[idx], LABEL_FILES[idx], new_image_path, new_label_path)
        print(f"DEBUG: generated {idx}")

def generate_aug_data_sheared():
    print("Running generate_aug_data")    
    print(f"DEBUG: img files = {len(IMG_FILES)}")
    print(f"DEBUG: label files = {len(LABEL_FILES)}")

    new_image_path = "./sheared_dataset/images"
    new_label_path = "./sheared_dataset/annotations"

    if os.path.exists(new_image_path):
        shutil.rmtree(new_image_path)
    if os.path.exists(new_label_path):
        shutil.rmtree(new_label_path)
    os.makedirs(new_image_path, exist_ok=True)
    os.makedirs(new_label_path, exist_ok=True)
    for idx in range(len(IMG_FILES)):
        generate_sheared_image_label_files(IMG_FILES[idx], LABEL_FILES[idx], new_image_path, new_label_path)
        print(f"DEBUG: generated {idx}")       

if __name__ == "__main__":
    # generate_aug_data_sheared()

    training_labels, testing_labels = split_training_testing_labels()
    copy_training_and_testing_data(training_labels, testing_labels)