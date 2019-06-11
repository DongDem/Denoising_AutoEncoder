from __future__ import division, print_function

import cv2
import numpy as np
import os

import shutil
from distutils.dir_util import copy_tree
from shutil import copyfile

basedir = './dataset/train/'
savedir = './dataset/train_augment/'
def flip(in_image):
    image  =cv2.imread(in_image)
    vertical_img = image.copy()
    vertical_img = cv2.flip(image,1)
    return vertical_img
def left_15(in_image):
    image = cv2.imread(in_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    M_left = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotate_left = cv2.warpAffine(image, M_left, (cols, rows))
    return rotate_left

def left_10(in_image):
    image = cv2.imread(in_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    M_left = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    rotate_left = cv2.warpAffine(image, M_left, (cols, rows))
    return rotate_left

def left_5(in_image):
    image = cv2.imread(in_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    M_left = cv2.getRotationMatrix2D((cols / 2, rows / 2), 5, 1)
    rotate_left = cv2.warpAffine(image, M_left, (cols, rows))
    return rotate_left

def right_15(in_image):
    image = cv2.imread(in_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    M_right = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
    rotate_right = cv2.warpAffine(image, M_right, (cols, rows))
    return rotate_right

def right_10(in_image):
    image = cv2.imread(in_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    M_right = cv2.getRotationMatrix2D((cols / 2, rows / 2), -10, 1)
    rotate_right = cv2.warpAffine(image, M_right, (cols, rows))
    return rotate_right

def right_5(in_image):
    image = cv2.imread(in_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    M_right = cv2.getRotationMatrix2D((cols / 2, rows / 2), -5, 1)
    rotate_right = cv2.warpAffine(image, M_right, (cols, rows))
    return rotate_right

if not os.path.exists(savedir):
    os.mkdir(savedir)

for image_file in sorted(os.listdir(basedir)):
    full_dir = os.path.join(basedir, image_file)
    print(full_dir)
    flip_image = flip(full_dir)
    if flip_image is not None:
        cv2.imwrite(os.path.join(savedir, str('flip_'+image_file)), flip_image)
    save_path = os.path.join(savedir, image_file)
    copyfile(full_dir, save_path)
    '''
        left_15_image = left_15(full_dir)
        left_15_image_copy = left_15_image.copy()
        left_15_flip_image = cv2.flip(left_15_image_copy,1)
        left_10_image = left_10(full_dir)
        left_10_image_copy = left_10_image.copy()
        left_10_flip_image = cv2.flip(left_10_image_copy,1)
        left_5_image = left_5(full_dir)
        left_5_image_copy = left_5_image.copy()
        left_5_flip_image = cv2.flip(left_5_image_copy,1)
        right_15_image = right_15(full_dir)
        right_15_image_copy = right_15_image.copy()
        right_15_flip_image = cv2.flip(right_15_image_copy,1)
        right_10_image = right_10(full_dir)
        right_10_image_copy = right_10_image.copy()
        right_10_flip_image = cv2.flip(right_10_image_copy,1)
        right_5_image = right_5(full_dir)
        right_5_image_copy = right_5_image.copy()
        right_5_flip_image = cv2.flip(right_5_image_copy,1)
        '''







