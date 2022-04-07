import glob
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def _get_files_path(data_path):
    """
    :param
    data_path: путь к основном папке, содержащей папки с файлами
    :return:
    сlasses_no - количество классов
    classes_names - названия классов
    all_pathes - словарь, где клюи- названия классов
    значения - пути к файлам
    © Walid Al-Haidri
    April 2022
    """
    dirct_cl = []  # пути классов
    classes_names = []  # названия классов
    files_no = []
    for dir, folders, files in os.walk(data_path):
        dirct_cl.append(os.path.join(dir, '*'))
        classes_names.append(folders)

    classes_names = classes_names[0]
    dirct_cl = dirct_cl[1:]
    classes_no = len(dirct_cl)
    all_pathes = {}
    files_num = []
    for i, j in enumerate(dirct_cl):
        files_paths = glob.glob(dirct_cl[i])
        files_mounts = len(files_paths)
        all_pathes.setdefault(classes_names[i], files_paths)
        files_num.append(files_mounts)

    return classes_no, classes_names, sum(files_num), all_pathes


def _dataset_generate(data_path, new_size=[256, 256], isRGB = True):
    '''

    :param data_path: the folder with classes subfolders
    :param new_size: bribg images to one size
    :param isRGB: defualt   True for RGB, false for grayscal images
    :return:
    x - training input
    y - training output
     © Walid Al-Haidri
    April 2022
    '''
    classes_no, classes_names, files_num, train_all_pathes = _get_files_path(data_path=data_path)
    cl_items_len = []
    y = []
    if isRGB:
        x = np.zeros((files_num, new_size[0], new_size[1], 3), dtype='uint8')
    else:
        x = np.zeros((files_num, new_size[0], new_size[1], 1), dtype='uint8')
    j = 0
    for i in range(classes_no):
        paths_no = len(train_all_pathes[classes_names[i]])
        cl_items_len.append(paths_no)
        for k, path in enumerate(train_all_pathes[classes_names[i]]):
            im = np.asarray(Image.open(path))
            resize_im = cv2.resize(im, (new_size[0], new_size[1]))
            x[j] = resize_im
            y.append(i)
            j += 1
    return x, y


