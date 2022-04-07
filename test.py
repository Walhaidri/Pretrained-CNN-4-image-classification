from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from fun import _get_files_path
import matplotlib.pyplot as plt

data_path = 'datasets/flower_photos'

classes_no, classes_names, files_num, all_pathes = _get_files_path(data_path=data_path)
tst = np.random.randint(0, 600, 81)
trained_model = tf.keras.models.load_model("ResNet50_model.h5")

n = 2
for k, i in enumerate(tst):
    file = all_pathes[classes_names[n]][i]
    im = np.asarray(Image.open(file))
    resize_im = cv2.resize(im, (256, 256))
    resize_im = np.expand_dims(resize_im, axis=0)
    pred = trained_model.predict(resize_im)
    print(i, pred)
    m = np.argmax(pred)
    plt.subplot(9, 9, k+1)
    plt.imshow(np.squeeze(resize_im))
    plt.xticks()
    plt.yticks()
    plt.title(classes_names[m])
plt.show()

