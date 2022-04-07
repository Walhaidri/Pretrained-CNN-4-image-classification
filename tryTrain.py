from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
import numpy as np
from fun import _dataset_generate
from fun import _dataset_generate, _get_files_path

train_data_path = r'datasets\flower_photos'

classes_no, classes_names, files_num, all_pathes = _get_files_path(data_path=train_data_path)
print(f'количество классов = {classes_no} \n'
      f' В датасете содежатся следующие классы {classes_names} \n'
      f' Объем датасета :{files_num}')
for key in all_pathes.keys():
    print('------------------------------------------------------')
    print(f'Данные из класса {key}')
    print('------------------------------------------------------')
    for path in all_pathes[key]:
        print(path)

x_train, train_labels = _dataset_generate(data_path=train_data_path, new_size=[256, 256])
x_train = x_train / 255
y_tr = to_categorical(train_labels)
np.save('train_labels.npy', train_labels)
np.save('x_train.npy', np.array(x_train))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x[i + 3000])
plt.show()

# ------------------------------------- load dataset

X = np.load('x_train.npy')
y = np.load('train_labels.npy')
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Dataset loaded')
print('Train Dataset size is :', X_train.shape)
print('Test Dataset size is :', X_test.shape)

# ================  Import pretraind model ResNet50  ===========
model_ResNet50 = ResNet50(include_top=False, weights='imagenet', input_shape=[256, 256, 3])
model_ResNet50.trainable = False

# Creat our model
model = Sequential()
model.add(model_ResNet50)
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
model.summary()
plot_model(model, 'my_model.png')
model.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])

with tf.device('/device:gpu:1'):                    #train on GPU
    history = model.fit(X_train, y_train,
                        epochs=15,
                        validation_split=0.2,
                        batch_size=8,
                        verbose=1)

    model.save('ResNet50_model.h5')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.show()
    plt.savefig('training results.png')

    # Evaluate the model on the test data
    trained_model = tf.keras.models.load_model("ResNet50_model.h5")
    print("Evaluate on test data")
    results = trained_model.evaluate(X_test, y_test, batch_size=1)
    print("test loss, test acc:", results)

