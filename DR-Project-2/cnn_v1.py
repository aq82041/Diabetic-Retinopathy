"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""

import numpy as np
from keras import backend as K, Model
from keras.layers import *
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import classification_report, confusion_matrix

# Start
train_data_path = 'F:\\PARAM\\DL\\diaretdb1_v_1_1\\diaretdb1_v_1_1\\resources\\images\\train'
test_data_path = 'F:\\PARAM\\DL\\diaretdb1_v_1_1\\diaretdb1_v_1_1\\resources\\images\\test'
img_rows = 1024
img_cols = 1024
epochs = 10
batch_size = 10
num_of_train_samples = 70
num_of_test_samples = 19

# Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

# Build model

inputs = Input((img_rows, img_cols, 3))
conv1 = Conv2D(128, (5, 5), activation='relu', strides=4)(inputs)
bnc1 = BatchNormalization()(conv1)
# bn1 = BatchNormalization()(pool1)
conv2 = Conv2D(64, (7, 7), padding='same', activation='relu', strides=2)
conv3 = Conv2D(64, (5, 5), padding='same', activation='relu', strides=2)
merg1 = concatenate([conv2(bnc1), conv3(bnc1)], axis=3)
bm1 = BatchNormalization()(merg1)
conv4 = Conv2D(128, (5, 5), activation='relu')(bm1)
# bnc4 = BatchNormalization()(conv4)
# conv5 = Conv2D(32, (3, 3), activation='relu')(bnc4)
# pool2 = MaxPooling2D(pool_size=(2, 2))(conv5)
# bn2 = BatchNormalization()(pool2)
# conv6 = Conv2D(32, (3, 3), padding='same', activation='relu')
# conv7 = Conv2D(32, (1, 1), padding='same', activation='relu')
# merg2 = concatenate([conv6(bn2), conv7(bn2)], axis=3)
# bm2 = BatchNormalization()(merg2)

# conv8 = Conv2D(32, (3, 3), activation='relu')(bm2)
# bn3 = BatchNormalization()(conv8)
# conv9 = Conv2D(32, (1, 1), activation='relu')(bn3)
# bn4 = BatchNormalization()(conv9)
# pool3 = MaxPooling2D(pool_size=(2, 2))(bm1)
# ft = Flatten()(pool3)
# dense1 = Dense(1000, activation='relu')(ft)
ft = GlobalAveragePooling2D()(conv4)
ft = Dense(100, activation='sigmoid')(ft)
dense2 = Dense(2, activation='softmax')(ft)

model = Model(input=inputs, output=dense2)

# from keras.utils.vis_utils import plot_model
#
# plot_model(model, to_file='model_db1_plot.png', show_shapes=True, show_layer_names=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Train
model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size)

# Confution Matrix and Classification Report
model.save('db11.h5')
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['exudates', 'non_exudates']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names)
plt.show()
