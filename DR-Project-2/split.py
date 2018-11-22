"""
__author__ = "Param Popat"
__version__ = "4"
__git__ = "https://github.com/parampopat/"
"""

import shutil
import os
import pandas as pd


def original_noexudates():
    """
    Finds Non - Exudate images
    :return:
    """
    images = []
    for file in os.listdir("F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/noexudates"):
        if file.endswith(".dot"):
            base = os.path.splitext(file)[0]
            os.rename("F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/noexudates/" + file,
                      "F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/noexudates/" + base + ".png")
            print(file)
            images.append(file)

    for file in os.listdir("F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images"):
        for i in images:
            if file == i:
                shutil.copy2(
                    'F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images/' + file,
                    'F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/noexudates_img/' + file)


def exudates_vs_no():
    """
    Seperates images containing exudates from those missing exudates
    :return:
    """
    no = []
    for file in os.listdir("F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/noexudates"):
        no.append(file)
    for file in os.listdir("F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images"):
        if file not in no:
            shutil.copy2('F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images/' + file,
                         'F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/exudates_img/' + file)


def make_csv():
    """
    Generates a csv containing label for each image in dataset
    0 : Non Exudate
    1 : Exudate
    :return:
    """
    for file in os.listdir("F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/noexudates_img"):
        fopen = open('F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0.csv', "a")
        st = file + "," + str(0)
        fopen.write("\n" + st)
        fopen.close()
    for file in os.listdir("F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/exudates_img"):
        fopen = open('F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0.csv', "a")
        st = file + "," + str(1)
        fopen.write("\n" + st)
        fopen.close()


def train_test_split_diaret_db_0():
    """
    Splits DIARETDB0 dataset in 80:20 ratio for train:test
    keeping the proportion of each label equal in both
    train and test.
    :return:
    """
    dataset = pd.read_csv('F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0.csv')
    label = dataset.iloc[:, 1].values
    image = dataset.iloc[:, 0].values
    non_exudates = []
    exudates = []
    for i in range(label.__len__()):
        if label[i] == 0:

            non_exudates.append(image[i])
        else:
            exudates.append(image[i])

    num_non_exudates = int(0.8 * len(non_exudates))
    num_exudates = int(0.8 * len(exudates))
    i = 0
    j = 0
    source = 'F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images/'
    destination = 'F:/PARAM/DL/diaretdb0_v_1_1/diaretdb0_v_1_1/resources/images/'
    for file in non_exudates:
        if i < num_non_exudates:
            shutil.copy2(source + file,
                         destination + 'train\\non_exudates\\' + file)
        else:
            shutil.copy2(source + file,
                         destination + 'test\\non_exudates\\' + file)
        i += 1

    for file in exudates:
        if j < num_exudates:
            shutil.copy2(source + file,
                         destination + 'train\\exudates\\' + file)
        else:
            shutil.copy2(source + file,
                         destination + 'test\\exudates\\' + file)
        j += 1


def train_test_split_diaret_db_1():
    """
    Splits DIARETDB1 dataset in 80:20 ratio for train:test
    keeping the proportion of each label equal in both
    train and test.
    :return:
    """
    dataset = pd.read_csv('F:\\PARAM\DL\\diaretdb1_v_1_1\\diaretdb1_v_1_1\\resources\\images\\diaretdb1.csv')
    label = dataset.iloc[:, 1].values
    image = dataset.iloc[:, 0].values
    non_exudates = []
    exudates = []
    for i in range(label.__len__()):
        if label[i] == 0:

            non_exudates.append(image[i] + '.png')
        else:
            exudates.append(image[i] + '.png')

    num_non_exudates = int(0.8 * len(non_exudates))
    num_exudates = int(0.8 * len(exudates))
    i = 0
    j = 0
    source = 'F:\\PARAM\DL\\diaretdb1_v_1_1\\diaretdb1_v_1_1\\resources\\images\\ddb1_fundusimages\\'
    destination = 'F:\\PARAM\DL\\diaretdb1_v_1_1\\diaretdb1_v_1_1\\resources\\images\\'
    for file in non_exudates:
        if i < num_non_exudates:
            shutil.copy2(
                source + file,
                destination + 'train\\non_exudates\\' + file)
        else:
            shutil.copy2(
                source + file,
                destination + 'test\\non_exudates\\' + file)
        i += 1

    for file in exudates:
        if j < num_exudates:
            shutil.copy2(source + file,
                         destination + 'train\\exudates\\' + file)
        else:
            shutil.copy2(source + file,
                         destination + 'test\\exudates\\' + file)
        j += 1


def train_test_split_messidor():
    """
    Splits Messidor dataset in 80:20 ratio for train:test
    keeping the proportion of each label equal in both
    train and test.
    :return:
    """
    names = ['Annotation_Base11', 'Annotation_Base12', 'Annotation_Base13', 'Annotation_Base14',
             'Annotation_Base21', 'Annotation_Base22', 'Annotation_Base23', 'Annotation_Base24', 'Annotation_Base31',
             'Annotation_Base32', 'Annotation_Base33', 'Annotation_Base34']

    for filename in names:
        dataset = pd.read_csv('F:/PARAM/DL/Messidor/messidordataset/' + filename + '.csv')
        image = dataset.iloc[:, 1].values
        label = dataset.iloc[:, 5].values
        non_exudates = []
        exudates = []
        for i in range(label.__len__()):
            if label[i] == 0:
                non_exudates.append(image[i])
            else:
                exudates.append(image[i])

        num_non_exudates = int(0.8 * len(non_exudates))
        num_exudates = int(0.8 * len(exudates))
        i = 0
        j = 0
        source = 'F:\\PARAM\\DL\\Messidor\\' + filename[11:17] + '\\'
        destination = 'F:\\PARAM\\DL\Messidor\\'
        for file in non_exudates:
            if i < num_non_exudates:
                shutil.copy2(
                    source + file,
                    destination + 'train\\non_exudates\\' + file)
            else:
                shutil.copy2(
                    source + file,
                    destination + 'test\\non_exudates\\' + file)
            i += 1

        for file in exudates:
            if j < num_exudates:
                shutil.copy2(source + file,
                             destination + 'train\\exudates\\' + file)
            else:
                shutil.copy2(source + file,
                             destination + 'test\\exudates\\' + file)
            j += 1


if __name__ == '__main__':
    data = int(input('0 for DIARETDB0\n1 for DIARETDB1\n2 for MESSIDOR\n'))
    if data == 0:
        train_test_split_diaret_db_0()
    elif data == 1:
        train_test_split_diaret_db_1()
    else:
        train_test_split_messidor()
