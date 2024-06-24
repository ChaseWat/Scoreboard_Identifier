import numpy as np
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle

SKIP_NEGATIVES = True
NEGATIVE_CLASS = "No-Circle"
classes = [0]


def xml_to_csv(path, skipNegatives):
    # Exports xml data to csv format file
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.find('object'):
            for member in root.findall('object'):
                bbx = member.find('bndbox')
                xmin = round(float(bbx.find('xmin').text))
                ymin = round(float(bbx.find('ymin').text))
                xmax = round(float(bbx.find('xmax').text))
                ymax = round(float(bbx.find('ymax').text))
                label = member.find('name').text
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         label,
                         xmin,
                         ymin,
                         xmax,
                         ymax
                         )
                # print(value)
                xml_list.append(value)
        elif not skipNegatives:
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     NEGATIVE_CLASS,
                     0,
                     0,
                     0,
                     0
                     )
            # print(value)
            xml_list.append(value)

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


datasets = ['training']
# xml to csv for data
for ds in datasets:
    # image_path = os.path.join(os.getcwd(), 'Images', ds)
    image_path = ('.\\\\dataset\\annot')
    xml_df = xml_to_csv(image_path, SKIP_NEGATIVES)
    xml_df.to_csv('dataset_01.csv'.format(ds), index=None)
    print('Successfully converted xml to csv.')


# File Path to csv and main image directory
TRAINING_CSV_FILE = ('.\\\\dataset.csv')
TRAINING_IMAGE_DIR = ('.\\\\dataset\\images')

image_records = pd.read_csv(TRAINING_CSV_FILE)
# Test Split (90/10)
training_image_records = image_records.sample(frac=0.9, random_state=40)
test_image_records = image_records.drop(training_image_records.index)
train_image_path = os.path.join(os.getcwd(), TRAINING_IMAGE_DIR)

# arrays to hold data
images = []
targets = []
labels = []

# loop that writes training data into the above arrays
for index, row in training_image_records.iterrows():
    (filename, width, height, class_name, xmin, ymin, xmax, ymax) = row

    train_image_fullpath = os.path.join(train_image_path, filename)
    train_img = keras.preprocessing.image.load_img(train_image_fullpath, target_size=(height, width))
    train_img_arr = keras.preprocessing.image.img_to_array(train_img)

    xmin = round(xmin / width, 2)
    ymin = round(ymin / height, 2)
    xmax = round(xmax / width, 2)
    ymax = round(ymax / height, 2)

    images.append(train_img_arr)
    targets.append((xmin, ymin, xmax, ymax))
    labels.append(classes.index(class_name))

images = np.array(images)
targets = np.array(targets)
labels = np.array(labels)

# Train / Validation Split (80/20)
train_images, validation_images, train_targets, validation_targets, train_labels, validation_labels = train_test_split(
    images, targets, labels, test_size=0.20, random_state=42)

# resolution
width = 256
height = 256
num_classes = 1

# create the common input layer
input_shape = (height, width, 3)
input_layer = tf.keras.layers.Input(input_shape)

# create the base layers
base_layers = layers.experimental.preprocessing.Rescaling(1. / 255, name='bl_1')(input_layer)
base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_7')(base_layers)
base_layers = layers.Flatten(name='bl_8')(base_layers)

locator_branch = layers.Dense(128, activation='relu', name='bb_1')(base_layers)
locator_branch = layers.Dense(64, activation='relu', name='bb_2')(locator_branch)
locator_branch = layers.Dense(32, activation='relu', name='bb_3')(locator_branch)
locator_branch = layers.Dense(4, activation='sigmoid', name='bb_head')(locator_branch)

# assemble the model architecture
model = tf.keras.Model(input_layer, outputs=[locator_branch])

# use MSE loss
losses = tf.keras.losses.MSE

model.compile(loss=losses, optimizer='Adam', metrics=['accuracy'])

trainTargets = train_targets
validationTargets = validation_targets

# detrmines number of epochs
training_epochs = 20

# TRAINING
# model.fit(train_images, trainTargets, validation_data=(validation_images, validationTargets), batch_size=4,epochs=training_epochs, shuffle=True,verbose=1)

# MODEL SAVING/LOADING
# Model Saver
# model.save("Scoreboard_Obj_Loc.keras")
# Model Loader
model = tf.keras.models.load_model("Scoreboard_Obj_Loc.keras")

# prints a png of the model architecture
plot_model(model, "app_01_model.png", show_shapes=True)

# arrays for test data
pred_images = []
pred_targets = []
pred_labels = []

# file paths for the training images
test_image_fullpaths = []

# repeat of for loop to write test data to test arrays
for index, row in test_image_records.iterrows():
    (filename, width, height, class_name, xmin, ymin, xmax, ymax) = row
    test_image_fullpath = (os.path.join(train_image_path, filename))
    test_image_fullpaths.append(test_image_fullpath)
    img = keras.preprocessing.image.load_img(test_image_fullpath, target_size=(height, width))
    img_arr = keras.preprocessing.image.img_to_array(img)
    xmin = round(xmin / width, 2)
    ymin = round(ymin / height, 2)
    xmax = round(xmax / width, 2)
    ymax = round(ymax / height, 2)

    pred_images.append(img_arr)
    pred_targets.append((xmin, ymin, xmax, ymax))
    pred_labels.append(classes.index(class_name))

pred_images = np.array(pred_images)
pred_targets = np.array(pred_targets)
pred_labels = np.array(pred_labels)


# method to plot model predictions
def plot_pred(img, pred):
    n = 10
    fig, ax = plt.subplots(nrows=2, ncols=n // 2)
    for i in range(n):
        image = PIL.Image.open(img[i])
        res_1 = 1920
        res_2 = 1080
        image = image.resize(size=(res_1, res_2))
        if i<5:
            r = 0
            c = i
        else:
            r = 1
            c = i - 5
        ax[r, c].imshow(image)
        prediction = model.predict(np.expand_dims(pred[i], axis=0))
        # rect = Rectangle(xy=(pred[i, 0] * res_1, pred[i, 1] * res_2), width=(pred[i, 2] - pred[i, 0]) * res_1, height=(pred[i, 3] - pred[i, 1]) * res_2, linewidth=2, edgecolor='g', facecolor='none')
        rect = Rectangle(xy=(prediction[0, 0] * res_1, prediction[0, 1] * res_2),
                         width=(prediction[0, 2] - prediction[0, 0]) * res_1,
                         height=(prediction[0, 3] - prediction[0, 1]) * res_2, linewidth=2, edgecolor='g',
                         facecolor='none')
        ax[r, c].add_patch(rect)
    plt.show()


# displays test results
results = model.evaluate(pred_images, pred_targets)
print(print("LOSS, ACCURACY=", results))
plot_pred(test_image_fullpaths, pred_images)
