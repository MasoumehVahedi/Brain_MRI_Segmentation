import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from data_preprocessing import ImageLoading, show_image
from Attention_UNet_model import Attention_UNet
from util import jacard_coef



def positive_negative_diagnosis(file_masks):
  mask = cv2.imread(file_masks)
  value = np.max(mask)
  if value > 0:
    return 1
  else:
    return 0


def data_loader(path):
    # Separate images and masks in the directory
    masks_dir = glob(path + "/*/*_mask*")
    images_dir = []
    for img in masks_dir:
        images_dir.append(img.replace("_mask", ""))

    # Create a new datafarme for brain images and masks
    data_brain = pd.DataFrame(data={
        "file_images": images_dir,
        "file_masks": masks_dir
    })
    print(data_brain.head())

    data_brain["mask"] = data_brain["file_masks"].apply(lambda x: positive_negative_diagnosis(x))
    print(data_brain.head())
    print(data_brain["mask"].value_counts())

    # Show some images and masks
    show_image(data_brain)

    image_loader = ImageLoading(images_dir, masks_dir)
    images_train = image_loader.resize_images()
    masks_train = image_loader.resize_masks()

    # Normalize images
    images_train = np.array(images_train) / 255.
    masks_train = np.expand_dims((np.array(masks_train)), 3) / 255.

    return images_train, masks_train



def train_model(path):
    # Load images and masks
    images_train, masks_train = data_loader(path)
    X_train, X_test, Y_train, Y_test = train_test_split(images_train, masks_train, test_size=0.2, random_state=42)

    print("X_train shape = {}".format(X_train.shape))
    print("X_test shape = {}".format(X_test.shape))
    print("Y_train shape = {}".format(Y_train.shape))
    print("Y_test shape = {}".format(Y_test.shape))

    # Set hyperparameters
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    # Binary class
    NUM_CLASS = 1
    EPOCHS = 25
    BATCH_SIZE = 8
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = Attention_UNet(input_shape=input_shape)
    attention_unet_model = model.build_attention_unit()
    attention_unet_model.compile(optimizer="adam",
                                 loss="binary_crossentropy",
                                 metrics=["accuracy", jacard_coef])

    history = attention_unet_model.fit(X_train, Y_train,
                                       verbose=1,
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE,
                                       validation_data=(X_test, Y_test),
                                       shuffle=False)

    # Save the model
    attention_unet_model.save("/input/attention_unet_model.hdf5")

    # Check and plot the loss and accuracy
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('loss')
    plt.xlabel('epoch')

    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Acuracy')
    plt.xlabel('epoch')

    plt.show()




if __name__== "__main__":
    path = "/input/kaggle_3m"
    train_model(path)