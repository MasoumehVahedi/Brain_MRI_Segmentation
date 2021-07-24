import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage import io
import cv2



########################## Load all images and masks #################################
class ImageLoading():

  def __init__(self, img_path, mask_path):
    self.img_path = img_path
    self.mask_path = mask_path
    self.IMG_HEIGHT = 256
    self.IMG_WIDTH = 256
    # The number of classes for segmentation (is a binary problem)
    self.NUM_CLASSES = 1

    # Load images
    self.images_training = self.resize_images()
    print(self.images_training.shape)
    # Load masks
    self.masks_training = self.resize_masks()
    print(self.masks_training.shape)


  # resise the image
  def resize_images(self):
    images_training = []
    for imagePath in self.img_path:
      image = cv2.imread(imagePath)
      image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
      images_training.append(image)
    # Convert to numpy array
    images_training = np.array(images_training)
    return images_training

  # resise the mask
  def resize_masks(self):
    masks_training = []
    for maskPath in self.mask_path:
      mask = cv2.imread(maskPath, 0)
      mask = cv2.resize(mask, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
      masks_training.append(mask)
    # Convert to numpy array
    masks_training = np.array(masks_training)
    return masks_training


######################### Data Visualization #############################
def show_image(df):
  fig, ax = plt.subplots(nrows=10, ncols=3, figsize=(20, 40))
  count = 0
  i = 0
  for mask in df["mask"]:
    if mask == 1:
      # Show images
      image = io.imread(df.file_images[i])
      ax[count][0].title.set_text("Brain MRI")
      ax[count][0].imshow(image)

      # Show masks
      mask = io.imread(df.file_masks[i])
      ax[count][1].title.set_text("Mask Brain MRI")
      ax[count][1].imshow(mask, cmap="gray")

      # Show MRI Brain with mask
      image[mask == 255] = (0, 255, 150)    # Here, we want to modify the color of pixel at the position of mask
      ax[count][2].title.set_text("MRI Brain with mask")
      ax[count][2].imshow(image)
      count += 1
    i += 1
    if count == 10:
      break
  fig.tight_layout()
  plt.show()