import math
import numpy as np
import os
from random import shuffle
from imageio import imread
from skimage.transform import resize
from preprocessing.data_preprocessing import adjust_contrast, adjust_hue, adjust_saturation, adjust_brightness, \
  horizontal_flip, ssd_random_crop, preprocess_input

class Generator(object):
  def __init__(self,
               gt, bbox_util, train_batch_size, val_batch_size,
               train_image_paths, val_image_paths,
               image_size, num_classes,
               mean_subtraction, scaling, data_aug_config):
    self.gt = gt
    self.bbox_util = bbox_util
    self.train_batch_size = train_batch_size


    self.val_batch_size = val_batch_size
    self.train_image_paths = train_image_paths
    self.val_image_paths = val_image_paths
    self.train_batches = int(math.ceil(float(len(train_image_paths)) / float(train_batch_size)))
    self.val_batches = int(math.ceil(float(len(val_image_paths)) / float(val_batch_size)))
    self.image_size = image_size
    self.num_classes = num_classes
    self.mean_subtraction = mean_subtraction
    self.scaling = scaling

    # Parse the data augmentation parameters
    self.hflip_prob = 0.
    if 'horizontal_flip' in data_aug_config:
      self.hflip_prob = data_aug_config['horizontal_flip']['flip_prob']

    self.saturation_prob = 0.
    if 'saturation' in data_aug_config:
      saturation_config = data_aug_config['saturation']
      self.saturation_prob = saturation_config['saturation_prob']
      self.saturation_lower = saturation_config['saturation_lower']
      self.saturation_upper = saturation_config['saturation_upper']

    self.brightness_prob = 0.
    if 'brightness' in data_aug_config:
      brightness_config = data_aug_config['brightness']
      self.brightness_prob = brightness_config['brightness_prob']
      self.brightness_delta = brightness_config['brightness_delta']

    self.contrast_prob = 0.
    if 'contrast' in data_aug_config:
      contrast_config = data_aug_config['contrast']
      self.contrast_prob = contrast_config['contrast_prob']
      self.contrast_lower = contrast_config['contrast_lower']
      self.contrast_upper = contrast_config['contrast_upper']

    self.hue_prob = 0.
    if 'hue' in data_aug_config:
      hue_config = data_aug_config['hue']
      self.hue_prob = hue_config['hue_prob']
      self.hue_delta = hue_config['hue_delta']

    if 'ssd_random_crops' in data_aug_config and len(data_aug_config['ssd_random_crops']) > 0:
      self.ssd_random_crops = data_aug_config['ssd_random_crops']

  def generate(self, train=True):
    while True:
      if train:
        shuffle(self.train_image_paths)
        keys = self.train_image_paths
      else:
        shuffle(self.val_image_paths)
        keys = self.val_image_paths
      inputs = []
      targets = []
      for img_path in keys:  # Either train or val image paths
        img = imread(img_path)

        # Check if this image has any detections, if not, append an empty array
        if os.path.basename(img_path) in self.gt:
          y = self.gt[os.path.basename(img_path)].copy()
        else:
          y = np.zeros((0, 4 + self.num_classes + 8))

        # Do random crops with the SSD method
        if train:
          img, y = ssd_random_crop(img, y, self.ssd_random_crops)
        img = resize(img, self.image_size, preserve_range=True)

        if train:
          # Do photometric transformations
          order_prob = np.random.random()
          if order_prob < 0.5:
            img = adjust_brightness(img, brightness_prob=self.brightness_prob,
                                    brightness_delta=self.brightness_delta)
            img = adjust_contrast(img, contrast_prob=self.contrast_prob,
                                  contrast_lower=self.contrast_lower,
                                  contrast_upper=self.contrast_upper)
            img = adjust_saturation(img, saturation_prob=self.saturation_prob,
                                    saturation_lower=self.saturation_lower,
                                    saturation_upper=self.saturation_upper)
            img = adjust_hue(img, hue_prob=self.hue_prob,
                             hue_delta=self.hue_delta)
          else:
            img = adjust_brightness(img, brightness_prob=self.brightness_prob,
                                    brightness_delta=self.brightness_delta)
            img = adjust_saturation(img, saturation_prob=self.saturation_prob,
                                    saturation_lower=self.saturation_lower,
                                    saturation_upper=self.saturation_upper)
            img = adjust_hue(img, hue_prob=self.hue_prob,
                             hue_delta=self.hue_delta)
            img = adjust_contrast(img, contrast_prob=self.contrast_prob,
                                  contrast_lower=self.contrast_lower,
                                  contrast_upper=self.contrast_upper)
          img, y = horizontal_flip(img, y, hflip_prob=self.hflip_prob)

        # Do mean subtraction and normalization
        img = preprocess_input(img, self.mean_subtraction, self.scaling)
        y = self.bbox_util.assign_boxes(y)
        inputs.append(img)
        targets.append(y)
        if (train and (len(targets) == self.train_batch_size)) or \
            (not train and (len(targets) == self.val_batch_size)):
          tmp_inp = np.array(inputs)
          tmp_targets = np.array(targets)
          inputs = []
          targets = []
          yield tmp_inp, tmp_targets
