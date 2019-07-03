import os
import os.path as osp
import sys
import cPickle as pickle
import numpy as np
import json
import cv2
import tensorflow as tf
from tqdm import tqdm

import pdb

CLASSES = ['chair', 'table', 'sofa_chair', 'desk', 'cabinet', 'bed', 'sofa', 
            'garbage_bin', 'door', 'shelf', 'counter', 'sink']

label2id = dict(map(reversed, enumerate(CLASSES)))

IMAGE_SHAPE = 100

def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print file_name
    full_name = os.path.join(data_path, file_name)
    with open(full_name) as finp:
      data = pickle.load(finp)
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_data(data_path, num_valids=5000):
  print "-" * 80
  print "Reading data"

  images, labels = {}, {}

  # if exist cache file, read from cache
  if 'images_dict_of_array.npz' in os.listdir(data_path) and 'labels_dict_of_array.npz' in os.listdir(data_path):
    print('exist cached file, recover from file')
    images_npz = np.load(osp.join(data_path, 'images_dict_of_array.npz'))
    for key in images_npz.files:
      images[key] = images_npz[key]

    labels_npz = np.load(osp.join(data_path, 'labels_dict_of_array.npz'))
    for key in labels_npz.files:
      labels[key] = labels_npz[key]
    print('recover finished')

    return images, labels

  # no cache
  def _load_annotation(file_path):
    with open(file_path, 'r') as f:
      anns = json.load(f)

    return anns

  tasks = ['train', 'test', 'valid']
  for task in tasks:
    print('reading {} data'.format(task))
    task_labels = []
    task_images = []
    # 1. read annotaion file
    anns = _load_annotation(osp.join(data_path, 'annotations/{}s_ann.json'.format(task)))
    # 2. for each ann, read label, read image
    for ann in tqdm(anns[:4000]):
      if ann['label'] not in CLASSES:
        continue

      task_labels.append(np.expand_dims(label2id[ann['label']], 0))
      img = cv2.imread(osp.join(data_path, 'images', ann['filename'])).astype(np.float32)
      # 3. post_processing: 
      # 3_a. resize image to form a [n, h, w, 3] np.ndarray
      img = cv2.resize(img, (IMAGE_SHAPE, IMAGE_SHAPE), interpolation=cv2.INTER_LINEAR)

      # TODO: read depth image and concatenate with img along axis 2
      task_images.append(np.expand_dims(img, 0))

    task_labels = np.concatenate(task_labels, axis=0).astype(np.int32)
    task_images = np.concatenate(task_images, axis=0)

    labels[task] = task_labels
    print('labels shape: {}'.format(task_labels.shape))
    images[task] = task_images
    print('images shape: {}'.format(task_images.shape))
    
  # 3_b. substrace mean, divide std
  print "Prepropcess: [subtract mean], [divide std]"
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)
  print "mean: {}".format(np.reshape(mean, [-1]))
  print "std: {}".format(np.reshape(std, [-1]))

  images["train"] = (images["train"] - mean) / std
  images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  print('cache images and labels to file')
  np.savez(osp.join(data_path, 'images_dict_of_array.npz'), **images)
  np.savez(osp.join(data_path, 'labels_dict_of_array.npz'), **labels)
  print('cache finished')

  return images, labels

