"""Provides data for the mturbo datasets.
"""
import os

import pandas as pd
import tensorflow as tf


slim = tf.contrib.slim

_FILE_PATTERN = 'mturbo_%s.tfrecord'

_SPLITS_TO_SIZES = {'train': 98316, 'valid': 54642, 'test': 40042}

_NUM_CLASSES = 3

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'image filepath',
    'label': 'A single integer between 0 and 2',
}

_IMAGE_SIZE = 96

# The bounding box used to crop the mturbo image.
_BBOX = [16, 128, 355, 256]

_LABELS_TO_NAME = {0: 'NORMAL', 1: '4C', 2: 'SA'}

class Image(slim.tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                 filename_key,
                 image_size,
                 split_name,
                 channels=1,
                 dtype=tf.uint8):
        """Initialize the image.
        Args:
            filename_key: the name of the TF-Example feature in which the
                filename is encoded.
            image_size: the size of the image after resizing.
            split_name: A string that specifies the name of the split.
            channels: the number of channels in the image.
            dtype: images will be decoded at this bit depth. Different formats
                support different bit depths.
        """
        super(Image, self).__init__(filename_key)
        self._channels = channels
        self._dtype = dtype
        self._image_size = _IMAGE_SIZE
        self._split_name = split_name

    def tensors_to_item(self, keys_to_tensors):
        """Read and preprocess the image given the filename.
        We crop the image if the split name contains ``mturbo``
            and then resize the image to self._image_size.
        """
        filename = keys_to_tensors[self._keys[0]]

        file_contents = tf.read_file(filename)
        image = tf.image.decode_png(file_contents, channels=1, dtype=tf.uint8)
        image = tf.image.crop_to_bounding_box(
                image, _BBOX[0], _BBOX[1], _BBOX[2], _BBOX[3])
        image = tf.image.resize_images(image, [self._image_size, self._image_size])

        return image


def get_split(split_name, dataset_dir='/data2/tmp/imalkin/',
              num_classes=_NUM_CLASSES):
    """Get a `Dataset` struct for reading the Aqua SF Horz dataset.
    Args:
        split_name: A train/test split name.
        image_size: The size of the image after resizing.
        dataset_dir: The base directory of the dataset resources.
        num_classes: An integer as the number of classes.
    Returns:
        A `Dataset` named tuple.
    """
    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
    print(file_pattern)

    keys_to_features = {
        'image/image_filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': Image('image/image_filename', _IMAGE_SIZE, split_name),
        'label':
            slim.tfexample_decoder.Tensor('image/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers,
    )

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_classes,
        labels_to_names=_LABELS_TO_NAME)