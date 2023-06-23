import os
import pandas as pd
import glob
import numpy as np
from absl import logging
import argparse
import tensorflow as tf
from tools.utils import _parse_features, read_tfrecord

# parser = argparse.ArgumentParser()
# parser.add_argument('--base_path', '-bp', help="path to use as base", default=base_path)
# parser.add_argument('--orig_width', '-w', help='original width', type=int, default=1918)
# parser.add_argument('--orig_height', '-h', help='original height', type=int, default=1280)
# # parser.add_argument('--path', '-p', help='path directory for data', default=os.getcwd())
# # parser.add_argument('--name', '-n', help='name for data directory', default='data')
# # parser.add_argument('--anchors', '-a', help='temp', type=int, default=1)
# # parser.add_argument('--classes', '-c', help='classes to predict', type=int, default=20)
# # parser.add_argument('--projection_dim', '-pd', help='dimension to project patches', type=int, default=32)
# # parser.add_argument('--num_heads', '-nh', help='num heads to use in ViT', type=int, default=4)
# # parser.add_argument('--patch_size_ratio', '-psr', help='ratio between img size and patch size', type=int, default=16)
# args = parser.parse_args()


def get_data_df(args):

    train_imgs_path = os.path.join(args.base_path, 'train', 'train')
    train_masks_path = os.path.join(args.base_path, 'train_masks', 'train_masks')
    
    train_masks_csv = os.path.join(args.base_path, r'train_masks.csv\train_masks.csv')
    try:
        df_train_mask = pd.read_csv(train_masks_csv)
    except FileNotFoundError:
        print(f"{train_masks_csv} is not a valid path")

    df_train_mask['img_id'] = df_train_mask['img'].map(lambda s: s.split('.')[0])
    df_train_mask = df_train_mask.set_index('img_id')

    all_img_df = pd.DataFrame(dict(path = sorted(glob.glob(os.path.join(train_imgs_path, '*.*')))))
    all_mask_df = pd.DataFrame(dict(mask_path = sorted(glob.glob(os.path.join(train_masks_path, '*.*')))))
    df_train = pd.concat([all_img_df, all_mask_df], axis=1)

    df_train['key_id'] = df_train['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0]) # Get the file name: ..\\00087a6bd4dc_01.jpg -> 00087a6bd4dc_01
    df_train['car_id'] = df_train['key_id'].map(lambda x: x.split('_')[0]) # Get car id: 00087a6bd4dc_01 -> 00087a6bd4dc
    df_train['exists'] = df_train['mask_path'].map(os.path.exists) # Identify if files exist exists
    df_train = df_train.set_index('key_id')
    df_train['rle_mask'] = df_train_mask['rle_mask']

    return df_train, df_train_mask


def get_train_val(df_train, aug=True):
    
    unique_cars = df_train.car_id.unique()
    np.random.shuffle(unique_cars)
    
    if aug == True:
        split = 0.55
    else:
        split = 0.8
    
    frac = int(len(unique_cars)*split)
    train_cars = unique_cars[:frac]
    val_cars = unique_cars[frac:]
    
    train = df_train[df_train.car_id.isin(train_cars)]
    val = df_train[df_train.car_id.isin(val_cars)]
    
    return train, val


def load_data(image_path, mask_path, img_size):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=img_size)
    image = image / 255. # tf.float32 [0, 1]


    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, expand_animations=False, channels=0)
    mask.set_shape([None, None, 3])
    mask = tf.image.resize(images=mask, size=img_size)
    mask = tf.image.rgb_to_grayscale(mask) / 255. # tf.float32 [0, 1]
    return image, mask


def data_generator(image_list, mask_list, split='train', img_size = (128, 128), ds_augment_func=None, batch_size=86):

    assert len(img_size) == 2, "img_size must be a tuple of length 2"

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.shuffle(8*batch_size) if split == 'train' else dataset 
    dataset = dataset.map(lambda x, y: load_data(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True) # [None, 2, H, W, C]
    if (ds_augment_func != None) and (split == 'train'):
        dataset = dataset.map(lambda x, y: ds_augment_func((x, y, tf.constant([0.5]))), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def data_generator_tfrecordFile(tfrecord_file, split='train', img_size=(128, 128), ds_augment_func=None, batch_size=86):

    assert len(img_size) == 2, "img_size must be a tuple of length 2"
    features = read_tfrecord(tfrecord_file, return_values=False)

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.shuffle(8*batch_size) if split == 'train' else dataset 
    dataset = dataset.map(lambda x: _parse_features(x, img_size, features), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if (ds_augment_func != None) and (split == 'train'):
        dataset = dataset.map(lambda x, y: ds_augment_func((x, y, tf.constant([0.5]))), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Use the prefetch transformation to overlap the work of a producer and consumer

# Parallelize the data reading transformation using the interleave transformation

# Parallelize the map transformation by setting the num_parallel_calls argument

# Use the cache transformation to cache data in memory during the first epoch

# Vectorize user-defined functions passed in to the map transformation -> Vectorize the user-defined function (that is, have it operate over a batch of inputs at once) and apply the batch transformation before the map transformation.

# Reduce memory usage when applying the interleave, prefetch, and shuffle transformations