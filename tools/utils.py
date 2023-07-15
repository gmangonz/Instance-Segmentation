from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from tqdm import tqdm
import numpy as np


###### Save to TFRecord ######

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_image_to_tfexample(image_path, label='image'):

    """Converts an image file to a tf.train.Example."""
    assert label in ['image', 'mask']

    with open(image_path, 'rb') as f:
        image_data = f.read()

    image = Image.open(image_path)
    width, height = image.size
    feature_dict = {
                    f'{label}/encoded': _bytes_feature(image_data),
                    f'{label}/width': _int64_feature(width),
                    f'{label}/height': _int64_feature(height),
                    f'{label}/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(image_path))),
                    }
    return feature_dict

def save_images_to_tfrecord(image_dir, mask_dir, output_file):

    """Saves images in a directory to a single TFRecord file."""
    
    image_paths = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))
    num_images = len(image_paths)
    
    mask_paths = tf.io.gfile.glob(os.path.join(mask_dir, '*.gif'))
    num_masks = len(mask_paths)

    assert num_images == num_masks

    with tf.io.TFRecordWriter(output_file) as writer:
        for i in tqdm(range(num_images)):

            image_feature_dict = convert_image_to_tfexample(image_paths[i], label='image')
            mask_feature_dict = convert_image_to_tfexample(mask_paths[i], label='mask')
            feature_dict = dict(**image_feature_dict, **mask_feature_dict)

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())

    print(f'All images saved to: {output_file}')


###### Load from TFRecord ######

def read_tfrecord(tfrecord_path, return_values=False):

    feature_dict = {}
    for rec in tf.data.TFRecordDataset([str(tfrecord_path)]):

        example_bytes = rec.numpy()
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        
        for key in example.features.feature:

            feature = example.features.feature[key]
            if feature.HasField('bytes_list'):
                values = feature.bytes_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.string)
            elif feature.HasField('float_list'):
                values = feature.float_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.float32)
            elif feature.HasField('int64_list'):
                values = feature.int64_list.value if return_values else tf.io.FixedLenFeature([], dtype=tf.int64)
            else:
                values = feature.WhichOneof('kind')

            feature_dict[key] = values
    return feature_dict


def decode_image(parsed_features, image_shape, label):

    image = tf.image.decode_image(parsed_features[f'{label}/encoded'], channels = 3)    
    h = parsed_features[f'{label}/height']
    w = parsed_features[f'{label}/width']
    image = tf.reshape(image, shape=[h, w, -1])
    image = tf.image.rgb_to_grayscale(image) if label == 'mask' else image
    image = tf.image.resize(image, size=image_shape, method='nearest') / 255
    return image

def _parse_features(example_proto, image_shape, features):
    
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image = decode_image(parsed_features, image_shape, 'image')
    mask = decode_image(parsed_features, image_shape, 'mask')

    return image, mask


###### Display images ######

def visualize_outputs(figsize, data, save_to=None):

    imgs, masks = data[0], data[1]
    num_images = len(imgs)
    titles = ['Image {}'.format(i+1) for i in range(num_images)]
    
    rows = np.floor(np.sqrt(num_images))
    cols = np.ceil(num_images / rows)

    fig, axes = plt.subplots(int(rows), int(cols), figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image = imgs[i]
            ax.imshow(np.array(image))

            ax.axis('off')
            ax.set_title(titles[i])
        else:
            ax.axis('off')
    plt.tight_layout()
    if save_to != None:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def display_imgs(*input_imgs):

    display_list = []
    for item in input_imgs:
        if len(item.shape) > 3:
            item = item[0]
        display_list.append(item)
    plt.figure(figsize=(8, 8))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(f"Img {i}")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def display_train_mask(row, input_shape=None):

    if input_shape == None:
        input_shape=[256, 256, 3] # (1280, 1918, 3)
    else:
        assert len(input_shape) == 3,'Input shape needs to have HWC, i.e [H, W, C]'
    if type(input_shape) == tuple:
        input_shape = list(input_shape)
    
    img = img_to_array(load_img(row['path'], target_size=input_shape))/255
    mask = img_to_array(load_img(row['mask_path'], target_size=input_shape[:-1], color_mode = 'grayscale')).reshape(input_shape[:-1] + [1])/255

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,9))
    ax1.imshow(img)
    ax1.set_title('Input Car Image')
    ax1.axis('off')
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Segmentation Map')
    ax2.axis('off')

    return img, mask # Returns 2 arrays
