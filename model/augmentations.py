import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Note: layers.Layer sublasses work best with call() rather than __call__()

class RandomGamma(layers.Layer):

  def __init__(self, 
               gamma=0.4, 
               gain=0.75, 
               **kwargs):

    super(RandomGamma, self).__init__(**kwargs)

    self.gamma = gamma
    self.gain = gain

  def call(self, inputs):

    img, mask = inputs 

    gamma = (0, self.gamma)
    gamma = tf.random.uniform(shape=(), minval=gamma[0], maxval=gamma[1])

    gain = (self.gain/2, self.gain)
    gain = tf.random.uniform(shape=(), minval=gain[0], maxval=gain[1]) 

    img_gamma = tf.image.adjust_gamma(img, gamma=1+gamma, gain=gain)
    mask_gamma = tf.image.adjust_gamma(mask, gamma=1+gamma, gain=gain)

    return img_gamma, mask_gamma


class RandomHorizontalFlip(layers.Layer):

  def __init__(self,
                seed=42,
                **kwargs):

    super(RandomHorizontalFlip, self).__init__(**kwargs)
    
    self.flip_img = layers.RandomFlip(mode='vertical', seed=seed, **kwargs)
    self.flip_mask = layers.RandomFlip(mode='vertical', seed=seed, **kwargs)
    
  def get_batch_wise(self, imgs, masks, rand):
        
      batch_size = tf.shape(imgs)[0]
      augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
      augmentation_bools = tf.math.less(augmentation_values, rand)
      flipped_imgs, flipped_masks = self.flip_img(imgs), self.flip_mask(masks)
      imgs = tf.where(augmentation_bools, flipped_imgs, imgs)
      masks = tf.where(augmentation_bools, flipped_masks, masks)
      return imgs, masks

  def call(self, inputs): # img: (h, w, c)

    imgs, masks, rand = inputs[0], inputs[1], inputs[2]
    return self.get_batch_wise(imgs, masks, rand)

  
class RandomMirror(layers.Layer):

    def __init__(self,
                 **kwargs):
      
      super(RandomMirror, self).__init__(**kwargs)
    
    def get_batch_wise(self, imgs, masks, rand):
        
      batch_size = tf.shape(imgs)[0]
      augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
      augmentation_bools = tf.math.less(augmentation_values, rand)
      mirrored_imgs, mirrored_masks = imgs[:, :, ::-1], masks[:, :, ::-1]
      imgs = tf.where(augmentation_bools, mirrored_imgs, imgs)
      masks = tf.where(augmentation_bools, mirrored_masks, masks)
      return imgs, masks

    def call(self, inputs):

      imgs, masks, rand = inputs[0], inputs[1], inputs[2]
      return self.get_batch_wise(imgs, masks, rand)


class RandomZoom(layers.Layer):

    def __init__(self,
                 height_factor = 0.2, 
                 width_factor = 0.2, 
                 seed = 42,
                 **kwargs):

        super(RandomZoom, self).__init__(**kwargs)

        self.zoom_img = layers.RandomZoom(height_factor=height_factor, width_factor=width_factor, fill_mode='nearest', interpolation='nearest', seed=seed, fill_value=0.0)
        self.zoom_mask = layers.RandomZoom(height_factor=height_factor, width_factor=width_factor, fill_mode='nearest', interpolation='nearest', seed=seed, fill_value=0.0)

    def get_batch_wise(self, imgs, masks, rand):
        
      batch_size = tf.shape(imgs)[0]
      augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
      augmentation_bools = tf.math.less(augmentation_values, rand)
      zoomed_imgs, zoomed_masks = self.zoom_img(imgs), self.zoom_mask(masks)
      imgs = tf.where(augmentation_bools, zoomed_imgs, imgs)
      masks = tf.where(augmentation_bools, zoomed_masks, masks)
      return imgs, masks

    def call(self, inputs):

        imgs, masks, rand = inputs[0], inputs[1], inputs[2]
        return self.get_batch_wise(imgs, masks, rand)
    

class RandomRotate(layers.Layer):

  def __init__(self,
               rot=0.3,
               seed=42,
               **kwargs):

    super(RandomRotate, self).__init__(**kwargs)
    
    self.rotate_img = layers.RandomRotation(factor=rot, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)
    self.rotate_mask = layers.RandomRotation(factor=rot, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)

  def get_batch_wise(self, imgs, masks, rand):
       
    batch_size = tf.shape(imgs)[0]
    augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
    augmentation_bools = tf.math.less(augmentation_values, rand)
    rotated_imgs, rotated_masks = self.rotate_img(imgs), self.rotate_mask(masks)
    imgs = tf.where(augmentation_bools, rotated_imgs, imgs)
    masks = tf.where(augmentation_bools, rotated_masks, masks)
    return imgs, masks

  def call(self, inputs):

    imgs, masks, rand = inputs[0], inputs[1], inputs[2]
    return self.get_batch_wise(imgs, masks, rand)


class RandomShift(layers.Layer):

  def __init__(self,
               translate=0.2, 
               seed=42,
               **kwargs):

    super(RandomShift, self).__init__(**kwargs)

    self.translate_img = layers.RandomTranslation(translate, translate, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)
    self.translate_mask = layers.RandomTranslation(translate, translate, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)

  def get_batch_wise(self, imgs, masks, rand):
       
    batch_size = tf.shape(imgs)[0]
    augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
    augmentation_bools = tf.math.less(augmentation_values, rand)
    shifted_imgs, shifted_masks = self.translate_img(imgs), self.translate_mask(masks)
    imgs = tf.where(augmentation_bools, shifted_imgs, imgs)
    masks = tf.where(augmentation_bools, shifted_masks, masks)
    return imgs, masks

  def call(self, inputs): # img: (h, w, c)

    imgs, masks, rand = inputs[0], inputs[1], inputs[2]
    return self.get_batch_wise(imgs, masks, rand)
  

class RandomBox(layers.Layer):

    def __init__(self, 
                 min_height, 
                 min_width, 
                 max_height, 
                 max_width, 
                 **kwargs):
        
        super(RandomBox, self).__init__(**kwargs)

        self.min_height = min_height
        self.min_width = min_width
        self.height = max_height
        self.width = max_width

    def make_boxes(self, inputs): # (h, w, c), (h, w, c)

        img, masks = inputs
        img_shape = tf.shape(img)

        block_height = tf.random.uniform(shape=[], minval=self.min_height, maxval=self.height, dtype=tf.int32)
        block_width = tf.random.uniform(shape=[], minval=self.min_width, maxval=self.width, dtype=tf.int32)

        pad_h = img_shape[0] - block_height
        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
        pad_bottom = pad_h - pad_top

        pad_w = img_shape[1] - block_width
        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
        pad_right = pad_w - pad_left
        
        box_mean = tf.math.reduce_mean(img[pad_top:pad_top+block_height, pad_left:pad_left+block_width, :])
        block = tf.ones(shape=[block_height, block_width, 3]) * box_mean
        block = tf.cast(block, img.dtype)
        block_img = tf.pad(block, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=1)

        return block_img, masks

    def call(self, inputs): # (bs, h, w, c), (bs, h, w, c)
        
        imgs, masks = inputs

        block_img, masks = tf.map_fn(self.make_boxes, elems=(imgs, masks)) # Make it so that images in the batch have different boxes.   
        blocked_img = tf.multiply(tf.cast(imgs, tf.float32), tf.cast(block_img, tf.float32))
        blocked_img = tf.cast(blocked_img, imgs.dtype)
    
        return blocked_img, masks

class OverlayBox(layers.Layer):

    def __init__(self, 
                 min_height, 
                 min_width, 
                 height, 
                 width, 
                 **kwargs):

        super(OverlayBox, self).__init__(**kwargs)
        self.random_box = RandomBox(min_height, min_width, height, width)

    def get_batch_wise(self, imgs, masks, rand):
       
        batch_size = tf.shape(imgs)[0]
        augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        augmentation_bools = tf.math.less(augmentation_values, rand)
        boxed_imgs, masks = self.random_box((imgs, masks))
        imgs = tf.where(augmentation_bools, boxed_imgs, imgs)
        return imgs, masks

    def call(self, inputs): # (bs, h, w, c), (bs, h, w, c)
        
        imgs, masks, rand = inputs[0], inputs[1], inputs[2]
        return self.get_batch_wise(imgs, masks, rand)
    

class AddNoise(layers.Layer):

    def __init__(self, 
                 **kwargs):

        super(AddNoise, self).__init__(**kwargs)

    def get_noisy_img(self, imgs):
       
        noise = tf.random.normal(tf.shape(imgs), mean=0, stddev=0.075, dtype=imgs.dtype)
        img_noisy = tf.clip_by_value(imgs + noise, 0.0, 1.0)
        img_noisy = tf.cast(img_noisy, imgs.dtype)
        return img_noisy

    def get_batch_wise(self, imgs, masks, rand):
       
        batch_size = tf.shape(imgs)[0]
        augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        augmentation_bools = tf.math.less(augmentation_values, rand)
        imgs = tf.where(augmentation_bools, self.get_noisy_img(imgs), imgs)
        return imgs, masks

    def call(self, inputs):

        imgs, masks, rand = inputs[0], inputs[1], inputs[2]
        return self.get_batch_wise(imgs, masks, rand)


class GaussianBlur(layers.Layer):

    def __init__(self, 
                 kernel_size, 
                 sigma, 
                 **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.sigma = sigma

    def gaussian_kernel(self, size=3, sigma=1):

        x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
        y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs**2 + ys**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))
        return tf.cast( kernel / tf.reduce_sum(kernel), tf.float32)

    def blur_image(self, img):
       
        kernel = self.gaussian_kernel(self.kernel_size, self.sigma)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
        
        r, g, b = tf.split(img, [1,1,1], axis=-1)
        r_blur = tf.nn.conv2d(r, kernel, [1,1,1,1], 'SAME', name='r_blur')
        g_blur = tf.nn.conv2d(g, kernel, [1,1,1,1], 'SAME', name='g_blur')
        b_blur = tf.nn.conv2d(b, kernel, [1,1,1,1], 'SAME', name='b_blur')

        blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
        blur_image = tf.cast(blur_image, img.dtype)

        return blur_image

    def get_batch_wise(self, imgs, masks, rand):
       
        batch_size = tf.shape(imgs)[0]
        augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        augmentation_bools = tf.math.less(augmentation_values, rand)
        imgs = tf.where(augmentation_bools, self.blur_image(imgs), imgs)
        return imgs, masks

    def call(self, inputs):

        imgs, masks, rand = inputs[0], inputs[1], inputs[2]
        return self.get_batch_wise(imgs, masks, rand)
    

class AugProbability(layers.Layer):
   
    def __init__(self, **kwargs):
      
      super(AugProbability, self).__init__(**kwargs)
      self.probability = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="p")
      
    def call(self, inputs): # (1, 1) e.g. [[x]]
       
       self.probability.assign(inputs[0][0])

       return self.probability


class Ada(tf.keras.Model):

    def __init__(self, 
                 img_size=(128, 128),
                 aug_functions=[],
                 initial_probability = 0.0,
                 switch = True,
                 **kwargs):
        
        super(Ada, self).__init__(**kwargs)
        
        self.probability = tf.Variable([[initial_probability]], name='ada_p')
        self.switch = switch
        self.augmenter = build_augmenter(aug_functions, img_size)
    
    def call(self, inputs, training=False):

        imgs, masks = inputs[0], inputs[1] # [bs, H, W, C], [bs, H, W, C]

        if training and self.switch:
          
          return self.augmenter((imgs, masks, self.probability), training)

        return inputs

    def update(self, loss): # Loss can be tf.constant of any shape

        # The more accurate the model, the probability of augmentations is higher.
        accuracy = K.mean(1. - loss)
        factor = (tf.math.exp(3. * accuracy) - 1) / (3. * tf.math.exp(accuracy) + 9.) # Arbitray function to map accuracy to 0-1 range.
        self.probability.assign( [[tf.clip_by_value(factor, 0.0, 1.0)]] )


def build_augmenter(aug_functions, img_size):

    if len(img_size) == 2:
        img_shape = img_size + (3,)
        mask_shape = img_size + (1,)

    if len(img_size) == 3:
        img_shape = img_size
        mask_shape = (img_size[0], img_size[1], 1)

    input_img = layers.Input(shape=img_shape) # (H, W, C)
    input_mask = layers.Input(shape=mask_shape)
    input_rand = layers.Input(shape=(1, ), batch_size=1)

    p = AugProbability()(input_rand)
    x = (input_img, input_mask, p)
    for i, func in enumerate(aug_functions):
        out_img, out_mask = func(x)

        if i <= len(aug_functions) - 1:
          x = (out_img, out_mask, p)

    augment_model = tf.keras.Model([input_img, input_mask, input_rand], [out_img, out_mask], name='obj_det_data_augmentation_function')
    return augment_model