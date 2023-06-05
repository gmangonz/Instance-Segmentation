import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow.keras import Sequential

class RandomHorizontalFlip(layers.Layer):

  def __init__(self,
               **kwargs):
  
    super(RandomHorizontalFlip, self).__init__(**kwargs)

  def __call__(self, inputs): # img: (h, w, c)

    img, mask = inputs
    img_flipped = img[:, ::-1, :] # Horizontal flip across the width (corresponds to x value)
    mask_flipped = mask[:, ::-1, :] # Horizontal flip across the width (corresponds to x value)

    return img_flipped, mask_flipped
  
class RandomMirror(layers.Layer):

    def __init__(self,
                **kwargs):
      
      super(RandomMirror, self).__init__(**kwargs)
    
    def __call__(self, inputs):

        img, mask = inputs
        img_mirror = img[:, ::-1]
        mask_mirror = mask[:, :, ::-1]

        return img_mirror, mask_mirror
    
class RandomZoom(layers.Layer):

  def __init__(self, 
               scale=0.25, 
               diff=True, 
               **kwargs):
  
    super(RandomZoom, self).__init__(**kwargs)
    self.scale = scale
    self.diff = diff
  
  @staticmethod
  def pad(resized_image, resized_shape, img_shape):
    
    pad_y = max(img_shape[0] - resized_shape[0], 0) # How much to pad height
    pad_y_before = pad_y//2
    pad_x = max(img_shape[1] - resized_shape[1], 0) # How much to pad width
    pad_x_before = pad_x//2

    paddings = [[pad_y_before, pad_y - pad_y_before], # Pad first channel (H)
                [pad_x_before, pad_x - pad_x_before], # Pad second channel (W)
                [           0,                    0]] # Pad third channel (C)
    
    resized_image = tf.pad(resized_image, paddings, "REFLECT") # Pad image to maintain original size
    return resized_image

  @staticmethod
  def zoom(img, scale_x, scale_y, img_shape):

    resize_scale_y = tf.cast( (1 + scale_y) * img_shape[0], tf.int32) # New y size
    resize_scale_x = tf.cast( (1 + scale_x) * img_shape[1], tf.int32) # New x size

    resized_image = tf.image.resize(tf.convert_to_tensor(img), (resize_scale_y, resize_scale_x), method='nearest') # Resize
    resized_image = RandomZoom.pad(resized_image, tf.shape(resized_image), img_shape)

    max_y = tf.cast(img_shape[0], dtype=tf.int32)
    max_x = tf.cast(img_shape[1], dtype=tf.int32)

    resized_image = resized_image[:max_y,:max_x,:] # Crop if needed

    return resized_image

  def __call__(self, inputs): # img: (h, w, c)

    img, mask = inputs

    scale = (max(-1, -self.scale), self.scale)
    img_shape = tf.shape(img) # h, w, c
    mask_shape = tf.shape(mask) # h, w, c
    assert img_shape == mask_shape, 'Image and mask image should be the same size'

    if self.diff:
        scale_xy = tf.random.uniform(shape=(2,), minval=scale[0], maxval=scale[1]) # Get values btwn 0-1 to use to scale x and y respectively
        scale_x = scale_xy[0]
        scale_y = scale_xy[1]
    else:
        scale_x = tf.random.uniform(shape=(1,), minval=scale[0], maxval=scale[1]) # Get value btwn 0-1 to use to scale both x and y
        scale_y = scale_x

    resized_image = RandomZoom.zoom(img, scale_x, scale_y, img_shape)
    resized_mask = RandomZoom.zoom(mask, scale_x, scale_y, mask_shape)

    return resized_image, resized_mask

class RandomTranslate(layers.Layer):

  def __init__(self, 
               translate = 0.2, 
               diff=True, **kwargs):
  
    super(RandomTranslate, self).__init__(**kwargs)
    self.translate = translate
    assert self.translate > 0 and self.translate < 1
    self.diff = diff

  def __call__(self, inputs): # img: (h, w, c)

    img, mask = inputs

    translate = (-self.translate, self.translate)
    img_shape = tf.shape(img) # h, w, c

    translate_xy = tf.random.uniform(shape=(2,), minval=translate[0]*img_shape[0], maxval=translate[1]*img_shape[1]) # Get values to use to translate x and y respectively 
    translate_x = translate_xy[0]
    translate_y = translate_xy[1]

    img_translate = tfa.image.translate(img, [-translate_x, -translate_y], interpolation='nearest', fill_mode='nearest')
    mask_translate = tfa.image.translate(mask, [-translate_x, -translate_y], interpolation='nearest', fill_mode='nearest')

    return img_translate, mask_translate

class RandomRotate(layers.Layer):

  def __init__(self, 
               rot = 60, 
               **kwargs):

    super(RandomRotate, self).__init__(**kwargs)

    self.angle = rot # Max angle to rotate

  def __call__(self, inputs):

    img, mask = inputs

    rotate = (-np.abs(self.angle), np.abs(self.angle))
    degree_angle = tf.random.uniform(shape=(), minval=rotate[0], maxval=rotate[1]) # Get values to use to translate x and y respectively

    img_Rotate = tfa.image.rotate(img, angles=degree_angle*(np.pi/180), interpolation='nearest', fill_mode='nearest')
    mask_Rotate = tfa.image.rotate(mask, angles=degree_angle*(np.pi/180), interpolation='nearest', fill_mode='nearest')

    return img_Rotate, mask_Rotate

class RandomGamma(layers.Layer):

  def __init__(self, gamma=0.4, gain=0.75, **kwargs):

    super(RandomGamma, self).__init__(**kwargs)

    self.gamma = gamma
    self.gain = gain

  def __call__(self, inputs):

    img, mask = inputs 

    gamma = (0, self.gamma)
    gamma = tf.random.uniform(shape=(), minval=gamma[0], maxval=gamma[1])

    gain = (self.gain/2, self.gain)
    gain = tf.random.uniform(shape=(), minval=gain[0], maxval=gain[1]) 

    img_gamma = tf.image.adjust_gamma(img, gamma=1+gamma, gain=gain)
    mask_gamma = tf.image.adjust_gamma(mask, gamma=1+gamma, gain=gain)

    return img_gamma, mask_gamma
  

def step(values): # "hard sigmoid", useful for binary accuracy calculation from logits. negative values -> 0.0, positive values -> 1.0  
    return 0.5 * (1.0 + tf.sign(values))

class Ada(layers.Layer):

    def __init__(self, 
                 translate = 0.2, 
                 translate_diff = False, 
                 rot = 60, 
                 scale = 0.25, 
                 zoom_diff = False,
                 **kwargs):
        
        super(Ada, self).__init__(**kwargs)
        
        self.probability = tf.Variable(0.0)
        self.target_accuracy = 0.85 # 0.85, 0.95
        self.integration_steps = 1500 # 1000, 2000

        self.augmenter = Sequential([layers.InputLayer(input_shape=(None, None, 3)), # (2, H, W, C)
                                     RandomHorizontalFlip(),
                                     RandomTranslate(translate, translate_diff),
                                     RandomRotate(rot),
                                     RandomZoom(scale, zoom_diff),
                                     RandomMirror()], name="ada")


    def call(self, inputs):

        imgs, masks, training = inputs # [bs, H, W, C], [bs, H, W, C], None

        if training:
          
          batch_size = tf.shape(imgs)[0]
          augmented_images = self.augmenter((imgs, masks), training)
          augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1, 1), minval=0.0, maxval=1.0) # Generate random numbers
          augmentation_bools = tf.math.less(augmentation_values, self.probability) # Get booleans in the indices where we want augmented images or not
          images = tf.where(augmentation_bools, augmented_images, imgs)

        return images

    def update(self, real_logits):

        current_accuracy = tf.reduce_mean(step(real_logits))
        # the augmentation probability is updated based on the dicriminator's accuracy on real images
        accuracy_error = current_accuracy - self.target_accuracy
        self.probability.assign( tf.clip_by_value(self.probability + accuracy_error / self.integration_steps, 0.0, 1.0) )








# class RandomShift(layers.Layer):

#   def __init__(self, translate=0.2, diff=True, **kwargs):

#     super(RandomShift, self).__init__(**kwargs)
#     self.translate = tf.constant(translate)
#     assert self.translate > 0 and self.translate < 1
#     self.diff = diff

#   def __call__(self, inputs): # img: (h, w, c)

#     img, mask = inputs

#     translate = (-self.translate, self.translate)
#     img_shape = tf.shape(img) # h, w, c

#     minval = translate[0]*tf.cast(img_shape[0], dtype=self.translate.dtype)
#     maxval = translate[1]*tf.cast(img_shape[1], dtype=self.translate.dtype)

#     translate_xy = tf.random.uniform(shape=(2,), minval=minval, maxval=maxval) # Get values to use to translate x and y respectively 
#     translate_x = translate_xy[0]
#     translate_y = translate_xy[1]

#     # tf.convert_to_tensor(img)
#     img_translate = tfa.image.translate(img, [-translate_x, -translate_y], interpolation='nearest', fill_mode='nearest')
#     mask_translate = tfa.image.translate(mask, [-translate_x, -translate_y], interpolation='nearest', fill_mode='nearest')

#     return img_translate, mask_translate

# class RandomRotate(layers.Layer):

#   def __init__(self, 
#                rot = 60, 
#                seed = 42,
#                **kwargs):

#     super(RandomRotate, self).__init__(**kwargs)
    
#     self.angle = rot # Max angle to rotate
#     self.rotate = layers.RandomRotation(factor=(0.6, 0.6), fill_mode='nearest', interpolation='nearest', seed=None, fill_value=0.0)
#     self.rotate_img = layers.RandomRotation(factor=0.3, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)
#     self.rotate_mask = layers.RandomRotation(factor=0.3, fill_mode='nearest', interpolation='nearest', fill_value=0.0, seed=seed)

#   def __call__(self, inputs):

#     img, mask = inputs

#     rotate = (-np.abs(self.angle), np.abs(self.angle))
#     degree_angle = tf.random.uniform(shape=(), minval=rotate[0], maxval=rotate[1]) # Get values to use to translate x and y respectively

#     img_rotate = self.rotate_img(img)
#     mask_rotate = self.rotate_mask(mask)
#     img_rotate = tfa.image.rotate(img, angles=degree_angle*(np.pi/180), interpolation='nearest', fill_mode='nearest')
#     mask_rotate = tfa.image.rotate(mask, angles=degree_angle*(np.pi/180), interpolation='nearest', fill_mode='nearest')

#     return img_rotate, mask_rotate

# class DoNothing(layers.Layer):

#     def __init__(self, **kwargs):
        
#         super(DoNothing, self).__init__(**kwargs)
    
#     def call(self, inputs):
#         return inputs

# class Conditional(layers.Layer):

#     def __init__(self, probability, **kwargs):

#         self.probability = probability
#         self.random_box = RandomBox(30, 30, 60, 60)
#         self.do_nothing = DoNothing()
#         super(Conditional, self).__init__(**kwargs)

#     def call(self, inputs):

#         return tf.where([tf.less(tf.random.uniform([]), self.probability)], RandomBox(30, 30, 60, 60), DoNothing())


# class RandomZoom(layers.Layer):

#   def __init__(self, scale=0.25, diff=True, **kwargs):
    

#     if isinstance(scale, tuple): scale=scale[0]
#     self.scale = scale
#     # self.diff = diff
#     super(RandomZoom, self).__init__(**kwargs)

    
#   @staticmethod
#   def pad(resized_image, resized_shape, old_shape):
    
#     pad_y = tf.math.maximum(old_shape[0] - resized_shape[0], 0) # How much to pad height
#     pad_y_before = pad_y//2

#     pad_x = tf.math.maximum(old_shape[1] - resized_shape[1], 0) # How much to pad width
#     pad_x_before = pad_x//2

#     paddings = [[           0,            0], # Pad batch channel (B)
#                 [pad_y_before, pad_y_before], # Pad first channel (H)
#                 [pad_x_before, pad_x_before], # Pad second channel (W)
#                 [           0,            0]] # Pad third channel (C)
    
#     padded_img = tf.pad(resized_image, paddings, "REFLECT") # Pad image to maintain original size
#     return padded_img

#   @staticmethod
#   def zoom(img, scale_h, scale_w, shape):

#     resize_scale_h = (1 + scale_h) * tf.cast(shape[1], dtype=scale_h.dtype)
#     resize_scale_w = (1 + scale_w) * tf.cast(shape[2], dtype=scale_w.dtype)

#     resize_scale_h = tf.cast( resize_scale_h, tf.int32 ) # New h size
#     resize_scale_w = tf.cast( resize_scale_w, tf.int32 ) # New w size

#     resized_image = tf.image.resize(tf.convert_to_tensor(img), (resize_scale_h, resize_scale_w), method='nearest') # Resize
#     resized_shape = tf.shape(resized_image)
#     resized_image = RandomZoom.pad(resized_image, resized_shape, shape)

#     max_h = tf.cast(shape[0], dtype=tf.int32)
#     max_w = tf.cast(shape[1], dtype=tf.int32)

#     resized_image = resized_image[:, :max_h, :max_w, :] # Crop if needed

#     return resized_image

#   def __call__(self, inputs): # img: (h, w, c)

#     img, mask = inputs

#     scale = (max(-1, -self.scale), self.scale)
#     img_shape = tf.shape(img) # bs, h, w, c
#     mask_shape = tf.shape(mask) # bs, h, w, c

#     # if self.diff:
#     scale_hw = tf.random.uniform(shape=(2,), minval=scale[0], maxval=scale[1]) # Get values btwn 0-1 to use to scale x and y respectively
#     scale_h = scale_hw[0]
#     scale_w = scale_hw[1]
#     # else:
#         # scale_x = tf.random.uniform(shape=(1,), minval=scale[0], maxval=scale[1]) # Get value btwn 0-1 to use to scale both x and y
#         # scale_y = scale_x

#     resized_image = RandomZoom.zoom(img, scale_h, scale_w, img_shape)
#     resized_mask = RandomZoom.zoom(mask, scale_h, scale_w, mask_shape)

#     return resized_image, resized_mask